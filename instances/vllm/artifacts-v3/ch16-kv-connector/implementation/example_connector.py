# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py
# ExampleConnector——官方 debug 参考实现（m14）：外部缓存=磁盘文件
# （safetensors），『第二个前缀缓存』的最小样例——文件系统就是外部缓存。
# inject/extract_kv_from_layer 示范 worker 直写池的 slot 寻址
# （block_id×block_size+offset）；get_num_new_matched_tokens/update_state_
# after_alloc 示范调度器侧最小实现（F 节对照）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 8 条 MLA 分支（isinstance(attn_metadata, MLACommonMetadata) 的
#     reshape 路径——L138-L145/L230-L232；非 MLA 主路径逐字保留）；
#   build_connector_meta 的 resumed-from-preemption 支（L341-L370——
#     需要 CachedRequestData 的全 token 回查；本章不驱动断连重入，
#     total_need_load 对账断言随支保留在读侧注释）。
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors
import safetensors.torch  # noqa: F401  (HOST SEAM：显式绑定 .torch 子模块——
#   真实环境由包的惰性属性提供；host 版需显式 import 才可访问
#   safetensors.torch.load_file/save_file)
import torch

from .base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from .hashing import safe_hash
from .output import SchedulerOutput

if TYPE_CHECKING:
    from .config import VllmConfig
    from .forward_context import ForwardContext
    from .kv_cache_interface import KVCacheConfig
    from .kv_cache_manager import KVCacheBlocks
    from .request import Request

import logging

logger = logging.getLogger(__name__)  # LOGGER SEAM


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L31
#   ReqMeta——worker 侧每请求载荷（token/slot 寻址/store 标记）
@dataclass
class ReqMeta:
    # Request tokens
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L33-L38
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool
    mm_hashes: list[str]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L41
    #   make_meta——slot 寻址算术（block_id×block_size+offset 展平）
    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "ReqMeta":
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L49-L64
        valid_num_tokens = align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L67
#   ExampleConnectorMetadata——不透明计划的实例形态（本实现：请求载荷表）
@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L68-L69
    requests: list[ReqMeta] = field(default_factory=list)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L71
    #   add_request
    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L79-L81
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L84
#   ExampleConnector 类头（NOTE 原话：debug 实现、覆写既有前缀缓存的
#   额外工作）
class ExampleConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.
    # It does extra work which will overwrite the existing prefix-cache in GPU
    # - to remove the overhead, need to add some "mask" in the ReqMeta class

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L90
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L96-L107
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Request] = {}
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        logger.info(self._kv_transfer_config)
        logger.info("Shared storage path is %s", self._storage_path)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L109
    #   start_load_kv——加载方向：逐请求逐层把磁盘 KV 注回 paged buffer
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L122
        #   inject_kv_into_layer——worker 直写池（slot 寻址的具体形态）
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
            attn_metadata: "AttentionMetadata",
        ) -> None:
            """Inject the KV cache into the layer.

            Args:
                dst_kv_cache_layer (torch.Tensor): the destination KV cache
                    layer. In shape [num_pages, page_size, xxx] for MLA,
                    [num_pages, 2, page_size, xxx] otherwise.
                src_kv_cache (torch.Tensor): the source KV cache.
                slot_mapping (torch.Tensor): the slot mapping. In shape
                    [num_tokens].
            """
            # SUBTRACTED: MLA reshape 分支（L138-L145——第 8 条）。
            # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L146-L149
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            dst_kv_cache_layer[block_idxs, :, offsets] = src_kv_cache

        # Get the metadata
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L151-L153
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L155-L158
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("In connector.start_load_kv, but the attn_metadata is None")
            return

        # Load the KV for each request each layer
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L160-L190
        for request in metadata.requests:
            if request.is_store:
                continue
            logger.info(
                "Inject KV cache of %d tokens to the paged memory",
                len(request.slot_mapping),
            )
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]

                # Only process layers that have kv_cache
                # attribute (attention layers) Skip non-attention
                # layers like FusedMoEFactory/MLP etc.
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue

                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = safetensors.torch.load_file(
                    filename, device=str(kv_cache_layer.device)
                )["kv_cache"]
                if isinstance(attn_metadata, dict):
                    inject_kv_into_layer(
                        kv_cache_layer,
                        kv_cache,
                        request.slot_mapping,
                        attn_metadata[layer_name],
                    )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L192
    #   wait_for_layer_load——本实现同步加载、无需等待（逐层重叠的空实现
    #   对照组：契约要求阻塞语义，磁盘版在 start_load_kv 已完成写入）
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L201
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L203
    #   save_kv_layer——存储方向：从 paged buffer 按 slot 抽出该层该请求
    #   的 KV、落盘
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: arguments for the save operation.
        """

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L221
        #   extract_kv_from_layer——save 方向的对称形态
        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (num_pages, page_size, xxx)
            for MLA, and (num_pages, 2, page_size, xxx) otherwise.
            """
            # SUBTRACTED: MLA reshape 分支（L230-L232——第 8 条）。
            # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L233-L235
            block_idxs = slot_mapping // self._block_size
            offsets = slot_mapping % self._block_size
            return layer[block_idxs, :, offsets]

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L237-L246
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._generate_filename_debug(
                    layer_name, request.token_ids, request.mm_hashes
                )
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                tensors = {"kv_cache": kv_cache.detach().cpu()}
                safetensors.torch.save_file(tensors, filename)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L248
    def wait_for_save(self):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L249
        #   （save 在 save_kv_layer 内同步完成——同步实现的对照组）
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L251
    #   get_num_new_matched_tokens——调度器侧原语①：磁盘有文件=命中
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L276-L277
        if not self._found_match_for_request(request):
            return 0, False

        logger.info("External Cache Hit!")

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L283-L286
        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)

        return num_tokens_to_check - num_computed_tokens, False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L288
    #   update_state_after_alloc——调度器侧原语②：num_external_tokens>0
    #   才登记待加载（判据不是 blocks 空否）
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L297-L298
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L300
    #   build_connector_meta——调度器侧原语③：产计划、清状态（调用即重置）
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L312-L339
        meta = ExampleConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            mm_hashes = [f.identifier for f in new_req.mm_features]
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                # NOTE(rob): for this debug implementation, we only cache
                # the original prompt tokens.
                if not self._found_match_for_prompt(token_ids, mm_hashes):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )

        # SUBTRACTED: resumed-from-preemption 支（L341-L370——CachedRequest
        #   Data 全 token 回查与 resumed 判定；断连重入的加载补发——
        #   本章不驱动，total_need_load 对账语义保留于 clear 前）。

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L372-L374
        #   （调用即重置：_requests_need_load 清空）
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L380
    #   _found_match_for_request
    def _found_match_for_request(
        self,
        request: "Request",
    ) -> bool:
        """Check if the cache is hit for the request."""
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L385-L388
        return self._found_match_for_prompt(
            list(request.prompt_token_ids or []),
            [f.identifier for f in request.mm_features],
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L390
    #   _found_match_for_prompt——外部缓存查命中=文件夹存在
    def _found_match_for_prompt(
        self,
        prompt_token_ids: list[int],
        mm_hashes: list[str],
    ) -> bool:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L395-L403
        num_tokens_to_check = align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        foldername = self._generate_foldername_debug(
            torch.tensor(prompt_token_ids)[:num_tokens_to_check],
            mm_hashes,
            create_folder=False,
        )
        return os.path.exists(foldername)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L405
    #   _generate_foldername_debug——文件夹名=safe_hash(token bytes+mm)
    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input
        ids.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L414-L425
        token_bytes = token_ids.numpy().tobytes()
        # Add mm_hashes to the bytes being hashed to avoid path traversal and
        # to create a canonical key.
        if mm_hashes:
            mm_str = "-".join(mm_hashes)
            token_bytes += mm_str.encode("utf-8")
        input_ids_hash = safe_hash(token_bytes, usedforsecurity=False).hexdigest()

        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L427
    #   _generate_filename_debug
    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        """Generate a file name based on the layer name and the hash
        of the bytes of the input ids.
        """
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L436-L439
        foldername = self._generate_foldername_debug(
            token_ids, mm_hashes=mm_hashes, create_folder=True
        )
        return os.path.join(foldername, f"{layer_name}.safetensors")


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L442
#   align_to_block_size——块对齐工具（(n-1)//bs*bs：留最后一个 token 重算）
def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L444
    return (num_tokens - 1) // block_size * block_size
