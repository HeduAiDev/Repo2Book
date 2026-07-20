# SPDX-License-Identifier: Apache-2.0
# Subtract-only companion for ch29《PD 分离的抽象与调度器集成》.
# 只做减法：与 vLLM 同名/同结构/同控制流，只删不增。
#
# 本文件是 vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py
# 的子集：把 KV cache 存/取磁盘 safetensors 的调试参考实现，给出每个抽象方法
# 最朴素的真实落地，作『role-split 契约如何被填实』的范例。
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .base import KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L32
@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    # Slot mappings, should have the same length as token_ids
    slot_mapping: torch.Tensor
    # Is store or load
    is_store: bool

    @staticmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L42
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> "ReqMeta":
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
        )
        # SUBTRACTED: mm_hashes 字段（多模态哈希参与文件名）与 KV-connector 契约
        # 无关，只影响磁盘 key（原 L37, L48, L63）。


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L68
@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L72
    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, block_ids, block_size, is_store)
        )


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L85
class ExampleConnector(KVConnectorBase_V1):
    # NOTE: This is Simple debug implementation of the KV connector.
    # It save / load the KV cache to / from the disk.

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L91
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, "Request"] = {}
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )

    # ==============================
    # Worker-side methods
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L110
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer."""

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L123
        def inject_kv_into_layer(
            dst_kv_cache_layer: torch.Tensor,
            src_kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            """Inject the KV cache into the layer.

            dst shape [2, num_pages, page_size, xxx]; src shape [2, num_tokens, xxx].
            """
            dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
            num_pages = dst_kv_cache_layer_shape[1]
            page_size = dst_kv_cache_layer_shape[2]
            dst_kv_cache_layer = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
            dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
            # SUBTRACTED: isinstance MLACommonMetadata / TritonAttentionMetadata 两支
            # 是不同 attention 后端的 reshape 差异，与 connector 契约无关；精简版保留
            # 默认（非 MLA）一支演示 KV inject 思路（原 L142-152）。

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ExampleConnectorMetadata)

        # Load the KV for each request each layer
        for request in metadata.requests:
            if request.is_store:
                continue
            for layer_name in forward_context.no_compile_layers:
                layer = forward_context.no_compile_layers[layer_name]
                kv_cache_layer = getattr(layer, "kv_cache", None)
                if kv_cache_layer is None:
                    continue
                filename = self._generate_filename_debug(layer_name, request.token_ids)
                kv_cache = self._load_file(filename)
                inject_kv_into_layer(kv_cache_layer, kv_cache, request.slot_mapping)
        # SUBTRACTED: attn_metadata None 检查、.cuda()、safetensors.load_file、
        # isinstance(attn_metadata, dict) 的后端分派 —— host 无 CUDA，精简版用普通
        # torch 张量 + 自带 _load_file 占位以保证可运行（原 L161-198）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L200
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. (this debug impl loads synchronously, so it's a no-op)"""
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L211
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Start saving the KV cache of the layer from vLLM's paged buffer
        to the connector."""

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L229
        def extract_kv_from_layer(
            layer: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> torch.Tensor:
            """Extract the KV cache from the layer.

            Assume the shape of the layer is (2, num_pages, page_size, xxx).
            """
            num_pages, page_size = layer.shape[1], layer.shape[2]
            return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]
            # SUBTRACTED: MLA / Triton 后端分支同 inject（原 L238-244）。

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, ExampleConnectorMetadata)
        for request in connector_metadata.requests:
            if request.is_store:
                filename = self._generate_filename_debug(layer_name, request.token_ids)
                kv_cache = extract_kv_from_layer(kv_layer, request.slot_mapping)
                self._save_file(filename, kv_cache.detach())
                # SUBTRACTED: safetensors.save_file({"kv_cache": ...cpu()}) 换成自带
                # _save_file 占位以保证 host 可运行（原 L256-257）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L259
    def wait_for_save(self):
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L262
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        """
        # NOTE: in this debug implementation, we assume that the prompt is
        # cached_prompt + newly_generated_single_token
        # Therefore, we use prompt_token_ids[:-1] to determine the folder name

        # NOTE: in current v1 scheduler, the num_computed_tokens is aligned
        # with the block granularity. And it expects the returned blocks and
        # num_computed_tokens to also be aligned with the block granularity.
        if not self._found_match_for_request(request):
            return 0, False

        # Now, first num_tokens_to_check tokens are hit, we need to prepare
        # the metadata for the worker connector to correctly load the KV
        token_ids = request.prompt_token_ids or []
        num_tokens_to_check = align_to_block_size(len(token_ids) - 1, self._block_size)

        return num_tokens_to_check - num_computed_tokens, False

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L299
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _requests_need_load,
        such that we load the KVs in the next forward pass.
        """
        if num_external_tokens > 0:
            self._requests_need_load[request.request_id] = request

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L311
    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        """
        meta = ExampleConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            if new_req.req_id in self._requests_need_load:
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                )
                total_need_load += 1
            else:
                # NOTE: here, we set the store and load being exclusive,
                # but a single request can have both store and load.
                if not self._found_match_for_prompt(token_ids):
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                    )

        # SUBTRACTED: scheduled_cached_reqs（被抢占后 resumed 请求的 load 计划重建）
        # 一段是 resumed 路径细节，非 role-split 主线；精简版只演示 new_reqs 的
        # load/store 分流（原 L352-381）。

        assert total_need_load == len(self._requests_need_load)
        self._requests_need_load.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L391
    def _found_match_for_request(self, request: "Request") -> bool:
        """Check if the cache is hit for the request."""
        return self._found_match_for_prompt(list(request.prompt_token_ids or []))

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L401
    def _found_match_for_prompt(self, prompt_token_ids: list[int]) -> bool:
        num_tokens_to_check = align_to_block_size(
            len(prompt_token_ids) - 1, self._block_size
        )
        foldername = self._generate_foldername_debug(
            torch.tensor(prompt_token_ids)[:num_tokens_to_check],
            create_folder=False,
        )
        return os.path.exists(foldername)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L416
    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        create_folder=False,
    ) -> str:
        """Generate a folder name based on the hash of the bytes of the input ids."""
        token_bytes = token_ids.numpy().tobytes()
        import hashlib

        input_ids_hash = hashlib.sha256(token_bytes).hexdigest()
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername
        # SUBTRACTED: mm_hashes 拼入哈希、safe_hash(usedforsecurity=False)；精简版
        # 用 hashlib.sha256 等价占位以脱离 vllm 依赖（原 L425-435）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L438
    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
    ) -> str:
        """Generate a file name based on the layer name and the hash of input ids."""
        foldername = self._generate_foldername_debug(token_ids, create_folder=True)
        return os.path.join(foldername, f"{layer_name}.safetensors")

    # _save_file / _load_file 是 safetensors.torch.{save,load}_file 的 host 占位，
    # 用 torch.save/load 等价落地，以便精简版在无 CUDA/safetensors 的 host 跑通。
    @staticmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L257 (safetensors.torch.save_file 的 host 占位)
    def _save_file(filename: str, tensor: torch.Tensor) -> None:
        torch.save({"kv_cache": tensor}, filename)

    @staticmethod
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L191 (safetensors.torch.load_file 的 host 占位)
    def _load_file(filename: str) -> torch.Tensor:
        return torch.load(filename)["kv_cache"]


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py:L453
def align_to_block_size(num_tokens: int, block_size) -> int:
    """Align the number of tokens to the block size."""
    return (num_tokens - 1) // block_size * block_size
