# vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
#   —— subtract-only companion（ch10 第 1 层「挑 layerwise 讲透」+ 第 2 层地址算术）
#
# 本章选 MooncakeLayerwiseConnector 一个连接器讲透 vLLM v1 连接器契约：
#   facade 按 KVConnectorRole 只持有 Scheduler 或 Worker 之一，把每个 base-class 钩子转发过去。
#   Scheduler 侧：get_num_new_matched_tokens（do_remote_prefill→拉整段 prompt）、
#     update_state_after_alloc（按 do_remote_prefill/do_remote_decode 分入 recv/send 队列 + 握手
#     POST 到 metaserver）、build_connector_meta（把待 recv/send 化为 ReqMeta）。
#   Worker 侧：start_load_kv（consumer 登记收 / producer 解析块映射）、save_kv_layer（每算完一层
#     立刻把该层 KV 入发送队列＝逐层 push）、get_finished（回收完成的传输）。
#   第 2 层地址算术：get_transfer_meta 把 (base_addr + block_id*block_len) 摊成 src/dst/length；
#     group_concurrent_contiguous 把连续块合并成批量拷贝。
#
# host 无 torch_npu / mooncake：NPU 重排/量化路径与真实跨节点 P2P 搬运不真跑，按 dossier 减法
# 计划删除（保留 pd_head_ratio==1、无量化的可读默认分支）；vllm 符号经 runtime_stub 接住。
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import numpy.typing as npt
import torch

from runtime_stub import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SchedulerOutput,
    SupportsHMA,
    get_ip,
    logger,
    round_down,
)
from mooncake_transfer_engine import global_te

# SUBTRACTED: import torch_npu / from mooncake.engine import TransferEngine / get_pcp_group /
#   get_tensor_model_parallel_rank / MambaSpec/SlidingWindowSpec/... / npu_stream_switch /
#   trans_nd_to_nz / kv_alltoall_and_rearrange 等（L1-L73）—— 仅服务被删的 NPU 重排/量化/
#   hybrid-cache/分布式拓扑路径；host 无 NPU/CANN 不可跑。原 mooncake_layerwise_connector.py:L1-L73

if TYPE_CHECKING:
    from runtime_stub import KVCacheBlocks, Request


@dataclass
class LayerMetadata:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L87
    tensor_group_idx: list[int]
    kv_caches_base_addr: list[int]
    block_len: list[int]
    block_size_scale: list[int]


# SUBTRACTED: class MooncakeAgentMetadata (msgspec.Struct, L95-L97) —— worker 间 RPC 元数据
#   序列化载体，属被删的握手序列化路径。原 mooncake_layerwise_connector.py:L95-L97


@dataclass
class ReqMeta:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L100
    local_block_ids: list[list[int]]
    token_ids: list[int] | None
    # Not None if layer-wise is disabled
    remote_block_ids: list[list[int]]
    remote_block_size: list[list[int]]
    remote_engine_id: str | None
    remote_host: str | None
    remote_port: int | None
    remote_te_rpc_port: int | None
    remote_layer_metadata: dict[str, LayerMetadata] | None
    metaserver: str | None
    remote_tp_size: int | None
    remote_pcp_size: int | None
    remote_dcp_size: int | None
    chunk_finish: bool = False
    prompt_len: int = 0
    trans_count: list[int] | None = None
    remote_cache_tokens: int = 0
    local_computed_tokens: int = 0
    local_transed_tokens: int = 0
    do_virtual: bool = False


@dataclass
class SendTask:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L125
    send_request: dict[str, ReqMeta] = field(default_factory=dict)
    # pd_head_ratio == 1 use
    wait_event: Any | None = None
    # pd_head_ratio > 1 use
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    # kv cache quantization layer use
    k_quant_cache: torch.Tensor | None = None
    v_quant_cache: torch.Tensor | None = None
    layer_idx: int = 0
    layer_name: str = ""
    # trans block info
    group_rearrange_block_ids: list[list[int]] | None = None
    # SUBTRACTED: group_num_blocks/group_num_tokens/group_block_table/group_block_len_tensor/
    #   group_seq_start_tensor（L140-L144）—— 仅 pd_head_ratio>1 重排路径用。


@dataclass
class TransferMeta:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L147
    src: list[int]
    dst: list[int]
    length: list[int]
    req_ids: list[str]


@dataclass
class SendReqInfo:  # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L155
    local_block_ids: list[list[int]]
    local_transferred_tokens: int
    local_computed_tokens: int
    request: "Request"

    def extend_local_block_ids(self, new_block_ids: list[list[int]]) -> None:  # SOURCE: L162
        """extend local block ids for this step"""
        for i, new_block_id in enumerate(new_block_ids):
            self.local_block_ids[i].extend(new_block_id)

    def update_computed_tokens(self, computed_tokens: int) -> None:  # SOURCE: L167
        """update local computen tokens for this step"""
        self.local_computed_tokens = computed_tokens

    def update_transferred_tokens(self, transferred_tokens: int) -> None:  # SOURCE: L171
        """update transferred tokens for this step"""
        self.local_transferred_tokens = transferred_tokens

    def unpack(self):  # SOURCE: L175
        return (
            self.local_block_ids,
            self.local_transferred_tokens,
            self.local_computed_tokens,
            self.request,
        )


# SUBTRACTED: class SizedDict (L184-L201) —— 远端元数据的有界 LRU 缓存，属被删的 worker 缓存层。


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L204
class KVCacheSendingLayerThread(threading.Thread):
    # SUBTRACTED: 真实 __init__ 的 ~40 个 NPU/量化/MLA/mamba 参数（L205-L264）—— 仅保留发送队列、
    #   引擎、逐层元数据与 block 算术所需字段，删去量化缓冲/重排流等 host 不可跑的状态。
    def __init__(self, engine, kv_cache_specs, layer_metadata, total_layers, ready_event):  # SOURCE: L205
        super().__init__(daemon=True, name="KVCacheSendingLayerThread")
        self.engine = engine
        self.kv_cache_specs = kv_cache_specs
        self.layer_metadata = layer_metadata
        self.total_layers = total_layers
        self.pd_head_ratio = 1
        self.enable_kv_quant = False
        self.enable_c8_quant = False
        self.send_queue: queue.Queue = queue.Queue()
        self.failed_reqs: set[str] = set()
        self.ready_event = ready_event

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L266
    def run(self):
        # SUBTRACTED: torch.npu.set_device 绑卡（L267-L270）—— host 无 NPU。
        self.ready_event.set()
        while True:
            send_task = self.send_queue.get()
            self._handle_request(send_task)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L275
    def _handle_request(self, send_task: SendTask):
        try:
            self._transfer_kv_cache(send_task)
        except Exception as e:
            logger.error("Failed to transfer KV cache. layer_idx=%s, error=%s.", send_task.layer_idx, e)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L285
    def get_transfer_meta(self, send_task: SendTask, req_id: str, req_meta: ReqMeta, layer_group_idx: int):
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []

        layer_name = send_task.layer_name
        remote_block_ids = req_meta.remote_block_ids[layer_group_idx]
        remote_layer_metadata = req_meta.remote_layer_metadata[layer_name]
        local_layer_metadata = self.layer_metadata[layer_name]
        local_block_ids = req_meta.local_block_ids[layer_group_idx]

        # SUBTRACTED: MambaSpec 分支（L297-L358）与 pd_head_ratio>1 重排分支（L407-L437）与
        #   量化分支（L367-L392）—— 保留默认分支：pd_head_ratio==1、无量化的 base-addr+block_len 算术。
        #   原 mooncake_layerwise_connector.py:L297-L437
        layer_local_kv_base_addr = local_layer_metadata.kv_caches_base_addr
        layer_remote_kv_base_addr = remote_layer_metadata.kv_caches_base_addr
        block_lens = local_layer_metadata.block_len
        grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
            remote_block_ids, local_block_ids
        )
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
            zip(layer_local_kv_base_addr, layer_remote_kv_base_addr)
        ):
            block_len = block_lens[k]
            for group_remote_block_id, group_local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                src = src_layer_base_addr + group_local_block_id[0] * block_len
                dst = dst_layer_base_addr + group_remote_block_id[0] * block_len
                length = len(group_local_block_id) * block_len
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)
        return (src_list, dst_list, length_list)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L440
    def _transfer_kv_cache(self, send_task: SendTask):
        layer_name = send_task.layer_name
        layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
        # SUBTRACTED: pd_head_ratio>1 / 量化的 NPU 缓冲拷贝与 resharding_stream 同步
        #   （L443-L458, L474-L485）—— host 无 NPU；默认分支直接按 wait_event 同步后发车。
        # Merge transmission tasks of the same session
        session_meta: dict[str, TransferMeta] = {}
        for req_id, req_meta in send_task.send_request.items():
            session_id = f"{req_meta.remote_host}:{req_meta.remote_te_rpc_port}"
            if session_id not in session_meta:
                session_meta[session_id] = TransferMeta(src=[], dst=[], length=[], req_ids=[])

            (src_list, dst_list, length_list) = self.get_transfer_meta(send_task, req_id, req_meta, layer_group_idx)

            session_meta[session_id].src.extend(src_list)
            session_meta[session_id].dst.extend(dst_list)
            session_meta[session_id].length.extend(length_list)
            session_meta[session_id].req_ids.append(req_id)

        for session_id, transfer_meta in session_meta.items():
            if len(transfer_meta.src) > 0:
                ret = self.engine.batch_transfer_sync_write(
                    session_id, transfer_meta.src, transfer_meta.dst, transfer_meta.length
                )
                if ret < 0:
                    logger.error("Mooncake transfer failed. session=%s, ret=%d.", session_id, ret)


# SUBTRACTED: class KVCacheRecvingLayerThread (L523-...) —— consumer 侧后台收线程；其握手/收块
#   生命周期是被删的线程内务，ch10 用 do_recving 集合的回收（get_finished）讲清异步闭环即可。


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L649
class MooncakeLayerwiseConnectorMetadata(KVConnectorMetadata):
    def __init__(self):  # SOURCE: mooncake_layerwise_connector.py:L650
        self.requests: dict[str, ReqMeta] = {}
        self.send_task: SendTask = SendTask()

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L654
    def add_new_req(
        self,
        request_id: str,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        token_ids: list[int] | None = None,
        chunk_finish: bool = False,
        prompt_len: int = 0,
        remote_cache_tokens: int = 0,
        local_computed_tokens: int = 0,
        local_transed_tokens: int = 0,
    ):
        self.requests[request_id] = ReqMeta(
            token_ids=token_ids or [],
            local_block_ids=local_block_ids,
            remote_block_ids=kv_transfer_params.get("remote_block_ids", []),
            remote_block_size=kv_transfer_params.get("remote_block_size", []),
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
            remote_host=kv_transfer_params.get("remote_host"),
            remote_port=kv_transfer_params.get("remote_port"),
            remote_te_rpc_port=kv_transfer_params.get("remote_te_rpc_port"),
            remote_layer_metadata=kv_transfer_params.get("remote_layer_metadata"),
            metaserver=kv_transfer_params.get("metaserver"),
            remote_tp_size=kv_transfer_params.get("remote_tp_size"),
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size"),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size"),
            do_virtual=kv_transfer_params.get("do_virtual"),
            chunk_finish=chunk_finish,
            remote_cache_tokens=remote_cache_tokens,
            local_computed_tokens=local_computed_tokens,
            prompt_len=prompt_len,
            local_transed_tokens=local_transed_tokens,
            trans_count=[],
        )


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L690
class MooncakeLayerwiseConnector(KVConnectorBase_V1, SupportsHMA):
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L691
    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeLayerwiseConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeLayerwiseConnectorScheduler | None = MooncakeLayerwiseConnectorScheduler(
                vllm_config, kv_cache_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeLayerwiseConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeLayerwiseConnectorWorker(vllm_config, kv_cache_config, str(self.engine_id))

    ############################################################
    # Scheduler Side Methods
    ############################################################

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L710
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L714
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L718
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L725
    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L733
    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L744
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L748
    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    # SUBTRACTED: get_block_ids_with_load_errors （L753-L756）—— 失败块上报，异步闭环的边角。

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L758
    def start_load_kv(self, forward_context, **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L763
    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeLayerwiseConnector does not do layerwise saving."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.wait_for_layer_load(layer_name)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L769
    def save_kv_layer(self, layer_name: str, kv_layer: list[torch.Tensor], attn_metadata, **kwargs) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeLayerwiseConnectorMetadata)
        self.connector_worker.save_kv_layer(layer_name, kv_layer, attn_metadata, self._connector_metadata)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L777
    def wait_for_save(self):
        """MooncakeLayerwiseConnector does not save explicitly."""
        pass


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L782
class MooncakeLayerwiseConnectorScheduler:
    """Implementation of Scheduler side methods"""

    # SUBTRACTED: __init__ 的 TLS/ssl 握手 plumbing 与 pcp 断言（L812-L825, L794-L804 部分）——
    #   transport 加固，与 P/D 路由逻辑正交。保留 block_size/engine_id/握手 host:port/recv·send 队列。
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L785
    def __init__(self, vllm_config, kv_cache_config, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.block_size = [group_spec.kv_cache_spec.block_size for group_spec in kv_cache_config.kv_cache_groups]
        self.engine_id = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank * vllm_config.parallel_config.tensor_parallel_size
        )
        # Requests that need to start recv / send. Added by update_state_after_alloc.
        self._reqs_need_recv: dict[str, tuple[Request, list[int], list[list[int]]]] = {}
        self._reqs_need_send_layerwise: dict[str, SendReqInfo] = {}
        self.executor = ThreadPoolExecutor(32)
        self.metaserver_client = httpx.Client(limits=httpx.Limits(max_connections=100000), timeout=None)

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L827
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.
        """
        params = request.kv_transfer_params

        if params is not None and params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert num_computed_tokens % min(self.block_size) == 0
            # Note: We use the full token count as transmit data here.
            count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
            return count, count > 0

        # No remote prefill for this request.
        return 0, False

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L860
    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params

        if params is not None and params.get("do_remote_prefill"):
            do_virtual = params.get("do_virtual", False)
            local_block_ids = (blocks.get_block_ids()) if num_external_tokens > 0 else []
            remote_cached_tokens = request.num_computed_tokens
            # Get unhashed blocks to pull from remote.
            self._reqs_need_recv[request.request_id] = (
                request,
                [],  # request._all_token_ids,
                local_block_ids,
            )

            params["do_remote_prefill"] = False

            # All parameters here should appear in the returned dict of
            # request_finished in the scheduler side except "request_id".
            external_req_id = get_external_request_id(request.request_id)
            kv_transfer_params = dict(
                token_ids=[],
                request_id=external_req_id,
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=local_block_ids,
                remote_block_size=self.block_size,
                remote_engine_id=self.engine_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                remote_tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_pcp_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
                remote_dcp_size=self.vllm_config.parallel_config.decode_context_parallel_size,
                remote_cached_tokens=remote_cached_tokens,
            )
            if not do_virtual:
                future = self.executor.submit(
                    self._access_metaserver, url=params.get("metaserver", None), message=kv_transfer_params
                )

                def handle_exception(future):  # SOURCE: mooncake_layerwise_connector.py:L909
                    if future.exception():
                        logger.error("Access metaserver fail. error=%s.", future.exception())

                future.add_done_callback(handle_exception)

        # Layerwise prefiller add request need send
        if params is not None and params.get("do_remote_decode"):
            local_block_ids = list(blocks.get_block_ids())
            remote_cache_tokens = params["remote_cached_tokens"]
            local_transferred_tokens = remote_cache_tokens
            local_computed_tokens = 0
            self._reqs_need_send_layerwise[request.request_id] = SendReqInfo(
                local_block_ids=local_block_ids,
                local_transferred_tokens=local_transferred_tokens,
                local_computed_tokens=local_computed_tokens,
                request=request,
            )

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L931
    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        meta = MooncakeLayerwiseConnectorMetadata()

        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            # Loop through scheduled reqs and convert to ReqMeta.
            for req_id, (req, token_ids, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    token_ids=token_ids,
                )
            # Clear the list once workers start the transfers
            self._reqs_need_recv.clear()
        else:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            new_reqs = scheduler_output.scheduled_new_reqs
            scheduled_spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
            for req_id, new_blocks in zip(cached_reqs.req_ids, cached_reqs.new_block_ids):
                if req_id in self._reqs_need_send_layerwise and new_blocks is not None:
                    self._reqs_need_send_layerwise[req_id].extend_local_block_ids(new_blocks)
            computed_tokens = dict(
                list(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))
                + [(x.req_id, x.num_computed_tokens) for x in new_reqs]
            )
            for req_id, scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
                if req_id in self._reqs_need_send_layerwise:
                    send_req_info = self._reqs_need_send_layerwise[req_id]
                    # update local transferred tokens
                    send_req_info.update_transferred_tokens(
                        round_down(send_req_info.local_computed_tokens, min(self.block_size))
                    )
                    # update local computed tokens, not transfer spec decode tokens
                    spec_decode_tokens = (
                        len(scheduled_spec_decode_tokens[req_id]) if (req_id in scheduled_spec_decode_tokens) else 0
                    )
                    send_req_info.update_computed_tokens(
                        computed_tokens.get(req_id, 0) + scheduled_tokens - spec_decode_tokens
                    )

                    def add_transfer_task(req_id, send_req_info: SendReqInfo, chunk_finish=False):  # SOURCE: L979
                        (local_block_ids, local_transed_tokens, local_computed_tokens, request) = send_req_info.unpack()
                        meta.add_new_req(
                            request_id=req_id,
                            local_block_ids=local_block_ids,
                            kv_transfer_params=request.kv_transfer_params,
                            token_ids=[],
                            chunk_finish=chunk_finish,
                            remote_cache_tokens=request.kv_transfer_params.get("remote_cached_tokens"),
                            prompt_len=len(request.all_token_ids),
                            local_computed_tokens=local_computed_tokens,
                            local_transed_tokens=local_transed_tokens,
                        )

                    # whether chunk finish
                    chunk_finish = send_req_info.local_computed_tokens >= len(send_req_info.request.all_token_ids)
                    add_transfer_task(req_id, send_req_info, chunk_finish=chunk_finish)
                    if chunk_finish:
                        self._reqs_need_send_layerwise.pop(req_id)
        return meta

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1020
    def _access_metaserver(self, url, message):
        success = False
        retry = 0
        while retry < 3 and success is False:
            retry += 1
            try:
                self.metaserver_client.post(url, json=message)
                success = True
            except Exception as e:
                logger.error("Failed to connect to metaserver. url=%s, retry=%d.", url, retry)
                if retry == 3:
                    raise e

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1033
    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        # layer_wise push, not need delay_free_blocks
        return False, None

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1045
    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        # layer_wise push, not need delay_free_blocks
        return False, None


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1058
class MooncakeLayerwiseConnectorWorker:
    """Implementation of Worker side methods"""

    # SUBTRACTED: __init__ 的 ~80 行 NPU/TP/PCP/DCP/量化/远端 socket 池/k_buffer 等状态
    #   （L1061-L1146）—— host 无 NPU/分布式拓扑。保留逐层 push 与异步收发闭环读到的核心字段。
    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1061
    def __init__(self, vllm_config, kv_cache_config, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.kv_cache_specs = [spec.kv_cache_spec for spec in kv_cache_config.kv_cache_groups]
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.engine = global_te.get_transfer_engine(get_ip(), device_name=None)
        self.kv_recv_layer_thread = None
        self.kv_send_layer_thread: KVCacheSendingLayerThread | None = None
        self.layer_metadata: dict[str, LayerMetadata] = {}
        self.current_layer = -1
        self.request_map: dict[str, str] = {}
        self.virtual_request: set[str] = set()
        self._invalid_block_ids: set[int] = set()
        self._recving_metadata: dict[str, ReqMeta] = {}

    # SUBTRACTED: register_kv_caches 真实体（L1178-...）—— 把 KV 张量按 group 登记给 mooncake 引擎、
    #   建 layer_metadata 的 base_addr/block_len，全是 NPU 内存布局算术；host 无 NPU。
    #   原 mooncake_layerwise_connector.py:L1178
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):  # SOURCE: L1178
        raise NotImplementedError("NPU KV registration is not runnable on host (subtracted).")

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1331
    def get_finished(self) -> tuple[set[str], set[str]]:
        done_recving = (
            self.kv_recv_layer_thread.get_and_clear_done_requests()
            if self.vllm_config.kv_transfer_config.is_kv_consumer
            else set()
        )
        done_recving = {self.request_map[s] for s in done_recving if s in self.request_map}
        done_recving.update(self.virtual_request)
        self.virtual_request = set()
        # SUBTRACTED: failed_recving 失败块回收与 _recving_metadata 清理（L1342-L1354）—— 失败重试边角。
        for req_id in done_recving:
            org_req_id = req_id[:-9]
            self.request_map.pop(org_req_id, None)
            self._recving_metadata.pop(req_id, None)
        return set(), done_recving

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1534
    def start_load_kv(self, metadata: MooncakeLayerwiseConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        self.current_layer = 0
        if self.vllm_config.kv_transfer_config.is_kv_consumer:
            for req_id, meta in metadata.requests.items():
                if meta.do_virtual:
                    self.virtual_request.add(req_id)
                    continue
                external_req_id = get_external_request_id(req_id)
                assert self.kv_recv_layer_thread is not None
                self.request_map[external_req_id] = req_id
                self._recving_metadata[req_id] = meta
        elif self.vllm_config.kv_transfer_config.is_kv_producer:
            # SUBTRACTED: producer 侧 _get_kv_split_metadata/_align_remote_block_ids 块映射解析与
            #   pd_head_ratio!=1 的 send_task 重排准备（L1547-L1619）—— NPU 拓扑/重排算术；host 无 NPU。
            pass

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1621
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, connector_metadata, **kwargs) -> None:
        """MooncakeLayerwiseConnector does not save explicitly."""
        if self.vllm_config.kv_transfer_config.is_kv_producer and connector_metadata.requests.keys():
            if self.current_layer >= self.total_layers:
                self.current_layer += 1
                return
            # SUBTRACTED: reshape_cache_event 取用 + pd_head_ratio>1/量化的 npu_paged_cache_load/
            #   alltoall/量化重排（L1634-L1733）—— host 无 NPU。默认分支：keys/values 直接走原 KV。
            send_task = connector_metadata.send_task
            layer_group_idx = self.layer_metadata[layer_name].tensor_group_idx[0]
            keys = None
            values = None
            # (this layer is pushed the instant it is computed = layer-by-layer pipelining)
            layer_send_task = SendTask(
                wait_event=None,  # SUBTRACTED: reshape_cache_event（NPU Event）
                k_cache=keys,
                v_cache=values,
                layer_idx=self.current_layer,
                layer_name=layer_name,
                group_rearrange_block_ids=send_task.group_rearrange_block_ids,
            )
            for req_id, req_meta in connector_metadata.requests.items():
                if len(req_meta.local_block_ids[layer_group_idx]) == 0:
                    continue
                req_meta_update = self.update_decoder_info(req_id, req_meta)
                layer_send_task.send_request[req_id] = req_meta_update

            self.kv_send_layer_thread.send_queue.put(layer_send_task)
            self.current_layer += 1

    # SUBTRACTED: update_decoder_info 真实体（L1795-...）—— 拉远端 layer_metadata/te_rpc_port 补全
    #   ReqMeta；属被删的握手元数据交换。host 无远端。原 mooncake_layerwise_connector.py:L1795
    def update_decoder_info(self, req_id, req_meta: ReqMeta):  # SOURCE: L1795
        raise NotImplementedError("Remote handshake resolution is not runnable on host (subtracted).")

    # SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1902
    def wait_for_layer_load(self, layer_name: str) -> None:
        pass


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L1922
def group_concurrent_contiguous(
    src: list[int], dst: list[int] | None = None
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    if dst is None:
        dst = []
    if not dst:
        src_only_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)

        if src_only_indices.size == 0:
            return [], []

        brk = np.where(np.diff(src_only_indices) != 1)[0] + 1
        src_groups = np.split(src_only_indices, brk)
        src_groups = [g.tolist() for g in src_groups]

        return src_groups, []

    else:
        src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
        dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

        if src_indices.size == 0:
            return [], []

        brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
        src_groups = np.split(src_indices, brk)
        dst_groups = np.split(dst_indices, brk)

        src_groups = [g.tolist() for g in src_groups]
        dst_groups = [g.tolist() for g in dst_groups]

        return src_groups, dst_groups


# SOURCE: vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:L2013
def get_external_request_id(request_id: str):
    # NOTE(zxr): vLLM PR #27987 add additional suffix
    # to EngineCore request_id with len(suffix) == 9
    return request_id[:-9]
