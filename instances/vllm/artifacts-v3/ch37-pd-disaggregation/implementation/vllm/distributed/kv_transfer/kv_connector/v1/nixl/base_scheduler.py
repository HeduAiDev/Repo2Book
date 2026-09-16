# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拉/推共用的调度器侧：side channel、四账本、租约常量、块剪裁。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L1-L505
# SUBTRACTED:
#   * Mamba/SSM 深分支（_get_remote_prefill_token_count 的 N−1 与
#     _truncate_mamba_request_for_prefill、_ssm_spec_blocks /
#     _ssm_state_slots_are_positional）——减法计划删除项 2；
#   * 双向回拉（is_bidirectional_kv_xfer_enabled / decoder_kv_blocks_ttl 的消费分支）
#     ——减法计划删除项 5；
#   * host buffer 保存路径（_build_save_meta 的 yield_req_data 循环）——删除项 3。
"""

import threading
import time
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import zmq_ctx
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L51-L505
class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L54-L181
    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self.kv_cache_config = kv_cache_config
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L64-L68
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L70-L76
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        # SUBTRACTED: self._has_mamba（Mamba 组探测，L91-L94）——删除项 2。
        # SUBTRACTED: HMA 使能日志（L96-L98）。

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, BlockIds]] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L123-L136
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.kv_cache_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]
        # SUBTRACTED: _ssm_spec_blocks / _ssm_state_slots_are_positional（L138-L150）——删除项 2。

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L152-L159
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L161-L167
        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        # SUBTRACTED: decoder_kv_blocks_ttl（D 侧块的 TTL，L168-L172）与其使能日志
        #   （L174-L181）——减法计划删除项 5；push_scheduler 的注册 watchdog 默认
        #   值曾借用它，现改为等值显式常量（push_scheduler.py 已标注）。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L183-L187
    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L189-L223
    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        if remote_engine_id not in self._heartbeat_by_engine:
            self._heartbeat_by_engine[remote_engine_id] = HeartbeatInfo(
                req_ids=set(),
                host=host,
                port=port,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L225-L233
    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                if not info.req_ids:
                    # Clean up empty engines so we don't leak a key when remote dies.
                    del self._heartbeat_by_engine[engine_id]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L235-L279
    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.
        """
        if len(block_ids) == 0 or not self._is_hma_required:
            # No blocks to clip eg Full prefix cache hit or not a hybrid model.
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable.
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            # SUBTRACTED: SSM 组的投机 scratch 槽剥离与单状态模式尾槽保留
            #   （clip_ssm 分支，L268-L277）——删除项 2；参数保留以对齐调用点签名。
            clipped.append(blocks)
        return tuple(clipped)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L281-L322
    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        encoded_data: dict[tuple[int, int], bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, tp_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_data[(pp_rank, tp_rank)] = encoder.encode(rank_metadata)
            logger.debug(
                "PP rank %d, TP rank %d: encoded NixlHandshakePayload size: %s bytes",
                pp_rank,
                tp_rank,
                str(len(encoded_data[(pp_rank, tp_rank)])),
            )

        # Only start the listener when we have metadata to serve.
        if self._nixl_handshake_listener_t is None:
            ready_event = threading.Event()
            self._nixl_handshake_listener_t = threading.Thread(
                target=self._nixl_handshake_listener,
                args=(
                    encoded_data,
                    ready_event,
                    self._stop_event,
                    self.side_channel_host,
                    self.side_channel_port,
                ),
                daemon=True,
                name="nixl_handshake_listener",
            )
            self._nixl_handshake_listener_t.start()
            ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L324-L365
    @staticmethod
    def _nixl_handshake_listener(
        encoded_data: dict[tuple[int, int], Any],
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
    ):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L324-L365
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while True:
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    if stop_event.is_set():
                        break
                    continue
                # Decode (GET_META_MSG, pp_rank, tp_rank).
                msg, target_pp_rank, target_tp_rank = msgspec.msgpack.decode(msg)
                logger.debug(
                    "Received message for pp rank %s, tp rank %s",
                    target_pp_rank,
                    target_tp_rank,
                )
                if msg != GET_META_MSG:
                    logger.warning("Connection listener got unexpected message %s", msg)
                # Echo our perf_counter so P can estimate the clock offset.
                # perf_counter is only comparable within a process, so this
                # listener must run in the same process that stamps the block
                # expiry deadline (`_reqs_need_send`).
                ts = msgspec.msgpack.encode(time.perf_counter())
                sock.send_multipart(
                    (identity, b"", encoded_data[(target_pp_rank, target_tp_rank)], ts)
                )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L367-L398
    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns the number of prompt tokens the remote holds.

        # SUBTRACTED: Mamba 模型的 N−1（解码侧要重算末 token 从 h(N−1) 出发，
        #   L370-L372）——删除项 2；非 Mamba 模型逐字等价。
        """
        return num_prompt_tokens

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L400-L435
    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """host buffer 模式下把「本步新分配的块」登记为待保存。

        # SUBTRACTED: yield_req_data 驱动的逐请求 Save ReqMeta 构建与 partial
        #   prefill 保留逻辑（L409-L435）——删除项 3（host buffer 旁路实现）。
        """
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L437-L476
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        # Clock reference for reqs_to_send: deadlines above are in this
        # process's perf_counter domain; workers (possibly on other nodes,
        # where perf_counter has a different epoch) rebase against this.
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}

        return meta

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L478-L481
    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L483-L484
    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L490-L493
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L495-L498
    def update_state_after_alloc(
        self, request: "Request", blocks: Any, num_external_tokens: int
    ):
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:L500-L505
    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError


__all__ = ["NixlBaseConnectorScheduler"]
