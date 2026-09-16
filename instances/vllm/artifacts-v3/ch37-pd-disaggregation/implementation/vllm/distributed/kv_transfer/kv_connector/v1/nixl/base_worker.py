# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拉/推共用的 worker 侧：NIXL 注册、带外握手、完成回收、租约心跳。

worker 侧是「数据的实际搬运者」：把 KV 张量登记成 NIXL 区域 → 与对端交换描述符
（握手）→ 发 READ/WRITE → 轮询 handle / 收 notif 判定完成。

# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1-L2598
# SUBTRACTED（逐条挂减法计划批准项）：
#   [1] 异构 TP 深分支：_build_local_splits_from_plan / src_xfer_handles_by_tp_ratio
#       的多读切分 / _map_block_ids_for_block_size_ratio / get_mapped_blocks /
#       _physical_blocks_per_logical_kv_block≠1 的重映射体；
#   [2] Mamba/HMA 深分支：_build_mamba_local/_build_mamba_remote/_stack_descs 的
#       conv 分解链、_sync_device_after_mamba_recv、ssm 描述符与 _conv_decomp；
#   [3] host buffer 旁路：initialize_host_xfer_buffer/save_kv_to_host/
#       sync_recved_kv_to_device/set_host_xfer_buffer_ops 的实现体（保留 use_host_buffer
#       判定与 wait_for_save 的判定注释）；
#   [4] packed / cross-layer 布局族：_register_packed_kv_cache、
#       register_cross_layers_kv_caches、跨层区域推断；
#   [0] 异构块尺寸的接收后处理：post_process_device_kv_on_receive*（含
#       heterogeneous_attn 与 host buffer 回拷）；
#   [6] 遥测：xfer_stats.record_* 的全部调用点与 get_xfer_telemetry；
#   [8] PP-aware 分支：pp_size>1 的握手 product 循环与 NotImplementedError 断言、
#       _remote_region_offset（pp_size 参数保留，默认 1）。
# 保留（减法计划明示 must_keep）：_apply_prefix_caching 的非 Mamba 尾裁剪路径、
#   _logical_to_kernel_block_ids（ratio==1 快路径）、_region_is_mla 形状位。
"""

import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    EngineTransferInfo,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
    ReqMeta,
    TransferHandle,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
    get_representative_spec_type,
    zmq_ctx,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.attention.backend import get_kv_cache_layout
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import select_common_block_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


# SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L90-L2598
class NixlBaseConnectorWorker:
    """Base implementation of Worker side methods shared by pull and push."""

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L93-L161
    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        """Compute NIXL descriptor IDs for given block IDs."""
        # SUBTRACTED: [2] SSM 区域的描述符计数与 conv 分解（num_ssm_regions 的
        #   计算与 `_has_mamba` 断言，L101-L107）——非 Mamba 模型恒 0 个 SSM 区域。
        num_ssm_regions = 0
        # SUBTRACTED: [1] block_size_ratio 的 kernel 块扩张（L110-L111）——对称
        #   P/D 的块尺寸相同，ratio 恒 1（调用点传 None 或 1）。
        num_blocks = dst_num_blocks
        num_fa_descs = self.num_regions * num_blocks

        # All-attention fast path: single vectorized broadcast.
        if num_ssm_regions == 0:
            # NOTE (NickLucche) With HMA, every kv group has the same number of layers
            # and layers from different groups share the same kv tensor.
            # eg block_ids=[[1, 2], [3]]->blocks [1, 2] need to be
            # read across all regions, same for [3], but group0-group1 blocks will
            # always differ (different areas). Therefore we can just flatten the
            # block_ids and compute the descs ids for all groups at once.
            block_arr = np.concatenate(block_ids)[None, :]
            region_ids = np.arange(self.num_regions)[:, None]
            return (region_ids * num_blocks + block_arr).flatten()

        # SUBTRACTED: [2] 逐组计算（FA 描述符与 SSM 描述符拼接、num_fa_descs 偏移）
        #   （L126-L161）——上面那条 all-attention 快路径就是本章的全部分支。
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L220-L246
    def _fa_desc_replicated(self, num_fa_descs: int) -> list[bool]:
        """Per-FA-descriptor replicate flag, in _build_fa_local emission order
        (region-major; one desc per block, with K/V packed). Length ``num_fa_descs``.
        """
        assert self.transfer_topo is not None
        n_regions = len(self.block_len_per_layer)
        if n_regions == 0 or self.num_regions == 0:
            return [False] * num_fa_descs
        nblk = num_fa_descs // self.num_regions
        flags: list[bool] = []
        for i in range(n_regions):
            replicated = self._is_region_replicated(i)
            flags.extend([replicated] * nblk)
        assert len(flags) == num_fa_descs, (
            f"FA desc flags {len(flags)} != num_fa_descs {num_fa_descs}"
        )
        return flags

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L238-L246
    def _is_region_replicated(self, region_idx: int) -> bool:
        """Whether region ``region_idx`` is transferred REPLICATE vs SPLIT.

        REPLICATE (MLA): identical on every rank, whole block read from one
        rank at offset 0, key-only. SPLIT (full-attn): head-sharded across TP.
        # SUBTRACTED: [1] MLA 区域的 REPLICATE 判定——本章 `_region_is_mla` 恒全
        #   False（MLA 规格的注册分支已删），故恒 SPLIT。
        """
        return region_idx < len(self._region_is_mla) and self._region_is_mla[region_idx]

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L248-L556
    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        # mypy will complain on re-assignment otherwise.
        self.block_size: int = cast(int, vllm_config.cache_config.block_size)

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L273-L277
        kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._lease_extension = kv_lease_duration * 2 // 3

        # SUBTRACTED: [5] 双向回拉旗标（L279-L283）——其消费分支已删；旗标保留在
        #   调度器侧（base_scheduler）以便读者看到部署门。

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )
        self.kv_cache_config = kv_cache_config
        self._layer_specs = {
            layer: group.kv_cache_spec
            for group in kv_cache_config.kv_cache_groups
            for layer in group.layer_names
        }
        self.hma_group_size = len(kv_cache_config.kv_cache_tensors)

        # ---- Model state (derived from model config) ----
        # SUBTRACTED: [2] Mamba conv 分解（_conv_decomp / mamba_ssm_size /
        #   is_conv_state_dim_first 断言，L300-L330）。`_has_mamba` 保留为常量位，
        #   值的来源仍是「有没有 MambaSpec 组」（宿主与本章用例恒 False）。
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        self._mamba_ssm_size = (0, 0)

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        self.nixl_wrapper = nixl_wrapper_cls(str(uuid.uuid4()), config)
        # Map of engine_id -> {(pp_rank, tp_rank): agent_name, ...}.
        # non-PP remote uses pp_rank 0, i.e. (0, tp_rank).
        self._remote_agents: dict[EngineId, dict[tuple[int, int], str]] = defaultdict(
            dict
        )
        # Map of engine_id -> clock offset.
        self._engine_clock_offset: dict[EngineId, float] = {}

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()

        self.num_blocks = kv_cache_config.num_blocks
        self.enable_permute_local_kv = False
        self.enable_heterogeneous_attn_post_process = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # reserve different cores for start_load_kv() from model_forward()
        if self.device_type == "cpu":
            numa_core_list = current_platform.discover_numa_topology()
            # setup one last core in each numa for kv transfer.
            rsv_cores_for_kv = [
                max(each_numa_core_list) for each_numa_core_list in numa_core_list
            ]

            if rsv_cores_for_kv:
                if not hasattr(os, "sched_setaffinity"):
                    raise NotImplementedError(
                        "os.sched_setaffinity is not available on this platform"
                    )
                os.sched_setaffinity(0, rsv_cores_for_kv)

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda", "xpu"]:
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0

        # SUBTRACTED: [8] PP 层切片（pp_size / _remote_region_offset 与两条
        #   NotImplementedError 断言，L434-L449）——pp_size 参数保留，默认 1。
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        # Keep heartbeat handshakes to a PP-sharded producer notif-only.
        self._hb_handshake_notif_only = False

        # nixl_prepped_dlist_handle.
        self.src_xfer_handles_by_block_size: dict[int, int] = {}
        # Local descriptor arrays per remote block size (block_size_ratio>1),
        # kept for building per-tp-ratio splits at the same granularity.
        self.src_blocks_data_by_block_size: dict[int, np.ndarray] = {}
        # SUBTRACTED: [1] src_xfer_handles_by_tp_ratio（异构 TP 的按源 rank 切分
        #   句柄表，L458-L460）。
        # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
        self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Invalid blocks from failed NIXL operations (thread-safe queue of block ids)
        self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()
        # requests that skipped transfer (handshake or transfer failures)
        # Uses Queue for thread-safe cross-thread coordination with the
        # background handshake thread, matching the _ready_requests pattern.
        self._failed_recv_reqs: queue.Queue[ReqId] = queue.Queue()

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[
            EngineId, Future[tuple[dict[tuple[int, int], str], float]]
        ] = {}
        # Protects _handshake_futures and _remote_agents.
        self._handshake_lock = threading.RLock()

        # TTL-based eviction of stale remote engine state.
        self._engine_last_active: dict[EngineId, float] = {}
        self._engine_ttl: float = vllm_config.kv_transfer_config.get_from_extra_config(
            "engine_ttl", 3600.0
        )

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config

        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.backend_name = self.attn_backends[0].get_name()

        self.kv_cache_layout = get_kv_cache_layout()
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.info(
            "Detected attention backend(s) %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)

        # lazy initialized in register_kv_caches
        self.compat_hash: str | None = None
        self.transfer_topo: TransferTopology | None = None

        # With heterogeneous TP, P must wait for all assigned D TP workers to
        # finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        # SUBTRACTED: [6] xfer_stats（NixlKVConnectorStats）——遥测旁路。

        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        # Unwrap UniformTypeKVCacheSpecs to get the representative spec type
        self._group_spec_types = tuple(
            get_representative_spec_type(g.kv_cache_spec)
            for g in self.kv_cache_config.kv_cache_groups
        )

        # Per-region MLA flag, 1:1 with block_len_per_layer. True -> REPLICATE
        # (MLA), False -> SPLIT (head-sharded full-attn).
        # SUBTRACTED: [1] MLA 区域的识别——列表保形（恒 False），使
        #   `_is_region_replicated` 的逐区域查询仍成立。
        self._region_is_mla = list[bool]()

        # Enable different block lengths for different layers *only* when MLA is used.
        # This is not used for SSM layers, which use the counterpart `mamba_ssm_size`.
        self.block_len_per_layer = list[int]()

        # Per-engine TP mappings. Generated during handshake.
        self.tp_mappings: dict[EngineId, TPMapping] = {}

        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L558-L575
    def _sync_block_size_with_kernel(self) -> None:
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        # Number of blocks not accounting for kernel block mismatches
        self._logical_num_blocks = self.num_blocks
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size
            self.num_blocks *= self._physical_blocks_per_logical_kv_block

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L577-L724
    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Do a NIXL handshake with a remote instance."""

        # the first time we connect to a remote agent.
        # be careful, the handshake happens in a background thread.
        # it does not have an active cuda context until any cuda runtime
        # call is made. when UCX fails to find a valid cuda context, it will
        # disable any cuda ipc communication, essentially disabling any NVLink
        # communication.
        # when we are using device buffers, we need to set the device
        # explicitly to make sure the handshake background thread has a valid
        # cuda context.
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        assert self.transfer_topo is not None
        p_remote_ranks = self.transfer_topo.handshake_target_ranks(remote_tp_size)
        remote_rank_to_agent_name: dict[tuple[int, int], str] = {}
        path = make_zmq_path("tcp", host, port)
        # Clock offset to the peer, estimated from the handshake round-trip.
        # Keep the lowest-RTT sample: hop cost is ~uniform across ranks, so a
        # higher RTT is just noise that skews the midpoint estimate.
        best_rtt = float("inf")
        best_offset: float | None = None

        with zmq_ctx(zmq.REQ, path) as sock:
            # SUBTRACTED: [8] PP 维度的 product（`range(remote_pp_size)`）与
            #   pp_size>1 的断言——1P1D 主线只有 pp_rank 0；remote_pp_size 参数
            #   保留以对齐调用点。
            for remote_rank in p_remote_ranks:
                remote_pp_rank = 0
                logger.debug(
                    "Querying metadata on path: %s at remote pp rank %s, tp rank %s",
                    path,
                    remote_pp_rank,
                    remote_rank,
                )

                # Send query for the request.
                msg = msgspec.msgpack.encode(
                    (GET_META_MSG, remote_pp_rank, remote_rank)
                )
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                start_time = time.perf_counter()
                sock.send(msg)
                reply_parts = sock.recv_multipart()
                recv_time = time.perf_counter()
                assert len(reply_parts) == 2
                handshake_bytes = reply_parts[0]

                remote_perf = msgspec.msgpack.decode(reply_parts[1])
                rtt = recv_time - start_time
                if rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = remote_perf - (start_time + recv_time) / 2

                # Decode handshake payload to get compatibility hash
                handshake_decoder = msgspec.msgpack.Decoder(NixlHandshakePayload)
                try:
                    handshake_payload = handshake_decoder.decode(handshake_bytes)
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hash BEFORE decoding agent metadata
                assert self.compat_hash is not None
                if (
                    self.enforce_compat_hash
                    and handshake_payload.compatibility_hash != self.compat_hash
                ):
                    raise RuntimeError(
                        f"NIXL compatibility hash mismatch. "
                        f"Local: {self.compat_hash}, "
                        f"Remote: {handshake_payload.compatibility_hash}. "
                        f"Prefill and decode instances have incompatible "
                        f"configurations. This may be due to: different vLLM versions,"
                        f" models, dtypes, KV cache layouts, attention backends, etc. "
                        f"Both instances must use identical configurations."
                        f"Disable this check using "
                        f'--kv-transfer-config \'{{"kv_connector_extra_config": '
                        f'{{"enforce_handshake_compat": false}}}}\''
                    )

                logger.info(
                    "NIXL compatibility check passed (hash: %s)",
                    handshake_payload.compatibility_hash,
                )

                # Decode agent metadata
                metadata_decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
                try:
                    metadata = metadata_decoder.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as e:
                    # This should not happen if hash matched
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )

                # Register Remote agent.
                if notif_agents_only:
                    remote_agent_name = self._add_notif_only_remote_agent(
                        metadata, remote_tp_size
                    )
                else:
                    remote_agent_name = self.add_remote_agent(
                        metadata, remote_rank, remote_tp_size
                    )
                setup_agent_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: add agent took: %s (notif_agents_only=%s)",
                    setup_agent_time - got_metadata_time,
                    notif_agents_only,
                )
                remote_ranks = (remote_pp_rank, remote_rank)
                remote_rank_to_agent_name[remote_ranks] = remote_agent_name

        assert best_offset is not None
        return remote_rank_to_agent_name, best_offset

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L726-L745
    def _add_notif_only_remote_agent(
        self, metadata: NixlAgentMetadata, remote_tp_size: int
    ) -> str:
        """Load a remote agent for notifs only on the push-mode decode side.

        Skips descriptor setup but records engine info for block accounting.
        """
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=metadata.block_lens[0],
                remote_physical_blocks_per_logical=(
                    metadata.physical_blocks_per_logical_kv_block
                ),
            ),
        )
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L747-L802
    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu).

        # SUBTRACTED: [3] 实现体（逐层 NHD/HND 形状换算与 CPU 张量分配，
        #   L752-L790）——宿主平台 device_type == "cpu" ⇒ use_host_buffer 恒 False。
        """
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L792-L801
    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used.

        # SUBTRACTED: [3] 实现体（kv_buffer_device/device_type 判定与
        #   copy_blocks 赋值）——调用点仍在 connector facade 的转发链上。
        """
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L803-L851
    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: BaseException | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": sum(
                        len(group) for group in meta.local_block_ids
                    ),
                    "num_remote_blocks": sum(
                        len(group) for group in meta.remote.block_ids
                    ),
                    "local_block_ids_sample": meta.local_block_ids[0][:10]
                    if meta.local_block_ids
                    else [],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L853-L909
    def _ensure_handshake(
        self,
        engine_id: EngineId,
        host: str,
        port: int,
        tp_size: int,
        pp_size: int = 1,
        notif_agents_only: bool = False,
    ) -> Future[tuple[dict[tuple[int, int], str], float]] | None:
        """
        Ensure a handshake is in-flight (or already done) for *engine_id*.

        Returns the ``Future`` if a handshake is pending (or was just
        started), or ``None`` if the handshake already completed
        successfully.  Callers can attach per-request callbacks to the
        returned future.
        Failures to handshake are logged and the request is marked as failed.
        """
        self._evict_stale_engines()
        with self._handshake_lock:
            if engine_id in self._remote_agents:
                return None
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
                pp_size,
                notif_agents_only,
            )
            self._handshake_futures[engine_id] = fut

            # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L889-L906
            def done_callback(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                eid=engine_id,
            ):
                with self._handshake_lock:
                    del self._handshake_futures[eid]
                    try:
                        remote_agents, clock_offset = f.result()
                        self._remote_agents[eid] = remote_agents
                        self._engine_clock_offset[eid] = clock_offset
                        self._engine_last_active[eid] = time.perf_counter()
                    except Exception as e:
                        self._log_failure(
                            failure_type="handshake_setup_failed",
                            req_id=None,
                            error=e,
                            remote_engine_id=eid,
                        )

            fut.add_done_callback(done_callback)
            return fut

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L911-L941
    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        assert meta.remote is not None
        fut = self._ensure_handshake(
            remote_engine_id,
            meta.remote.host,
            meta.remote.port,
            meta.tp_size,
        )
        if fut is None:
            # Already handshaked — only happens if caller does not pre-check.
            self._ready_requests.put((req_id, meta))
            return

        # Check handshake success before proceeding with request.
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L928-L940
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            try:
                f.result()
                self._ready_requests.put(entry)
            except Exception as e:
                self._log_failure(
                    failure_type="handshake_failed",
                    req_id=req_id,
                    error=e,
                    meta=meta,
                )
                self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L943-L952
    def register_cross_layers_kv_caches(self, kv_cache: torch.Tensor) -> None:
        """Register a cross-layers KV cache tensor with NIXL."""
        # SUBTRACTED: [4] 跨层单张量注册——走同一 register_kv_caches 入口即可，
        #   本章不做跨层布局优化（opt-in 默认关）。
        first_layer = next(iter(self._layer_specs))
        self.register_kv_caches({first_layer: kv_cache})

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1037-L1277
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        # SUBTRACTED: [4] packed 分配探测（同一 storage 多 data_ptr 时走
        #   _register_packed_kv_cache，L1040-L1052）。

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, self.backend_name, self.transfer_topo.cross_layers_blocks
        )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        # With hybrid allocator, layers can share a kv cache tensor
        seen_base_addresses: list[int] = []

        # K and V are packed into the content dim, so each attention layer is a
        # single NIXL region whose block transfers as one unit.
        tensor_size_bytes = None

        for layer_name, cache in xfer_buffers.items():
            # NOTE (NickLucche) Hybrid SSM mamba/FA physical page_size may differ when
            # kernel requires a specific block size. This leads to SSM and FA layers
            # having different num_blocks.
            # `_physical_blocks_per_logical_kv_block` ratio is used to adjust for this.
            layer_spec = self._layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s as no KVCache spec is present. "
                    "This is likely because the layer is sharing its KV cache",
                    layer_name,
                )
                continue
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                # DSA Indexer case: UniformTypeKVCacheSpecs merges kv_cache_specs
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            # `layer_spec.page_size_bytes` only accounts for logical page_size, that is
            # the page_size assuming constant `self._logical_num_blocks`.
            # SUBTRACTED: [2] MambaSpec 的页大小直通臂——SSM 组归 ch14/ch16。
            physical_page_size = (
                layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            num_blocks = self.num_blocks
            # `page_size` accounts for physical blocks, st KVCache is always
            # [`num_blocks` * `page_size`]
            curr_tensor_size_bytes = num_blocks * physical_page_size

            base_addr = cache.data_ptr()
            # SUBTRACTED: [1] MLA 区域识别（MLAAttentionSpec / SlidingWindowMLASpec
            #   的 isinstance，L1142-L1144）——`_region_is_mla` 保形但恒 False。
            is_mla_region = False
            if base_addr in seen_base_addresses:
                # NOTE (NickLucche) HMA employs memory pooling to share tensors
                # across groups. This results in skipping all tensors but the ones
                # pointed to by group0. Also, generally we will have more blocks
                # per tensor but fewer regions.
                idx = seen_base_addresses.index(base_addr)
                self._region_is_mla[idx] |= is_mla_region
                logger.debug("Skipping %s because it's already seen", layer_name)
                continue
            logger.debug(
                "Registering layer %s with cache shape: %s", layer_name, cache.shape
            )
            seen_base_addresses.append(base_addr)
            # SUBTRACTED: [2] Mamba 页长的 `physical_page_size //
            #   physical_blocks_per_logical` 折算（L1162-L1165）——非 Mamba 层
            #   逐字保留：
            self.block_len_per_layer.append(physical_page_size)
            self._region_is_mla.append(is_mla_region)

            if not is_mla_region:
                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All non-MLA kv cache tensors must have the same size"
                )

            # When there's a mismatch between kbs<>bs, we rely on HMA to ensure
            # caches are either [NB, PS] or [NB*r, PS/r] where r is bs/kbs.
            if (
                self._physical_blocks_per_logical_kv_block == 1
                and cache.shape[0] != num_blocks
            ):
                raise AssertionError(
                    "All kv cache tensors must have the same number of "
                    f"blocks; layer={layer_name}, "
                    f"expected_num_blocks={num_blocks}, "
                    f"cache_shape={tuple(cache.shape)}, "
                    f"cache_stride={tuple(cache.stride())}, "
                    f"layer_spec={type(layer_spec).__name__}, "
                    f"backend={self.backend_name}, "
                    "all_backends="
                    f"{[backend.get_name() for backend in self.attn_backends]}, "
                    f"kv_cache_layout={self.kv_cache_layout}"
                )

            # Need to make sure the device ID is non-negative for NIXL,
            # Torch uses -1 to indicate CPU tensors.
            self.device_id = max(cache.get_device(), 0)
            caches_data.append((base_addr, curr_tensor_size_bytes, self.device_id, ""))

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert (
            len(self.block_len_per_layer)
            == len(seen_base_addresses)
            == len(self._region_is_mla)
        )

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(caches_data)

        # SUBTRACTED: [8] PP 层区间的远端区域偏移（L1213-L1220）。

        # Total local FA descriptors (boundary between FA and mamba descs).
        self.num_descs = self.num_regions * self.num_blocks

        descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
        logger.debug("Registering descs: %s", caches_data)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        logger.debug("Done registering descs")
        self._registered_descs.append(descs)

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        # SUBTRACTED: [2] Hybrid SSM 注册日志（L1234-L1246）。

        # Register local/src descr for NIXL xfer.
        self.src_xfer_handles_by_block_size[self.block_size], self.src_blocks_data = (
            self.register_local_xfer_handler(self.block_size)
        )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1384-L1402
    def _build_fa_local(
        self,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> np.ndarray:
        """Build local FA descriptors for all layers as an Nx3 uint64 array."""
        assert self.transfer_topo is not None
        assert base_addresses, "Local KV cache base addresses must not be empty."
        num_blocks = self.num_blocks * block_size_ratio
        device_id = self.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        parts: list[np.ndarray] = []
        for i, base_addr in enumerate(base_addresses):
            # K/V are packed into the content dim, so the whole block transfers
            # as one unit: desc length equals the block stride.
            block_len = self.block_len_per_layer[i] // block_size_ratio
            addrs = base_addr + block_arange * block_len
            parts.append(self._stack_descs(addrs, block_len, device_id))
        return np.concatenate(parts)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1376-L1382
    @staticmethod
    def _stack_descs(addrs: np.ndarray, length: int, device_id: int) -> np.ndarray:
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1376-L1382
        out = np.empty((addrs.shape[0], 3), dtype=np.uint64)
        out[:, 0] = addrs
        out[:, 1] = length
        out[:, 2] = device_id
        return out

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1404-L1445
    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
    ) -> np.ndarray:
        """Build remote FA descriptors for all layers as an Nx3 uint64 array."""
        assert self.transfer_topo is not None
        assert nixl_agent_meta.kv_caches_base_addr, (
            "Remote KV cache base addresses must not be empty."
        )
        # SPLIT regions read their head slice from this many remote ranks at a
        # per-rank offset; REPLICATE regions read the whole block once.
        # SUBTRACTED: [1] `split_reads = len(plan.source_ranks_per_group[fa_idx])`
        #   与按头切分的 num_reads/rank_offset（L1418-L1440）——对称 TP 下每个
        #   区域都是「整块读一次」，局部块长即传输单位。
        num_blocks = nixl_agent_meta.num_blocks
        device_id = nixl_agent_meta.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        parts: list[np.ndarray] = []
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            replicated = self._is_region_replicated(i)
            # Read our whole local region size from remote..
            local_block_len = self.block_len_per_layer[i]
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                # ..using remote kv_block_len as transfer unit
                local_block_len = remote_kv_block_len

            # REPLICATE reads the whole block once at offset 0; SPLIT gathers
            # its head slice from `split_reads` remote ranks at a per-rank offset.
            num_reads = 1
            rank_offset = 0
            local_block_len = local_block_len // num_reads

            page_size = nixl_agent_meta.block_lens[i]
            addrs = base_addr + rank_offset + block_arange * page_size
            parts.append(self._stack_descs(addrs, local_block_len, device_id))
        return np.concatenate(parts)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1447-L1488
    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, np.ndarray]:
        """
        Function used for register local xfer handler with local block_size or
        Remote block_size.

        When local block_size is same as remote block_size, we use local block_size
        to register local_xfer_handler during init.
        """
        assert self.transfer_topo is not None
        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data = self._build_fa_local(local_base_addresses, block_size_ratio)
        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            len(blocks_data),
            self.engine_id,
            self.tp_rank,
            self.device_id,
        )
        # SUBTRACTED: [2] Mamba 的 4 区域拼接（_build_mamba_local，L1474-L1484）。

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        return self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs), blocks_data

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1490-L1696
    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.

        # SUBTRACTED: [1] 异构 TP 的 rank_offset 说明与 split handles（原 docstring
        #   的 D>P / P>D 示例段与 L1637-L1660 的 src_xfer_handles_by_tp_ratio 注册）；
        #   [2] Mamba hetero-TP 的 3-read 说明。
        """
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if (0, remote_tp_rank) in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][(0, remote_tp_rank)]

        # SUBTRACTED: [8] PP 层窗口切片（L1550-L1565）——本 worker 持全模型层。

        ### Register remote engine in TransferTopology (idempotent).
        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        # Example:
        # block_size_ratio > 1:
        # remote:               | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|
        # local origin:|          0|          1|          8|         12|
        # local mapped:| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
        block_size_ratio = transfer_topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

        # This is 1 when P and D `--tensor-parallel-size` match. Otherwise,
        # this is the ratio between the two sizes.
        tp_ratio = transfer_topo.tp_ratio(remote_tp_size)

        logger.debug(
            "Registering remote agent (%s, rank %s) memory regions with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        plan = self.tp_mappings[engine_id]

        ### (Optional) Register a local handler at the remote engine's block
        ### granularity (remote/prefill blocks smaller than local).
        # SUBTRACTED: [1] block_size_ratio>1 时的额外本地句柄注册（L1623-L1634）
        #   与 tp_ratio<0 的 per-source 切分句柄（L1637-L1660）——对称部署下
        #   块尺寸相同、ratio 恒 1，本地句柄就是初始化时那一个。
        remote_block_size = nixl_agent_meta.block_size
        src_blocks_data = self.src_blocks_data

        ### Register remote agent memory regions
        # With homogeneous TP, D pulls the whole kv cache from corresponding rank.

        # Register all remote blocks, but only the corresponding kv heads.
        blocks_data = self._build_fa_remote(
            plan,
            nixl_agent_meta,
            block_size_ratio,
        )
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )
        # SUBTRACTED: [2] Mamba 远端区域拼接（_build_mamba_remote，L1681-L1688）。

        # Register with NIXL.
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        self.dst_xfer_side_handles[engine_id][remote_tp_rank] = (
            self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        )

        return remote_agent_name

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1698-L1867
    def _validate_remote_agent_handshake(
        self, nixl_agent_meta: NixlAgentMetadata, remote_tp_size: int
    ):
        """
        Validate the remote agent handshake metadata ensuring the
        invariants hold true.

        # SUBTRACTED: [1] 异构 TP 的 kv 复制/头切分断言与 enable_permute_local_kv、
        #   [2] Mamba 的 physical_per_logical 与 ssm_sizes 校验——对称同构部署下
        #   只剩本条主断言：两侧的层数、块长与块数必须逐项相等。
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )

        # Heterogeneous TP requires head-splitting, which only works with
        # HND layout.
        # SUBTRACTED: [1] abs(tp_ratio)!=1 与 is_kv_replicated 的分支（L1788-L1801）。

        # Per-region block_len validation enforcing the P/D invariant.
        assert len(self.block_len_per_layer) == len(nixl_agent_meta.block_lens), (
            "Number of KV layers must match between prefill and decode"
        )
        for i, local_len in enumerate(self.block_len_per_layer):
            replicated = self.use_mla or self._is_region_replicated(i)
            remote_len = nixl_agent_meta.block_lens[i]
            if replicated:
                assert local_len // block_size_ratio == remote_len, (
                    "KV cache sizes must match between P and D when "
                    f"replicated (region {i}: local={local_len}, "
                    f"remote={remote_len}, bsr={block_size_ratio})."
                )
            else:
                # Symmetric TP: the head shard per rank is the same on both sides.
                # SUBTRACTED: [1] `remote_len == local_len * remote_heads //
                #   local_heads // bsr` 的异构断言（L1843-L1862）——两侧同 TP
                #   时 local_heads == remote_heads，退化为逐项相等。
                assert remote_len * block_size_ratio == local_len, (
                    f"SPLIT region {i}: remote P KV block_len {remote_len} "
                    f"must equal local {local_len} (bsr={block_size_ratio}) "
                    "under homogeneous TP."
                )

        # TP workers that handhshake with same remote have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        # Same number of regions/~layers.
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(self.block_len_per_layer)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1869-L1890
    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device.

        # SUBTRACTED: [3] 实现体（逐组的 h2d copy_blocks 调用）——host buffer 旁路。
        """
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L1892-L1916
    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer.

        # SUBTRACTED: [3] 实现体（逐请求 d2h copy_blocks）——host buffer 旁路。
        """
        raise NotImplementedError

    # SUBTRACTED: [0] post_process_device_kv_on_receive /
    #   post_process_device_kv_on_receive_heterogeneous_attn（L1934-L2042）与
    #   _attention_kv_caches（L1918-L1932）——异构块尺寸/布局的接收后处理。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2044-L2164
    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        done_recving = self._pop_done_transfers(self._recving_transfers)

        # Drain queue of requests where handshake or transfer setup failed.
        failed_recv_reqs = set[ReqId]()
        while not self._failed_recv_reqs.empty():
            try:
                failed_recv_reqs.add(self._failed_recv_reqs.get_nowait())
            except queue.Empty:
                break

        # Add failed requests to done_recving for scheduler tracking
        # (blocks are already marked invalid, scheduler will handle recompute)
        done_recving.update(failed_recv_reqs)

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving (%s failed)",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
                len(failed_recv_reqs),
            )

        # SUBTRACTED: [0] 接收后的 host buffer 回拷与异构块/布局后处理
        #   （L2076-L2143，含 _sync_device_after_mamba_recv 的 [2] Mamba 同步）。

        # Handle timeout to avoid stranding blocks on remote.
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2145-L2162
        now = time.perf_counter()
        while self._reqs_to_send:
            req_id, expires = next(iter(self._reqs_to_send.items()))
            # Sorted dict, oldest requests are put first so we can exit early.
            if now < expires:
                break
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d remote worker(s) before lease expired.",
                req_id,
                count,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            done_sending.add(req_id)

        return done_sending, done_recving

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2182-L2187
    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer notification.

        Subclasses must implement this to handle mode-specific notifications.
        """
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2189-L2208
    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases for requests referenced in a heartbeat.

        Args:
            payload: comma-separated P-side request IDs, e.g.
                     "req_abc,req_def".
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)
                logger.debug(
                    "Heartbeat extended lease for request %s "
                    "by %ds (old_expiry=%.1f, new_expiry=%.1f)",
                    req_id,
                    self._lease_extension,
                    old,
                    new_expiry,
                )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2210-L2255
    def _pop_done_transfers(self, transfers: dict[str, list[int]]) -> set[str]:
        """
        Pop completed xfers by checking for DONE state.
        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids that have all done xfers
        """
        done_req_ids: set[str] = set()
        for req_id, handles in list(transfers.items()):
            in_progress = []
            for handle in handles:
                try:
                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                    if xfer_state == "DONE":
                        # SUBTRACTED: [6] get_xfer_telemetry 与 xfer_stats.record_transfer。
                        self.nixl_wrapper.release_xfer_handle(handle)
                    elif xfer_state == "PROC":
                        in_progress.append(handle)
                        continue
                    else:
                        self._log_failure(
                            failure_type="transfer_failed",
                            msg="Marking blocks as invalid",
                            req_id=req_id,
                            xfer_state=xfer_state,
                        )
                        self._handle_failed_transfer(req_id, handle)
                except Exception as e:
                    self._log_failure(
                        failure_type="transfer_exception",
                        msg="Marking blocks as invalid",
                        req_id=req_id,
                        error=e,
                    )
                    self._handle_failed_transfer(req_id, handle)

            if not in_progress:
                # Only report request as completed when all transfers are done.
                done_req_ids.add(req_id)
                del transfers[req_id]
            else:
                transfers[req_id] = in_progress
        return done_req_ids

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2257-L2273
    def _handle_failed_transfer(self, req_id: str, handle: int | None):
        """
        Handle a failed transfer by marking all (logical) blocks as invalid and
        recording the failure.

        Args:
            req_id: The request ID.
            handle: The transfer handle.
        """
        # Use .get() here as the metadata cleanup is handled by get_finished()
        # TODO (NickLucche) handle failed transfer for HMA.
        if (meta := self._recving_metadata.get(req_id)) and not self._is_hma_required:
            self._invalid_block_ids.put(set(meta.local_block_ids[0]))
        self._failed_recv_reqs.put(req_id)
        if handle is not None:
            self.nixl_wrapper.release_xfer_handle(handle)
        # SUBTRACTED: [6] xfer_stats.record_failed_transfer。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2275-L2305
    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """
        Send heartbeat notifications to remote engines, extending lease on KV blocks.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            if (
                self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    hb_info.pp_size,
                    self._hb_handshake_notif_only and hb_info.pp_size > 1,
                )
                is not None
            ):
                continue  # handshake is still pending

            # Build the heartbeat message: "HB:req1,req2,..."
            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    # SUBTRACTED: [1] get_mapped_blocks / _map_block_ids_for_block_size_ratio
    #   （L2307-L2365）——异构块尺寸的块号重映射。

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2367-L2396
    def _logical_to_kernel_block_ids(self, block_ids: BlockIds, ratio: int) -> BlockIds:
        """
        Convert block ids to kernel physical block ids.
        This is required when the logical block size (the one set by the user)
        does not match the one required by the attn backend.
        `ratio` is the number of physical blocks per logical block.
        We always receive logical blocks from the engine, so we expand them here eg:
        logical block ids: [(SW-clipped) [1], (FA) [2, 3]], ratio=2
        physical block ids: [(SW-clipped) [2, 3], (FA) [4, 5, 6, 7]]
        """
        if ratio == 1:
            # Noop when physical and logical block sizes are the same
            return block_ids
        # SUBTRACTED: [1] ratio>1 的逐块扩张（BlockTable.map_to_kernel_blocks，
        #   L2380-L2396）——逻辑块尺寸与 kernel 块尺寸相同时不触发；宿主的
        #   注意力后端接受任意块尺寸，ratio 恒 1。
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2398-L2486
    def _apply_prefix_caching(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        remote_physical_per_logical: int,
    ) -> tuple[BlockIds, list]:
        """Apply prefix caching by trimming local/remote block ID lists.

        For non-Mamba models: end-trim remote to match local count, so that
        already-cached prefix blocks are skipped in the transfer.
        """
        # Partial prefix cache hit: just read uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        # (This is the general path: D may hold a partial local prefix cache,
        # eg a shared system prompt reused across requests.)
        remote_block_ids = list(remote_block_ids)
        if not self._has_mamba:
            for i, remote_group in enumerate(remote_block_ids):
                num_local_blocks = len(local_block_ids[i])
                assert num_local_blocks <= len(remote_group)
                if num_local_blocks < len(remote_group):
                    remote_block_ids[i] = remote_group[-num_local_blocks:]
        else:
            # SUBTRACTED: [2] Mamba 混合模型的 front-trim（SSM 槽位对齐与
            #   physical_per_logical 取整裁剪，L2423-L2485）——混合模型归
            #   ch14/ch16 的 HMA 语义。
            raise NotImplementedError
        return local_block_ids, remote_block_ids

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2488-L2495
    def get_kv_connector_stats(self) -> Any | None:
        """
        Get the KV transfer stats for the connector.

        # SUBTRACTED: [6] xfer_stats 克隆/复位（L2492-L2495）——遥测旁路；
        #   契约签名保留，恒返回 None。
        """
        return None

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2497-L2511
    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        # Drain the queue (thread-safe, no lock needed).
        result: set[int] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2513-L2536
    def _evict_stale_engines(self) -> None:
        """Scan for and evict remote engines that have exceeded their TTL.

        Called from the main thread in when a new remote engine appears.
        We can only go OOM as we discover and register a new remote, therefore we make
        sure we clean up stale engine data structures before then. This invariant
        prevents us from using background threads, though memory usage is not guaranteed
        to be "optimal" until a new handshake is performed.

        Engines with active transfers or pending handshakes cannot be stale:
        - Active transfers touch _engine_last_active in start_load_kv.
        - Pending handshakes don't have an _engine_last_active entry yet
        """
        if self._engine_ttl <= 0:
            return

        now = time.perf_counter()
        for eid, last_active in list(self._engine_last_active.items()):
            if now - last_active > self._engine_ttl:
                self._cleanup_remote_engine(eid)

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2538-L2572
    def _cleanup_remote_engine(
        self, engine_id: EngineId, *, log_eviction: bool = True
    ) -> None:
        """Remove all state for a single remote engine.

        Releases NIXL resources (dlist handles, remote agents) and clears
        all per-engine data structures. Used by both TTL eviction and
        shutdown.
        """
        assert engine_id in self._remote_agents

        # Notif-only engines (push-mode D side) have no descriptor state.
        for handle in self.dst_xfer_side_handles.pop(engine_id, {}).values():
            self.nixl_wrapper.release_dlist_handle(handle)
        for agent_name in self._remote_agents.pop(engine_id).values():
            self.nixl_wrapper.remove_remote_agent(agent_name)

        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_num_blocks.pop(engine_id, None)
        self.tp_mappings.pop(engine_id, None)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

        # Drop the cached clock offset; it is re-measured on the next handshake.
        self._engine_clock_offset.pop(engine_id, None)
        # Push P-side engines are tracked in _remote_agents but not in
        # _engine_last_active (they don't participate in stale eviction), so
        # tolerate a missing entry.
        last_active = self._engine_last_active.pop(engine_id, None)
        if log_eviction and last_active is not None:
            logger.info(
                "Evicted stale remote engine %s (inactive for %.1fs).",
                engine_id,
                time.perf_counter() - last_active,
            )

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2574-L2575
    def __del__(self):
        self.shutdown()

    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:L2577-L2598
    def shutdown(self):
        """Shutdown the connector worker."""
        if not hasattr(self, "_handshake_initiation_executor"):
            # error happens during init, no need to shutdown
            return
        self._handshake_initiation_executor.shutdown(wait=False)
        for handles in self._recving_transfers.values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        self._recving_transfers.clear()
        for handle in self.src_xfer_handles_by_block_size.values():
            self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_block_size.clear()
        # SUBTRACTED: [1] src_xfer_handles_by_tp_ratio 的释放（L2590-L2593）。
        for engine_id in list(self._remote_agents):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()


__all__ = ["NixlBaseConnectorWorker"]
