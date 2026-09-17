# ch38 测试脚手架：把「一个引擎的调度器半边 + worker 半边」装配进同一进程。
#
# 保留的边界（与真实部署同构）：
#   * 调度器侧 connector ↔ worker 侧 connector 只经 OffloadingConnectorMetadata
#     通讯（build_connector_meta → worker.start_kv_transfers / prepare_store_kv /
#     get_finished → OffloadingWorkerMetadata 回传）；
#   * CPUOffloadingManager（块池账本：准入/驱逐/ref_cnt/四态 lookup）是**真身**；
#   * 文件描述符级 DMA 寻址（compute_sub_block_ptrs）是**真身**。
#
# 唯一被替换的是 CUDA DMA 事件循环（host 无 CUDA，见 vllm/platforms/__init__.py
# 的 seam 说明）：HostOffloadingWorker 用同一套描述符数学同步搬字节——
# 真源的 stream/event 时序（wait_stream 计算流 / 同向保序 / SRC_ACCESS_ORDER_ANY）
# 无法在 host 观察，控制流与字节流不变。
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

from vllm.config import (  # noqa: E402
    CacheConfig,
    KVEventsConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.kv_transfer import KVTransferConfig  # noqa: E402
from vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: E402
from vllm.v1.core.sched.output import (  # noqa: E402
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_offload.base import (  # noqa: E402
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

BLOCK_SIZE = 4
NUM_KV_HEADS = 2
HEAD_SIZE = 8
LAYER_NAMES = ["layer.0.attn.kv", "layer.1.attn.kv"]
NUM_BLOCKS = 8


# ═══════════════════════ 配置装配 ═══════════════════════


def make_kv_config(
    num_blocks: int = NUM_BLOCKS,
    block_size: int = BLOCK_SIZE,
    num_kv_heads: int = NUM_KV_HEADS,
    head_size: int = HEAD_SIZE,
    layer_names: list[str] | None = None,
) -> KVCacheConfig:
    layer_names = list(layer_names or LAYER_NAMES)
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float32,
    )
    tensor_bytes = spec.page_size_bytes * num_blocks
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_bytes, shared_by=list(layer_names))
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)
        ],
    )


def make_kv_caches(kv_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """按页几何分配：共享同一 KVCacheTensor 的层共享同一物理存储（源码断言同 data_ptr）。"""
    caches: dict[str, torch.Tensor] = {}
    for kv_cache_tensor in kv_config.kv_cache_tensors:
        shared_names = kv_cache_tensor.shared_by
        spec = next(
            g.kv_cache_spec
            for g in kv_config.kv_cache_groups
            if shared_names[0] in g.layer_names
        )
        base = torch.zeros(
            (kv_config.num_blocks, spec.page_size_bytes), dtype=torch.int8
        )
        for name in shared_names:
            caches[name] = base
    return caches


def make_vllm_config(
    *,
    engine_id: str = "test-engine",
    extra_config: dict | None = None,
    world_size: int = 1,
    tp_size: int = 1,
    block_size: int = BLOCK_SIZE,
    enable_prefix_caching: bool = True,
    enable_kv_events: bool = False,
    self_describing_kv_events: bool = False,
    connector: str = "OffloadingConnector",
) -> VllmConfig:
    extra = dict(extra_config or {})
    if self_describing_kv_events:
        extra["self_describing_kv_events"] = True
    return VllmConfig(
        model_config=ModelConfig(model="dummy-model", dtype="float32"),
        cache_config=CacheConfig(
            block_size=block_size, enable_prefix_caching=enable_prefix_caching
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            rank=0,
            world_size=world_size,
        ),
        scheduler_config=SchedulerConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            engine_id=engine_id,
            kv_connector_extra_config=extra,
        ),
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=enable_kv_events,
            self_describing_kv_events=self_describing_kv_events,
        ),
    )


def make_request(
    request_id: str,
    prompt_token_ids: list[int],
    kv_transfer_params: dict[str, Any] | None = None,
) -> Request:
    req = Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
    )
    if kv_transfer_params is not None:
        req.kv_transfer_params = dict(kv_transfer_params)
    return req


def fill_block_hashes(req: Request, num_hashes: int, seed: int = 0) -> list[bytes]:
    """链式块哈希（测试里用确定性字节序列；真实链式哈希算法属 ch15）。"""
    hashes: list[bytes] = []
    for i in range(num_hashes):
        h = (seed * 1_000_003 + i).to_bytes(32, "big")
        hashes.append(h)
    req.block_hashes.extend(hashes)
    return hashes


# ═══════════════════════ GPU 块视图（update_state_after_alloc 的入参） ═══════════════════════


@dataclass
class FakeBlock:
    block_id: int
    is_null: bool = False
    block_hash: bytes | None = None


def make_blocks(group_block_ids: list[list[int]], hashed: int | None = None) -> KVCacheBlocks:
    """hashed=N：前 N 个块带哈希（=本地前缀缓存已命中区），其余 block_hash=None。"""
    groups: tuple[list[FakeBlock], ...] = tuple(
        [
            FakeBlock(bid, block_hash=(f"h{bid}".encode() if (hashed is not None and i < hashed) else None))
            for i, bid in enumerate(ids)
        ]
        for ids in group_block_ids
    )
    return KVCacheBlocks(blocks=groups)


# ═══════════════════════ host DMA 替身（唯一的 SEAM） ═══════════════════════


class HostOffloadingWorker(OffloadingWorker):
    """同步执行真描述符数学的 host DMA 替身。

    与 SingleDirectionOffloadingHandler 相同的寻址语义（compute_sub_block_ptrs
    + group_sizes/block_indices 半块对齐），只是同步完成、事件即完即报。
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_blocks: int,
    ) -> None:
        self.blocks_per_chunk = blocks_per_chunk
        self.gpu_tensors = [t.tensor for t in kv_caches.tensors]
        self.cpu_tensors = [
            torch.zeros(
                (num_cpu_blocks, t.page_size_bytes * blocks_per_chunk),
                dtype=torch.int8,
            )
            for t in kv_caches.tensors
        ]
        self.group_refs = kv_caches.group_data_refs
        self._finished: list[TransferResult] = []
        self.submitted: list[tuple[int, str]] = []  # (job_id, direction) 观测面

    def _copy(
        self, job_id: int, src_spec: BlockIDsLoadStoreSpec, dst_spec: BlockIDsLoadStoreSpec
    ) -> None:
        gpu_spec = src_spec if isinstance(src_spec, GPULoadStoreSpec) else dst_spec
        gpu_to_cpu = isinstance(src_spec, GPULoadStoreSpec)
        src_bpc = 1 if gpu_to_cpu else self.blocks_per_chunk
        dst_bpc = self.blocks_per_chunk if gpu_to_cpu else 1

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        src_off = dst_off = 0
        for group_size, block_idx, group_data_refs in zip(
            gpu_spec.group_sizes, gpu_spec.block_indices, self.group_refs
        ):
            if group_size == 0:
                continue
            src_skip = block_idx % src_bpc
            dst_skip = block_idx % dst_bpc
            src_count = (group_size + src_skip + src_bpc - 1) // src_bpc
            dst_count = (group_size + dst_skip + dst_bpc - 1) // dst_bpc
            group_src = src_blocks[src_off : src_off + src_count]
            group_dst = dst_blocks[dst_off : dst_off + dst_count]
            for data_ref in group_data_refs:
                src_ptrs = np.empty(group_size, dtype=np.uint64)
                dst_ptrs = np.empty(group_size, dtype=np.uint64)
                compute_sub_block_ptrs(
                    group_src, src_bpc, src_ptrs,
                    self.gpu_tensors[data_ref.tensor_idx] if gpu_to_cpu
                    else self.cpu_tensors[data_ref.tensor_idx],
                    skip_count=src_skip,
                )
                compute_sub_block_ptrs(
                    group_dst, dst_bpc, dst_ptrs,
                    self.cpu_tensors[data_ref.tensor_idx] if gpu_to_cpu
                    else self.gpu_tensors[data_ref.tensor_idx],
                    skip_count=dst_skip,
                )
                page = data_ref.page_size_bytes
                for sp, dp in zip(src_ptrs, dst_ptrs):
                    ctypes.memmove(int(dp), int(sp), page)
            src_off += src_count
            dst_off += dst_count

    def submit_store(self, job_id, src_spec, dst_spec) -> bool:
        self.submitted.append((job_id, "store"))
        self._copy(job_id, src_spec, dst_spec)
        self._finished.append(TransferResult(job_id=job_id, success=True))
        return True

    def submit_load(self, job_id, src_spec, dst_spec) -> bool:
        self.submitted.append((job_id, "load"))
        self._copy(job_id, src_spec, dst_spec)
        self._finished.append(TransferResult(job_id=job_id, success=True))
        return True

    def get_finished(self) -> list[TransferResult]:
        out, self._finished = self._finished, []
        return out

    def wait(self, job_ids: set[int]) -> None:
        return None


# ═══════════════════════ 引擎装配：调度器半边 + worker 半边 ═══════════════════════


@dataclass
class Engine:
    """一次装配 = OffloadingConnector 两个 role 半边 + host DMA 替身。"""

    scheduler: Any  # OffloadingConnectorScheduler
    worker: Any  # OffloadingConnectorWorker
    host_worker: HostOffloadingWorker
    kv_caches: dict[str, torch.Tensor]

    @property
    def manager(self):
        return self.scheduler.manager


def assemble_engine(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor],
) -> Engine:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
        OffloadingConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    sched_connector = OffloadingConnector(
        vllm_config, KVConnectorRole.SCHEDULER, kv_cache_config
    )
    work_connector = OffloadingConnector(
        vllm_config, KVConnectorRole.WORKER, kv_cache_config
    )

    connector_worker = work_connector.connector_worker
    assert connector_worker is not None
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(vllm_config, kv_cache_config)
    )
    captured: dict[str, HostOffloadingWorker] = {}

    def _host_init_worker(self, canonical: CanonicalKVCaches) -> None:
        host = HostOffloadingWorker(
            canonical,
            blocks_per_chunk=spec.blocks_per_chunk,
            num_cpu_blocks=spec.num_blocks,
        )
        captured["host"] = host
        self.worker = host

    orig_init = OffloadingConnectorWorker._init_worker
    OffloadingConnectorWorker._init_worker = _host_init_worker  # type: ignore[method-assign]
    try:
        connector_worker.register_kv_caches(kv_caches)
    finally:
        OffloadingConnectorWorker._init_worker = orig_init  # type: ignore[method-assign]

    return Engine(
        scheduler=sched_connector.connector_scheduler,
        worker=connector_worker,
        host_worker=captured["host"],
        kv_caches=kv_caches,
    )


def make_scheduler_output(
    *,
    new_reqs: list[dict[str, Any]] | None = None,
    cached_reqs: list[dict[str, Any]] | None = None,
    num_scheduled_tokens: dict[str, int] | None = None,
    finished_req_ids: set[str] | None = None,
    preempted_req_ids: set[str] | None = None,
) -> SchedulerOutput:
    """new_reqs/cached_reqs 条目：{req_id, block_ids: tuple[list[int],...]}。"""
    new = [
        NewRequestData(
            req_id=entry["req_id"],
            block_ids=entry.get("block_ids"),
            prompt_token_ids=entry.get("prompt_token_ids"),
        )
        for entry in (new_reqs or [])
    ]
    cached = cached_reqs or []
    cached_data = CachedRequestData(
        req_ids=[e["req_id"] for e in cached],
        new_block_ids=[e.get("block_ids", ()) for e in cached],
        num_computed_tokens=[e.get("num_computed_tokens", 0) for e in cached],
    )
    return SchedulerOutput(
        scheduled_new_reqs=new,
        scheduled_cached_reqs=cached_data,
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        finished_req_ids=set(finished_req_ids or ()),
        preempted_req_ids=set(preempted_req_ids or ()),
    )


def block_checksum(tensor: torch.Tensor, block_id: int, page: int) -> bytes:
    flat = tensor.reshape(-1)
    return bytes(flat[block_id * page : (block_id + 1) * page].numpy().tobytes())


# ═══════════════════════ 测试替身：MultiConnector 子连接器 / out-of-tree 策略 ═══════════════════════


class FakePoolA:
    """MultiConnector 的 A 腿：hits 控制查询返回（None=仍在查）。"""

    hits: int | None = 0
    alloc_calls: list[tuple[str, int]] = []

    def __init__(self, vllm_config, role, kv_cache_config):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (FakePoolA.hits, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        FakePoolA.alloc_calls.append((request.request_id, num_external_tokens))

    def build_connector_meta(self, scheduler_output):
        return None

    def get_finished(self, finished_req_ids):
        return set(), set()

    def request_finished(self, request, block_ids):
        return False, None

    def register_kv_caches(self, kv_caches):
        return None


class FakePoolB(FakePoolA):
    hits: int | None = 5
    alloc_calls: list[tuple[str, int]] = []

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (FakePoolB.hits, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        FakePoolB.alloc_calls.append((request.request_id, num_external_tokens))


from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy


class MyTinyPolicy(LRUCachePolicy):
    """out-of-tree CachePolicy 替身（CachePolicyFactory 模块路径加载）。"""


from vllm.v1.kv_offload.base import OffloadingSpec as _OffloadingSpec


class OutTreeSpecSentinel(_OffloadingSpec):
    """out-of-tree spec 替身（factory 的 issubclass 断言要它是真 spec 子类）。"""

    def get_manager(self):
        return None

    def get_worker(self, kv_caches):
        return None
