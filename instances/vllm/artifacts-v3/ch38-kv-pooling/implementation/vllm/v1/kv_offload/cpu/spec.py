# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPUOffloadingSpec：单层 CPU 池的定容与工厂（调度器半边 manager / worker 半边 worker）。"""

from typing import Any

import torch
from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion


# SOURCE: vllm/v1/kv_offload/cpu/spec.py:L27-L186
class CPUOffloadingSpec(OffloadingSpec):
    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L28
    BLOCK_SIZE_ALIGNMENT = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L30-L75
    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, Any]:
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L30-L75
        # SUBTRACTED: 四条 gauge/histogram 指标定义——减法计划删除项 7。
        """Return Prometheus metric definitions emitted by this spec."""
        return {}

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L77-L121
    def __init__(self, config: OffloadingConfig):
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L78-L121
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0
        # SUBTRACTED: replicated_layout 的 rank=0 单拷贝布局（减法计划删除项 1：
        #   TP 复制页写放大优化，默认 replicated_layout=False 直通）——
        #   基类 OffloadingSpec.__init__ 已置 False，教学主线=每 rank 各存自己分片。
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            num_copies = world_size
            kv_bytes_per_block = config.worker_kv_bytes_per_block * num_copies
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // num_copies

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            # or |--- B0 (single copy) ---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L123-L142
    @override
    def get_manager(self) -> OffloadingManager:
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L124-L142
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L144-L147
    def _uses_shared_region(self) -> bool:
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L144-L147
        """Whether the worker CPU buffer is the shared mmap region (vs a private
        per-rank tensor); replicated-layout dedup is gated on this being True."""
        return current_platform.is_cuda_alike()

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L149-L173
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L150-L167
        mmap_region: SharedOffloadRegion | None = None
        # num_blocks == 0 would size the region to zero bytes, which cannot be
        # mmap'd; fall back to the tensor path (empty tensors) as before.
        if self._uses_shared_region() and self.num_blocks > 0:
            # SUBTRACTED: replicated_layout 的 rank=0 单拷贝槽位
            #   （减法计划删除项 1）——每 rank 各占自己按设备序号的 slot。
            world_size = self.config.parallel.world_size
            rank = torch.accelerator.current_device_index() % world_size
            mmap_region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_blocks=self.num_blocks,
                rank=rank,
                kv_bytes_per_block=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
            )
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L168-L173
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
            mmap_region=mmap_region,
        )

    # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L175-L186
    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        # SOURCE: vllm/v1/kv_offload/cpu/spec.py:L176-L186
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
