# SPDX-License-Identifier: Apache-2.0
"""
Worker-side async transfer layer.

This module mirrors:
  vllm/v1/kv_offload/worker/worker.py    (176 LOC) — OffloadingHandler ABC
  vllm/v1/kv_offload/cpu/gpu_worker.py   (433 LOC) — CUDA-stream impl
  vllm/v1/simple_kv_offload/copy_backend.py (97 LOC) — DMA backend (TWO streams)
  vllm/v1/simple_kv_offload/cuda_mem_ops.py (153 LOC) — pin via cudaHostRegister
  vllm/v1/simple_kv_offload/worker.py    (305 LOC) — pin + register_kv_caches

Why TWO CUDA streams (HARD GATE design decision 7):
  Loads (CPU→GPU) and stores (GPU→CPU) can run concurrently on independent
  PCIe lanes (modern GPUs have separate copy engines for the two directions).
  Using a single stream would serialize them, halving achievable PCIe BW.
  REFERENCE: vllm/v1/simple_kv_offload/copy_backend.py:L43-L44 — load_stream
  and store_stream are explicitly separate.

Why pinned memory (HARD GATE design decision 8):
  Pinned (page-locked) host memory bypasses the driver's pageable→pinned
  bounce buffer, doubling effective H2D bandwidth. PyTorch's
  CUDACachingHostAllocator rounds every `pin_memory=True` allocation up
  to the next power of 2 (a 100 GB request becomes 128 GB pinned!),
  which is unacceptable for production-scale offload. The simple variant
  uses a raw `torch.empty(...)` then `cudaHostRegister` to pin without
  the rounding penalty.
  REFERENCE: vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25 (pin_tensor docstring).

Trap C: a single 16 MB block transfer at PCIe Gen5 = 16 MB / 64 GB/s ≈ 250 µs.
That is 0.5% of one decode step (~50 ms). PCIe is bandwidth-bound for
LLM workloads, NOT latency-bound — the transfer ALPHA (~10 µs) is
negligible compared to BETA × 16 MB (= ~250 µs). The break-even block
size where alpha equals beta×bytes is ~660 KB; vLLM's 16 MB blocks
are ~24× past break-even.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from .offload_spec import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    LoadStoreSpec,
    PCIE_GEN5_BANDWIDTH_GB_PER_S,
    PCIE_OVERHEAD_ALPHA_US,
    PCIE_OVERHEAD_BETA_US_PER_BYTE,
)


# REFERENCE: vllm/v1/kv_offload/worker/worker.py:L9-L23 — TransferSpec / Result
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]
TransferType = tuple[str, str]


@dataclass
class TransferResult:
    """Result of a completed transfer (returned by get_finished)."""

    job_id: int
    success: bool
    transfer_size: Optional[int] = None  # bytes
    transfer_time: Optional[float] = None  # seconds
    transfer_type: Optional[TransferType] = None


# REFERENCE: vllm/v1/kv_offload/worker/worker.py:L26-L74 — OffloadingHandler ABC
class OffloadingHandler(ABC):
    """Worker-side async transfer handler.

    `transfer_async(job_id, spec)` kicks off a copy on a CUDA stream;
    `get_finished()` polls completed jobs (non-blocking);
    `wait(job_ids)` synchronizes on specific jobs (blocking).

    The handler is OWNED by the worker (one per rank) and runs entirely
    in the worker process. The scheduler-side manager NEVER touches CUDA.
    """

    @abstractmethod
    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """Start an async transfer. Returns True on success submission."""

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """Drain all newly completed transfers."""

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """Block until all `job_ids` have completed."""

    def shutdown(self) -> None:
        """Release CUDA streams + events."""


@dataclass
class Transfer:
    """Per-job state inside the handler queue.

    REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L30-L36
    """

    job_id: int
    stream: Any  # torch.cuda.Stream in production
    start_event: Any  # torch.Event
    end_event: Any  # torch.Event
    num_bytes: int
    submit_t: float = 0.0  # wall-clock at submission (sim only)
    finish_t: float = 0.0  # wall-clock at finish (sim only)


# REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L111-L173 — handler
class SingleDirectionOffloadingHandler(OffloadingHandler):
    """Handles transfers in ONE direction (GPU→CPU OR CPU→GPU).

    The two-direction split is critical: loads and stores get separate
    CUDA streams so they overlap on the PCIe link.

    SIMPLIFIED: this implementation is CPU-only (no torch / no CUDA).
    We simulate transfer latency using the alpha-beta model so the demos
    can run on any machine. Real vLLM uses
    `swap_blocks_batch(batch_src, batch_dst, batch_sizes)` on a `cuda.Stream`.
    REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L308-L321 — CUDA stream record.
    """

    def __init__(
        self,
        gpu_block_bytes: int,
        cpu_block_bytes: int,
        gpu_to_cpu: bool,
        bandwidth_gb_per_s: float = PCIE_GEN5_BANDWIDTH_GB_PER_S,
        alpha_us: float = PCIE_OVERHEAD_ALPHA_US,
        beta_us_per_byte: float = PCIE_OVERHEAD_BETA_US_PER_BYTE,
    ):
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L120-L177
        assert gpu_block_bytes > 0
        assert cpu_block_bytes >= gpu_block_bytes, (
            f"cpu_block_bytes={cpu_block_bytes} < gpu_block_bytes="
            f"{gpu_block_bytes}; cpu blocks must be >= gpu blocks "
            f"(block_size_factor >= 1)"
        )
        self.gpu_block_bytes = gpu_block_bytes
        self.cpu_block_bytes = cpu_block_bytes
        self.gpu_to_cpu = gpu_to_cpu
        self.bandwidth_bytes_per_us = (bandwidth_gb_per_s * 1e9) / 1e6
        self.alpha_us = alpha_us
        self.beta_us_per_byte = beta_us_per_byte

        self.transfer_type: TransferType = (
            ("GPU", "CPU") if gpu_to_cpu else ("CPU", "GPU")
        )

        # job_id → end-event (sim: float wall-clock finish time)
        self._transfer_events: dict[int, float] = {}
        self._transfers: deque[Transfer] = deque()

    def _alpha_beta_us(self, num_bytes: int) -> float:
        """Alpha-beta latency model.

        latency = alpha + beta * bytes
        For PCIe Gen5 we use alpha=10us, beta=1.5e-5 us/byte.
        This is the model used in `wisdom/debugging.md` for cross-rank
        comm and applies equally well to PCIe transfers.
        """
        return self.alpha_us + self.beta_us_per_byte * num_bytes

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L179-L334
        src_spec, dst_spec = spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        num_blocks = max(len(src_spec.block_ids), len(dst_spec.block_ids))
        num_bytes = num_blocks * (
            self.cpu_block_bytes if self.gpu_to_cpu else self.gpu_block_bytes
        )

        latency_us = self._alpha_beta_us(num_bytes)

        submit_t = time.perf_counter()
        # Simulated end time: now + latency. The model matches the CUDA
        # stream semantics — calls return immediately, completion is
        # reported via `get_finished()` polling.
        finish_t = submit_t + latency_us / 1e6

        # CRITICAL: enforce in-order completion within a direction.
        # If a previous transfer hasn't finished, our finish must be
        # at least its finish + this latency. This mirrors the
        # `stream.wait_event(last_event)` in real vLLM.
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L311-L315
        if self._transfers:
            last_finish = self._transfers[-1].finish_t
            if last_finish > submit_t:
                finish_t = last_finish + latency_us / 1e6

        transfer = Transfer(
            job_id=job_id,
            stream=None,  # SIMPLIFIED: no real CUDA stream in sim
            start_event=submit_t,
            end_event=finish_t,
            num_bytes=num_bytes,
            submit_t=submit_t,
            finish_t=finish_t,
        )
        self._transfers.append(transfer)
        self._transfer_events[job_id] = finish_t
        return True

    def get_finished(self) -> list[TransferResult]:
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L336-L356
        now = time.perf_counter()
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].finish_t <= now:
            t = self._transfers.popleft()
            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=True,
                    transfer_size=t.num_bytes,
                    transfer_time=max(0.0, t.finish_t - t.submit_t),
                    transfer_type=self.transfer_type,
                )
            )
            del self._transfer_events[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L358-L362
        for job_id in job_ids:
            finish = self._transfer_events.get(job_id)
            if finish is not None:
                # Spin until the simulated finish time is past.
                while time.perf_counter() < finish:
                    time.sleep(0.0001)

    def shutdown(self) -> None:
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L364-L372
        self._transfers.clear()
        self._transfer_events.clear()

    # --- inspectors used by tests ---

    def num_inflight(self) -> int:
        return len(self._transfers)


# REFERENCE: vllm/v1/kv_offload/worker/worker.py:L77-L177 — multi-handler dispatch
class OffloadingWorker:
    """Aggregate multiple OffloadingHandlers and dispatch by transfer type.

    The worker is one per rank. It owns a (CPU→GPU) handler AND a
    (GPU→CPU) handler (and possibly more for SSD/NVMe in future).
    `transfer_async` looks up the right handler by (src.medium(), dst.medium()).
    """

    def __init__(self) -> None:
        self.handlers: set[OffloadingHandler] = set()
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L96-L97
        self.transfer_type_to_handler: dict[
            TransferType, OffloadingHandler
        ] = {}

    def register_handler(
        self,
        src_cls: type[LoadStoreSpec],
        dst_cls: type[LoadStoreSpec],
        handler: OffloadingHandler,
    ) -> None:
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L99-L116
        transfer_type = (src_cls.medium(), dst_cls.medium())
        assert transfer_type not in self.transfer_type_to_handler, (
            f"Handler for {transfer_type} already registered"
        )
        self.handlers.add(handler)
        self.transfer_type_to_handler[transfer_type] = handler

    def transfer_async(
        self, job_id: int, spec: TransferSpec
    ) -> bool:
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L118-L150
        src, dst = spec
        transfer_type = (src.medium(), dst.medium())
        handler = self.transfer_type_to_handler.get(transfer_type)
        assert handler is not None, (
            f"No handler for transfer type {transfer_type}"
        )
        return handler.transfer_async(job_id, spec)

    def get_finished(self) -> list[TransferResult]:
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L152-L162
        finished: list[TransferResult] = []
        for handler in self.handlers:
            finished.extend(handler.get_finished())
        return finished

    def wait(self, job_ids: set[int]) -> None:
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L164-L172
        for handler in self.handlers:
            handler.wait(job_ids)

    def shutdown(self) -> None:
        # REFERENCE: vllm/v1/kv_offload/worker/worker.py:L174-L176
        for handler in self.handlers:
            handler.shutdown()


# REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L375-L433 — CpuGpuOffloadingHandlers
class CpuGpuOffloadingHandlers:
    """Bundle of two SingleDirectionOffloadingHandlers (G→C and C→G).

    SIMPLIFIED: we don't allocate real CPU/GPU tensors here (production
    uses `torch.zeros(..., pin_memory=True)` + cudaHostRegister). Instead
    we record the byte sizes and run the alpha-beta latency simulator.
    The PUBLIC API (`gpu_to_cpu_handler`, `cpu_to_gpu_handler`) matches
    the production class so OffloadingSpec.get_handlers can yield the
    same tuples regardless.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ):
        # Total bytes per CPU block = sum(layer page_size_bytes) * factor.
        # REFERENCE: vllm/v1/kv_offload/cpu/gpu_worker.py:L383-L417
        total_layer_bytes = 0
        for canon_tensor in kv_caches.tensors:
            total_layer_bytes += canon_tensor.page_size_bytes
        gpu_block_bytes = total_layer_bytes
        cpu_block_bytes = gpu_block_bytes * block_size_factor

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            gpu_block_bytes=gpu_block_bytes,
            cpu_block_bytes=cpu_block_bytes,
            gpu_to_cpu=True,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            gpu_block_bytes=gpu_block_bytes,
            cpu_block_bytes=cpu_block_bytes,
            gpu_to_cpu=False,
        )
        self.num_cpu_blocks = num_cpu_blocks
        self.gpu_block_bytes = gpu_block_bytes
        self.cpu_block_bytes = cpu_block_bytes


def alpha_beta_latency_us(
    num_bytes: int,
    alpha_us: float = PCIE_OVERHEAD_ALPHA_US,
    beta_us_per_byte: float = PCIE_OVERHEAD_BETA_US_PER_BYTE,
) -> float:
    """Public alpha-beta helper used by tests / demos / impl-notes."""
    return alpha_us + beta_us_per_byte * num_bytes


def break_even_block_bytes(
    alpha_us: float = PCIE_OVERHEAD_ALPHA_US,
    beta_us_per_byte: float = PCIE_OVERHEAD_BETA_US_PER_BYTE,
) -> int:
    """Block size at which transfer cost equals fixed overhead.

    bytes_be such that beta * bytes_be == alpha → bytes_be = alpha / beta.
    For PCIe Gen5 (alpha=10us, beta=1.5e-5 us/byte): 10/1.5e-5 ≈ 666_667 bytes.
    vLLM's 16 MB block is ~24× past this break-even.
    """
    return int(math.ceil(alpha_us / beta_us_per_byte))
