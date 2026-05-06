# REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L71-L275
"""Memory snapshot + profiling — what vLLM measures, when, and how.

vLLM divides GPU memory into three categories:

    1. memory used by anything OTHER THAN the current vLLM instance
    2. memory used by torch in the current vLLM instance
    3. memory used by the current vLLM instance but NOT by torch
       (NCCL buffers, attention-backend workspaces, CUDA context)

A `MemorySnapshot` captures all three at one moment via two POSIX-level reads:

    free, total = current_platform.mem_get_info(device)   # NVML/CUDA
    torch_reserved = torch.accelerator.memory_reserved()  # PyTorch's books

`memory_profiling()` is the context manager that bookends a profile run and
diffs two snapshots to attribute peak activations vs. non-torch growth.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from typing import Generator


# REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L70-L157
@dataclass
class MemorySnapshot:
    """A point-in-time capture of GPU memory state.

    Fields match vLLM's `MemorySnapshot` exactly. `auto_measure=False` is the
    "construct empty, fill in later via subtraction" mode used inside
    `MemoryProfilingResult`.
    """

    # REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L74-L80
    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    # In vLLM `device` is a `torch.types.Device` and `auto_measure=True` runs
    # `measure()` in __post_init__. We accept the same fields but make
    # measurement explicit for testability without a real GPU.
    device: str | None = None
    auto_measure: bool = False

    def __post_init__(self) -> None:
        if self.auto_measure:
            self.measure()

    # REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L96-L126
    def measure(self) -> None:
        """In production: read NVML + torch reserved. Here: a no-op stub.

        vLLM's `measure()` does the following:
            torch_peak     = torch.accelerator.memory_stats[allocated_bytes.all.peak]
            free, total    = current_platform.mem_get_info(device)
            torch_memory   = torch.accelerator.memory_reserved(device)
            cuda_memory    = total - free
            non_torch_mem  = cuda_memory - torch_memory

        Our demo populates these fields directly with simulated values to show
        the math without requiring a real CUDA device.
        """
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    # REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L128-L145
    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device,
            auto_measure=False,
        )


# REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L160-L187
@dataclass
class MemoryProfilingResult:
    """All numbers in bytes. Constructed empty and filled by `memory_profiling`."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: int = 0
    profile_time: float = 0.0

    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)


# REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L190-L275
@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """Wrap a profile run. After exit, the result holds the breakdown.

    Quantitative example from the source's docstring (mem_utils.py:L209-L235):

        Before vLLM creation:        cat1=1, cat2=0, cat3=0          GiB
        After model load:            cat1=1, cat2=2 (weights), cat3=0.5 (NCCL)
        Peak during profile:         cat1=1, cat2=4 (acts +2), cat3=1
        After profile (gc'd):        cat1=1, cat2=3,           cat3=1

    The result attributes:
        weights_memory       = 2 GiB (from arg)
        torch_peak_increase  = 4 - 2 = 2 GiB (peak activation)
        non_torch_increase   = 1 - 0 = 1 GiB (NCCL etc.)
        non_kv_cache_memory  = weights + peak_activation + non_torch = 5 GiB

    SIMPLIFIED: vLLM calls `gc.collect()` and `torch.accelerator.empty_cache()`
    at entry/exit to evict reusable allocator chunks. Our pedagogical version
    skips these — the caller fills `before_profile` / `after_profile` manually.
    """
    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        weights_memory=weights_memory,
    )
    # In vLLM: result.before_profile.measure() reads live torch peak.
    # REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L256
    yield result

    # REFERENCE: instances/vllm/source/vllm/utils/mem_utils.py:L263-L275
    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = (
        result.non_torch_increase
        + result.torch_peak_increase
        + result.weights_memory
    )


def format_gib(bytes_: int) -> str:
    """Match vLLM's logging format. REFERENCE: vllm/utils → format_gib helper."""
    return f"{bytes_ / (1024 ** 3):.2f}"
