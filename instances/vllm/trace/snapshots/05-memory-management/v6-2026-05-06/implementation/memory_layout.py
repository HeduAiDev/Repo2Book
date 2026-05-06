# REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L352-L505
"""determine_available_memory — how vLLM picks num_gpu_blocks at startup.

The flow at engine boot:

    1. Read total GPU memory from NVML.
    2. requested_memory = total * gpu_memory_utilization        (default 0.92)
    3. Load model weights → measure via `DeviceMemoryProfiler`.
    4. Run a profile pass (dummy max-batch forward) → measure peak activation
       AND non-torch growth (NCCL buffers, attention workspace).
    5. If CUDA graphs are on, profile that too.
    6. available_kv_cache = requested - weights - peak_activation
                                      - non_torch - cuda_graph
    7. num_gpu_blocks = available_kv_cache // (page_size_bytes * num_layers)

The 8% margin (1.0 - 0.92) covers:
    a. PyTorch caching allocator fragmentation
    b. Activation memory variance across input shapes
    c. CUDA context overhead that NVML doesn't credit to torch
    d. Safety against transient OOM during sampling

This module re-creates the calculation deterministically so a reader can plug
in their own numbers and see the breakdown without booting a real engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kv_cache_spec import AttentionSpec, get_num_blocks
from .mem_snapshot import MemoryProfilingResult, MemorySnapshot, format_gib


# REFERENCE: pedagogical — vLLM doesn't have one struct, the breakdown is
# scattered across MemoryProfilingResult + Worker fields.
@dataclass
class MemoryLayout:
    """The full breakdown of how `total_gpu_memory` is partitioned."""

    total_gpu_memory: int
    gpu_memory_utilization: float
    requested_memory: int

    # Categories (all in bytes).
    weights: int
    peak_activation: int
    cudagraph_memory: int
    non_torch: int  # NCCL, attention workspace, CUDA context
    available_kv_cache: int  # what's left for KV cache

    # KV-cache derived numbers.
    page_size_bytes: int
    num_layers: int
    num_gpu_blocks: int
    wasted_bytes: int  # available_kv - num_blocks * page_size * num_layers

    @property
    def util_margin(self) -> int:
        # 1 - gpu_memory_utilization fraction, in bytes.
        return self.total_gpu_memory - self.requested_memory

    def report(self) -> str:
        rows = [
            ("Total GPU memory",          self.total_gpu_memory),
            ("Requested (util=" + f"{self.gpu_memory_utilization:.2f})", self.requested_memory),
            ("├─ Model weights",          self.weights),
            ("├─ Peak activation",        self.peak_activation),
            ("├─ CUDA graph memory",      self.cudagraph_memory),
            ("├─ Non-torch (NCCL etc.)",  self.non_torch),
            ("└─ Available for KV cache", self.available_kv_cache),
            ("Util margin (unused safety)", self.util_margin),
        ]
        lines = []
        for label, value in rows:
            lines.append(f"  {label:36s} {format_gib(value):>8} GiB")
        lines.append("")
        lines.append(f"  Page size                            "
                     f"{self.page_size_bytes / 1024:>8.1f} KiB / layer")
        lines.append(f"  Num layers                           {self.num_layers:>8d}")
        lines.append(f"  num_gpu_blocks                       {self.num_gpu_blocks:>8d}")
        lines.append(f"  Wasted (rounding)                    "
                     f"{self.wasted_bytes / (1024 ** 2):>8.1f} MiB")
        return "\n".join(lines)


# REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L352-L505
def determine_available_memory(
    init_snapshot: MemorySnapshot,
    profile_result: MemoryProfilingResult,
    cudagraph_memory: int,
    gpu_memory_utilization: float,
    spec: AttentionSpec,
    num_layers: int,
) -> MemoryLayout:
    """Compute num_gpu_blocks the same way vLLM's worker does at boot.

    Inputs come from a real run:
        - init_snapshot.total_memory     : NVML
        - init_snapshot.free_memory      : NVML at boot
        - profile_result.weights_memory  : passed in from model loader
        - profile_result.torch_peak_increase : measured during profile_run()
        - profile_result.non_torch_increase  : measured during profile_run()
        - cudagraph_memory               : profile_cudagraph_memory()
        - gpu_memory_utilization         : config.cache_config

    Returns a MemoryLayout with the breakdown and num_gpu_blocks.
    """
    # REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L364-L382
    # (The kv_cache_memory_bytes override path is omitted.)
    total = init_snapshot.total_memory
    requested = int(total * gpu_memory_utilization)

    # REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L411-L415
    # non_kv_cache_memory = non_torch + peak_activation + weights
    non_kv_cache = (
        profile_result.non_torch_increase
        + profile_result.torch_peak_increase
        + profile_result.weights_memory
    )

    # REFERENCE: instances/vllm/source/vllm/v1/worker/gpu_worker.py:L441-L445
    available_kv_cache = requested - non_kv_cache - cudagraph_memory
    available_kv_cache = max(available_kv_cache, 0)

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L930-L947
    page_size = spec.page_size_bytes
    num_blocks = get_num_blocks(available_kv_cache, num_layers, page_size)
    used_bytes = num_blocks * page_size * num_layers
    wasted = available_kv_cache - used_bytes

    return MemoryLayout(
        total_gpu_memory=total,
        gpu_memory_utilization=gpu_memory_utilization,
        requested_memory=requested,
        weights=profile_result.weights_memory,
        peak_activation=profile_result.torch_peak_increase,
        cudagraph_memory=cudagraph_memory,
        non_torch=profile_result.non_torch_increase,
        available_kv_cache=available_kv_cache,
        page_size_bytes=page_size,
        num_layers=num_layers,
        num_gpu_blocks=num_blocks,
        wasted_bytes=wasted,
    )


def estimate_max_concurrency(
    layout: MemoryLayout,
    avg_seq_len: int,
    block_size: int,
) -> float:
    """How many concurrent average-length requests fit in the KV cache?

    REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L872-L890
    (`get_max_concurrency_for_kv_cache_config` does this for real configs).

    blocks_per_request = ceil(avg_seq_len / block_size)
    concurrency        = num_gpu_blocks / blocks_per_request
    """
    blocks_per_request = (avg_seq_len + block_size - 1) // block_size
    if blocks_per_request == 0:
        return 0.0
    return layout.num_gpu_blocks / blocks_per_request
