"""
GPU Memory Profiler & Partition Calculator — Our Reimplementation.

REFERENCE sources:
    MemorySnapshot:        vllm/utils/mem_utils.py:L71
    memory_profiling():    vllm/utils/mem_utils.py:L190
    determine_available_memory(): vllm/v1/worker/gpu_worker.py:L352
    request_memory():      vllm/v1/worker/utils.py:L403
    KVCacheSpec:           vllm/v1/kv_cache_interface.py:L81
    get_kv_cache_configs(): vllm/v1/core/kv_cache_utils.py:L1922
    _allocate_kv_cache():  vllm/v1/worker/gpu/attn_utils.py:L128

ARCHITECTURE:
    vLLM does NOT implement its own GPU memory allocator for KV cache.
    It uses PyTorch's torch.zeros() to allocate flat int8 tensors, then
    reshapes them to backend-specific layouts.

    The memory flow at startup:
        1. Measure total GPU memory (nvidia-smi / rocm-smi)
        2. requested_memory = total * gpu_memory_utilization (default 0.92)
        3. Load model weights → measured directly
        4. Profile run (dummy forward) → measure peak activation + CUDA graph
        5. available_kv_cache = requested - weights - activations - cuda_graph - non_torch
        6. num_blocks = available_kv_cache / block_bytes
        7. Allocate KV cache tensor: torch.zeros(total_block_bytes, dtype=int8)
        8. Reshape to attention backend layout

    The 8% margin (1.0 - 0.92) is for:
        - PyTorch allocator fragmentation
        - Varying activation memory across inputs
        - CUDA context overhead
        - Safety buffer against OOM
"""

import math
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# MemorySnapshot — vllm/utils/mem_utils.py:L71
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MemorySnapshot:
    """
    A point-in-time capture of GPU memory state.

    REFERENCE: vllm/utils/mem_utils.py:L71-L120
    In production, uses current_platform.mem_get_info() for real GPU data.
    We model this explicitly for educational clarity.
    """
    total_memory: int       # bytes
    free_memory: int        # bytes
    torch_peak: int         # PyTorch's peak allocated bytes
    torch_current: int      # PyTorch's current allocated bytes

    @property
    def used_memory(self) -> int:
        return self.total_memory - self.free_memory

    @property
    def non_torch_memory(self) -> int:
        """Memory NOT tracked by PyTorch: NCCL buffers, CUDA context, etc."""
        return self.used_memory - self.torch_current


# ═══════════════════════════════════════════════════════════════════════════
# ModelConfig — simplified
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LlamaModelMemory:
    """Memory breakdown for a Llama-style model."""
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    dtype_bytes: int = 2

    def weight_size(self) -> int:
        """
        Approximate model weight size.

        REFERENCE: vLLM measures this via DeviceMemoryProfiler during actual loading.
        We compute analytically for educational clarity.

        Major components:
        - QKV projection: d_model * 3*d_model (or d_model * (d_model + 2*kv_dim))
        - Output projection: d_model * d_model
        - MLP gate/up: 2 * d_model * intermediate (intermediate ≈ 8/3 * d_model)
        - MLP down: intermediate * d_model
        - RMSNorm: 2 * d_model
        """
        d = self.d_model
        kv_dim = self.num_kv_heads * (d // self.num_heads)
        intermediate = int(8/3 * d)  # Llama's SwiGLU intermediate size

        per_layer = (
            d * d + 2 * kv_dim * (d // self.num_heads) * self.num_heads // self.num_heads  # QKV
            + d * d                                                                          # O
            + 2 * d * intermediate                                                           # gate + up
            + intermediate * d                                                               # down
            + 2 * d                                                                          # 2×RMSNorm
        )
        # Actually, let's keep it simpler and more accurate
        # Llama-3.2-1B: d=2048, 32 layers, 8 kv heads
        # Weight: ~2.5B params × 2 bytes ≈ 5 GB for original... wait 1B params × 2 bytes = 2GB
        params = (
            self.num_layers * (
                d * d * 4          # QKV (3×) + O
                + 3 * d * intermediate  # gate + up + down
                + 2 * d             # RMSNorm × 2
            )
            + d * d                 # lm_head
            + d * 128000            # embedding (vocab_size ≈ 128K for Llama 3)
        )
        return int(params * self.dtype_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Size Calculator
# REFERENCE: vllm/v1/kv_cache_interface.py:L81-L195
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KVCacheSpec:
    """
    How much memory one KV cache block takes.

    REFERENCE: vllm/v1/kv_cache_interface.py:L81 (base)
               and FullAttentionSpec:L174 (attention-specific)
    """
    block_size: int          # tokens per block
    num_kv_heads: int
    head_dim: int
    num_layers: int
    dtype_bytes: int = 2

    def page_size_bytes(self) -> int:
        """
        Bytes per block, per layer.

        REFERENCE: vllm/v1/kv_cache_interface.py:L138 — AttentionSpec.page_size_bytes

        2 × block_size × num_kv_heads × head_dim × dtype_bytes
        ↑ K+V
        """
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def total_block_bytes(self, num_blocks: int) -> int:
        """Total bytes across all layers."""
        return num_blocks * self.page_size_bytes() * self.num_layers


# ═══════════════════════════════════════════════════════════════════════════
# Memory Profiler — vllm/v1/worker/gpu_worker.py:L352
# ═══════════════════════════════════════════════════════════════════════════

class MemoryProfiler:
    """
    Simulates vLLM's memory profiling flow.

    REFERENCE: vllm/v1/worker/gpu_worker.py:L352 — determine_available_memory()
               vllm/utils/mem_utils.py:L190 — memory_profiling()

    Steps:
        1. requested_memory = total * gpu_memory_utilization
        2. non_kv_cache = weights + peak_activation + cuda_graph + non_torch
        3. available_kv_cache = requested - non_kv_cache
        4. num_blocks = available_kv_cache / block_bytes
    """

    def __init__(
        self,
        total_gpu_memory: int,       # bytes
        gpu_memory_utilization: float = 0.92,
    ):
        self.total = total_gpu_memory
        self.utilization = gpu_memory_utilization

    def profile(
        self,
        model: LlamaModelMemory,
        peak_activation: int,        # bytes — from profiling run
        cuda_graph_memory: int = 0,  # bytes
        non_torch_overhead: int = 0, # bytes
        kv_cache_spec: Optional[KVCacheSpec] = None,
    ) -> dict:
        """
        Run the profiling calculation.

        REFERENCE: gpu_worker.py:L386-L445
        """
        requested = int(self.total * self.utilization)

        # REFERENCE: gpu_worker.py:L299 — weights measured by DeviceMemoryProfiler
        weights = model.weight_size()

        # Non-KV-cache memory (everything that's NOT KV cache)
        # REFERENCE: gpu_worker.py:L441-L445
        non_kv_cache = weights + peak_activation + cuda_graph_memory + non_torch_overhead

        # What's left for KV cache
        available_kv_cache = requested - non_kv_cache

        result = {
            "total_gpu_memory": self.total,
            "gpu_memory_utilization": self.utilization,
            "requested_memory": requested,
            "breakdown": {
                "model_weights": weights,
                "peak_activation": peak_activation,
                "cuda_graph_memory": cuda_graph_memory,
                "non_torch_overhead": non_torch_overhead,
                "total_non_kv_cache": non_kv_cache,
            },
            "available_kv_cache_bytes": available_kv_cache,
        }

        # Calculate number of KV cache blocks
        if kv_cache_spec and available_kv_cache > 0:
            per_layer_block_bytes = kv_cache_spec.page_size_bytes()
            total_block_bytes_per_block = per_layer_block_bytes * kv_cache_spec.num_layers
            num_blocks = available_kv_cache // total_block_bytes_per_block
            result["num_gpu_blocks"] = num_blocks
            result["block_bytes_per_layer"] = per_layer_block_bytes
            result["total_block_bytes_per_block"] = total_block_bytes_per_block
            result["wasted_bytes"] = available_kv_cache - num_blocks * total_block_bytes_per_block

        return result


# ═══════════════════════════════════════════════════════════════════════════
# Real-world examples
# ═══════════════════════════════════════════════════════════════════════════

def llama_3_2_1b_h100_example():
    """
    Memory profile for Llama-3.2-1B on H100 (80 GB).

    REFERENCE: These numbers are representative of real vLLM deployments.
    Exact values depend on batch size, model config, and CUDA version.
    """
    profiler = MemoryProfiler(total_gpu_memory=80 * 1024**3)  # 80 GB
    model = LlamaModelMemory(
        d_model=2048,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        dtype_bytes=2,
    )
    spec = KVCacheSpec(
        block_size=16,
        num_kv_heads=8,
        head_dim=128,
        num_layers=32,
        dtype_bytes=2,
    )

    result = profiler.profile(
        model=model,
        peak_activation=2 * 1024**3,     # ~2 GB for activation peak
        cuda_graph_memory=512 * 1024**2, # ~0.5 GB for CUDA graphs
        non_torch_overhead=512 * 1024**2,
        kv_cache_spec=spec,
    )

    print("Llama-3.2-1B on H100 (80 GB)")
    print("=" * 50)
    print(f"Requested memory:     {result['requested_memory']/(1024**3):.1f} GB")
    print(f"  Model weights:      {result['breakdown']['model_weights']/(1024**3):.1f} GB")
    print(f"  Peak activation:    {result['breakdown']['peak_activation']/(1024**3):.1f} GB")
    print(f"  CUDA graph:         {result['breakdown']['cuda_graph_memory']/(1024**3):.1f} GB")
    print(f"  Non-torch:          {result['breakdown']['non_torch_overhead']/(1024**3):.1f} GB")
    print(f"  Total non-KV:       {result['breakdown']['total_non_kv_cache']/(1024**3):.1f} GB")
    print(f"Available KV cache:   {result['available_kv_cache_bytes']/(1024**3):.1f} GB")
    print(f"Block size:           {spec.page_size_bytes()} B/layer")
    print(f"Number of blocks:     {result.get('num_gpu_blocks', 'N/A')}")
    print(f"KV cache per block:   {result.get('total_block_bytes_per_block', 0)/(1024**2):.1f} MB")

    return result


def gpu_memory_breakdown_pie():
    """
    Generate memory breakdown data for the chapter's visualization.

    Returns a dict suitable for rendering as a pie chart or table.
    """
    profiler = MemoryProfiler(total_gpu_memory=80 * 1024**3)
    model = LlamaModelMemory(2048, 32, 32, 8)
    spec = KVCacheSpec(16, 8, 128, 32)

    result = profiler.profile(
        model=model,
        peak_activation=2 * 1024**3,
        cuda_graph_memory=256 * 1024**2,
        non_torch_overhead=384 * 1024**2,
        kv_cache_spec=spec,
    )

    categories = {
        "Model Weights": result["breakdown"]["model_weights"] / result["total_gpu_memory"],
        "KV Cache": result["available_kv_cache_bytes"] / result["total_gpu_memory"],
        "Activations": result["breakdown"]["peak_activation"] / result["total_gpu_memory"],
        "CUDA Graph": result["breakdown"]["cuda_graph_memory"] / result["total_gpu_memory"],
        "Non-Torch Overhead": result["breakdown"]["non_torch_overhead"] / result["total_gpu_memory"],
        "Unused (gpu_memory_utilization margin)": 1.0 - result["requested_memory"] / result["total_gpu_memory"],
    }

    return categories


if __name__ == "__main__":
    llama_3_2_1b_h100_example()
    print("\n\nMemory Breakdown:")
    for k, v in gpu_memory_breakdown_pie().items():
        print(f"  {k:30s}: {v:5.1%}")
