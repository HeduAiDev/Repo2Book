# REFERENCE: vllm/v1/worker/gpu_worker.py → determine_available_memory()
# REFERENCE: vllm/v1/core/kv_cache_utils.py → get_num_blocks(), L930-L948
# REFERENCE: vllm/v1/core/kv_cache_utils.py → max_memory_usage_bytes(), L726-L732
"""
GPU memory profiling: determine how much memory is available for KV cache.

vLLM's startup profiling pipeline:
1. Measure total GPU memory
2. Load model weights → record torch memory consumed
3. Dummy forward with max_num_batched_tokens → record peak activation memory
4. Profile CUDA Graph memory (if enabled)
5. available_kv_cache = requested_memory - non_kv_cache_memory
   where requested_memory = total_gpu * gpu_memory_utilization
   and non_kv_cache_memory = weights + peak_act + cudagraph + non_torch

This module simulates the profiling pipeline for educational purposes.
"""

import math
from dataclasses import dataclass


# ─── Memory Budget Calculation ─────────────────────────────────────────────

@dataclass
class MemoryBudget:
    """Results of GPU memory profiling — the three-category budget.

    REFERENCE: vllm/v1/worker/gpu_worker.py → determine_available_memory()
    The original classifies all GPU memory into:
    - weights_memory: model parameters (quantized or full precision)
    - non_kv_cache_memory: weights + peak_activations + cudagraph + non_torch
    - available_kv_cache_memory: the remainder, allocated for KV cache blocks
    """
    total_gpu_memory: int          # bytes — total GPU memory
    gpu_memory_utilization: float  # fraction reserved for vLLM (e.g., 0.92)
    weights_memory: int            # bytes — model weights
    peak_activation_memory: int    # bytes — peak forward-pass activations
    cudagraph_memory: int          # bytes — CUDA Graph replay buffers
    non_torch_memory: int          # bytes — cuBLAS, NCCL, other processes
    available_kv_cache_memory: int # bytes — what's left for KV cache

    def __repr__(self) -> str:
        def gib(b: int) -> str:
            return f"{b / (1024**3):.2f} GiB"
        return (
            f"MemoryBudget(\n"
            f"  total_gpu_memory            = {gib(self.total_gpu_memory)}\n"
            f"  gpu_memory_utilization       = {self.gpu_memory_utilization}\n"
            f"  weights_memory              = {gib(self.weights_memory)}\n"
            f"  peak_activation_memory      = {gib(self.peak_activation_memory)}\n"
            f"  cudagraph_memory            = {gib(self.cudagraph_memory)}\n"
            f"  non_torch_memory            = {gib(self.non_torch_memory)}\n"
            f"  available_kv_cache_memory   = {gib(self.available_kv_cache_memory)}\n"
            f")"
        )


def profile_gpu_memory(
    total_gpu_memory: int,
    gpu_memory_utilization: float,
    model_params_count: int,
    dtype_bytes: int = 2,            # fp16/bf16 = 2 bytes
    max_num_batched_tokens: int = 8192,
    num_layers: int = 32,
    hidden_size: int = 4096,
    cudagraph_memory: int = 0,
    non_torch_memory: int = 0,
) -> MemoryBudget:
    """Simulate vLLM's startup memory profiling pipeline.

    In the real vLLM, this runs on actual GPU hardware with torch.cuda.
    We simulate it with formulas that match the real output.

    REFERENCE: vllm/v1/worker/gpu_worker.py::determine_available_memory()

    Args:
        total_gpu_memory: Total GPU memory in bytes (e.g., 16 GiB for T4).
        gpu_memory_utilization: Fraction reserved for vLLM (default 0.92).
        model_params_count: Number of model parameters.
        dtype_bytes: Bytes per parameter (2 for fp16/bf16, 1 for int8).
        max_num_batched_tokens: Max tokens per forward pass.
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size.
        cudagraph_memory: Estimated CUDA Graph replay memory.
        non_torch_memory: Memory outside PyTorch's allocator.

    Returns:
        MemoryBudget with all categories filled.
    """
    # Step 1: Compute model weights memory (simple: count_params × bytes_per_param)
    weights_memory = model_params_count * dtype_bytes

    # Step 2: Estimate peak activation memory.
    # In vLLM, this is determined by running a dummy forward pass with
    # max_num_batched_tokens and recording torch.cuda.max_memory_allocated().
    # The dominant term is the attention output + MLP intermediate activations.
    #
    # For a transformer layer, peak activation per token is approximately:
    #   4 × hidden_size × dtype_bytes  (QKV projection output)
    # + 4 × hidden_size × dtype_bytes  (attention output)
    # + 4 × hidden_size × dtype_bytes  (MLP intermediate, assuming 4x expansion)
    # = 12 × hidden_size × dtype_bytes per token per layer
    activation_per_token_per_layer = 12 * hidden_size * dtype_bytes
    peak_activation_memory = (
        num_layers * max_num_batched_tokens * activation_per_token_per_layer
    )

    # Step 3: Compute requested memory for vLLM
    # gpu_memory_utilization < 1.0 because:
    # - 8% buffer for cuBLAS workspace growth, NCCL buffers, CUDA context
    # - PyTorch CUDACachingAllocator fragmentation ("reserved but not usable")
    # - CUDA Graph capture peaks (capture > replay memory)
    requested_memory = int(total_gpu_memory * gpu_memory_utilization)

    # Step 4: Compute non-KV-cache memory
    non_kv_cache_memory = (
        weights_memory + peak_activation_memory + cudagraph_memory + non_torch_memory
    )

    # Step 5: Available KV cache memory = requested - non_kv_cache
    available_kv_cache_memory = max(0, requested_memory - non_kv_cache_memory)

    return MemoryBudget(
        total_gpu_memory=total_gpu_memory,
        gpu_memory_utilization=gpu_memory_utilization,
        weights_memory=weights_memory,
        peak_activation_memory=peak_activation_memory,
        cudagraph_memory=cudagraph_memory,
        non_torch_memory=non_torch_memory,
        available_kv_cache_memory=available_kv_cache_memory,
    )


# ─── KV Cache Block Calculation ────────────────────────────────────────────

@dataclass
class KVCacheConfig:
    """KV cache configuration derived from memory profiling.

    REFERENCE: vllm/v1/core/kv_cache_utils.py::get_num_blocks() L930-L948
    """
    num_blocks: int        # Total blocks in the pool
    block_size: int        # Tokens per block
    num_layers: int        # Number of layers sharing the pool
    page_size_bytes: int   # Bytes per block per layer (K+V)
    num_kv_heads: int      # Number of KV heads
    head_size: int         # Dimension of each head
    dtype_bytes: int       # Bytes per element (2 for fp16)


def compute_kv_cache_config(
    budget: MemoryBudget,
    block_size: int = 16,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_size: int = 128,
    dtype_bytes: int = 2,
) -> KVCacheConfig:
    """Compute KV cache configuration from available memory.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L930-L948

    page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_bytes
    (factor of 2 for K + V)

    num_blocks = available_kv_cache_memory // page_size_bytes // num_layers
    """
    # Each block stores K and V for block_size tokens
    # K: [block_size, num_kv_heads, head_size]
    # V: [block_size, num_kv_heads, head_size]
    page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_bytes

    if page_size_bytes * num_layers == 0:
        num_blocks = 0
    else:
        num_blocks = budget.available_kv_cache_memory // page_size_bytes // num_layers

    return KVCacheConfig(
        num_blocks=max(num_blocks, 0),
        block_size=block_size,
        num_layers=num_layers,
        page_size_bytes=page_size_bytes,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype_bytes=dtype_bytes,
    )


# ─── Helpers ────────────────────────────────────────────────────────────────

def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024**3:
        return f"{b / 1024**3:.2f} GiB"
    elif b >= 1024**2:
        return f"{b / 1024**2:.2f} MiB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KiB"
    return f"{b} B"


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# ─── Demonstration ──────────────────────────────────────────────────────────

def main():
    """Demonstrate GPU memory profiling for a Llama-3.2-1B-like model on T4."""
    print("=" * 70)
    print("GPU Memory Profiling: Llama-3.2-1B on T4 16GB")
    print("=" * 70)

    # T4 has ~15 GiB usable (16 GiB total, some reserved by driver)
    total_gpu_memory = 15 * 1024**3  # 15 GiB

    # Llama-3.2-1B: ~1.24B parameters
    model_params = 1_240_000_000

    # Profile memory
    budget = profile_gpu_memory(
        total_gpu_memory=total_gpu_memory,
        gpu_memory_utilization=0.92,
        model_params_count=model_params,
        dtype_bytes=2,                     # bfloat16
        max_num_batched_tokens=8192,
        num_layers=32,
        hidden_size=2048,
        cudagraph_memory=200 * 1024**2,    # ~200 MiB for CUDA Graph
        non_torch_memory=300 * 1024**2,    # ~300 MiB for NCCL, cuBLAS, etc.
    )
    print(budget)

    # Compute KV cache config
    config = compute_kv_cache_config(
        budget,
        block_size=16,
        num_layers=32,
        num_kv_heads=8,      # GQA: 8 KV heads
        head_size=128,       # Llama-3.2-1B head_size = 128
        dtype_bytes=2,
    )
    print(f"\n--- KV Cache Configuration ---")
    print(f"  block_size         = {config.block_size} tokens")
    print(f"  page_size_bytes    = {format_bytes(config.page_size_bytes)}")
    print(f"  num_layers         = {config.num_layers}")
    print(f"  num_blocks (total) = {config.num_blocks}")

    # Maximum concurrency
    max_model_len = 131072
    max_blocks_per_request = cdiv(max_model_len, config.block_size)
    max_concurrency = config.num_blocks / max_blocks_per_request
    print(f"\n--- Concurrency Estimate ---")
    print(f"  max_model_len               = {max_model_len}")
    print(f"  max_blocks_per_request      = {max_blocks_per_request}")
    print(f"  theoretical max concurrency = {max_concurrency:.1f}")

    # Show what happens if utilization was 1.0
    budget_full = profile_gpu_memory(
        total_gpu_memory=total_gpu_memory,
        gpu_memory_utilization=1.0,
        model_params_count=model_params,
        dtype_bytes=2,
        max_num_batched_tokens=8192,
        num_layers=32,
        hidden_size=2048,
        cudagraph_memory=200 * 1024**2,
        non_torch_memory=300 * 1024**2,
    )
    print(f"\n--- Comparison: gpu_memory_utilization ---")
    print(f"  utilization=0.92: KV cache = {format_bytes(budget.available_kv_cache_memory)}")
    print(f"  utilization=1.0 : KV cache = {format_bytes(budget_full.available_kv_cache_memory)}")
    print(f"  Difference: {format_bytes(budget_full.available_kv_cache_memory - budget.available_kv_cache_memory)}")
    print(f"  Why not 1.0? 8% buffer for cuBLAS workspace growth, NCCL buffers,")
    print(f"  PyTorch allocator fragmentation, and CUDA Graph capture peaks.")


if __name__ == "__main__":
    main()
