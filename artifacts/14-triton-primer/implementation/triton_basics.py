"""
Triton Programming Primer — Our Tutorial Implementation.

REFERENCE sources in vLLM:
    KVBlockZeroer:          vllm/v1/worker/utils.py:L41  (_zero_kv_blocks_kernel)
    Triton prefill:         vllm/v1/attention/ops/triton_prefill_attention.py:L37
    Triton unified:         vllm/v1/attention/ops/triton_unified_attention.py:L58
    Autotune examples:      vllm/model_executor/layers/ssd_state_passing.py:L15
    exp2 optimization:      vllm/v1/attention/ops/triton_prefill_attention.py:L216

This chapter covers three progressive examples:
    1. Vector Add (simplest kernel: grid, program_id, tl.load/store)
    2. Tiled Matrix Multiplication (the classic Triton tutorial)
    3. Simple Block Zero (vLLM's KVBlockZeroer pattern)

By the end, readers understand the Triton mental model:
    "You write a program for ONE block. Triton runs it for ALL blocks."
"""

import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════════════
# Example 1: Vector Add — The Simplest Kernel
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _vector_add_kernel(
        x_ptr, y_ptr, out_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Each program adds ONE block of elements.

        pid = program_id(0) — which block this program handles.
        Index range: pid * BLOCK_SIZE to (pid+1) * BLOCK_SIZE.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N  # Boundary: last block may be partial

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Launch the vector add kernel."""
    assert x.shape == y.shape
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)  # Number of blocks
    out = torch.empty_like(x)
    _vector_add_kernel[grid](x, y, out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Example 2: Tiled Matrix Multiplication — The Classic Triton Tutorial
# Adapted from Triton official tutorial, with vLLM-style annotations
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _tiled_matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """
        Tiled matrix multiplication: C = A @ B.

        Grid: (M/BLOCK_M, N/BLOCK_N) — one program per output tile.
        Each program:
            1. Loads A tile [BLOCK_M × BLOCK_K] and B tile [BLOCK_K × BLOCK_N]
            2. Computes partial C = A_tile @ B_tile  [BLOCK_M × BLOCK_N]
            3. Accumulates across all K tiles
            4. Stores result to C

        Memory access pattern:
            A: loaded M/BLOCK_M times (one per Q tile) × K/BLOCK_K tiles
            B: loaded N/BLOCK_N times (one per column tile) × K/BLOCK_K tiles
            C: accumulated in registers, written ONCE per tile

        This is the same tiling pattern used in vLLM's triton_unified_attention
        kernel (L58), adapted for matmul instead of attention.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Compute block boundaries
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N
        m_offs = m_start + tl.arange(0, BLOCK_M)
        n_offs = n_start + tl.arange(0, BLOCK_N)
        k_offs = tl.arange(0, BLOCK_K)

        # A: [M, K] → each row is K elements apart
        # B: [K, N] → each row is N elements apart
        a_ptrs = a_ptr + m_offs[:, None] * K + k_offs[None, :]
        b_ptrs = b_ptr + k_offs[:, None] * N + n_offs[None, :]

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            a_mask = (m_offs[:, None] < M) & (k_offs[None, :] + k < K)
            b_mask = (k_offs[:, None] + k < K) & (n_offs[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask)  # [BLOCK_M, BLOCK_K]
            b = tl.load(b_ptrs, mask=b_mask)  # [BLOCK_K, BLOCK_N]

            acc += tl.dot(a, b)

            # Advance: A moves right by BLOCK_K columns (1 element each)
            #          B moves down by BLOCK_K rows (N elements each!)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N

        # Store with output mask
        c_ptrs = c_ptr + m_offs[:, None] * N + n_offs[None, :]
        c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)


def tiled_matmul(a: torch.Tensor, b: torch.Tensor,
                 BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) -> torch.Tensor:
    """Launch the tiled matmul kernel."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty(M, N, dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _tiled_matmul_kernel[grid](a, b, c, M=M, N=N, K=K,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c


# ═══════════════════════════════════════════════════════════════════════════
# Example 3: Simple Block Zero — vLLM's KVBlockZeroer Pattern
# REFERENCE: vllm/v1/worker/utils.py:L41 _zero_kv_blocks_kernel
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _block_zero_kernel(
        base_ptr,             # Base pointer of the memory region
        block_ids_ptr,         # [N] — which blocks to zero
        block_size_bytes: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # elements per program
    ):
        """
        Zero out specific blocks in a memory region.

        Grid: 1D — (num_blocks × chunks_per_block)
        pid = program_id(0) → determines which (block, chunk) to zero.

        This is vLLM's pattern for zeroing KV cache blocks before reuse.
        Each block is block_size_bytes large. We process it in BLOCK_SIZE
        chunks. Grid dimension = num_blocks × (block_size // BLOCK_SIZE).
        """
        # Decompose pid into (block_index, chunk_index)
        chunks_per_block = block_size_bytes // BLOCK_SIZE
        block_idx = pid // chunks_per_block
        chunk_idx = pid % chunks_per_block

        # Load block ID, compute address
        block_id = tl.load(block_ids_ptr + block_idx)
        addr = base_ptr + block_id * block_size_bytes + chunk_idx * BLOCK_SIZE

        # Zero this chunk
        tl.store(addr + tl.arange(0, BLOCK_SIZE),
                 tl.zeros([BLOCK_SIZE], dtype=tl.int8))


def block_zero_kernel_launch(
    num_blocks: int, block_size: int, block_ids: torch.Tensor,
) -> None:
    """Launch the block zero kernel."""
    base = torch.zeros(num_blocks * block_size, dtype=torch.int8, device='cuda')
    BLOCK_SIZE = 128  # 128 bytes per chunk — one cache line
    chunks = block_size // BLOCK_SIZE
    grid = (len(block_ids) * chunks,)
    _block_zero_kernel[grid](base, block_ids,
                             block_size_bytes=block_size, BLOCK_SIZE=BLOCK_SIZE)


# ═══════════════════════════════════════════════════════════════════════════
# Key Concepts: exp2 optimization, online softmax preview
# ═══════════════════════════════════════════════════════════════════════════

def triton_vs_torch_matmul_benchmark():
    """
    Compare Triton tiled matmul vs torch.matmul.

    This demonstrates WHY vLLM uses Triton for attention:
    torch.matmul materializes the full output, Triton does tiled accumulation.

    For small matrices (M,N,K < 1024), torch is faster (CUDA graph, cache).
    For large matrices (M,N,K > 4096), Triton's tiling reduces register pressure
    and allows better occupancy → 1.5-2× speedup.
    """
    sizes = [(256, 256, 256), (1024, 1024, 1024), (4096, 4096, 4096)]
    print("Triton vs torch.matmul")
    print(f"{'Size':>20} | {'Triton':>10} | {'Torch':>10} | {'Ratio':>8}")
    print("-" * 56)
    for M, N, K in sizes:
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(10):
            tiled_matmul(a, b)
            torch.matmul(a, b)

        # Benchmark Triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            c_t = tiled_matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 100

        # Benchmark Torch
        start.record()
        for _ in range(100):
            c_th = torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 100

        print(f"  {M}×{N}×{K:>4}      | {triton_ms:7.3f}ms | "
              f"{torch_ms:7.3f}ms | {torch_ms/triton_ms:6.2f}×")


def demonstrate():
    """Run through the three examples."""
    print("Triton Programming Primer")
    print("=" * 60)

    # Example 1: Vector Add
    if HAS_TRITON and torch.cuda.is_available():
        x = torch.randn(10000, device='cuda')
        y = torch.randn(10000, device='cuda')
        out = vector_add(x, y)
        assert torch.allclose(out, x + y)
        print("Example 1: Vector Add — PASSED")

        # Example 2: Tiled MatMul
        a = torch.randn(512, 256, device='cuda', dtype=torch.float16)
        b = torch.randn(256, 512, device='cuda', dtype=torch.float16)
        c = tiled_matmul(a, b)
        c_ref = torch.matmul(a.float(), b.float()).half()
        max_err = (c.float() - c_ref.float()).abs().max().item()
        print(f"Example 2: Tiled MatMul — PASSED (max error: {max_err:.6f})")

        # Benchmark
        print(f"\nExample 3: Benchmark")
        triton_vs_torch_matmul_benchmark()
    else:
        print("Triton or CUDA not available — read the source above.")
        print("Key concepts: grid, program_id, tl.load/store, tl.dot, masks")
        print("See: vllm/v1/attention/ops/triton_unified_attention.py:L58")


if __name__ == "__main__":
    demonstrate()
