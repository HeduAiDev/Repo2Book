"""
PagedAttention + FlashAttention — Our Reimplementation.

Shows the FUSION of two optimizations:
    1. FlashAttention: tiled online softmax (IO-aware compute)
    2. PagedAttention: block_table indirection (virtual memory for KV cache)

REFERENCE sources (vLLM):
    CUDA kernel:        csrc/attention/attention_kernels.cuh:L85-L490
    CUDA V1 launcher:   csrc/attention/paged_attention_v1.cu:L160
    CUDA V2 launcher:   csrc/attention/paged_attention_v2.cu
    Triton decode:      vllm/v1/attention/ops/triton_decode_attention.py:L60
    Triton unified:     vllm/v1/attention/ops/triton_unified_attention.py:L58
    FA backend:         vllm/v1/attention/backends/flash_attn.py:L682
    FA interface:       vllm/vllm_flash_attn/flash_attn_interface.py:L176

Architecture:
    Naive Attention:      Q@K^T → [seq²] in HBM → softmax → [seq²] → @V
    FlashAttention:       Tiled Q@K^T in SRAM → online softmax → never write [seq²]
    PagedAttention:       KV stored in non-contiguous blocks → block_table maps
    FlashAttention + PA:  Tiled attention + block_table lookup fused in one kernel
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════════════
# 3.2 HBM Traffic Analysis: Naive vs FlashAttention
# ═══════════════════════════════════════════════════════════════════════════

# REFERENCE: vllm/v1/attention/backends/flash_attn.py:L682 — FlashAttentionImpl.forward() HBM traffic
# REFERENCE: Dao et al. "FlashAttention" (2022) — IO complexity analysis
def calculate_hbm_traffic(
    seq_len: int, num_heads: int, head_dim: int, dtype_bytes: int = 2,
    num_layers: int = 1,
) -> dict:
    """
    Calculate HBM read/write bytes for one attention computation.

    This is the quantitative basis for WHY FlashAttention matters.
    """
    # Naive attention
    naive_read = (
        num_heads * seq_len * head_dim * dtype_bytes     # Q
        + num_heads * seq_len * head_dim * dtype_bytes   # K
        + num_heads * seq_len * head_dim * dtype_bytes   # V
    )
    naive_write_intermediate = (
        num_heads * seq_len * seq_len * 4                # S (fp32)
        + num_heads * seq_len * seq_len * 4              # P (fp32)
    )
    naive_write_output = num_heads * seq_len * head_dim * dtype_bytes  # O
    naive_total = naive_read + naive_write_intermediate + naive_write_output

    # FlashAttention (approximate — tiled, no O(n²) intermediates written)
    # Q loaded once per Q tile: seq/BLOCK_Q passes
    # K,V loaded once per (Q tile × KV tile): (seq²)/(BLOCK_Q×BLOCK_KV) passes
    # O written once
    BLOCK_Q, BLOCK_KV = 64, 64
    fa_read = (
        num_heads * seq_len * head_dim * dtype_bytes                            # Q (once)
        + num_heads * seq_len * head_dim * dtype_bytes * (seq_len // BLOCK_Q)   # K (re-read)
        + num_heads * seq_len * head_dim * dtype_bytes * (seq_len // BLOCK_Q)   # V (re-read)
    )
    fa_write = num_heads * seq_len * head_dim * dtype_bytes  # O only
    fa_total = fa_read + fa_write

    return {
        "seq_len": seq_len,
        "dtype_bytes": dtype_bytes,
        "naive": {
            "read_bytes": naive_read,
            "write_intermediate_bytes": naive_write_intermediate,
            "write_output_bytes": naive_write_output,
            "total_gb": round(naive_total / (1024**3), 3),
        },
        "flashattention": {
            "read_bytes": fa_read,
            "write_bytes": fa_write,
            "total_gb": round(fa_total / (1024**3), 3),
            "speedup_vs_naive": round(naive_total / fa_total, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.3 PagedAttention: Block Table Indirection
# REFERENCE: csrc/attention/attention_kernels.cuh:L202, L252-L253
# ═══════════════════════════════════════════════════════════════════════════

# REFERENCE: csrc/attention/attention_kernels.cuh:L202,L252-L253 — block_table indirection
# REFERENCE: vllm/v1/attention/ops/triton_decode_attention.py:L119-L126 — Triton decode block_table
def paged_attention_with_block_table(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Paged Attention: attention over non-contiguous KV cache blocks.

    REFERENCE: csrc/attention/attention_kernels.cuh:L85-L490
               → paged_attention_kernel() implements this logic in CUDA

    The key indirection (from .cuh L202, L252-L253):
        const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
        const int64_t physical_block_number = block_table[block_idx];

    Args:
        Q: [batch, num_heads, head_dim] — single token query (decode phase)
        K_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        V_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: [batch, max_num_blocks] — maps logical→physical block
        seq_lens: [batch] — actual sequence length per batch item
        block_size: tokens per block (typical: 16)

    Returns:
        output: [batch, num_heads, head_dim]

    NOTE: vLLM's production kernel does this in CUDA with online softmax
    and tiled computation. We do it step-by-step in PyTorch for clarity.
    """
    B, H, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.zeros_like(Q)

    for b in range(B):
        L = seq_lens[b].item()
        num_blocks = (L + block_size - 1) // block_size

        # Gather all K,V for this sequence from non-contiguous blocks
        K_seq = []
        V_seq = []
        for blk in range(num_blocks):
            phys_blk = block_table[b, blk].item()  # ← THE indirection
            # phys_blk = block_table[seq_idx][logical_block_idx]

            # Tokens in this block (last block may be partial)
            tokens_in_block = min(block_size, L - blk * block_size)
            K_seq.append(K_cache[phys_blk, :tokens_in_block])
            V_seq.append(V_cache[phys_blk, :tokens_in_block])

        K_b = torch.cat(K_seq, dim=0)  # [L, H_kv, D]
        V_b = torch.cat(V_seq, dim=0)  # [L, H_kv, D]

        # Handle GQA: repeat KV heads to match Q heads
        if K_b.size(1) != H:
            reps = H // K_b.size(1)
            K_b = K_b.repeat_interleave(reps, dim=1)
            V_b = V_b.repeat_interleave(reps, dim=1)

        # Attention — single query over all keys
        # Q[b]: [H, D], K_b: [L, H, D] → scores: [H, L]
        scores = torch.einsum('hd,lhd->hl', Q[b], K_b) * scale
        attn = F.softmax(scores, dim=-1)   # [H, L]
        output[b] = torch.einsum('hl,lhd->hd', attn, V_b)  # [H, D]

    return output


# ═══════════════════════════════════════════════════════════════════════════
# 3.5 Fused Attention + Block Table (Educational Triton-like logic)
# ═══════════════════════════════════════════════════════════════════════════

# REFERENCE: csrc/attention/attention_kernels.cuh:L85 — paged_attention_kernel (CUDA)
# REFERENCE: vllm/v1/attention/ops/triton_unified_attention.py:L58 — unified Triton kernel
def fused_paged_attention_tiled(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scale: Optional[float] = None,
    BLOCK_Q: int = 32,
    BLOCK_KV: int = 32,
) -> torch.Tensor:
    """
    Fused FlashAttention + PagedAttention: tiled attention OVER block_table.

    This is what vLLM's actual kernel does — combines three things:
        1. Tiled computation (FlashAttention): Q blocks × KV blocks
        2. Online softmax: running m/l for numerical stability
        3. Block table indirection (PagedAttention): logical→physical mapping

    REFERENCE: This is the algorithm that CUDA paged_attention_kernel (.cuh:L85)
               and triton_decode_attention (triton_decode_attention.py:L60)
               both implement, adapted to readable Python.

    The core loop:
        for each Q block:
            m, l, O_acc = -inf, 0, 0
            for each KV logical block:
                phys_blk = block_table[seq_idx, logical_blk]  ← PA indirection
                K_block = K_cache[phys_blk]                   ← load from physical
                V_block = V_cache[phys_blk]
                S = Q_block @ K_block^T / sqrt(d_k)           ← FA: in SRAM only
                m_new = max(m, row_max(S))
                P = exp(S - m_new)
                correction = exp(m - m_new)
                l_new = correction * l + row_sum(P)
                O_acc = correction * O_acc + P @ V_block      ← fused accumulate
                m, l = m_new, l_new
            O_block = O_acc / l
    """
    B, H, D = Q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    output = torch.zeros_like(Q)

    for b in range(B):
        L = seq_lens[b].item()
        num_blocks = (L + block_size - 1) // block_size

        # Initialize online softmax state
        m = torch.full([H], float("-inf"), device=Q.device, dtype=torch.float32)
        l = torch.zeros([H], device=Q.device, dtype=torch.float32)
        O_acc = torch.zeros([H, D], device=Q.device, dtype=torch.float32)

        for blk in range(num_blocks):
            phys_blk = block_table[b, blk].item()  # ← PagedAttention indirection
            tokens = min(block_size, L - blk * block_size)

            K_blk = K_cache[phys_blk, :tokens]  # [block_size, H_kv, D]
            V_blk = V_cache[phys_blk, :tokens]

            # GQA handling
            if K_blk.size(1) != H:
                reps = H // K_blk.size(1)
                K_blk = K_blk.repeat_interleave(reps, dim=1)
                V_blk = V_blk.repeat_interleave(reps, dim=1)

            # Q[b]: [H, D], K_blk: [block_size, H, D] → S: [H, block_size]
            S = torch.einsum('hd,bhd->hb', Q[b], K_blk) * scale

            # Online softmax update
            m_new = torch.maximum(m, S.max(dim=-1).values)
            P = torch.exp(S - m_new.unsqueeze(-1))
            correction = torch.exp(m - m_new)
            l_new = correction * l + P.sum(dim=-1)

            # Fused accumulation: O_acc = correction * O_acc + P @ V
            # P: [H, block_size], V_blk: [block_size, H, D]
            update = torch.einsum('hb,bhd->hd', P, V_blk)
            O_acc = correction.unsqueeze(-1) * O_acc + update

            m, l = m_new, l_new

        output[b] = O_acc / l.unsqueeze(-1)

    return output


# ═══════════════════════════════════════════════════════════════════════════
# 3.4 Block Table Utilities
# ═══════════════════════════════════════════════════════════════════════════

def build_block_table(
    seq_lens: List[int], num_gpu_blocks: int, block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate vLLM's block allocation to build block_table.

    REFERENCE: vllm/v1/core/kv_cache_manager.py:L225 → allocate_slots()
               vllm/v1/core/block_pool.py:L322 → get_new_blocks()

    In vLLM, the Scheduler calls allocate_slots(), which uses BlockPool
    to allocate physical blocks. The resulting block_table is passed to
    the attention kernel.

    Here we simulate a simple first-fit allocator.
    """
    B = len(seq_lens)
    max_blocks_per_seq = 0
    for sl in seq_lens:
        n = (sl + block_size - 1) // block_size
        max_blocks_per_seq = max(max_blocks_per_seq, n)

    block_table = torch.full((B, max_blocks_per_seq), -1, dtype=torch.int32)
    free_blocks = list(range(num_gpu_blocks))  # simple free list

    for b, sl in enumerate(seq_lens):
        n = (sl + block_size - 1) // block_size
        for blk in range(n):
            if not free_blocks:
                raise RuntimeError("OOM: no free blocks")
            block_table[b, blk] = free_blocks.pop(0)

    return (
        block_table,
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor([num_gpu_blocks - len(free_blocks)], dtype=torch.int32),
    )


def hbm_traffic_comparison_table():
    """Generate the data for the chapter's HBM comparison figure."""
    print("HBM Traffic: Naive vs FlashAttention + PagedAttention")
    print("=" * 70)
    print(f"{'Seq Len':>8} | {'Naive HBM':>12} | {'FA+PA HBM':>12} | {'Speedup':>8}")
    print("-" * 70)
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        r = calculate_hbm_traffic(seq_len, num_heads=32, head_dim=128)
        print(f"{seq_len:8} | {r['naive']['total_gb']:9.3f} GB | "
              f"{r['flashattention']['total_gb']:9.3f} GB | "
              f"{r['flashattention']['speedup_vs_naive']:6.1f}x")


if __name__ == "__main__":
    hbm_traffic_comparison_table()
