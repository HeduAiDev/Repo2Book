"""
DCP/PCP Context Parallelism — Our Reimplementation.

REFERENCE sources:
    DCP attention:           vllm/v1/attention/backends/flash_attn.py:L884-L981
    PCP prefill (FlashInfer):vllm/v1/attention/backends/flashinfer.py:L212-L323
    CP communication ops:    vllm/v1/attention/ops/common.py:L212-L234
    DCP A2A:                 vllm/v1/attention/ops/dcp_alltoall.py:L393-L458
    CP KV cache interleave:  vllm/v1/worker/gpu/cp_utils.py:L36-L61
    Device mesh:             vllm/distributed/parallel_state.py:L1569-L1668
    ParallelConfig:          vllm/config/parallel.py:L108-L341

DCP (Decode Context Parallelism): shards KV cache along the TIME dimension
    within the TP group. When tp_size > num_kv_heads (e.g., DeepSeek-R1
    with 1 KV head on 8 GPUs), KV cache is replicated ×8 — wasting 87.5%
    of KV cache memory. DCP shards KV cache across those GPUs instead of
    replicating, enabling larger batches and higher throughput.

PCP (Prefill Context Parallelism): shards the prefill QUERY across GPUs
    to reduce TTFT for long prompts. Currently supports:
    1. Partial query, full KV (gather KV across GPUs)
    2. Partial query, partial KV (ring attention-style)

AG+RS (AllGather + ReduceScatter) — default communication:
    Step 1: AllGather LSEs across CP group
    Step 2: Rescale partial outputs with corrected LSE
    Step 3: ReduceScatter outputs back to original GPUs
    2 NCCL calls per attention layer

A2A (All-to-All) — alternative communication (fewer NCCL calls):
    Pack (output + LSE) → single All-to-All → Unpack and merge
    1 NCCL call per attention layer (good for MLA with large communication)
"""

import math
import torch
from dataclasses import dataclass
from typing import Tuple, List


# ═══════════════════════════════════════════════════════════════════════════
# KV Cache Replication Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_kv_cache_replication(
    tp_size: int, num_kv_heads: int, seq_len: int,
    head_dim: int, num_layers: int, dtype_bytes: int = 2,
) -> dict:
    """
    Quantify KV cache replication when tp_size > num_kv_heads.

    In standard TP, each GPU holds num_kv_heads/tp_size KV heads
    (or all of them if tp_size > num_kv_heads).

    Replication factor = max(1, tp_size / num_kv_heads)

    Example: DeepSeek-R1 (1 KV head MLA, tp=8)
        Replication = 8/1 = 8× → each GPU stores ALL KV cache
        With DCP=8: each GPU stores 1/8 of KV cache
        Memory saving = 8× → can fit 8× larger batch
    """
    tp_kv_heads_per_gpu = max(1, num_kv_heads // tp_size)
    replication_factor = (num_kv_heads * tp_size) / (tp_kv_heads_per_gpu * tp_size)
    replication_factor = max(1.0, tp_size / num_kv_heads)

    per_gpu_kv_bytes = (
        2 * seq_len * tp_kv_heads_per_gpu * head_dim * num_layers * dtype_bytes
    )
    total_wasted = per_gpu_kv_bytes * (replication_factor - 1) * tp_size

    # With DCP: each GPU stores seq_len / dcp_size tokens
    dcp_max = tp_size // num_kv_heads if num_kv_heads > 0 else tp_size
    optimal_dcp = min(tp_size, tp_size // num_kv_heads) if num_kv_heads > 0 else tp_size

    return {
        "tp_size": tp_size,
        "num_kv_heads": num_kv_heads,
        "replication_factor": round(replication_factor, 1),
        "kv_cache_per_gpu_gb": round(per_gpu_kv_bytes / (1024**3), 2),
        "total_wasted_gb": round(total_wasted / (1024**3), 2),
        "optimal_dcp_size": optimal_dcp,
        "note": (
            f"With TP={tp_size} and {num_kv_heads} KV heads, KV cache is "
            f"replicated {replication_factor:.0f}× across GPUs. "
            f"Optimal DCP size = {optimal_dcp} to eliminate all replication."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DCP Attention Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_dcp_attention(
    seq_len: int, num_heads: int, head_dim: int,
    dcp_size: int, cp_interleave: int = 1,
) -> dict:
    """
    Simulate how KV cache is distributed across DCP ranks.

    REFERENCE: vllm/v1/worker/gpu/cp_utils.py:L36-L61
               prepare_dcp_local_seq_lens() — Triton kernel for local seq lens

    The interleave strategy:
        With cp_interleave=1 (token-level):
            Token 0 → Rank 0, Token 1 → Rank 1, ..., Token N-1 → Rank N-1 % dcp_size

        With cp_interleave=block_size (block-level):
            Tokens 0..15 → Rank 0, Tokens 16..31 → Rank 1, ...

    Each rank stores approximately seq_len / dcp_size tokens.
    """
    # Compute local sequence lengths per rank
    local_seq_lens = []
    for rank in range(dcp_size):
        # Round-robin distribution of tokens
        rounds = seq_len // (dcp_size * cp_interleave)
        remainder = seq_len % (dcp_size * cp_interleave)
        remainder_rank = max(remainder - rank * cp_interleave, 0)
        remainder_rank = min(remainder_rank, cp_interleave)
        local = rounds * cp_interleave + remainder_rank
        local_seq_lens.append(local)

    # Communication: AG+RS pattern
    # AllGather LSE: each rank broadcasts its [B, H_local] LSE → [dcp_size, B, H_local]
    lse_ag_bytes = num_heads * 4  # fp32 LSE per head
    # ReduceScatter output: [B, dcp_size*H_local, D] → [B, H_local, D]
    out_rs_bytes = num_heads * head_dim * 2  # bf16 output

    return {
        "seq_len": seq_len,
        "dcp_size": dcp_size,
        "interleave_size": cp_interleave,
        "local_seq_lens": local_seq_lens,
        "max_local_seq_len": max(local_seq_lens),
        "imbalance_ratio": round(max(local_seq_lens) / (seq_len / dcp_size), 2),
        "ag_rs_comm_per_rank_bytes": lse_ag_bytes * dcp_size + out_rs_bytes * dcp_size,
        "a2a_comm_per_rank_bytes": (num_heads * head_dim * 2 + num_heads * 4) * dcp_size,
        "comm_savings_a2a_vs_agrs": (
            "A2A uses 1 NCCL call vs AG+RS 2 calls. "
            "For MLA models with large comm, A2A reduces latency ~30%."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# LSE Merge Algorithm
# REFERENCE: vllm/v1/attention/ops/merge_attn_states.py
#            arxiv.org/pdf/2501.01005 Section 2.2
# ═══════════════════════════════════════════════════════════════════════════

def lse_weighted_merge(
    partial_outputs: List[torch.Tensor],  # [P][B, H, D] — one per CP rank
    lses: List[torch.Tensor],            # [P][B, H] — log-sum-exp per rank
) -> torch.Tensor:
    """
    Merge partial attention outputs using LSE rescaling.

    Math:
        lse_global = max(lse_0, lse_1, ..., lse_{P-1})
        w_i = exp(lse_i - lse_global) / sum_j exp(lse_j - lse_global)
        output = sum_i w_i * partial_output_i

    This is the same online-softmax merge used in FlashAttention's
    partitioned softmax (Chapter 3).
    """
    P = len(partial_outputs)

    # Stack LSEs: [P, B, H]
    lse_stacked = torch.stack(lses, dim=0)

    # Global LSE = max across ranks
    lse_global = lse_stacked.max(dim=0).values  # [B, H]

    # Weights: softmax(LSE_i - LSE_global)
    lse_centered = lse_stacked - lse_global.unsqueeze(0)  # [P, B, H]
    weights = torch.softmax(lse_centered, dim=0)  # [P, B, H]

    # Weighted merge of outputs
    output = sum(
        w.unsqueeze(-1) * out
        for w, out in zip(weights.unbind(0), partial_outputs)
    )  # [B, H, D]

    return output


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Decision Helper
# ═══════════════════════════════════════════════════════════════════════════

def recommend_cp_config(
    model_name: str, num_kv_heads: int, tp_size: int,
    max_seq_len: int, head_dim: int, num_layers: int,
) -> dict:
    """Recommend DCP/PCP configuration for a given model setup."""
    replication = tp_size / num_kv_heads if num_kv_heads > 0 else tp_size

    dcp_rec = 1
    if replication > 1.5:
        dcp_rec = min(tp_size, int(tp_size // num_kv_heads)) if num_kv_heads > 0 else tp_size

    return {
        "model": model_name,
        "tp_size": tp_size,
        "num_kv_heads": num_kv_heads,
        "kv_replication": round(replication, 1),
        "recommended_dcp": dcp_rec,
        "reason": (
            f"KV cache replicated {replication:.0f}× across GPUs. "
            f"DCP={dcp_rec} eliminates replication by sharding KV along sequence dimension."
        ) if dcp_rec > 1 else "No replication — DCP not needed.",
    }


def demonstrate():
    print("DCP/PCP Context Parallelism Analysis")
    print("=" * 60)

    # DeepSeek-R1: 1 KV head, TP=8
    r1 = analyze_kv_cache_replication(8, 1, seq_len=32768, head_dim=128, num_layers=64)
    print(f"\nDeepSeek-R1 (1 KV head, TP=8):")
    print(f"  Replication: {r1['replication_factor']}×")
    print(f"  KV per GPU: {r1['kv_cache_per_gpu_gb']} GB")
    print(f"  Wasted: {r1['total_wasted_gb']} GB across GPUs")
    print(f"  Optimal DCP: {r1['optimal_dcp_size']}")

    # Llama 3.2 70B: 8 KV heads, TP=8
    r2 = analyze_kv_cache_replication(8, 8, seq_len=32768, head_dim=128, num_layers=80)
    print(f"\nLlama 3.2 70B (8 KV heads, TP=8):")
    print(f"  Replication: {r2['replication_factor']}×")
    print(f"  Optimal DCP: {r2['optimal_dcp_size']} (no replication → DCP not needed)")

    # DCP attention simulation
    print(f"\nDCP=4, seq=4096, token-level interleave:")
    s = simulate_dcp_attention(4096, 32, 128, dcp_size=4, cp_interleave=1)
    print(f"  Local lens: {s['local_seq_lens']}")
    print(f"  Max local: {s['max_local_seq_len']}")
    print(f"  {s['comm_savings_a2a_vs_agrs']}")


if __name__ == "__main__":
    demonstrate()
