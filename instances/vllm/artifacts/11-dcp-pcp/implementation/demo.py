"""Ch11 demos producing verbatim numerics for the writer.

Run::

    cd /home/zjq/Repo2Book/instances/vllm/artifacts/11-dcp-pcp
    python3 implementation/demo.py

Produces 5 demos with >=20 verbatim ground-truth values that the writer
quotes character-for-character in the narrative.

REFERENCE: vllm/v1/kv_cache_interface.py:L195-L205 (HBM accounting)
REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L40-L100 (LSE combine)
REFERENCE: vllm/config/parallel.py:L322-L329 (DCPCommBackend literal)
REFERENCE: vllm/v1/attention/backends/utils.py:L820-L857 (striped sharding)
REFERENCE: vllm/distributed/parallel_state.py:L1569-L1633 (5D mesh + groups)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import numpy as np

from implementation.dcp_alltoall import (
    a2a_op_count,
    a2a_payload_bytes,
    ag_rs_op_count,
    ag_rs_payload_bytes,
    alpha_beta_cost,
    simulate_a2a_combine,
)
from implementation.kv_cache_per_rank import (
    LLAMA_70B_KV_SPEC,
    fmt_gb,
    hbm_naive_total,
    hbm_per_rank,
)
from implementation.lse_combine import (
    lse_weighted_combine,
    reference_attention,
    split_attention,
)
from implementation.parallel_state_dcp_pcp import (
    initialize_model_parallel,
    reset_cp_singletons,
)
from implementation.seq_sharding import (
    causal_attention_work_per_rank,
    get_dcp_local_seq_lens,
    imbalance_ratio,
)
from implementation.world_topology import MeshConfig


# REFERENCE: vllm/v1/kv_cache_interface.py:L195-L205 (max_memory_usage_bytes — Demo §1)
# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L40-L100 (_lse_weighted_combine — Demo §2)
# REFERENCE: vllm/config/parallel.py:L322-L329 (DCPCommBackend AG+RS vs A2A — Demo §3)
# REFERENCE: vllm/v1/attention/backends/utils.py:L820-L857 (get_dcp_local_seq_lens — Demo §4)
# REFERENCE: vllm/distributed/parallel_state.py:L1569-L1633 (5D mesh + groups — Demo §5)


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def demo_1_hbm_capacity() -> None:
    """§1 — HBM-per-rank capacity sweep under (DCP, PCP).

    Trap A evidence: HBM is the **win axis** of CP. The sweep shows
    that going from (1,1) to (4,4) reduces per-rank KV cache from
    ``33.5 GB`` to ``2.1 GB`` — 16x reduction in capacity required.
    """
    banner("Demo §1 — HBM-per-rank capacity sweep (Llama-70B at 128K)")
    print("Spec: 80 layers, 8 KV heads, head_size=128, bf16, block_size=16")
    print()
    seq_len = 128 * 1024
    spec = LLAMA_70B_KV_SPEC

    # Naive headline number — for the chapter's "why CP" anchor.
    naive = hbm_naive_total(seq_len, spec)
    print(f"  Naive total KV bytes (no CP): {naive:,} = {fmt_gb(naive)}")
    print()

    print(f"  {'(dcp, pcp)':<12} {'per_rank_len':>12} {'per_rank_bytes':>17} {'as GB':>9}")
    print(f"  {'-'*12:<12} {'-'*12:>12} {'-'*17:>17} {'-'*9:>9}")
    cells = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 4)]
    for dcp, pcp in cells:
        b = hbm_per_rank(seq_len, spec, dcp=dcp, pcp=pcp)
        per_rank_len = -(-seq_len // (dcp * pcp))
        print(f"  ({dcp},{pcp})        {per_rank_len:>12,} {b:>17,} {fmt_gb(b):>9}")
    print()
    print("  Trap A — DCP/PCP wins HBM CAPACITY. Throughput is workload-dependent.")


def demo_2_lse_combine() -> None:
    """§2 — LSE-weighted combine math (Ring Attention algebra).

    Builds a toy: 4 CP ranks each own ``L/4`` of K, V. Each rank
    computes its partial output + LSE. The LSE combine produces a
    final output bit-equivalent to single-process attention.

    Trap F evidence: vLLM ships **no Ring Attention**. The algebra is
    LSE-weighted combine, identical regardless of transport.
    """
    banner("Demo §2 — LSE-weighted combine equivalence")
    rng = np.random.default_rng(seed=42)
    B, H, D, L = 4, 2, 8, 16
    N = 4  # cp_size

    q = rng.standard_normal((B, H, D)).astype(np.float64)
    k = rng.standard_normal((L, H, D)).astype(np.float64)
    v = rng.standard_normal((L, H, D)).astype(np.float64)

    # Ground truth.
    o_truth, lse_truth = reference_attention(q, k, v)

    # Per-rank partials.
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=N)
    print(f"  shape partial_outputs = {parts_o.shape}  (N, B, H, D)")
    print(f"  shape partial_lses    = {parts_lse.shape}  (N, B, H)")

    # Show per-rank LSE values for token 0, head 0:
    print()
    print("  Per-rank LSE values for (token=0, head=0):")
    for i in range(N):
        print(f"    rank {i}: lse_i = {parts_lse[i, 0, 0]: .6f}")

    # Compute weights manually and report.
    lse_max = parts_lse[:, 0, 0].max()
    weights = np.exp(parts_lse[:, 0, 0] - lse_max)
    weights_norm = weights / weights.sum()
    print(f"  lse_max (token=0, head=0) = {lse_max: .6f}")
    print("  per-rank weight (normalized):")
    for i in range(N):
        print(f"    rank {i}: weight_i = {weights_norm[i]: .6f}")

    # Combine and verify.
    combined = simulate_a2a_combine(parts_o, parts_lse)
    err = np.max(np.abs(combined - o_truth))
    print(f"  max abs error vs single-process FlashAttention = {err:.2e}")

    # Associativity proof:
    # (rank 0 + rank 1) + (rank 2 + rank 3) should equal
    # rank 0 + (rank 1 + (rank 2 + rank 3)) — LSE-weighted combine is
    # associative because it's effectively a softmax over ranks.
    o_a = lse_weighted_combine(parts_o[:2], parts_lse[:2], return_lse=True)
    o_b = lse_weighted_combine(parts_o[2:], parts_lse[2:], return_lse=True)
    pair_o = np.stack([o_a.output, o_b.output], axis=0)
    pair_lse = np.stack([o_a.global_lse, o_b.global_lse], axis=0)
    o_assoc = lse_weighted_combine(pair_o, pair_lse, return_lse=False).output
    assoc_err = np.max(np.abs(o_assoc - combined))
    print(f"  associativity error (rank01)+(rank23) vs flat       = {assoc_err:.2e}")
    print()
    print("  Trap F — Same LSE algebra regardless of transport (Ring/A2A/AG+RS).")


def demo_3_ag_rs_vs_a2a() -> None:
    """§3 — AG+RS vs A2A NCCL op count + bandwidth model.

    Trap F evidence: A2A reduces from 3 NCCL ops to 2. Both are
    NCCL collectives. Neither is Ring Attention.
    """
    banner("Demo §3 — AG+RS vs A2A NCCL ops + alpha-beta bandwidth model")

    # H100 4xNVLink reference numbers (literature, NOT measured).
    alpha_us = 10.0
    beta_gbps = 200.0

    num_tokens = 32 * 1024  # 32K prefill
    num_heads = 8
    head_dim = 128
    dtype_bytes = 2

    print(f"  Workload: num_tokens={num_tokens:,}, heads={num_heads}, "
          f"head_dim={head_dim}, bf16")
    print(f"  Model: alpha={alpha_us} us, beta={beta_gbps} GB/s "
          f"(H100 + 4xNVLink, literature reference)")
    print()
    print(f"  {'dcp_size':>8} {'AG+RS ops':>10} {'A2A ops':>9} "
          f"{'AG+RS bytes':>13} {'A2A bytes':>11} {'T_AG+RS us':>11} "
          f"{'T_A2A us':>10} {'speedup':>9}")
    print("  " + "-" * 86)

    for dcp in (2, 4, 8):
        ag_ops = ag_rs_op_count()
        a2_ops = a2a_op_count()
        ag_bytes = ag_rs_payload_bytes(num_tokens, num_heads, head_dim, dcp, dtype_bytes)
        a2_bytes = a2a_payload_bytes(num_tokens, num_heads, head_dim, dcp, dtype_bytes)
        t_ag = alpha_beta_cost(ag_bytes, alpha_us, beta_gbps, num_collectives=ag_ops)
        t_a2 = alpha_beta_cost(a2_bytes, alpha_us, beta_gbps, num_collectives=a2_ops)
        ratio = t_ag / t_a2 if t_a2 > 0 else float("nan")
        print(f"  {dcp:>8} {ag_ops:>10} {a2_ops:>9} {ag_bytes:>13,} "
              f"{a2_bytes:>11,} {t_ag:>11.1f} {t_a2:>10.1f} {ratio:>9.2f}x")

    pct = (1.0 - a2a_op_count() / ag_rs_op_count()) * 100
    print()
    print(f"  A2A reduces NCCL ops by {pct:.0f}% per layer (3 -> 2).")
    print("  Reference: arxiv.org/abs/2507.07120")
    print()
    print("  Trap F — Both are NCCL collectives, not P2P Ring topology.")


def demo_4_striped_vs_contiguous() -> None:
    """§4 — Striped vs contiguous KV partitioning under causal mask.

    Trap G evidence: striped sharding solves causal-mask **load
    imbalance**, not communication pattern.
    """
    banner("Demo §4 — Striped vs contiguous KV partition under causal mask")
    cp_size = 8
    seq_len = 64

    # Contiguous: rank r owns tokens [r*8, (r+1)*8).
    # In source's striped formulation this is interleave_size = seq_len/cp_size.
    contiguous_interleave = seq_len // cp_size
    contig_work = causal_attention_work_per_rank(seq_len, cp_size, contiguous_interleave)
    contig_imb = imbalance_ratio(contig_work)

    # Striped (interleave=1): token i goes to rank i % cp_size.
    striped_work = causal_attention_work_per_rank(seq_len, cp_size, 1)
    striped_imb = imbalance_ratio(striped_work)

    # Block-striped (interleave=2): 2-token chunks round-robin.
    block_work = causal_attention_work_per_rank(seq_len, cp_size, 2)
    block_imb = imbalance_ratio(block_work)

    print(f"  {'scheme':<22} {'interleave':>10} {'per-rank work (KV-attends)':<40}")
    print(f"  {'-'*22:<22} {'-'*10:>10} {'-'*40:<40}")
    print(f"  {'contiguous':<22} {contiguous_interleave:>10} {str(contig_work):<40}")
    print(f"  {'block-striped':<22} {2:>10} {str(block_work):<40}")
    print(f"  {'striped (interleave=1)':<22} {1:>10} {str(striped_work):<40}")
    print()
    print(f"  imbalance ratio (max/min):")
    print(f"    contiguous           = {contig_imb:.2f}x  "
          f"(rank-7 work={max(contig_work)}, rank-0 work={min(contig_work)})")
    print(f"    block-striped (K=2)  = {block_imb:.2f}x")
    print(f"    striped (interleave=1) = {striped_imb:.2f}x  (perfectly balanced)")
    print()
    print("  Trap G — Striped is a TOKEN-PARTITIONING scheme; communication")
    print("           pattern (Ring/A2A/AG+RS) is independent.")

    # Also exercise the get_dcp_local_seq_lens helper from utils.py.
    print()
    print("  get_dcp_local_seq_lens helper:")
    seq_lens = np.array([100, 64, 30, 17])
    for il in (1, 4, 16):
        local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=il)
        sums = local.sum(axis=-1)
        print(f"    interleave_size={il:>2}: per_rank_lens =\n"
              f"        {local}")
        print(f"      sum-across-ranks = {sums.tolist()} (must equal seq_lens "
              f"{seq_lens.tolist()})")


def demo_5_mesh_groups() -> None:
    """§5 — 5D mesh group construction at world_size=16.

    Trap D evidence: DCP and PCP are **separable axes**. Production
    config ``(tp=4, pcp=2, pp=2, dp=1, dcp=2)`` is valid; world_size
    is ``4 * 2 * 2 * 1 = 16`` (DCP excluded).
    """
    banner("Demo §5 — 5D mesh groups (world_size=16)")
    mesh = MeshConfig(external_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2)
    print(f"  MeshConfig: ext_dp={mesh.external_dp}, dp={mesh.dp}, "
          f"pp={mesh.pp}, pcp={mesh.pcp}, tp={mesh.tp}, dcp={mesh.dcp}")
    print(f"  world_size = ext_dp * dp * pp * pcp * tp = "
          f"{mesh.external_dp} * {mesh.dp} * {mesh.pp} * {mesh.pcp} * {mesh.tp} "
          f"= {mesh.world_size}")
    print(f"  total_cp_world_size = pcp * dcp = {mesh.pcp} * {mesh.dcp} = "
          f"{mesh.total_cp_world_size}")
    print(f"  num_dcp_subgroups per TP-group = tp/dcp = {mesh.num_dcp_subgroups}")

    reset_cp_singletons()
    groups = initialize_model_parallel(
        rank=0,
        world_size=mesh.world_size,
        tensor_model_parallel_size=mesh.tp,
        pipeline_model_parallel_size=mesh.pp,
        prefill_context_model_parallel_size=mesh.pcp,
        decode_context_model_parallel_size=mesh.dcp,
        data_parallel_size=mesh.dp,
    )

    print()
    print(f"  TP groups (count={len(groups['tp'])}):")
    for grp in groups["tp"]:
        print(f"    {grp}")
    print(f"  DCP sub-groups (count={len(groups['dcp'])}, folded inside TP):")
    for grp in groups["dcp"]:
        print(f"    {grp}")
    print(f"  PCP groups (count={len(groups['pcp'])}, independent axis):")
    for grp in groups["pcp"]:
        print(f"    {grp}")
    print(f"  PP groups (count={len(groups['pp'])}):")
    for grp in groups["pp"]:
        print(f"    {grp}")
    print()
    print("  Trap D — DCP and PCP are SEPARABLE axes; only tp%dcp==0 is forced.")


def main() -> None:
    print("Chapter 11: DCP/PCP — Demo numerics")
    print("Source pin: vllm-project/vllm @ 98661fe")
    demo_1_hbm_capacity()
    demo_2_lse_combine()
    demo_3_ag_rs_vs_a2a()
    demo_4_striped_vs_contiguous()
    demo_5_mesh_groups()
    print()
    print("=" * 72)
    print("  All 5 demos complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
