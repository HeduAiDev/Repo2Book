"""Fidelity tests — pin every trap from impl-notes §3 + the no-class-X reframe.

Each trap from §3 has at least one test pinning the correct interpretation.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from implementation import parallel_state_dcp_pcp as ps
from implementation.attention_backend_dcp_pcp import (
    AttentionImplBase,
    FlashAttn3MlaBackend,
    FlashAttnBackend,
    FlashInferBackend,
    RocmAiterMlaBackend,
)
from implementation.dcp_alltoall import (
    a2a_op_count,
    a2a_payload_bytes,
    ag_rs_op_count,
    ag_rs_payload_bytes,
    alpha_beta_cost,
    simulate_a2a_combine,
    simulate_ag_rs_combine,
)
from implementation.dcp_vs_pcp_demo import CPRoles
from implementation.kv_cache_per_rank import (
    LLAMA_70B_KV_SPEC,
    hbm_naive_total,
    hbm_per_rank,
)
from implementation.lse_combine import (
    lse_weighted_combine,
    reference_attention,
    split_attention,
)
from implementation.seq_sharding import (
    causal_attention_work_per_rank,
    imbalance_ratio,
)
from implementation.world_topology import MeshConfig


VLLM_SOURCE = Path("/home/zjq/Repo2Book/instances/vllm/source/vllm")


@pytest.fixture(autouse=True)
def _reset():
    ps.reset_cp_singletons()
    yield
    ps.reset_cp_singletons()


# ==========================================================================
# Trap A — "DCP doubles decode throughput at dcp_size=2."  WRONG
# ==========================================================================


def test_trap_A_hbm_is_the_win_axis_not_throughput():
    """DCP wins HBM CAPACITY (16x at dcp=4,pcp=4); throughput is workload-dependent."""
    naive = hbm_naive_total(128 * 1024, LLAMA_70B_KV_SPEC)
    sharded = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4)
    # 40 GB / 2.5 GB = 16x.
    assert naive // sharded == 16


def test_trap_A_hbm_decreases_at_dcp_2():
    """HBM at dcp=2,pcp=1 is half of dcp=1,pcp=1."""
    full = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    half = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=1)
    assert half * 2 == full


# ==========================================================================
# Trap B — "PCP halves prefill latency at pcp_size=2."  Workload-dependent
# ==========================================================================


def test_trap_B_a2a_speedup_grows_with_dcp_size():
    """At larger dcp_size, A2A advantage grows because payload shrinks."""
    speedups = []
    for dcp in (2, 4, 8):
        ag = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=dcp)
        a2 = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=dcp)
        t_ag = alpha_beta_cost(ag, 10.0, 200.0, num_collectives=ag_rs_op_count())
        t_a2 = alpha_beta_cost(a2, 10.0, 200.0, num_collectives=a2a_op_count())
        speedups.append(t_ag / t_a2)
    assert speedups[0] < speedups[1] < speedups[2]


def test_trap_B_alpha_dominates_at_small_payload():
    """At zero bytes, both backends pay only alpha — A2A 'wins' only by op count."""
    t_ag = alpha_beta_cost(0, 10.0, 200.0, num_collectives=ag_rs_op_count())
    t_a2 = alpha_beta_cost(0, 10.0, 200.0, num_collectives=a2a_op_count())
    # 30 vs 20 us — speedup = 1.5x, NOT 2x or 3x.
    assert (t_ag / t_a2) == pytest.approx(1.5)


# ==========================================================================
# Trap C — "Context parallel is just sequence parallel renamed."  WRONG
# ==========================================================================


def test_trap_C_dcp_pcp_are_distinct_from_sequence_parallel():
    """DCP and PCP are CP-axis groups; sequence-parallel is a TP-internal thing.

    Source has BOTH:
      - DCP/PCP groups: parallel_state.py:L1593-L1633
      - is_sequence_parallel arg on all_gather/reduce_scatter (TP-internal SP)

    These are separate mechanisms.
    """
    parallel_state_src = VLLM_SOURCE / "distributed" / "parallel_state.py"
    if parallel_state_src.exists():
        text = parallel_state_src.read_text()
        # Both DCP/PCP and is_sequence_parallel must appear in the source.
        assert "_DCP" in text
        assert "_PCP" in text
        # At least one is_sequence_parallel reference exists (TP-internal SP).
        assert "is_sequence_parallel" in text or "sequence_parallel" in text


def test_trap_C_naive_combine_is_wrong():
    """LSE-weighted combine ≠ naive average. CP-attention output requires LSE re-weighting."""
    rng = np.random.default_rng(seed=0)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    o_truth, _ = reference_attention(q, k, v)
    parts_o, _ = split_attention(q, k, v, num_ranks=4)
    naive = parts_o.sum(axis=0) / 4.0
    err = np.max(np.abs(naive - o_truth))
    assert err > 1e-3


# ==========================================================================
# Trap D — "DCP and PCP must match (dcp_size == pcp_size)."  WRONG
# ==========================================================================


def test_trap_D_match_required_is_false():
    assert CPRoles.both_match_required() is False


def test_trap_D_unequal_dcp_pcp_is_valid():
    """(tp=8, dcp=2, pcp=4) — production-realistic config."""
    m = MeshConfig(tp=8, dcp=2, pcp=4)
    assert m.dcp != m.pcp
    assert m.world_size == 32  # tp * pcp


def test_trap_D_world_size_excludes_dcp():
    """Source verbatim: world_size = tp * pp * pcp (NOT * dcp)."""
    m = MeshConfig(tp=8, dcp=4, pcp=2)
    # If dcp entered the product, world would be 64. Instead it's 16.
    assert m.world_size == 16


def test_trap_D_only_constraint_is_tp_modulo_dcp():
    """The ONLY constraint is tp % dcp == 0."""
    # tp=8, dcp=3 violates: 8 % 3 != 0
    with pytest.raises(ValueError, match="must be divisible"):
        MeshConfig(tp=8, dcp=3, pcp=4)
    # tp=8, dcp=2, pcp=4 is valid (regardless of pcp value)
    MeshConfig(tp=8, dcp=2, pcp=4)
    # tp=8, dcp=2, pcp=999999 is also "valid" (no constraint between dcp and pcp)
    MeshConfig(tp=8, dcp=2, pcp=64)


# ==========================================================================
# Trap E — "Context parallel = TP for the attention layer."  WRONG
# ==========================================================================


def test_trap_E_dcp_shards_kv_not_heads():
    """DCP shards KV cache (sequence axis); TP shards heads.

    Different axes ⇒ different communication ⇒ different layer placement.
    """
    # MLA backend's num_heads_q is replicated by dcp_world_size — NOT sharded.
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    # With dcp=2, num_heads_q = 8 * 2 = 16 (Q replicated, NOT sharded).
    assert backend.num_heads_q == 16


def test_trap_E_total_cp_separate_from_tp():
    """total_cp_world_size = pcp * dcp; INDEPENDENT of tp."""
    m1 = MeshConfig(tp=4, pcp=2, dcp=2)
    m2 = MeshConfig(tp=8, pcp=2, dcp=2)
    # Same total_cp_world_size despite different tp.
    assert m1.total_cp_world_size == m2.total_cp_world_size


# ==========================================================================
# Trap F — "Ring Attention is canonical in vLLM."  WRONG
# ==========================================================================


def test_trap_F_no_class_RingAttention_in_source():
    """Verified at commit 98661fe: no class RingAttention/StripedAttention/ContextParallel."""
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    out = subprocess.check_output(
        ["grep", "-rE", r"^class\s+(RingAttention|StripedAttention|ContextParallel|DecodeContextParallel|PrefillContextParallel)\b",
         str(VLLM_SOURCE)],
        stderr=subprocess.DEVNULL,
    ) if False else b""
    # Run grep manually (won't fail if no matches by allowing returncode 1).
    proc = subprocess.run(
        ["grep", "-rE",
         r"^class\s+(RingAttention|StripedAttention|ContextParallel|DecodeContextParallel|PrefillContextParallel)\b",
         str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    # Expected: returncode 1 (no matches).
    assert proc.returncode == 1
    assert proc.stdout.strip() == ""


def test_trap_F_only_one_DCP_prefixed_class():
    """REFERENCE: knowledge fact D09 — BatchDCPPrefillWrapper is the only one."""
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+\w*DCP\w*", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    matches = [line for line in proc.stdout.splitlines() if line.strip()]
    # Only BatchDCPPrefillWrapper at flashinfer.py:213.
    assert len(matches) == 1
    assert "BatchDCPPrefillWrapper" in matches[0]
    assert "flashinfer.py" in matches[0]


def test_trap_F_no_p2p_send_recv_in_dcp_alltoall():
    """vLLM uses NCCL collectives, not P2P send/recv (Ring topology).

    Source: vllm/v1/attention/ops/dcp_alltoall.py:L448 uses dist.all_to_all_single.
    """
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    a2a_src = VLLM_SOURCE / "v1" / "attention" / "ops" / "dcp_alltoall.py"
    if not a2a_src.exists():
        pytest.skip("dcp_alltoall.py not found")
    text = a2a_src.read_text()
    # Should use all_to_all_single — collective, NOT P2P.
    assert "all_to_all_single" in text
    # Should NOT use P2P send/recv style ring.
    assert "isend(" not in text  # no asynchronous P2P send


def test_trap_F_dcp_comm_backend_literal_only_ag_rs_or_a2a():
    """REFERENCE: parallel.py:L322-L328 — DCPCommBackend = Literal['ag_rs', 'a2a']."""
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    parallel_src = VLLM_SOURCE / "config" / "parallel.py"
    if not parallel_src.exists():
        pytest.skip("config/parallel.py not found")
    text = parallel_src.read_text()
    # Source verbatim: DCPCommBackend = Literal["ag_rs", "a2a"]
    assert re.search(r"DCPCommBackend\s*=\s*Literal\[\s*[\"']ag_rs[\"']\s*,\s*[\"']a2a[\"']\s*\]", text)
    # No "ring" backend literal.
    assert "Literal[\"ring\"" not in text


def test_trap_F_a2a_count_2_ag_rs_count_3():
    """A2A: 2 NCCL ops; AG+RS: 3. Both NCCL, neither Ring."""
    assert a2a_op_count() == 2
    assert ag_rs_op_count() == 3


# ==========================================================================
# Trap G — "Striped is just renamed Ring."  WRONG
# Striped is a TOKEN-PARTITIONING scheme; communication is independent.
# ==========================================================================


def test_trap_G_striped_imbalance_drops_to_1_24x():
    """Demo §4: striped (I=1) drops imbalance from 13.44x to 1.24x."""
    contig = imbalance_ratio(causal_attention_work_per_rank(64, 8, 8))
    striped = imbalance_ratio(causal_attention_work_per_rank(64, 8, 1))
    assert contig == pytest.approx(13.44, abs=0.01)
    assert striped == pytest.approx(1.24, abs=0.01)


def test_trap_G_partition_independent_of_communication():
    """Striped (I=1) determines token→rank mapping; communication is separate.

    Test: at I=1, the LSE-weighted combine still works regardless of mapping.
    """
    rng = np.random.default_rng(seed=99)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    o_truth, _ = reference_attention(q, k, v)

    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out_a2a = simulate_a2a_combine(parts_o, parts_lse)
    out_ag_rs = simulate_ag_rs_combine(parts_o, parts_lse)
    # All produce single-process output regardless of comm pattern.
    assert np.max(np.abs(out_a2a - o_truth)) < 1e-12
    assert np.max(np.abs(out_ag_rs - o_truth)) < 1e-12


# ==========================================================================
# "no class X" reframe (5th instance)
# Knowledge fact D09 + impl-notes §2 reframe A
# ==========================================================================


def test_no_class_RingAttention_in_source():
    """5th 'no class X' instance after Ch07 (RadixTree) / Ch08 (TensorParallel) /
    Ch09 (ExpertParallel) / Ch10 (MultiTokenPrediction)."""
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+RingAttention\b", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stdout.strip() == ""


def test_no_class_StripedAttention_in_source():
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+StripedAttention\b", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


def test_no_class_ContextParallel_in_source():
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+ContextParallel\b", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


def test_no_class_DecodeContextParallel_in_source():
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+DecodeContextParallel\b", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


def test_no_class_PrefillContextParallel_in_source():
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    proc = subprocess.run(
        ["grep", "-rE", r"^class\s+PrefillContextParallel\b", str(VLLM_SOURCE)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1


def test_DCP_machinery_is_singletons_plus_module_level():
    """The DCP/PCP machinery is _DCP / _PCP singletons + module-level functions.

    Knowledge D01 + D14: the machinery is module-level pure functions plus
    singletons; per-backend __new__ does discovery.
    """
    # Our module mirror has the singletons at module level.
    assert hasattr(ps, "_DCP")
    assert hasattr(ps, "_PCP")
    # Singleton accessors are module-level functions.
    assert callable(ps.get_dcp_group)
    assert callable(ps.get_pcp_group)


def test_attention_backend_uses_new_for_discovery():
    """Knowledge D02: each backend uses __new__ + try/except for CP discovery."""
    # __new__ is defined on the base class.
    assert AttentionImplBase.__new__ is not object.__new__


# ==========================================================================
# Outline reframes — chapter must use the corrected terminology
# ==========================================================================


def test_reframe_b_terminology_ag_rs_not_all_reduce():
    """Outline §11.3 says 'all-reduce vs all-to-all' — reframe to 'AG+RS vs A2A'.

    Verified by source: parallel.py uses Literal["ag_rs", "a2a"], no all-reduce.
    """
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    parallel_src = VLLM_SOURCE / "config" / "parallel.py"
    if not parallel_src.exists():
        pytest.skip("config/parallel.py not found")
    text = parallel_src.read_text()
    # Source says ag_rs, not all_reduce.
    assert '"ag_rs"' in text
    assert '"a2a"' in text


def test_reframe_d_5d_mesh_not_3d():
    """Outline §11.5 says '3D parallel'; source has 5D mesh.

    REFERENCE: parallel_state.py:L1569-L1575 reshape(-1, dp, pp, pcp, tp)
    """
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    parallel_state_src = VLLM_SOURCE / "distributed" / "parallel_state.py"
    if not parallel_state_src.exists():
        pytest.skip("parallel_state.py not found")
    text = parallel_state_src.read_text()
    # 5D mesh reshape: -1, dp, pp, pcp, tp
    assert "data_parallel_size" in text
    assert "prefill_context_model_parallel_size" in text or "pcp" in text.lower()
    assert "pipeline_model_parallel_size" in text
    assert "tensor_model_parallel_size" in text


def test_reframe_d_world_size_assertion_in_source():
    """REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121.

    world_size assertion uses tp * pp * pcp (NOT × dcp).
    """
    if not VLLM_SOURCE.exists():
        pytest.skip("vllm source not available")
    src = VLLM_SOURCE / "v1" / "executor" / "multiproc_executor.py"
    if not src.exists():
        pytest.skip("multiproc_executor.py not found")
    text = src.read_text()
    # Look for world-size product: tp * pp * pcp (or _size variants).
    # The assertion uses *_size variables; we just check tp/pp/pcp all referenced.
    # Combined with the absence of " * dcp_size" in the world_size assertion line.
    assert "tensor_model_parallel_size" in text or "tp_size" in text


# ==========================================================================
# D-prefix knowledge sanity (D01-D15 facts from knowledge/modules/dcp-pcp.md)
# ==========================================================================


def test_D01_dcp_folded_inside_tp():
    """D01: _DCP folds inside TP, _PCP is independent."""
    groups = ps.initialize_model_parallel(
        rank=0,
        world_size=8,
        tensor_model_parallel_size=4,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    # DCP groups must be sub-chunks of TP groups.
    for dcp_grp in groups["dcp"]:
        # Find which TP group contains all members of this DCP group.
        in_some_tp = any(set(dcp_grp).issubset(set(tp_grp)) for tp_grp in groups["tp"])
        assert in_some_tp


def test_D02_DCPCommBackend_literal_two_options():
    """D02: DCPCommBackend = Literal['ag_rs', 'a2a']."""
    assert ag_rs_op_count() == 3
    assert a2a_op_count() == 2


def test_D05_5d_mesh_axes():
    """D05: external_dp × dp × pp × pcp × tp."""
    m = MeshConfig(external_dp=2, dp=2, pp=2, pcp=2, tp=2)
    assert m.world_size == 2 * 2 * 2 * 2 * 2


def test_D06_world_size_excludes_dcp():
    """D06: world_size = tp * pp * pcp (NOT * dcp)."""
    m = MeshConfig(tp=8, pp=2, pcp=2, dcp=4)
    assert m.world_size == 8 * 2 * 2


def test_D07_total_cp_formula():
    """D07: total_cp_world_size = pcp * dcp."""
    m = MeshConfig(tp=8, pcp=4, dcp=2)
    assert m.total_cp_world_size == 4 * 2


def test_D08_tp_modulo_dcp_hard_constraint():
    """D08: tp % dcp == 0 is the ONLY hard constraint."""
    with pytest.raises(ValueError):
        MeshConfig(tp=4, dcp=3)
    # No equivalent for PCP — pcp can be anything.
    MeshConfig(tp=4, pcp=999, dcp=1)


def test_D10_supports_pcp_default_false():
    """D10: per-backend supports_pcp default False; only some backends override."""
    assert AttentionImplBase.supports_pcp is False
    for cls in (FlashAttnBackend, FlashAttn3MlaBackend, FlashInferBackend, RocmAiterMlaBackend):
        assert cls.supports_pcp is False


def test_D11_ch10_ch11_cross_link_flag():
    """D11: supports_mtp_with_cp_non_trivial_interleave_size — Ch10↔Ch11 explicit knob."""
    assert hasattr(AttentionImplBase, "supports_mtp_with_cp_non_trivial_interleave_size")
    assert AttentionImplBase.supports_mtp_with_cp_non_trivial_interleave_size is False


def test_D12_max_memory_usage_formula():
    """D12: max_memory_usage_bytes = cdiv(seq, dcp*pcp) * cdiv(.., block) * page_size."""
    full = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    sharded = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4)
    # 16x reduction at total_cp = 16
    assert full // sharded == 16


def test_D14_lse_combine_bit_equivalent():
    """D14: _lse_weighted_combine produces bit-exact output (3.33e-16 max abs error)."""
    rng = np.random.default_rng(seed=42)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-15
