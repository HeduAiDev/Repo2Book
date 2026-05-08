"""End-to-end integration: 5D mesh + LSE combine + per-rank KV + backend discovery.

This file exercises *cross-module* invariants the writer must trust as load-bearing.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation import parallel_state_dcp_pcp as ps
from implementation.attention_backend_dcp_pcp import (
    AttentionImplBase,
    FlashAttn3MlaBackend,
)
from implementation.dcp_alltoall import (
    a2a_op_count,
    ag_rs_op_count,
    simulate_a2a_combine,
    simulate_ag_rs_combine,
)
from implementation.dcp_vs_pcp_demo import per_rank_kv_chunk
from implementation.kv_cache_per_rank import LLAMA_70B_KV_SPEC, hbm_per_rank
from implementation.lse_combine import (
    lse_weighted_combine,
    reference_attention,
    split_attention,
)
from implementation.seq_sharding import (
    causal_attention_work_per_rank,
    get_dcp_local_seq_lens,
    imbalance_ratio,
)
from implementation.world_topology import MeshConfig, per_rank_kv_fraction


@pytest.fixture(autouse=True)
def _reset():
    ps.reset_cp_singletons()
    yield
    ps.reset_cp_singletons()


# --------------------------------------------------------------------------
# Cross-module invariant: backend.total_cp_world_size matches mesh.total_cp_world_size
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tp,pcp,dcp", [
    (4, 2, 2),
    (8, 1, 4),
    (8, 4, 2),
    (4, 4, 1),
    (2, 1, 1),
])
def test_backend_total_cp_matches_mesh_total_cp(tp, pcp, dcp):
    mesh = MeshConfig(tp=tp, pcp=pcp, dcp=dcp)
    ps.initialize_model_parallel(
        rank=0,
        world_size=mesh.world_size,
        tensor_model_parallel_size=tp,
        prefill_context_model_parallel_size=pcp,
        decode_context_model_parallel_size=dcp,
    )
    backend = AttentionImplBase.__new__(AttentionImplBase)
    assert backend.total_cp_world_size == mesh.total_cp_world_size


def test_backend_dcp_world_size_matches_mesh_dcp():
    mesh = MeshConfig(tp=8, pcp=2, dcp=4)
    ps.initialize_model_parallel(
        rank=0,
        world_size=mesh.world_size,
        tensor_model_parallel_size=mesh.tp,
        prefill_context_model_parallel_size=mesh.pcp,
        decode_context_model_parallel_size=mesh.dcp,
    )
    backend = AttentionImplBase.__new__(AttentionImplBase)
    assert backend.dcp_world_size == mesh.dcp


def test_backend_pcp_world_size_matches_mesh_pcp():
    mesh = MeshConfig(tp=8, pcp=2, dcp=4)
    ps.initialize_model_parallel(
        rank=0,
        world_size=mesh.world_size,
        tensor_model_parallel_size=mesh.tp,
        prefill_context_model_parallel_size=mesh.pcp,
        decode_context_model_parallel_size=mesh.dcp,
    )
    backend = AttentionImplBase.__new__(AttentionImplBase)
    assert backend.pcp_world_size == mesh.pcp


# --------------------------------------------------------------------------
# Cross-module: mesh.total_cp_world_size matches HBM scaling
# --------------------------------------------------------------------------


def test_hbm_scaling_matches_per_rank_kv_fraction():
    """HBM at (dcp, pcp) = no_cp * (1/total_cp) modulo cdiv rounding."""
    mesh = MeshConfig(tp=4, pcp=2, dcp=2)
    full = hbm_per_rank(64 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1)
    sharded = hbm_per_rank(64 * 1024, LLAMA_70B_KV_SPEC, dcp=mesh.dcp, pcp=mesh.pcp)
    assert sharded == int(full * per_rank_kv_fraction(mesh))


def test_hbm_scaling_independent_of_axis_assignment():
    """(dcp=4, pcp=2) and (dcp=2, pcp=4) produce same per-rank HBM."""
    a = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=2)
    b = hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=4)
    assert a == b


# --------------------------------------------------------------------------
# End-to-end attention: split → run per-rank → combine == single-process
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tp,pcp,dcp,L", [
    (4, 1, 4, 16),
    (4, 2, 2, 16),
    (8, 4, 2, 32),
    (4, 4, 1, 16),
    (2, 1, 1, 4),
])
def test_e2e_attention_at_various_meshes(tp, pcp, dcp, L):
    """For any valid mesh, split-and-combine == single-process attention."""
    rng = np.random.default_rng(seed=tp * pcp * dcp)
    B, H, D = 4, 2, 8
    q = rng.standard_normal((B, H, D)).astype(np.float64)
    k = rng.standard_normal((L, H, D)).astype(np.float64)
    v = rng.standard_normal((L, H, D)).astype(np.float64)

    mesh = MeshConfig(tp=tp, pcp=pcp, dcp=dcp)
    total_cp = mesh.total_cp_world_size
    if L % total_cp != 0:
        pytest.skip("L must be divisible by total_cp")

    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=total_cp)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-11


# --------------------------------------------------------------------------
# AG+RS and A2A backends must produce IDENTICAL output (Trap F anchor)
# --------------------------------------------------------------------------


def test_ag_rs_and_a2a_produce_identical_output_at_dcp_2():
    rng = np.random.default_rng(seed=1)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=2)
    out_a = simulate_a2a_combine(parts_o, parts_lse)
    out_b = simulate_ag_rs_combine(parts_o, parts_lse)
    assert np.array_equal(out_a, out_b)


def test_ag_rs_and_a2a_produce_identical_output_at_dcp_4():
    rng = np.random.default_rng(seed=2)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out_a = simulate_a2a_combine(parts_o, parts_lse)
    out_b = simulate_ag_rs_combine(parts_o, parts_lse)
    assert np.array_equal(out_a, out_b)


# --------------------------------------------------------------------------
# Demo §1 + §5 wiring: world=16, mesh=(tp=4, pcp=2, pp=2, dcp=2), total_cp=4
# --------------------------------------------------------------------------


def test_demo_5_world_size_total_cp():
    mesh = MeshConfig(external_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2)
    assert mesh.world_size == 16
    assert mesh.total_cp_world_size == 4


def test_demo_5_per_rank_kv_chunk_at_seq_64k():
    """At (dcp=2, pcp=2), per-rank chunk = seq/4."""
    mesh = MeshConfig(external_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2)
    assert per_rank_kv_chunk(64 * 1024, mesh) == 16 * 1024


# --------------------------------------------------------------------------
# Mesh group construction matches singleton population
# --------------------------------------------------------------------------


def test_mesh_groups_match_singleton_dcp():
    mesh = MeshConfig(tp=4, dcp=2)
    groups = ps.initialize_model_parallel(
        rank=2,
        world_size=mesh.world_size,
        tensor_model_parallel_size=mesh.tp,
        decode_context_model_parallel_size=mesh.dcp,
    )
    # Find the DCP group containing rank 2.
    matches = [g for g in groups["dcp"] if 2 in g]
    assert len(matches) == 1
    assert ps.get_dcp_group().ranks == matches[0]


def test_mesh_groups_match_singleton_pcp():
    mesh = MeshConfig(tp=2, pcp=2)
    groups = ps.initialize_model_parallel(
        rank=3,
        world_size=mesh.world_size,
        tensor_model_parallel_size=mesh.tp,
        prefill_context_model_parallel_size=mesh.pcp,
    )
    matches = [g for g in groups["pcp"] if 3 in g]
    assert len(matches) == 1
    assert ps.get_pcp_group().ranks == matches[0]


# --------------------------------------------------------------------------
# Ch10 ↔ Ch11 cross-link: supports_mtp_with_cp_non_trivial_interleave_size knob
# REFERENCE: vllm/v1/attention/backend.py:L705-L706
# --------------------------------------------------------------------------


def test_ch10_cross_link_flag_present_on_attention_base():
    """Ch10's MTP-with-CP knob is wired on AttentionImplBase."""
    assert hasattr(AttentionImplBase, "supports_mtp_with_cp_non_trivial_interleave_size")


def test_ch10_cross_link_flag_default_false():
    """Default is False — backends must opt in."""
    assert AttentionImplBase.supports_mtp_with_cp_non_trivial_interleave_size is False


# --------------------------------------------------------------------------
# Striped-vs-contiguous load balance + LSE combine work together
# (Demo §4 + Demo §2 cross-validation)
# --------------------------------------------------------------------------


def test_load_balance_drops_with_smaller_interleave():
    """Demo §4: decreasing I lowers imbalance (13.44x → 1.55x → 1.24x)."""
    cp = 8
    seq = 64
    contig = imbalance_ratio(causal_attention_work_per_rank(seq, cp, 8))
    block = imbalance_ratio(causal_attention_work_per_rank(seq, cp, 2))
    striped = imbalance_ratio(causal_attention_work_per_rank(seq, cp, 1))
    assert contig > block > striped


def test_get_dcp_local_seq_lens_matches_demo_invariant():
    """Per-rank lens always sum to global seq_len."""
    seq_lens = np.array([100, 64, 30, 17])
    for I in (1, 4, 16):
        local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=I)
        np.testing.assert_array_equal(local.sum(axis=-1), seq_lens)


# --------------------------------------------------------------------------
# Configuration sanity: world_size formula NEVER includes dcp
# REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tp,pcp,dcp", [
    (8, 1, 1),
    (8, 1, 2),
    (8, 1, 4),
    (8, 1, 8),
    (8, 2, 1),
    (8, 2, 2),
    (8, 2, 4),
    (8, 2, 8),
    (8, 4, 1),
    (8, 4, 2),
    (8, 4, 4),
])
def test_world_size_excludes_dcp_grid(tp, pcp, dcp):
    """For any valid (tp, pcp, dcp), world_size = tp * pcp (no dcp)."""
    m = MeshConfig(tp=tp, pcp=pcp, dcp=dcp)
    assert m.world_size == tp * pcp


# --------------------------------------------------------------------------
# Closing seal: NCCL op counts are constants (sanity for Demo §3 anchor)
# --------------------------------------------------------------------------


def test_op_count_constants_match_arxiv_2507_07120():
    """3 vs 2 NCCL ops — arxiv.org/abs/2507.07120 headline."""
    assert ag_rs_op_count() == 3
    assert a2a_op_count() == 2
