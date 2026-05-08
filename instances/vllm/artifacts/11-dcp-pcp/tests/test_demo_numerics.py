"""Pin every headline number from demo-output.txt verbatim.

Writer must quote these character-for-character. Each cell is a separate test.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation import parallel_state_dcp_pcp as ps
from implementation.dcp_alltoall import (
    a2a_op_count,
    a2a_payload_bytes,
    ag_rs_op_count,
    ag_rs_payload_bytes,
    alpha_beta_cost,
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
from implementation.seq_sharding import (
    causal_attention_work_per_rank,
    get_dcp_local_seq_lens,
    imbalance_ratio,
)
from implementation.world_topology import MeshConfig


@pytest.fixture(autouse=True)
def _reset():
    ps.reset_cp_singletons()
    yield
    ps.reset_cp_singletons()


# ==========================================================================
# §1 — HBM per-rank capacity sweep at Llama-70B 128K
# ==========================================================================


def test_section_1_naive_total_bytes_42_949_672_960():
    """Demo §1: naive total = 42,949,672,960 bytes."""
    assert hbm_naive_total(128 * 1024, LLAMA_70B_KV_SPEC) == 42_949_672_960


def test_section_1_naive_total_40_0_GB():
    """Demo §1: 40.0 GB headline."""
    assert fmt_gb(hbm_naive_total(128 * 1024, LLAMA_70B_KV_SPEC)) == "40.0 GB"


def test_section_1_cell_1_1_per_rank_len_131_072():
    """Cell (1,1): per_rank_len = 131,072."""
    seq = 128 * 1024
    assert -(-seq // (1 * 1)) == 131_072


def test_section_1_cell_1_1_bytes_42_949_672_960():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=1) == 42_949_672_960


def test_section_1_cell_1_2_per_rank_len_65_536():
    seq = 128 * 1024
    assert -(-seq // (1 * 2)) == 65_536


def test_section_1_cell_1_2_bytes_21_474_836_480():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=2) == 21_474_836_480


def test_section_1_cell_1_2_GB_20_0():
    assert fmt_gb(hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=2)) == "20.0 GB"


def test_section_1_cell_2_1_bytes_21_474_836_480():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=1) == 21_474_836_480


def test_section_1_cell_2_2_bytes_10_737_418_240():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=2) == 10_737_418_240


def test_section_1_cell_2_2_GB_10_0():
    assert fmt_gb(hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=2)) == "10.0 GB"


def test_section_1_cell_1_4_per_rank_len_32_768():
    seq = 128 * 1024
    assert -(-seq // (1 * 4)) == 32_768


def test_section_1_cell_1_4_bytes_10_737_418_240():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=1, pcp=4) == 10_737_418_240


def test_section_1_cell_4_1_bytes_10_737_418_240():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=1) == 10_737_418_240


def test_section_1_cell_2_4_per_rank_len_16_384():
    seq = 128 * 1024
    assert -(-seq // (2 * 4)) == 16_384


def test_section_1_cell_2_4_bytes_5_368_709_120():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=4) == 5_368_709_120


def test_section_1_cell_2_4_GB_5_0():
    assert fmt_gb(hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=2, pcp=4)) == "5.0 GB"


def test_section_1_cell_4_4_per_rank_len_8_192():
    seq = 128 * 1024
    assert -(-seq // (4 * 4)) == 8_192


def test_section_1_cell_4_4_bytes_2_684_354_560():
    assert hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4) == 2_684_354_560


def test_section_1_cell_4_4_GB_2_5():
    assert fmt_gb(hbm_per_rank(128 * 1024, LLAMA_70B_KV_SPEC, dcp=4, pcp=4)) == "2.5 GB"


# ==========================================================================
# §2 — LSE combine equivalence
# ==========================================================================


def _section_2_qkv():
    """Replicate Demo §2 setup: B=4, H=2, D=8, L=16, seed=42."""
    rng = np.random.default_rng(seed=42)
    q = rng.standard_normal((4, 2, 8)).astype(np.float64)
    k = rng.standard_normal((16, 2, 8)).astype(np.float64)
    v = rng.standard_normal((16, 2, 8)).astype(np.float64)
    return q, k, v


def test_section_2_partial_outputs_shape_4_4_2_8():
    q, k, v = _section_2_qkv()
    parts_o, _ = split_attention(q, k, v, num_ranks=4)
    assert parts_o.shape == (4, 4, 2, 8)


def test_section_2_partial_lses_shape_4_4_2():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse.shape == (4, 4, 2)


def test_section_2_rank_0_lse_1_448093():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse[0, 0, 0] == pytest.approx(1.448093, abs=1e-6)


def test_section_2_rank_1_lse_0_996192():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse[1, 0, 0] == pytest.approx(0.996192, abs=1e-6)


def test_section_2_rank_2_lse_2_106473():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse[2, 0, 0] == pytest.approx(2.106473, abs=1e-6)


def test_section_2_rank_3_lse_1_629767():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse[3, 0, 0] == pytest.approx(1.629767, abs=1e-6)


def test_section_2_lse_max_2_106473():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_lse[:, 0, 0].max() == pytest.approx(2.106473, abs=1e-6)


def test_section_2_weight_rank_0_0_209762():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse[:, 0, 0]
    weights = np.exp(lses - lses.max())
    weights /= weights.sum()
    assert weights[0] == pytest.approx(0.209762, abs=1e-6)


def test_section_2_weight_rank_1_0_133496():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse[:, 0, 0]
    weights = np.exp(lses - lses.max())
    weights /= weights.sum()
    assert weights[1] == pytest.approx(0.133496, abs=1e-6)


def test_section_2_weight_rank_2_0_405190():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse[:, 0, 0]
    weights = np.exp(lses - lses.max())
    weights /= weights.sum()
    assert weights[2] == pytest.approx(0.405190, abs=1e-6)


def test_section_2_weight_rank_3_0_251552():
    q, k, v = _section_2_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse[:, 0, 0]
    weights = np.exp(lses - lses.max())
    weights /= weights.sum()
    assert weights[3] == pytest.approx(0.251552, abs=1e-6)


def test_section_2_max_abs_error_3_33e_minus_16():
    """Demo §2: max abs error ≤ 3.33e-16."""
    q, k, v = _section_2_qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = lse_weighted_combine(parts_o, parts_lse).output
    err = np.max(np.abs(out - o_truth))
    assert err < 1e-15


def test_section_2_associativity_error_2_22e_minus_16():
    """Demo §2: associativity error ≤ 2.22e-16."""
    q, k, v = _section_2_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    flat = lse_weighted_combine(parts_o, parts_lse).output
    pair01 = lse_weighted_combine(parts_o[:2], parts_lse[:2])
    pair23 = lse_weighted_combine(parts_o[2:], parts_lse[2:])
    assoc = lse_weighted_combine(
        np.stack([pair01.output, pair23.output], axis=0),
        np.stack([pair01.global_lse, pair23.global_lse], axis=0),
    ).output
    err = np.max(np.abs(assoc - flat))
    assert err < 1e-15


# ==========================================================================
# §3 — AG+RS vs A2A NCCL ops + alpha-beta
# ==========================================================================


def test_section_3_ag_rs_ops_3():
    assert ag_rs_op_count() == 3


def test_section_3_a2a_ops_2():
    assert a2a_op_count() == 2


def test_section_3_ag_rs_bytes_67_108_864():
    """Demo §3: AG+RS bytes = 67,108,864 (independent of dcp)."""
    assert ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2) == 67_108_864
    assert ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=4) == 67_108_864
    assert ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=8) == 67_108_864


def test_section_3_a2a_bytes_dcp_2_34_078_720():
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2) == 34_078_720


def test_section_3_a2a_bytes_dcp_4_17_039_360():
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4) == 17_039_360


def test_section_3_a2a_bytes_dcp_8_8_519_680():
    assert a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8) == 8_519_680


def test_section_3_T_AG_RS_dcp_2_1036_6():
    """Demo §3: T_AG+RS(dcp=2) = 1036.6 us."""
    bytes_ = ag_rs_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    t = alpha_beta_cost(bytes_, 10.0, 200.0, num_collectives=ag_rs_op_count())
    assert t == pytest.approx(1036.6, abs=0.5)


def test_section_3_T_A2A_dcp_2_360_8():
    bytes_ = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=2)
    t = alpha_beta_cost(bytes_, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(360.8, abs=0.5)


def test_section_3_T_A2A_dcp_4_190_4():
    bytes_ = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=4)
    t = alpha_beta_cost(bytes_, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(190.4, abs=0.5)


def test_section_3_T_A2A_dcp_8_105_2():
    bytes_ = a2a_payload_bytes(32 * 1024, 8, 128, dcp_size=8)
    t = alpha_beta_cost(bytes_, 10.0, 200.0, num_collectives=a2a_op_count())
    assert t == pytest.approx(105.2, abs=0.5)


def test_section_3_speedup_dcp_2_2_87x():
    ag = alpha_beta_cost(ag_rs_payload_bytes(32 * 1024, 8, 128, 2), 10.0, 200.0,
                         num_collectives=ag_rs_op_count())
    a2 = alpha_beta_cost(a2a_payload_bytes(32 * 1024, 8, 128, 2), 10.0, 200.0,
                         num_collectives=a2a_op_count())
    assert (ag / a2) == pytest.approx(2.87, abs=0.02)


def test_section_3_speedup_dcp_4_5_44x():
    ag = alpha_beta_cost(ag_rs_payload_bytes(32 * 1024, 8, 128, 4), 10.0, 200.0,
                         num_collectives=ag_rs_op_count())
    a2 = alpha_beta_cost(a2a_payload_bytes(32 * 1024, 8, 128, 4), 10.0, 200.0,
                         num_collectives=a2a_op_count())
    assert (ag / a2) == pytest.approx(5.44, abs=0.05)


def test_section_3_speedup_dcp_8_9_85x():
    ag = alpha_beta_cost(ag_rs_payload_bytes(32 * 1024, 8, 128, 8), 10.0, 200.0,
                         num_collectives=ag_rs_op_count())
    a2 = alpha_beta_cost(a2a_payload_bytes(32 * 1024, 8, 128, 8), 10.0, 200.0,
                         num_collectives=a2a_op_count())
    assert (ag / a2) == pytest.approx(9.85, abs=0.05)


def test_section_3_a2a_reduces_33_percent():
    pct = (1.0 - a2a_op_count() / ag_rs_op_count()) * 100
    assert round(pct) == 33


# ==========================================================================
# §4 — Striped vs contiguous load balance
# ==========================================================================


def test_section_4_contiguous_imbalance_13_44x():
    work = causal_attention_work_per_rank(64, 8, 8)
    assert imbalance_ratio(work) == pytest.approx(13.44, abs=0.01)


def test_section_4_block_striped_imbalance_1_55x():
    work = causal_attention_work_per_rank(64, 8, 2)
    assert imbalance_ratio(work) == pytest.approx(1.55, abs=0.01)


def test_section_4_striped_imbalance_1_24x():
    work = causal_attention_work_per_rank(64, 8, 1)
    assert imbalance_ratio(work) == pytest.approx(1.24, abs=0.01)


def test_section_4_contiguous_per_rank_work():
    """Demo §4 verbatim: [36, 100, 164, 228, 292, 356, 420, 484]."""
    work = causal_attention_work_per_rank(64, 8, 8)
    assert work == [36, 100, 164, 228, 292, 356, 420, 484]


def test_section_4_block_striped_per_rank_work():
    work = causal_attention_work_per_rank(64, 8, 2)
    assert work == [204, 220, 236, 252, 268, 284, 300, 316]


def test_section_4_striped_per_rank_work():
    work = causal_attention_work_per_rank(64, 8, 1)
    assert work == [232, 240, 248, 256, 264, 272, 280, 288]


def test_section_4_contiguous_rank_0_work_36():
    work = causal_attention_work_per_rank(64, 8, 8)
    assert work[0] == 36


def test_section_4_contiguous_rank_7_work_484():
    work = causal_attention_work_per_rank(64, 8, 8)
    assert work[7] == 484


def test_section_4_get_dcp_local_seq_lens_I_1_verbatim():
    """Demo §4 verbatim."""
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=1)
    expected = np.array([
        [25, 25, 25, 25],
        [16, 16, 16, 16],
        [8, 8, 7, 7],
        [5, 4, 4, 4],
    ])
    np.testing.assert_array_equal(local, expected)


def test_section_4_get_dcp_local_seq_lens_I_4_verbatim():
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=4)
    expected = np.array([
        [28, 24, 24, 24],
        [16, 16, 16, 16],
        [8, 8, 8, 6],
        [5, 4, 4, 4],
    ])
    np.testing.assert_array_equal(local, expected)


def test_section_4_get_dcp_local_seq_lens_I_16_verbatim():
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=16)
    expected = np.array([
        [32, 32, 20, 16],
        [16, 16, 16, 16],
        [16, 14, 0, 0],
        [16, 1, 0, 0],
    ])
    np.testing.assert_array_equal(local, expected)


# ==========================================================================
# §5 — 5D mesh group construction at world=16
# ==========================================================================


def test_section_5_world_size_16():
    m = MeshConfig(external_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2)
    assert m.world_size == 16


def test_section_5_total_cp_4():
    m = MeshConfig(tp=4, pcp=2, dcp=2)
    assert m.total_cp_world_size == 4


def test_section_5_num_dcp_subgroups_2():
    m = MeshConfig(tp=4, dcp=2)
    assert m.num_dcp_subgroups == 2


def test_section_5_tp_groups_count_4():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert len(groups["tp"]) == 4


def test_section_5_dcp_subgroups_count_8():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert len(groups["dcp"]) == 8


def test_section_5_pcp_groups_count_8():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert len(groups["pcp"]) == 8


def test_section_5_pp_groups_count_8():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert len(groups["pp"]) == 8


def test_section_5_first_tp_group_0_1_2_3():
    """Demo §5 verbatim: first TP group is [0, 1, 2, 3]."""
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert groups["tp"][0] == [0, 1, 2, 3]


def test_section_5_first_dcp_subgroup_0_1():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert groups["dcp"][0] == [0, 1]


def test_section_5_first_pcp_group_0_4():
    """Demo §5 verbatim: first PCP group is [0, 4]."""
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert groups["pcp"][0] == [0, 4]


def test_section_5_first_pp_group_0_8():
    """Demo §5 verbatim: first PP group is [0, 8]."""
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    assert groups["pp"][0] == [0, 8]


def test_section_5_full_dcp_subgroups_verbatim():
    """Demo §5 verbatim: all 8 DCP sub-groups."""
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    expected = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
    assert groups["dcp"] == expected


def test_section_5_full_pcp_groups_verbatim():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    expected = [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]
    assert groups["pcp"] == expected


def test_section_5_full_pp_groups_verbatim():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    expected = [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]
    assert groups["pp"] == expected


def test_section_5_full_tp_groups_verbatim():
    groups = ps.initialize_model_parallel(
        rank=0, world_size=16,
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    expected = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert groups["tp"] == expected
