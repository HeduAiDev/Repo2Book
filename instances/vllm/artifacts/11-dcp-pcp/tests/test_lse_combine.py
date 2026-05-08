"""Tests for lse_combine.py — LSE-weighted combine math (FlashAttention online softmax across ranks).

Source: vllm/v1/attention/ops/dcp_alltoall.py:L39-L103
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from implementation.lse_combine import (
    CombineResult,
    lse_weighted_combine,
    reference_attention,
    split_attention,
)


# --------------------------------------------------------------------------
# Bit-exact equivalence (fp64 max abs error < 1e-12)
# --------------------------------------------------------------------------


def _build_qkv(B=4, H=2, D=8, L=16, seed=42):
    rng = np.random.default_rng(seed=seed)
    q = rng.standard_normal((B, H, D)).astype(np.float64)
    k = rng.standard_normal((L, H, D)).astype(np.float64)
    v = rng.standard_normal((L, H, D)).astype(np.float64)
    return q, k, v


def test_combine_matches_single_process_attention_4_ranks():
    """Demo §2 verbatim: 4 ranks, B=4, H=2, D=8, L=16 → max abs error 3.33e-16."""
    q, k, v = _build_qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-12


def test_combine_demo_max_abs_error_under_1e_15():
    """Demo §2 reports 3.33e-16; we assert < 1e-15 with the same seed."""
    q, k, v = _build_qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-15


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8, 16])
def test_combine_equivalence_across_rank_counts(num_ranks):
    """For any num_ranks dividing L, combined output == single-process attention."""
    q, k, v = _build_qkv(L=32)
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=num_ranks)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-12


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100, 12345])
def test_combine_equivalence_across_seeds(seed):
    """Identity holds for any random Q, K, V."""
    q, k, v = _build_qkv(seed=seed)
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.max(np.abs(out - o_truth)) < 1e-12


# --------------------------------------------------------------------------
# Associativity (online softmax is associative + commutative)
# REFERENCE: dcp_alltoall.py:L39-L103
# --------------------------------------------------------------------------


def test_associativity_split_then_combine_pairs():
    """(rank01) + (rank23) combined pairwise == flat 4-way combine."""
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)

    # Flat combine across 4 ranks.
    out_flat = lse_weighted_combine(parts_o, parts_lse).output

    # Pairwise: combine ranks (0,1), (2,3), then combine the two results.
    pair01 = lse_weighted_combine(parts_o[:2], parts_lse[:2])
    pair23 = lse_weighted_combine(parts_o[2:], parts_lse[2:])
    pair_o = np.stack([pair01.output, pair23.output], axis=0)
    pair_lse = np.stack([pair01.global_lse, pair23.global_lse], axis=0)
    out_assoc = lse_weighted_combine(pair_o, pair_lse).output

    assert np.max(np.abs(out_flat - out_assoc)) < 1e-12


def test_associativity_demo_value_under_1e_15():
    """Demo §2 reports 2.22e-16 associativity error."""
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    out_flat = lse_weighted_combine(parts_o, parts_lse).output
    pair01 = lse_weighted_combine(parts_o[:2], parts_lse[:2])
    pair23 = lse_weighted_combine(parts_o[2:], parts_lse[2:])
    out_assoc = lse_weighted_combine(
        np.stack([pair01.output, pair23.output], axis=0),
        np.stack([pair01.global_lse, pair23.global_lse], axis=0),
    ).output
    assert np.max(np.abs(out_flat - out_assoc)) < 1e-15


def test_commutativity_permuted_ranks():
    """Permuting the rank order produces the same combined output."""
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)

    out_in_order = lse_weighted_combine(parts_o, parts_lse).output

    perm = [3, 1, 0, 2]
    out_permuted = lse_weighted_combine(parts_o[perm], parts_lse[perm]).output
    assert np.max(np.abs(out_in_order - out_permuted)) < 1e-12


# --------------------------------------------------------------------------
# Naive sum is WRONG — Trap C anchor (need LSE re-weighting, not naive average)
# --------------------------------------------------------------------------


def test_naive_unweighted_sum_is_wrong():
    """Trap C/Trap F anchor: naive (1/N)*sum_i(O_i) is NOT the correct combine.

    The combine MUST be LSE-weighted. Without weights, the combined output
    differs from single-process attention by a non-trivial margin.
    """
    q, k, v = _build_qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, _ = split_attention(q, k, v, num_ranks=4)

    naive_sum = parts_o.sum(axis=0) / parts_o.shape[0]
    err_naive = np.max(np.abs(naive_sum - o_truth))
    assert err_naive > 1e-3, (
        f"Naive sum should differ from truth; got err={err_naive:.2e}"
    )


def test_lse_combine_strictly_better_than_naive():
    """LSE-weighted combine must beat naive average by orders of magnitude."""
    q, k, v = _build_qkv()
    o_truth, _ = reference_attention(q, k, v)
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)

    naive = parts_o.sum(axis=0) / parts_o.shape[0]
    lse = lse_weighted_combine(parts_o, parts_lse).output

    err_naive = np.max(np.abs(naive - o_truth))
    err_lse = np.max(np.abs(lse - o_truth))

    # LSE should be at least 10 orders of magnitude better.
    assert err_lse < err_naive / 1e10


# --------------------------------------------------------------------------
# Numerical stability (lse_max subtraction prevents overflow)
# --------------------------------------------------------------------------


def test_combine_with_large_lses_does_not_overflow():
    """LSE values can be very negative; subtracting lse_max keeps weights in [0, 1]."""
    parts_o = np.zeros((2, 1, 1, 1), dtype=np.float64)
    parts_o[0, 0, 0, 0] = 1.0
    parts_o[1, 0, 0, 0] = 2.0
    parts_lse = np.array([[[1000.0]], [[1001.0]]])
    out = lse_weighted_combine(parts_o, parts_lse).output
    # Should not be NaN/inf.
    assert np.isfinite(out).all()


def test_combine_with_negative_lses_does_not_underflow():
    parts_o = np.ones((2, 1, 1, 1), dtype=np.float64)
    parts_lse = np.array([[[-1000.0]], [[-1001.0]]])
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.isfinite(out).all()


def test_combine_handles_inf_lse_as_negative_inf():
    """Source line 66-70 sanitizes NaN/+inf to -inf."""
    parts_o = np.array([[[[1.0]]], [[[5.0]]]], dtype=np.float64)
    # rank 0 has finite lse, rank 1 has +inf (sanitized to -inf → weight 0)
    parts_lse = np.array([[[1.0]], [[float("inf")]]])
    out = lse_weighted_combine(parts_o, parts_lse).output
    # Combined should be approximately rank 0's output.
    assert np.allclose(out, [[[1.0]]])


def test_combine_handles_nan_lse_as_negative_inf():
    parts_o = np.array([[[[1.0]]], [[[5.0]]]], dtype=np.float64)
    parts_lse = np.array([[[1.0]], [[float("nan")]]])
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.allclose(out, [[[1.0]]])


def test_combine_all_minus_inf_lse_safe():
    """All ranks contribute zero attention; lse_max gets clipped to 0."""
    parts_o = np.zeros((2, 1, 1, 1), dtype=np.float64)
    parts_lse = np.full((2, 1, 1), -math.inf)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------
# Single-rank reduction is identity
# --------------------------------------------------------------------------


def test_combine_single_rank_returns_input():
    """N=1: the combine must return the partial output unchanged."""
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=1)
    out = lse_weighted_combine(parts_o, parts_lse).output
    assert np.allclose(out, parts_o[0])


# --------------------------------------------------------------------------
# Shape contract
# --------------------------------------------------------------------------


def test_combine_returns_combine_result_namedtuple():
    parts_o = np.zeros((2, 4, 2, 8), dtype=np.float64)
    parts_lse = np.zeros((2, 4, 2), dtype=np.float64)
    result = lse_weighted_combine(parts_o, parts_lse)
    assert isinstance(result, CombineResult)
    assert result.output.shape == (4, 2, 8)
    assert result.global_lse.shape == (4, 2)


def test_combine_output_shape_independent_of_num_ranks():
    for n in (1, 2, 4, 8):
        parts_o = np.zeros((n, 4, 2, 8), dtype=np.float64)
        parts_lse = np.zeros((n, 4, 2), dtype=np.float64)
        out = lse_weighted_combine(parts_o, parts_lse).output
        assert out.shape == (4, 2, 8)


def test_combine_lse_shape_assertion():
    parts_o = np.zeros((2, 4, 2, 8), dtype=np.float64)
    parts_lse_wrong = np.zeros((2, 4, 3), dtype=np.float64)
    with pytest.raises(AssertionError):
        lse_weighted_combine(parts_o, parts_lse_wrong)


# --------------------------------------------------------------------------
# is_lse_base_on_e=False — base 2 LSE
# --------------------------------------------------------------------------


def test_combine_base2_does_not_match_basee_in_general():
    """Base-2 and base-e LSEs disagree in general."""
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)

    out_base_e = lse_weighted_combine(parts_o, parts_lse, is_lse_base_on_e=True).output
    # Base-2 with the same numeric LSE input would weight differently.
    out_base_2 = lse_weighted_combine(parts_o, parts_lse, is_lse_base_on_e=False).output

    # They are NOT bit-equivalent (different exp base for the same LSE).
    assert not np.allclose(out_base_e, out_base_2)


def test_combine_global_lse_returned_when_requested():
    parts_o = np.zeros((2, 4, 2, 8), dtype=np.float64)
    parts_lse = np.zeros((2, 4, 2), dtype=np.float64)
    result = lse_weighted_combine(parts_o, parts_lse, return_lse=True)
    assert result.global_lse is not None
    assert result.global_lse.shape == (4, 2)


# --------------------------------------------------------------------------
# Demo §2 verbatim numerics — per-rank LSE values for (token=0, head=0)
# --------------------------------------------------------------------------


def test_demo_per_rank_lse_values():
    """Demo §2: rank LSEs are 1.448093 / 0.996192 / 2.106473 / 1.629767."""
    q, k, v = _build_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    expected = [1.448093, 0.996192, 2.106473, 1.629767]
    for i, ev in enumerate(expected):
        assert parts_lse[i, 0, 0] == pytest.approx(ev, abs=1e-6)


def test_demo_per_rank_lse_max_value():
    """Demo §2: lse_max(token=0, head=0) = 2.106473."""
    q, k, v = _build_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lse_max = parts_lse[:, 0, 0].max()
    assert lse_max == pytest.approx(2.106473, abs=1e-6)


def test_demo_per_rank_normalized_weights():
    """Demo §2: normalized weights at (token=0, head=0)."""
    q, k, v = _build_qkv()
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse[:, 0, 0]
    weights = np.exp(lses - lses.max())
    weights /= weights.sum()
    expected = [0.209762, 0.133496, 0.405190, 0.251552]
    for i, w in enumerate(expected):
        assert weights[i] == pytest.approx(w, abs=1e-6)


def test_weights_sum_to_one_after_normalization():
    """Per the math derivation: normalized weights sum to 1 along rank axis."""
    q, k, v = _build_qkv(B=8, H=4)
    _, parts_lse = split_attention(q, k, v, num_ranks=4)
    lses = parts_lse - parts_lse.max(axis=0, keepdims=True)
    weights = np.exp(lses)
    weights /= weights.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)


# --------------------------------------------------------------------------
# reference_attention sanity (used as ground truth for combine tests)
# --------------------------------------------------------------------------


def test_reference_attention_output_shape():
    q, k, v = _build_qkv()
    o, lse = reference_attention(q, k, v)
    assert o.shape == q.shape
    assert lse.shape == q.shape[:2]


def test_reference_attention_softmax_normalization():
    """Output is a convex combination of V rows; all components in [min v, max v]."""
    q, k, v = _build_qkv()
    o, _ = reference_attention(q, k, v)
    assert (o.min() >= v.min() - 1e-9) and (o.max() <= v.max() + 1e-9)


# --------------------------------------------------------------------------
# split_attention contract
# --------------------------------------------------------------------------


def test_split_attention_requires_divisibility():
    q, k, v = _build_qkv(L=15)
    with pytest.raises(AssertionError):
        split_attention(q, k, v, num_ranks=4)


def test_split_attention_partial_shapes():
    q, k, v = _build_qkv()
    parts_o, parts_lse = split_attention(q, k, v, num_ranks=4)
    assert parts_o.shape == (4,) + q.shape  # (N, B, H, D)
    assert parts_lse.shape == (4,) + q.shape[:2]  # (N, B, H)
