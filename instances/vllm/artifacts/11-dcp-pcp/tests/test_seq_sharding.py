"""Tests for seq_sharding.py — get_dcp_local_seq_lens + causal load balance.

Source: vllm/v1/attention/backends/utils.py:L820-L857 + parallel.py:L330-L342
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.seq_sharding import (
    causal_attention_work_per_rank,
    get_dcp_local_seq_lens,
    imbalance_ratio,
)


# --------------------------------------------------------------------------
# Mass conservation: sum across ranks == global seq_len (Demo §4 invariant)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_sum_across_ranks_equals_global_seq_lens(interleave):
    """Demo §4 verbatim: sum-across-ranks must equal seq_lens regardless of I."""
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=interleave)
    np.testing.assert_array_equal(local.sum(axis=-1), seq_lens)


@pytest.mark.parametrize("dcp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 2, 8, 16])
def test_mass_conservation_grid(dcp_size, interleave):
    seq_lens = np.array([100, 64, 30, 17, 256, 1024])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=dcp_size, cp_kv_cache_interleave_size=interleave)
    np.testing.assert_array_equal(local.sum(axis=-1), seq_lens)


def test_mass_conservation_random_seq_lens():
    rng = np.random.default_rng(seed=42)
    seq_lens = rng.integers(1, 4096, size=64)
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=8, cp_kv_cache_interleave_size=4)
    np.testing.assert_array_equal(local.sum(axis=-1), seq_lens)


# --------------------------------------------------------------------------
# Demo §4 verbatim per-rank lens at interleave=1, 4, 16
# --------------------------------------------------------------------------


def test_demo_interleave_1_per_rank_lens():
    """Demo §4 verbatim: I=1, dcp=4, seq_lens=[100,64,30,17]."""
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=1)
    expected = np.array([
        [25, 25, 25, 25],
        [16, 16, 16, 16],
        [8, 8, 7, 7],
        [5, 4, 4, 4],
    ])
    np.testing.assert_array_equal(local, expected)


def test_demo_interleave_4_per_rank_lens():
    """Demo §4 verbatim: I=4, dcp=4."""
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=4)
    expected = np.array([
        [28, 24, 24, 24],
        [16, 16, 16, 16],
        [8, 8, 8, 6],
        [5, 4, 4, 4],
    ])
    np.testing.assert_array_equal(local, expected)


def test_demo_interleave_16_per_rank_lens():
    """Demo §4 verbatim: I=16, dcp=4."""
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=16)
    expected = np.array([
        [32, 32, 20, 16],
        [16, 16, 16, 16],
        [16, 14, 0, 0],
        [16, 1, 0, 0],
    ])
    np.testing.assert_array_equal(local, expected)


# --------------------------------------------------------------------------
# dcp_rank=None returns full [num_requests, dcp_size]
# --------------------------------------------------------------------------


def test_dcp_rank_none_returns_full_2d():
    seq_lens = np.array([100, 64])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=1)
    assert local.shape == (2, 4)


def test_dcp_rank_specific_returns_1d():
    seq_lens = np.array([100, 64])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, dcp_rank=2, cp_kv_cache_interleave_size=1)
    assert local.shape == (2,)


def test_dcp_rank_specific_matches_full():
    """The dcp_rank=r slice must equal the full[..., r] result."""
    seq_lens = np.array([100, 64, 30, 17])
    full = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=1)
    for r in range(4):
        per_rank = get_dcp_local_seq_lens(
            seq_lens, dcp_size=4, dcp_rank=r, cp_kv_cache_interleave_size=1
        )
        np.testing.assert_array_equal(per_rank, full[:, r])


# --------------------------------------------------------------------------
# Edge cases: dcp=1 trivially returns seq_lens
# --------------------------------------------------------------------------


def test_dcp_1_returns_seq_lens_unchanged():
    seq_lens = np.array([100, 64, 30, 17])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=1, cp_kv_cache_interleave_size=1)
    np.testing.assert_array_equal(local.squeeze(-1), seq_lens)


def test_dcp_1_at_any_interleave():
    """dcp=1 means every token belongs to that single rank."""
    seq_lens = np.array([100, 64, 30, 17])
    for I in (1, 4, 16, 64):
        local = get_dcp_local_seq_lens(seq_lens, dcp_size=1, cp_kv_cache_interleave_size=I)
        np.testing.assert_array_equal(local.squeeze(-1), seq_lens)


# --------------------------------------------------------------------------
# Causal-mask load imbalance: striped (I=1) << contiguous (I=seq/cp)
# Demo §4: contiguous=13.44x, striped=1.24x at cp=8, seq=64
# --------------------------------------------------------------------------


def test_demo_contiguous_imbalance_13_44x():
    """Demo §4: contiguous (I = seq_len/cp_size = 8) → imbalance 13.44x."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=8)
    assert imbalance_ratio(work) == pytest.approx(13.44, abs=0.01)


def test_demo_block_striped_imbalance_1_55x():
    """Demo §4: block-striped (I=2) → imbalance 1.55x."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=2)
    assert imbalance_ratio(work) == pytest.approx(1.55, abs=0.01)


def test_demo_striped_imbalance_1_24x():
    """Demo §4: striped (I=1) → imbalance 1.24x (perfectly balanced)."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=1)
    assert imbalance_ratio(work) == pytest.approx(1.24, abs=0.01)


def test_demo_contiguous_extreme_values():
    """Demo §4: contiguous → rank-0 work=36, rank-7 work=484."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=8)
    assert work == [36, 100, 164, 228, 292, 356, 420, 484]


def test_demo_block_striped_pattern():
    """Demo §4: I=2 → [204, 220, 236, 252, 268, 284, 300, 316]."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=2)
    assert work == [204, 220, 236, 252, 268, 284, 300, 316]


def test_demo_striped_pattern():
    """Demo §4: I=1 → [232, 240, 248, 256, 264, 272, 280, 288]."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=8, cp_kv_cache_interleave_size=1)
    assert work == [232, 240, 248, 256, 264, 272, 280, 288]


def test_striped_strictly_more_balanced_than_contiguous():
    """Striped (I=1) imbalance < contiguous (I=seq/cp) at any seq, cp."""
    for seq_len, cp_size in [(64, 8), (128, 4), (32, 4)]:
        contig = imbalance_ratio(causal_attention_work_per_rank(
            seq_len, cp_size, seq_len // cp_size))
        striped = imbalance_ratio(causal_attention_work_per_rank(seq_len, cp_size, 1))
        assert striped < contig


def test_imbalance_monotone_in_interleave():
    """Larger interleave → larger imbalance under causal mask."""
    cp = 8
    seq = 256
    ratios = []
    for I in (1, 2, 4, 8, 16, 32):
        work = causal_attention_work_per_rank(seq, cp, I)
        ratios.append(imbalance_ratio(work))
    # Non-decreasing.
    assert all(ratios[i] <= ratios[i + 1] for i in range(len(ratios) - 1))


# --------------------------------------------------------------------------
# Total work invariant: sum equals seq_len*(seq_len+1)/2 regardless of partition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seq_len,cp_size,I", [
    (64, 8, 1),
    (64, 8, 2),
    (64, 8, 4),
    (64, 8, 8),
    (256, 4, 1),
    (256, 4, 4),
    (256, 4, 16),
])
def test_total_work_equals_triangular_sum(seq_len, cp_size, I):
    """Total causal work = sum_{i=0}^{seq-1} (i+1) = seq*(seq+1)/2."""
    work = causal_attention_work_per_rank(seq_len, cp_size, I)
    expected = seq_len * (seq_len + 1) // 2
    assert sum(work) == expected


# --------------------------------------------------------------------------
# imbalance_ratio edge cases
# --------------------------------------------------------------------------


def test_imbalance_ratio_perfect_balance_is_1():
    assert imbalance_ratio([10, 10, 10, 10]) == 1.0


def test_imbalance_ratio_zero_min_is_inf():
    """If a rank has zero work, ratio is infinite."""
    assert imbalance_ratio([0, 100]) == float("inf")


def test_imbalance_ratio_large_skew():
    """Huge skew: 1 vs 1000 → 1000x."""
    assert imbalance_ratio([1, 1000]) == 1000.0


# --------------------------------------------------------------------------
# Source-formula bookkeeping: base + remainder formula correctness
# --------------------------------------------------------------------------


def test_formula_base_plus_remainder_at_specific_seq():
    """Exact source formula at seq=100, dcp=4, I=1.

    base = 100 // 1 // 4 * 1 = 25; remainder = 100 - 25*4 = 0.
    Each rank gets 25 + 0 = 25.
    """
    seq_lens = np.array([100])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=1)
    np.testing.assert_array_equal(local, np.array([[25, 25, 25, 25]]))


def test_formula_with_uneven_seq_at_interleave_4():
    """seq=30, dcp=4, I=4: base = 30 // 4 // 4 * 4 = 4; remainder = 30 - 4*4 = 14.

    Distribute remainder of 14 in I-sized chunks: rank 0 gets 4 (from 14,0,...), rank 1 gets 4,
    rank 2 gets 4, rank 3 gets clip(14-12, 0, 4)=2. Plus base 4 each → [8,8,8,6].
    """
    seq_lens = np.array([30])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=4, cp_kv_cache_interleave_size=4)
    np.testing.assert_array_equal(local, np.array([[8, 8, 8, 6]]))


# --------------------------------------------------------------------------
# Parametrized cp_size + seq_lens exercise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [10, 100, 1024])
def test_per_rank_lens_nonnegative(cp_size, seq_len):
    seq_lens = np.array([seq_len])
    local = get_dcp_local_seq_lens(seq_lens, dcp_size=cp_size, cp_kv_cache_interleave_size=1)
    assert (local >= 0).all()


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_token_owner_round_robin_at_interleave_1(cp_size):
    """At I=1, token i is on rank i % cp_size."""
    work = causal_attention_work_per_rank(seq_len=64, cp_size=cp_size, cp_kv_cache_interleave_size=1)
    # Rank r owns token positions r, r+cp, r+2cp, ...
    expected_per_rank = []
    for r in range(cp_size):
        owned = sum(i + 1 for i in range(64) if i % cp_size == r)
        expected_per_rank.append(owned)
    assert work == expected_per_rank
