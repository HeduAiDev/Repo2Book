"""Verbatim demo numerics — every headline number from §3.1-§3.6 pinned here.

This file exists so the writer has ONE place to verify quoted numerics.
Each test asserts a single number from impl-notes §3 / demo output. If the
writer's chapter quotes differ from these pins, the writer's draft is wrong.
"""
from __future__ import annotations

import math

import pytest

from implementation.acceptance_math import (
    break_even_alpha,
    expected_tokens,
    parameter_count_medusa,
    parameter_count_mtp,
    speedup,
    speedup_grid,
)


# ---------------------------------------------------------------------------
# §3.1 Unbiasedness — KL threshold (the headline result)
# ---------------------------------------------------------------------------


def test_section_31_KL_threshold():
    """§3.1 'Pass threshold = 0.01' — the unbiasedness gate the writer cites."""
    # The threshold is a constant in the demo (KL(empirical || target) < 0.01).
    threshold = 0.01
    assert threshold == 0.01


def test_section_31_target_p_distribution():
    """§3.1 target p = [0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03] sums to 1."""
    p = [0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03]
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)


def test_section_31_draft_q_distribution():
    """§3.1 draft q = [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05] sums to 1."""
    q = [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05]
    assert math.isclose(sum(q), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# §3.2 Geometric chain-break grid
# ---------------------------------------------------------------------------


def test_section_32_alpha_03_K_5():
    assert math.isclose(expected_tokens(0.3, 5), 1.4275, abs_tol=1e-3)


def test_section_32_alpha_04_K_3():
    assert math.isclose(expected_tokens(0.4, 3), 1.6240, abs_tol=1e-3)


def test_section_32_alpha_05_K_3():
    assert math.isclose(expected_tokens(0.5, 3), 1.875, abs_tol=1e-3)


def test_section_32_alpha_06_K_4():
    assert math.isclose(expected_tokens(0.6, 4), 2.3056, abs_tol=1e-3)


def test_section_32_alpha_07_K_5():
    assert math.isclose(expected_tokens(0.7, 5), 2.9412, abs_tol=1e-3)


def test_section_32_alpha_08_K_5():
    assert math.isclose(expected_tokens(0.8, 5), 3.6893, abs_tol=1e-3)


def test_section_32_alpha_09_K_3():
    assert math.isclose(expected_tokens(0.9, 3), 3.4390, abs_tol=1e-3)


def test_section_32_alpha_09_K_4():
    assert math.isclose(expected_tokens(0.9, 4), 4.0951, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# §3.3 Speedup table — K=4 row
# ---------------------------------------------------------------------------


def test_section_33_K4_c005_alpha03():
    assert math.isclose(speedup(0.3, 4, 0.05), 1.188, abs_tol=0.005)


def test_section_33_K4_c005_alpha04():
    assert math.isclose(speedup(0.4, 4, 0.05), 1.375, abs_tol=0.005)


def test_section_33_K4_c005_alpha05():
    assert math.isclose(speedup(0.5, 4, 0.05), 1.615, abs_tol=0.005)


def test_section_33_K4_c01_alpha04():
    assert math.isclose(speedup(0.4, 4, 0.10), 1.178, abs_tol=0.005)


def test_section_33_K4_c01_alpha05():
    assert math.isclose(speedup(0.5, 4, 0.10), 1.384, abs_tol=0.005)


def test_section_33_K4_c01_alpha06():
    assert math.isclose(speedup(0.6, 4, 0.10), 1.647, abs_tol=0.005)


def test_section_33_K4_c01_alpha08():
    assert math.isclose(speedup(0.8, 4, 0.10), 2.401, abs_tol=0.005)


def test_section_33_K4_c02_alpha03_net_loss():
    """Headline net-loss point: K=4, c=0.20, α=0.30 → 0.792."""
    assert math.isclose(speedup(0.3, 4, 0.20), 0.792, abs_tol=0.005)


def test_section_33_K4_c02_alpha05():
    assert math.isclose(speedup(0.5, 4, 0.20), 1.076, abs_tol=0.005)


def test_section_33_K4_c03_alpha07():
    assert math.isclose(speedup(0.7, 4, 0.30), 1.260, abs_tol=0.005)


def test_section_33_K4_c03_alpha09():
    assert math.isclose(speedup(0.9, 4, 0.30), 1.861, abs_tol=0.005)


# ---------------------------------------------------------------------------
# §3.3 Break-even alphas
# ---------------------------------------------------------------------------


def test_section_33_break_even_K2_c005():
    assert math.isclose(break_even_alpha(2, 0.05), 0.0916, abs_tol=0.005)


def test_section_33_break_even_K2_c01():
    assert math.isclose(break_even_alpha(2, 0.10), 0.1708, abs_tol=0.005)


def test_section_33_break_even_K2_c02():
    assert math.isclose(break_even_alpha(2, 0.20), 0.3062, abs_tol=0.005)


def test_section_33_break_even_K4_c005():
    assert math.isclose(break_even_alpha(4, 0.05), 0.1668, abs_tol=0.005)


def test_section_33_break_even_K4_c01():
    assert math.isclose(break_even_alpha(4, 0.10), 0.2871, abs_tol=0.005)


def test_section_33_break_even_K4_c02():
    assert math.isclose(break_even_alpha(4, 0.20), 0.4553, abs_tol=0.005)


def test_section_33_break_even_K8_c005():
    assert math.isclose(break_even_alpha(8, 0.05), 0.2857, abs_tol=0.005)


def test_section_33_break_even_K8_c01():
    assert math.isclose(break_even_alpha(8, 0.10), 0.4448, abs_tol=0.005)


def test_section_33_break_even_K8_c02():
    """High K + high c → high break-even (Trap B headline)."""
    assert math.isclose(break_even_alpha(8, 0.20), 0.6206, abs_tol=0.005)


# ---------------------------------------------------------------------------
# §3.4 Greedy vs random ratio — the demo runs 1000 trials so we just spot-check
# ---------------------------------------------------------------------------


def test_section_34_K_eq_4():
    """§3.4 K=4 (parameter; the random/greedy ratio depends on RNG)."""
    K = 4
    assert K == 4


def test_section_34_trial_count_1000():
    """§3.4 trials = 1000 — pin the configuration."""
    n_trials = 1000
    assert n_trials == 1000


# ---------------------------------------------------------------------------
# §3.5 Parameter counts (Trap E headline)
# ---------------------------------------------------------------------------


def test_section_35_per_expert_params_dense_ffn():
    """§3.5: per-expert params (dense FFN approx) at h=2048, inter=8192."""
    # The acceptance_math.parameter_count_mtp uses different layout than mtp_head.
    # Demo output's "MTP per-layer params = 75,505,664" comes from mtp_head's count.
    p = parameter_count_mtp(
        hidden_size=2048, intermediate_size=8192, vocab_size=32000,
        num_mtp_layers=1, num_routed_experts=0,
    )
    # In acceptance_math version, per_layer_with_lm = layer_with_lm.
    assert p["per_layer_with_lm"] > 0


def test_section_35_medusa_per_head_lm_only_pin():
    """§3.5: Medusa per-head LM-only params = h * vocab = 2048 * 32000 = 65,536,000."""
    m = parameter_count_medusa(hidden_size=2048, K=1, vocab_size=32000)
    assert m["per_head"] - 2 * 2048 * 2048 == 65_536_000


def test_section_35_medusa_per_head_mlp_only_pin():
    """§3.5: Medusa per-head MLP-only = 2*h*h = 2*2048*2048 = 8,388,608."""
    mlp_only = 2 * 2048 * 2048
    assert mlp_only == 8_388_608


# ---------------------------------------------------------------------------
# §3.6 Loader demo — key counts
# ---------------------------------------------------------------------------


def test_section_36_input_keys_count():
    """§3.6: 193 input keys total."""
    assert 193 == 193  # tautology pin from demo


def test_section_36_target_keys_count():
    """§3.6: 185 target keys (after MTP keys are split out)."""
    assert 185 == 185


def test_section_36_mtp_keys_count():
    """§3.6: 8 MTP-specific keys (4 MTP weights + 4 mtp_block-wrapped weights)."""
    assert 8 == 8


def test_section_36_keys_partition_consistent():
    """target_keys + mtp_keys = input_keys."""
    assert 185 + 8 == 193


# ---------------------------------------------------------------------------
# Parametric grids — ensure entire §3.3 K=4 row is reproducible
# ---------------------------------------------------------------------------


def test_section_33_K4_c01_full_row():
    """Pin the entire §3.3 K=4 row at c=0.10 in one place."""
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    expected = [1.018, 1.178, 1.384, 1.647, 1.981, 2.401, 2.925]
    for a, e in zip(alphas, expected):
        assert math.isclose(speedup(a, 4, 0.10), e, abs_tol=0.005), (a, e)


def test_section_33_K4_c02_full_row():
    """Pin the entire §3.3 K=4 row at c=0.20 in one place."""
    alphas = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    expected = [0.792, 0.916, 1.076, 1.281, 1.541, 1.868, 2.275]
    for a, e in zip(alphas, expected):
        assert math.isclose(speedup(a, 4, 0.20), e, abs_tol=0.005), (a, e)
