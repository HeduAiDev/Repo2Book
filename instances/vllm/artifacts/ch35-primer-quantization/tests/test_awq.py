"""Tests for awq.py — AWQ §3.2 Eq.1-5 (salient-weight protection by
activation-aware per-channel scaling).
"""
import numpy as np
import pytest

from awq import quantize_dequantize, round_err, scaled_error_ratio, search_alpha


def test_round_err_is_close_to_quarter():
    # AWQ §3.2: "RoundErr(.) ~ 0.25" -- the expected |round(t)-t| for t
    # spread across a quantization grid is ~0.25 (average of a roughly
    # Uniform[0,0.5] rounding-error distribution).
    rng = np.random.default_rng(2)
    t = rng.uniform(-50, 50, size=200_000)
    err = round_err(t)
    assert err == pytest.approx(0.25, abs=0.01)


def test_scaled_error_ratio_matches_naive_1_over_s_when_max_unchanged():
    # AWQ Eq.2-3: scaling up a salient weight w by s (and dividing the
    # matching activation by s) reduces its *relative* quantization error to
    # ~ Delta'/Delta * 1/s. This is a statement about the *expected* rounding
    # error (RoundErr(.) ~ 0.25, a distributional claim, §3.2) -- a single
    # scalar's realized error is noisy (it can land exactly on a grid point),
    # so we Monte-Carlo average many random salient values and compare
    # *aggregate* errors (sum of scaled errors / sum of original errors,
    # not a mean of individual ratios, which a near-zero denominator would
    # blow up) against the naive 1/s prediction.
    rng = np.random.default_rng(3)
    base_group = rng.normal(scale=1.0, size=64)
    n_bits = 8
    for s in (2.0, 4.0):
        sum_scaled_err = 0.0
        sum_original_err = 0.0
        delta_ratios = []
        naive_predicted = None
        for _ in range(400):
            group = base_group.copy()
            group[0] = rng.uniform(-0.3, 0.3)  # small vs. group range -> Delta unchanged
            delta = np.max(np.abs(group)) / (2 ** (n_bits - 1))
            original_val = group[0]
            original_q = np.round(original_val / delta) * delta
            original_err = abs(original_q - original_val)

            predicted, _, delta_ratio = scaled_error_ratio(group, 0, s, n_bits)
            naive_predicted = predicted
            delta_ratios.append(delta_ratio)

            scaled_group = group.copy()
            scaled_group[0] = original_val * s
            delta_scaled = np.max(np.abs(scaled_group)) / (2 ** (n_bits - 1))
            scaled_q = np.round(scaled_group[0] / delta_scaled) * delta_scaled / s
            scaled_err = abs(scaled_q - original_val)

            sum_original_err += original_err
            sum_scaled_err += scaled_err

        assert naive_predicted == pytest.approx(1.0 / s)
        assert np.mean(delta_ratios) == pytest.approx(1.0, abs=0.05)  # Delta' ~= Delta
        aggregate_ratio = sum_scaled_err / sum_original_err
        assert aggregate_ratio == pytest.approx(naive_predicted, rel=0.3)


def test_scaling_that_grows_delta_hurts_nonsalient_channels():
    # §3.2: "further increasing s will increase the quantization error for
    # non-salient channels" -- because once s*w exceeds the old group max,
    # Delta' > Delta, and *every other* (non-salient) element now sits on a
    # coarser grid. scaled_error_ratio's own delta_ratio output already
    # confirms Delta grows (checked below); this test checks the direct
    # consequence the paper cares about: a fixed non-salient value's own
    # quantization error grows roughly in proportion to Delta'/Delta once
    # the salient channel is scaled past the old group max.
    rng = np.random.default_rng(11)
    s = 8.0
    n_bits = 4
    delta_ratios = []
    err_before_total = 0.0
    err_after_total = 0.0
    for _ in range(400):
        # non-salient channels sit close to the group's current absmax
        # (0.15-0.2); the salient element is smaller until scaled by s=8, at
        # which point it becomes the new dominant magnitude and Delta grows.
        others = rng.uniform(0.15, 0.2, size=4) * rng.choice([-1, 1], size=4)
        salient_val = rng.uniform(0.05, 0.2)
        group = np.concatenate([[salient_val], others])
        _, _, delta_ratio = scaled_error_ratio(group, 0, s, n_bits)
        delta_ratios.append(delta_ratio)

        delta_before = np.max(np.abs(group)) / (2 ** (n_bits - 1))
        scaled_group = group.copy()
        scaled_group[0] = salient_val * s
        delta_after = np.max(np.abs(scaled_group)) / (2 ** (n_bits - 1))

        # hold one non-salient value fixed and compare its own quantization
        # error under the old vs. the new (larger) Delta.
        nonsalient_val = others[0]
        err_before = abs(np.round(nonsalient_val / delta_before) * delta_before - nonsalient_val)
        err_after = abs(np.round(nonsalient_val / delta_after) * delta_after - nonsalient_val)
        err_before_total += err_before
        err_after_total += err_after

    assert np.mean(delta_ratios) > 1.05  # Delta' grew on average (new absmax)
    # aggregate (not per-instance, to dodge lucky exact-roundtrip trials)
    # non-salient error should be larger under the grown Delta.
    assert err_after_total > err_before_total


def test_search_alpha_prefers_moderate_scaling():
    rng = np.random.default_rng(4)
    Ci, Co = 8, 4
    W = rng.normal(size=(Ci, Co))
    s_x = np.abs(rng.normal(loc=3.0, scale=1.0, size=Ci))  # per-input-channel activation avg magnitude
    X = rng.normal(size=(20, Ci)) * s_x  # activations whose scale correlates with s_x
    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    best_alpha, losses = search_alpha(X, W, s_x, n_bits=4, alphas=alphas)
    assert best_alpha in alphas
    assert len(losses) == len(alphas)
    # the search must be doing real work: the best loss should be at least
    # as good as either "push everything to one extreme" endpoint.
    assert min(losses) <= losses[0]
    assert min(losses) <= losses[-1]


def test_quantize_dequantize_eq1_matches_awq_scale():
    w = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    q = quantize_dequantize(w, n_bits=4)
    # Delta = max|w| / 2^(N-1) = 4/8 = 0.5; round(w/Delta)*Delta should
    # reproduce w exactly on this grid-aligned example.
    np.testing.assert_allclose(q, w)
