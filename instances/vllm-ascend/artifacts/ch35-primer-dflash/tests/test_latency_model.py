"""
Tests for the DFlash speedup/latency model (arXiv:2602.06036 §3.1-3.2,
Eq.(1)(2)(3)).
"""
import pytest

from latency_model import (
    autoregressive_draft_cost,
    diffusion_draft_cost,
    per_token_latency,
    speedup,
    speedup_for_mode,
)


class TestPerTokenLatencyEq1:
    def test_matches_definition(self):
        # L = (T_draft + T_verify) / tau
        assert per_token_latency(t_draft=4.0, t_verify=6.0, tau=2.0) == pytest.approx(5.0)

    def test_rejects_nonpositive_tau(self):
        # tau is an expected accepted-token count in [1, gamma+1]; <= 0 is
        # nonsensical and must not silently produce a negative/inf latency.
        with pytest.raises(ValueError):
            per_token_latency(t_draft=1.0, t_verify=1.0, tau=0.0)

    def test_speedup_is_ratio_of_latencies(self):
        # eta = L_target / L
        assert speedup(l_target=10.0, l=2.0) == pytest.approx(5.0)


class TestAutoregressiveCostEq2:
    def test_scales_linearly_with_gamma(self):
        t_step = 3.0
        costs = [autoregressive_draft_cost(gamma, t_step) for gamma in (1, 2, 4, 8)]
        # T_draft = gamma * t_step: doubling gamma doubles cost exactly.
        assert costs == [3.0, 6.0, 12.0, 24.0]

    def test_zero_gamma_is_zero_cost(self):
        assert autoregressive_draft_cost(0, t_step=5.0) == 0.0


class TestDiffusionCostEq3:
    def test_independent_of_gamma(self):
        # T_draft = t_parallel regardless of how many tokens the block holds.
        t_parallel = 7.5
        costs = {diffusion_draft_cost(t_parallel, gamma=g) for g in (1, 4, 8, 16, 32)}
        assert costs == {7.5}


class TestSpeedupForMode:
    def test_autoregressive_and_diffusion_agree_at_gamma_1(self):
        # At gamma=1 a single autoregressive step *is* a single parallel
        # block step, so both modes should cost the same and thus give the
        # same speedup for identical t_step == t_parallel.
        kwargs = dict(t_step=2.0, t_parallel=2.0, t_verify=1.0, tau=1.5, l_target=10.0)
        ar = speedup_for_mode("autoregressive", gamma=1, **kwargs)
        diff = speedup_for_mode("diffusion", gamma=1, **kwargs)
        assert ar == pytest.approx(diff)

    def test_diffusion_overtakes_autoregressive_as_gamma_grows(self):
        # This is the qualitative claim behind Fig.3: because T_draft for
        # autoregressive drafting grows linearly with gamma (Eq.2) while
        # diffusion's stays flat (Eq.3), a diffusion drafter with a larger
        # per-step constant cost still wins once gamma is large enough.
        kwargs = dict(t_step=1.0, t_parallel=3.0, t_verify=1.0, tau=8.0, l_target=20.0)
        ar_small = speedup_for_mode("autoregressive", gamma=2, **kwargs)
        diff_small = speedup_for_mode("diffusion", gamma=2, **kwargs)
        ar_large = speedup_for_mode("autoregressive", gamma=16, **kwargs)
        diff_large = speedup_for_mode("diffusion", gamma=16, **kwargs)
        # at gamma=2 the higher constant cost of diffusion may still lose...
        assert ar_small >= diff_small
        # ...but at gamma=16 the linear-growth mode has fallen far behind.
        assert diff_large > ar_large

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            speedup_for_mode("quantum", gamma=1, t_step=1.0, t_parallel=1.0,
                              t_verify=1.0, tau=1.0, l_target=1.0)
