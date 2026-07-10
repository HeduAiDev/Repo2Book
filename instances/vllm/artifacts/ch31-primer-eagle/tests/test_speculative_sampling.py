"""Tests for speculative_sampling.py -- PAPER: arXiv:2401.15077 §2 (accept/reject
rule + residual distribution) and Appendix A.2 Algorithm 1 (multi-round
speculative sampling for verifying a *tree* of candidates).

Note: the accept/reject rule's distribution-preservation proof is NOT
re-derived here -- that is ch28 §28.5's job (this chapter links to it). These
tests only check that the recursion in Algorithm 1 is implemented as written:
try each sibling in order against the current (possibly residual-adjusted)
distribution, stop at the first acceptance, fall back to a fresh sample from
the final residual distribution if every sibling is rejected.
"""
import numpy as np

from speculative_sampling import (
    accept_reject,
    multi_round_speculative_sampling,
    residual_distribution,
)


class TestAcceptReject:
    def test_accepts_when_target_prob_at_least_draft_prob(self):
        # PAPER §2: ratio = p(t)/p_hat(t) >= 1 -> min(1, ratio) == 1 -> always accept.
        p = np.array([0.5, 0.5])
        p_hat = np.array([0.2, 0.8])
        assert accept_reject(p, p_hat, token=0, u=0.999999) is True

    def test_boundary_u_below_ratio_accepts_above_rejects(self):
        p = np.array([0.3, 0.7])
        p_hat = np.array([0.6, 0.4])
        ratio = p[0] / p_hat[0]  # 0.5
        assert accept_reject(p, p_hat, token=0, u=ratio - 0.01) is True
        assert accept_reject(p, p_hat, token=0, u=ratio + 0.01) is False

    def test_zero_draft_prob_treated_as_ratio_one_no_division_by_zero(self):
        p = np.array([0.1, 0.9])
        p_hat = np.array([0.0, 1.0])
        # p_hat[token]==0 guard -> ratio defined as 1.0 -> accept whenever u<1.
        assert accept_reject(p, p_hat, token=0, u=0.5) is True


class TestResidualDistribution:
    def test_sums_to_one_and_matches_formula(self):
        p = np.array([0.5, 0.3, 0.2])
        p_hat = np.array([0.1, 0.4, 0.5])
        residual = residual_distribution(p, p_hat)
        expected_unnorm = np.maximum(p - p_hat, 0.0)
        expected = expected_unnorm / expected_unnorm.sum()
        np.testing.assert_allclose(residual, expected)
        assert np.isclose(residual.sum(), 1.0)

    def test_negative_diffs_clamped_to_zero(self):
        p = np.array([0.9, 0.1])
        p_hat = np.array([0.95, 0.05])
        residual = residual_distribution(p, p_hat)
        # p[0]-p_hat[0] < 0 -> clamped to 0; only mass 0.05 on index 1 survives.
        np.testing.assert_allclose(residual, [0.0, 1.0])

    def test_identical_distributions_guarded_uniform_fallback(self):
        p = np.array([0.4, 0.6])
        residual = residual_distribution(p, p)
        np.testing.assert_allclose(residual, [0.5, 0.5])


class TestMultiRoundSpeculativeSampling:
    def test_first_candidate_accepted_returns_immediately(self):
        rng = np.random.default_rng(0)
        p_target = np.array([0.9, 0.1])
        candidate_tokens = [0, 1]
        candidate_dists = [np.array([0.1, 0.9]), np.array([0.5, 0.5])]
        # ratio for token 0: 0.9/0.1 = 9 -> min(1, 9) = 1 -> always accepted.
        us = [0.0, 0.0]
        token, rounds = multi_round_speculative_sampling(
            p_target, candidate_tokens, candidate_dists, us, rng
        )
        assert token == 0
        assert rounds == 1

    def test_second_candidate_accepted_after_first_rejected(self):
        rng = np.random.default_rng(1)
        p_target = np.array([0.9, 0.1])
        candidate_tokens = [1, 0]
        # round 1: propose token 1, ratio = p[1]/p_hat0[1] = 0.1/0.9 ~ 0.111
        # -> u=0.5 > ratio -> rejected. Residual becomes norm(max(p-p_hat0,0))
        # = norm([0.8, 0.0]) = [1.0, 0.0].
        # round 2: propose token 0 against that residual, ratio = 1.0/0.6 ->
        # min(1,ratio)=1 -> accepted for any u<1.
        candidate_dists = [np.array([0.1, 0.9]), np.array([0.6, 0.4])]
        us = [0.5, 0.0]
        token, rounds = multi_round_speculative_sampling(
            p_target, candidate_tokens, candidate_dists, us, rng
        )
        assert token == 0
        assert rounds == 2

    def test_all_rejected_falls_back_to_fresh_sample_from_residual(self):
        rng = np.random.default_rng(42)
        p_target = np.array([1.0, 0.0, 0.0])
        candidate_tokens = [1, 2]
        # both candidates propose tokens the target assigns zero mass to ->
        # ratio = 0/p_hat = 0 -> always rejected regardless of u.
        candidate_dists = [np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        us = [0.0, 0.0]
        token, rounds = multi_round_speculative_sampling(
            p_target, candidate_tokens, candidate_dists, us, rng
        )
        # after two rejections residual collapses all mass back onto token 0
        # (p_target's only nonzero entry, since both candidates only ever
        # subtracted from indices 1 and 2).
        assert token == 0
        assert rounds == len(candidate_tokens)

    def test_single_candidate_accepted_matches_plain_accept_reject(self):
        rng = np.random.default_rng(2)
        p_target = np.array([0.5, 0.5])
        p_hat = np.array([0.2, 0.8])
        u = 0.3  # ratio = 0.5/0.2 = 2.5 -> min(1,ratio)=1 -> accepted for any u<1.
        assert accept_reject(p_target, p_hat, 0, u) is True
        token, rounds = multi_round_speculative_sampling(
            p_target, [0], [p_hat], [u], rng
        )
        assert token == 0
        assert rounds == 1

    def test_single_candidate_rejected_falls_back_to_residual_sample(self):
        rng = np.random.default_rng(3)
        p_target = np.array([0.1, 0.9])
        p_hat = np.array([0.9, 0.1])
        u = 0.99  # ratio = 0.1/0.9 -> tiny -> rejected.
        assert accept_reject(p_target, p_hat, 0, u) is False
        token, rounds = multi_round_speculative_sampling(
            p_target, [0], [p_hat], [u], rng
        )
        assert rounds == 1
        # after rejection, residual mass is entirely on token 1 (p[1]-p_hat[1]
        # = 0.8 > 0, p[0]-p_hat[0] = -0.8 clamped to 0), so the fresh sample
        # is deterministically token 1.
        assert token == 1
