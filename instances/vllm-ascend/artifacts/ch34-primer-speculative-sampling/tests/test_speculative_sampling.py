"""
Tests for the paper-faithful reference implementation of speculative sampling
(arXiv:2211.17192, Leviathan, Kalman, Matias, "Fast Inference from Transformers
via Speculative Decoding").

These tests check that the implementation reproduces the paper's *claims*
(Algorithm 1's acceptance rule, the residual distribution, the
distribution-preserving theorem of Appendix A.1, the acceptance-rate formula
of Corollary 3.6, the expected-length formula of Eq.1, and the walltime
formula of Theorem 3.8/Corollary 3.9) — not just internal self-consistency.
"""
import numpy as np
import pytest

from speculative_sampling import (
    acceptance_rate,
    expected_generated_tokens,
    lukaszyk_karmowski_divergence,
    optimal_gamma,
    propose_and_check,
    residual_distribution,
    speculative_decoding_step,
    speculative_sampling_step,
    walltime_improvement_factor,
)


def _rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Algorithm 1 core: accept-reject rule (paper.md §2.3)
# ---------------------------------------------------------------------------

class TestAcceptReject:
    def test_accepts_when_q_le_p(self):
        # q(x) <= p(x) at the *originally drawn* x => always kept (accept_prob = 1)
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        rng = _rng()
        # x=0: q(0)=0.5 <= p(0)=0.9 -> always accepted when x=0 is drawn
        accepted_flags = []
        for _ in range(2000):
            x, accepted = propose_and_check(p, q, rng)
            if x == 0:
                accepted_flags.append(accepted)
        assert len(accepted_flags) > 0
        assert all(accepted_flags)

    def test_acceptance_probability_matches_min_1_p_over_q(self):
        # x=1: q(1)=0.5 > p(1)=0.1 -> accept with prob p/q = 0.2
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        rng = np.random.default_rng(1)
        n = 200_000
        accepts_at_1 = 0
        draws_at_1 = 0
        for _ in range(n):
            x, accepted = propose_and_check(p, q, rng)
            if x == 1:
                draws_at_1 += 1
                accepts_at_1 += int(accepted)
        empirical = accepts_at_1 / draws_at_1
        assert empirical == pytest.approx(0.2, abs=0.02)


class TestResidualDistribution:
    def test_residual_is_normalized_max_0_p_minus_q(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        p_prime = residual_distribution(p, q)
        # unnormalized residual: max(0, p-q) = [0.4, 0]
        assert p_prime == pytest.approx(np.array([1.0, 0.0]))
        assert p_prime.sum() == pytest.approx(1.0)

    def test_residual_normalizing_constant_is_1_minus_beta(self):
        # paper.md §A.1: normalizing constant of p'(x) is 1 - beta,
        # where beta = sum_x min(p(x), q(x))
        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        beta = acceptance_rate(p, q)
        raw_residual = np.maximum(p - q, 0.0)
        assert raw_residual.sum() == pytest.approx(1.0 - beta, abs=1e-9)


class TestDistributionPreserving:
    """paper.md §A.1: for ANY p, q, samples from speculative sampling are
    distributed identically to samples from p alone (no bias from q)."""

    @pytest.mark.parametrize("seed", [10, 11, 12])
    def test_output_matches_target_distribution_p(self, seed):
        rng = np.random.default_rng(seed)
        vocab = 5
        p = rng.dirichlet(np.ones(vocab) * 2.0)
        q = rng.dirichlet(np.ones(vocab) * 2.0)

        n = 300_000
        counts = np.zeros(vocab)
        sample_rng = np.random.default_rng(seed + 100)
        for _ in range(n):
            x, _ = speculative_sampling_step(p, q, sample_rng)
            counts[x] += 1
        empirical = counts / n
        # loose Monte-Carlo tolerance (3-sigma-ish for n=300k, vocab=5)
        assert np.max(np.abs(empirical - p)) < 0.01

    def test_preserves_distribution_even_when_q_has_zero_support(self):
        # q(x)=0 for some x that p(x)>0: speculative sampling must still be
        # able to recover x via the residual distribution p'.
        p = np.array([0.5, 0.5])
        q = np.array([1.0, 0.0])
        rng = np.random.default_rng(3)
        n = 100_000
        counts = np.zeros(2)
        for _ in range(n):
            x, _ = speculative_sampling_step(p, q, rng)
            counts[x] += 1
        empirical = counts / n
        assert np.max(np.abs(empirical - p)) < 0.02


class TestAcceptanceRateAlpha:
    """paper.md §3.2 Lemma 3.3 / Theorem 3.5 / Corollary 3.6."""

    def test_alpha_equals_sum_min_p_q(self):
        p = np.array([0.6, 0.3, 0.1])
        q = np.array([0.2, 0.3, 0.5])
        alpha = acceptance_rate(p, q)
        assert alpha == pytest.approx(np.minimum(p, q).sum())
        assert alpha == pytest.approx(0.2 + 0.3 + 0.1)

    def test_alpha_is_1_minus_lk_divergence(self):
        rng = np.random.default_rng(4)
        p = rng.dirichlet(np.ones(4))
        q = rng.dirichlet(np.ones(4))
        alpha = acceptance_rate(p, q)
        d_lk = lukaszyk_karmowski_divergence(p, q)
        assert alpha == pytest.approx(1.0 - d_lk)

    def test_alpha_is_1_when_p_equals_q(self):
        p = np.array([0.2, 0.3, 0.5])
        assert acceptance_rate(p, p.copy()) == pytest.approx(1.0)

    def test_alpha_matches_monte_carlo_acceptance_frequency(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        alpha_theory = acceptance_rate(p, q)
        rng = np.random.default_rng(5)
        n = 200_000
        accepted = 0
        for _ in range(n):
            _, was_accepted = speculative_sampling_step(p, q, rng)
            accepted += int(was_accepted)
        assert accepted / n == pytest.approx(alpha_theory, abs=0.01)


class TestExpectedGeneratedTokens:
    """paper.md §3.1 Eq.1: E[#tokens] = (1 - alpha^(gamma+1)) / (1 - alpha)."""

    @pytest.mark.parametrize("alpha,gamma", [(0.5, 3), (0.8, 5), (0.9, 10), (0.0, 4)])
    def test_matches_capped_geometric_sum(self, alpha, gamma):
        # E[#tokens] = sum_{k=0}^{gamma} alpha^k  (direct definition)
        direct = sum(alpha ** k for k in range(gamma + 1))
        assert expected_generated_tokens(alpha, gamma) == pytest.approx(direct)

    def test_alpha_1_gives_gamma_plus_1(self):
        assert expected_generated_tokens(1.0, 7) == pytest.approx(8.0)

    def test_monotonic_in_gamma(self):
        vals = [expected_generated_tokens(0.7, g) for g in range(6)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))


class TestWalltimeSpeedup:
    """paper.md §3.3 Theorem 3.8, Corollary 3.9, Table 1."""

    def test_table_1_alpha_0_8_gamma_5(self):
        # Table 1: alpha=0.8, gamma=5, c=0 -> speed 3.69X
        factor = walltime_improvement_factor(alpha=0.8, gamma=5, c=0.0)
        assert factor == pytest.approx(3.69, abs=0.01)

    def test_table_1_alpha_0_9_gamma_10(self):
        # Table 1: alpha=0.9, gamma=10, c=0 -> speed 6.86X
        factor = walltime_improvement_factor(alpha=0.9, gamma=10, c=0.0)
        assert factor == pytest.approx(6.86, abs=0.01)

    def test_corollary_3_9_gamma_1_lower_bound(self):
        alpha, c = 0.75, 0.1
        factor = walltime_improvement_factor(alpha=alpha, gamma=1, c=c)
        assert factor == pytest.approx((1 + alpha) / (1 + c))

    def test_optimal_gamma_matches_brute_force_argmax(self):
        alpha, c = 0.8, 0.05
        best = optimal_gamma(alpha, c, gamma_max=20)
        factors = [walltime_improvement_factor(alpha, g, c) for g in range(21)]
        assert best == int(np.argmax(factors))


# ---------------------------------------------------------------------------
# Algorithm 1 full multi-token version
# ---------------------------------------------------------------------------

class TestSpeculativeDecodingStep:
    def test_perfect_draft_model_always_accepts_all_gamma(self):
        # q == p at every position => ratio == 1 everywhere => r_i > 1 never
        # happens (r_i ~ U(0,1)) => n == gamma always.
        gamma = 4
        vocab = 3
        rng = np.random.default_rng(6)
        fixed_dist = rng.dirichlet(np.ones(vocab))

        def get_dist(prefix):
            return fixed_dist

        for _ in range(50):
            _, n = speculative_decoding_step(get_dist, get_dist, prefix=[], gamma=gamma, rng=rng)
            assert n == gamma

    def test_returns_between_1_and_gamma_plus_1_tokens(self):
        gamma = 3
        vocab = 4
        rng = np.random.default_rng(7)
        p_dist = rng.dirichlet(np.ones(vocab))
        q_dist = rng.dirichlet(np.ones(vocab))

        for _ in range(200):
            seq, n = speculative_decoding_step(
                lambda prefix: q_dist, lambda prefix: p_dist, prefix=[], gamma=gamma, rng=rng
            )
            assert 0 <= n <= gamma
            assert len(seq) == n + 1  # n draft tokens kept + 1 (recovered or bonus)

    def test_first_token_marginal_matches_p1_distribution_preserving(self):
        # paper.md §A.1 correctness proof applies position-by-position: the
        # first token of the returned suffix must be distributed as p_1(x),
        # regardless of gamma or of q's quality.
        gamma = 2
        vocab = 4
        rng = np.random.default_rng(8)
        p_dist = rng.dirichlet(np.ones(vocab) * 2.0)
        q_dist = rng.dirichlet(np.ones(vocab) * 2.0)

        n_trials = 200_000
        counts = np.zeros(vocab)
        sample_rng = np.random.default_rng(9)
        for _ in range(n_trials):
            seq, _ = speculative_decoding_step(
                lambda prefix: q_dist, lambda prefix: p_dist, prefix=[], gamma=gamma, rng=sample_rng
            )
            counts[seq[0]] += 1
        empirical = counts / n_trials
        assert np.max(np.abs(empirical - p_dist)) < 0.01
