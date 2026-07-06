"""Monte Carlo distribution-preservation check for
speculative_sampling.multi_round_speculative_sampling -- PAPER: arXiv:2401.15077
Appendix A.2 Algorithm 1.

The dossier/impl-notes are explicit that this chapter does NOT re-derive the
distribution-preservation proof for the plain (single-candidate) accept/reject
rule -- that full proof (with worked example + Monte Carlo verification) is
ch28 Sec.28.5's job, linked rather than repeated here.

But Algorithm 1's *k-candidate, multi-round* recursion is a genuinely new
claim beyond the single-candidate case: its stated guarantee (paper, Appendix
A.2: "Output: a sample x ~ p") is that the returned token's marginal
distribution equals the target distribution p for ANY k candidate
distributions p_hat_1..p_hat_k, as long as each t_i is itself drawn from its
own p_hat_i. This is exactly what draft_tree.verify_tree relies on to claim a
*tree* draft (multiple sibling candidates per node, not a single chain
candidate) preserves the output distribution. The existing
test_speculative_sampling.py only checks a handful of hand-picked
accept/reject boundary scenarios with fixed candidate tokens and u draws --
it does not check the marginal-distribution guarantee itself. This module
adds that check, per the tester contract's primer-chapter rule: distribution-
preserving mechanisms get a statistical test (fixed seed, loose tolerance, no
flakiness), not just spot-checks.
"""
from __future__ import annotations

import numpy as np

from speculative_sampling import multi_round_speculative_sampling

# A single fixed seed for the whole module -- deterministic, no flakiness.
_SEED = 20260706
_N_TRIALS = 40_000
# Loose enough that this can't flake given N_TRIALS and _SEED fixed, but tight
# enough to catch a real bug (e.g. an off-by-one in the recursion, or the
# residual-adjustment step being skipped/misapplied).
_TOL = 0.02


def _empirical_marginal(
    p_target: np.ndarray,
    p_hats: list[np.ndarray],
    n_trials: int,
    seed: int,
) -> np.ndarray:
    """
    Runs Algorithm 1 `n_trials` times. Each trial draws its own k candidate
    tokens t_i ~ p_hat_i (independently, matching the theorem's precondition
    "t_i is sampled from p_hat_i") and its own U(0,1) draws, then records the
    returned token. Returns the empirical frequency vector over the vocab.
    """
    rng = np.random.default_rng(seed)
    vocab_size = len(p_target)
    counts = np.zeros(vocab_size)
    for _ in range(n_trials):
        candidate_tokens = [int(rng.choice(vocab_size, p=p_hat)) for p_hat in p_hats]
        us = rng.uniform(0.0, 1.0, size=len(p_hats))
        token, _rounds = multi_round_speculative_sampling(
            p_target, candidate_tokens, p_hats, us, rng
        )
        counts[token] += 1
    return counts / n_trials


class TestSingleCandidateMarginalMatchesTarget:
    """k=1 sanity check: Algorithm 1 degenerates to plain speculative
    sampling's accept/reject + residual-resample rule (PAPER §2), whose
    marginal output is p regardless of p_hat -- this is the base case the
    k>=2 recursion below builds on."""

    def test_marginal_distribution_matches_p_target(self):
        p_target = np.array([0.4, 0.3, 0.2, 0.1])
        p_hat = np.array([0.1, 0.2, 0.3, 0.4])  # deliberately mismatched
        empirical = _empirical_marginal(p_target, [p_hat], _N_TRIALS, _SEED)
        np.testing.assert_allclose(empirical, p_target, atol=_TOL)


class TestMultiCandidateMarginalMatchesTarget:
    """PAPER Appendix A.2 Algorithm 1's actual claim: for k>=2 sibling
    candidates (the tree-verification case draft_tree.verify_tree relies on),
    each drawn from its OWN p_hat_i, the returned token's marginal
    distribution still equals p_target -- i.e. having multiple, mutually
    different draft distributions to fall back on does not bias the output
    away from the target distribution."""

    def test_two_candidates_with_different_draft_distributions(self):
        p_target = np.array([0.5, 0.25, 0.15, 0.1])
        p_hat_1 = np.array([0.1, 0.1, 0.4, 0.4])  # underweights token 0 badly
        p_hat_2 = np.array([0.7, 0.1, 0.1, 0.1])  # overweights token 0
        empirical = _empirical_marginal(
            p_target, [p_hat_1, p_hat_2], _N_TRIALS, _SEED
        )
        np.testing.assert_allclose(empirical, p_target, atol=_TOL)

    def test_three_candidates_still_matches_target(self):
        p_target = np.array([0.1, 0.6, 0.2, 0.1])
        p_hat_1 = np.array([0.4, 0.2, 0.2, 0.2])
        p_hat_2 = np.array([0.25, 0.25, 0.25, 0.25])
        p_hat_3 = np.array([0.05, 0.05, 0.05, 0.85])
        empirical = _empirical_marginal(
            p_target, [p_hat_1, p_hat_2, p_hat_3], _N_TRIALS, _SEED
        )
        np.testing.assert_allclose(empirical, p_target, atol=_TOL)

    def test_target_equals_one_of_the_drafts_still_matches(self):
        # Edge case: p_hat_1 == p_target exactly (first candidate is always
        # accepted whenever the accept/reject draw succeeds; the recursion
        # must still land on the target marginal, not something skewed by
        # the second, badly-mismatched candidate).
        p_target = np.array([0.3, 0.3, 0.2, 0.2])
        p_hat_1 = np.array([0.3, 0.3, 0.2, 0.2])
        p_hat_2 = np.array([0.9, 0.05, 0.03, 0.02])
        empirical = _empirical_marginal(
            p_target, [p_hat_1, p_hat_2], _N_TRIALS, _SEED
        )
        np.testing.assert_allclose(empirical, p_target, atol=_TOL)
