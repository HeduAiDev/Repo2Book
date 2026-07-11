"""
Tests for the DDTree best-first draft-tree construction (arXiv:2604.12989
§4.2 Eq.(4)-(8), §4.3 Algorithm 1) -- the "single-trajectory verification vs.
tree verification" extension built on top of DFlash's per-position marginals.
"""
import itertools
import math

import pytest

from ddtree import best_first_tree, expected_acceptance_length_surrogate, prefix_log_prob


def _make_topk_log_probs(depths, top_k, seed=0):
    import random
    rng = random.Random(seed)
    result = []
    for _ in range(depths):
        raw = sorted((rng.uniform(0.05, 1.0) for _ in range(top_k)), reverse=True)
        total = sum(raw)
        probs = [p / total for p in raw]
        result.append([math.log(p) for p in probs])
    return result


def _brute_force_top_b(topk_log_probs, budget, top_k):
    depths = len(topk_log_probs)
    all_prefixes = []
    for d in range(1, depths + 1):
        for ranks in itertools.product(range(1, top_k + 1), repeat=d):
            all_prefixes.append(ranks)
    all_prefixes.sort(key=lambda rho: prefix_log_prob(rho, topk_log_probs), reverse=True)
    return all_prefixes[:budget]


class TestPrefixLogProb:
    def test_matches_direct_sum(self):
        topk = [[math.log(0.5), math.log(0.3)], [math.log(0.4), math.log(0.2)]]
        assert prefix_log_prob((1, 2), topk) == math.log(0.5) + math.log(0.2)


class TestBestFirstTreeOptimality:
    def test_matches_brute_force_top_b_for_small_instance(self):
        depths, top_k, budget = 3, 3, 8
        topk = _make_topk_log_probs(depths, top_k, seed=1)
        tree = best_first_tree(topk, budget=budget, top_k=top_k)
        brute = _brute_force_top_b(topk, budget=budget, top_k=top_k)
        assert set(tree) == set(brute)

    def test_returns_at_most_budget_nodes(self):
        topk = _make_topk_log_probs(depths=4, top_k=4, seed=2)
        tree = best_first_tree(topk, budget=10, top_k=4)
        assert len(tree) <= 10

    def test_root_is_always_included(self):
        topk = _make_topk_log_probs(depths=3, top_k=3, seed=3)
        tree = best_first_tree(topk, budget=5, top_k=3)
        assert (1,) in tree

    def test_tree_is_prefix_closed(self):
        # Prop. 2's remark: because probability is non-increasing along a
        # path, taking the top-B prefixes automatically yields a valid
        # (ancestor-closed) tree -- every non-root node's parent prefix must
        # also be present.
        topk = _make_topk_log_probs(depths=4, top_k=4, seed=4)
        tree = best_first_tree(topk, budget=12, top_k=4)
        tree_set = set(tree)
        for rho in tree:
            if len(rho) > 1:
                assert rho[:-1] in tree_set


class TestExpectedAcceptanceLengthSurrogate:
    def test_single_chain_tree_sums_prefix_probabilities(self):
        # Eq.(8): E[alpha_T] = sum_{u in T} q(u|c,b). For a single root-to-
        # leaf chain, this is just the sum of the chain's own prefix
        # probabilities -- no combinatorics over sibling branches involved.
        depths, top_k = 3, 2
        topk = _make_topk_log_probs(depths, top_k, seed=5)
        chain = [(1,), (1, 1), (1, 1, 1)]
        expected = sum(math.exp(prefix_log_prob(rho, topk)) for rho in chain)
        assert expected_acceptance_length_surrogate(chain, topk) == pytest.approx(expected)

    def test_larger_budget_never_decreases_surrogate_value(self):
        # Since best_first_tree greedily adds the next highest-probability
        # prefix each step, the running surrogate objective (8) must be
        # monotonically non-decreasing as the budget grows.
        topk = _make_topk_log_probs(depths=3, top_k=3, seed=6)
        prev = 0.0
        for budget in (1, 2, 4, 8, 16):
            tree = best_first_tree(topk, budget=budget, top_k=3)
            value = expected_acceptance_length_surrogate(tree, topk)
            assert value >= prev - 1e-9
            prev = value
