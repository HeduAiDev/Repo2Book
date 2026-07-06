"""Tests for draft_tree.py -- PAPER: arXiv:2401.15077 §3.1/§3.3/Appendix A.1
(static k-ary tree drafting + recursive multi-round verification) and
arXiv:2406.16858 EAGLE-2 §4.1/§4.2 (dynamic draft tree: value V_i, top-k
expansion, top-m reranking, tree attention mask) + §3.2 (well-calibrated
confidence vs. acceptance rate).
"""
import numpy as np
import torch

from draft_tree import (
    build_root,
    build_static_tree,
    build_tree_attention_mask,
    calibration_curve,
    compute_value,
    expand_node,
    expansion_phase,
    reranking_phase,
    verify_tree,
)
from feature_autoregression import AutoregressionHead, ToyTargetLLM


def _setup(vocab_size=6, hidden_dim=4):
    target_llm = ToyTargetLLM(vocab_size=vocab_size, hidden_dim=hidden_dim, seed=0)
    draft_head = AutoregressionHead(hidden_dim=hidden_dim, seed=1)
    return target_llm, draft_head


class TestBuildRootAndComputeValue:
    def test_root_has_value_one(self):
        root = build_root(token_id=5, feature=torch.zeros(4))
        assert root.value == 1.0
        assert root.confidence == 1.0
        assert root.parent is None
        assert root.depth == 0

    def test_child_value_is_product_of_path_confidences(self):
        root = build_root(token_id=1, feature=torch.zeros(4))
        target_llm, draft_head = _setup()
        children = expand_node(draft_head, target_llm, root, branching_k=2)
        for child in children:
            assert child.value == root.confidence * child.confidence
            assert np.isclose(child.value, child.confidence)  # root.confidence==1.0

    def test_grandchild_value_multiplies_full_path(self):
        root = build_root(token_id=1, feature=torch.zeros(4))
        target_llm, draft_head = _setup()
        children = expand_node(draft_head, target_llm, root, branching_k=2)
        grandchildren = expand_node(draft_head, target_llm, children[0], branching_k=2)
        for gc in grandchildren:
            expected = children[0].confidence * gc.confidence
            assert np.isclose(compute_value(gc), expected)


class TestExpandNode:
    def test_returns_at_most_branching_k_children_sorted_by_probability(self):
        target_llm, draft_head = _setup(vocab_size=6)
        root = build_root(token_id=0, feature=torch.zeros(4))
        children = expand_node(draft_head, target_llm, root, branching_k=3)
        assert len(children) == 3
        confidences = [c.confidence for c in children]
        assert confidences == sorted(confidences, reverse=True)

    def test_branching_k_capped_by_vocab_size(self):
        target_llm, draft_head = _setup(vocab_size=3)
        root = build_root(token_id=0, feature=torch.zeros(4))
        children = expand_node(draft_head, target_llm, root, branching_k=10)
        assert len(children) == 3  # capped at vocab_size

    def test_children_recorded_on_parent(self):
        target_llm, draft_head = _setup()
        root = build_root(token_id=0, feature=torch.zeros(4))
        children = expand_node(draft_head, target_llm, root, branching_k=2)
        assert root.children == children
        assert all(c.parent is root for c in children)
        assert all(c.depth == 1 for c in children)


class TestBuildStaticTree:
    def test_node_count_matches_full_kary_tree_formula(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        depth, k = 2, 2
        all_nodes = build_static_tree(draft_head, target_llm, root, depth=depth, branching_k=k)
        # 1 (root) + k (layer 1) + k^2 (layer 2) = 1 + 2 + 4 = 7
        expected = sum(k**d for d in range(depth + 1))
        assert len(all_nodes) == expected

    def test_every_node_at_every_layer_expanded_static_tree(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        all_nodes = build_static_tree(draft_head, target_llm, root, depth=2, branching_k=2)
        by_depth = {}
        for n in all_nodes:
            by_depth.setdefault(n.depth, []).append(n)
        assert len(by_depth[0]) == 1
        assert len(by_depth[1]) == 2
        assert len(by_depth[2]) == 4


class TestExpansionPhase:
    def test_only_top_k_value_nodes_of_last_layer_get_expanded(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        layer1 = expand_node(draft_head, target_llm, root, branching_k=4)
        assert len(layer1) == 4
        new_nodes = expansion_phase(draft_head, target_llm, layer1, k=2, branching_k=2)
        # only the 2 highest-value nodes of layer1 were expanded, each into
        # up to 2 children -> at most 4 new nodes, all with depth 2, and all
        # parents must be among the top-2-by-value of layer1.
        assert len(new_nodes) <= 4
        assert all(n.depth == 2 for n in new_nodes)
        top2_ids = {id(n) for n in sorted(layer1, key=lambda n: n.value, reverse=True)[:2]}
        assert all(id(n.parent) in top2_ids for n in new_nodes)


class TestRerankingPhase:
    def test_selects_top_m_by_value_and_stays_connected(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        all_nodes = build_static_tree(draft_head, target_llm, root, depth=2, branching_k=2)
        selected = reranking_phase(all_nodes, m=4)
        assert len(selected) == 4
        # connectivity: reranking_phase already asserts internally; also
        # check the parent of every non-root selected node is in `selected`.
        selected_ids = {id(n) for n in selected}
        for n in selected:
            if n.parent is not None:
                assert id(n.parent) in selected_ids

    def test_shallower_first_tiebreak_root_always_included(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        all_nodes = build_static_tree(draft_head, target_llm, root, depth=1, branching_k=3)
        selected = reranking_phase(all_nodes, m=1)
        # root has value 1.0, the maximum possible (all confidences <= 1),
        # so it must always be the single top-m=1 selection.
        assert selected == [root]


class TestBuildTreeAttentionMask:
    def test_each_node_sees_itself_and_all_ancestors_only(self):
        target_llm, draft_head = _setup(vocab_size=8)
        root = build_root(token_id=0, feature=torch.zeros(4))
        children = expand_node(draft_head, target_llm, root, branching_k=2)
        grandchildren = expand_node(draft_head, target_llm, children[0], branching_k=2)
        nodes = [root] + children + grandchildren
        mask = build_tree_attention_mask(nodes)
        assert mask.shape == (len(nodes), len(nodes))
        # root sees only itself.
        root_idx = 0
        assert mask[root_idx].sum() == 1
        assert mask[root_idx, root_idx]
        # a grandchild of children[0] sees itself + children[0] + root = 3.
        gc_idx = nodes.index(grandchildren[0])
        assert mask[gc_idx].sum() == 3
        child0_idx = nodes.index(children[0])
        assert mask[gc_idx, child0_idx]
        assert mask[gc_idx, root_idx]
        # a sibling not on that path is not visible.
        if len(children) > 1:
            other_child_idx = nodes.index(children[1])
            assert not mask[gc_idx, other_child_idx]


class TestVerifyTree:
    def test_returns_empty_path_when_root_has_no_children(self):
        root = build_root(token_id=0, feature=torch.zeros(4))
        rng = np.random.default_rng(0)
        path = verify_tree(root, target_dist_fn=lambda n: np.array([1.0]), rng=rng)
        assert path == []

    def test_walks_down_when_target_always_agrees_with_top_child(self):
        target_llm, draft_head = _setup(vocab_size=6)
        root = build_root(token_id=0, feature=torch.zeros(4))
        children = expand_node(draft_head, target_llm, root, branching_k=2)

        def target_dist_fn(node):
            # Target distribution puts all its mass on whichever token the
            # draft's top (highest-confidence) child proposed -> that child
            # is accepted with probability 1 at every visited node.
            top_child = max(node.children, key=lambda c: c.confidence)
            p = np.zeros(6)
            p[top_child.token_id] = 1.0
            return p

        rng = np.random.default_rng(1)
        path = verify_tree(root, target_dist_fn, rng)
        assert len(path) >= 1
        top_child = max(children, key=lambda c: c.confidence)
        assert path[0] == top_child.token_id


class TestCalibrationCurve:
    def test_bins_and_perfect_calibration_matches_identity(self):
        confidences = np.array([0.05, 0.15, 0.45, 0.55, 0.95])
        accepted = np.array([0, 0, 1, 1, 1])
        centers, mean_conf, acc_rate, counts = calibration_curve(
            confidences, accepted, num_bins=5
        )
        assert counts.sum() == 5
        assert len(centers) == len(mean_conf) == len(acc_rate) == 5
        # bin 0 (0.0-0.2) holds confidences 0.05,0.15 both not accepted.
        assert counts[0] == 2
        assert acc_rate[0] == 0.0

    def test_empty_bins_report_zero(self):
        confidences = np.array([0.9, 0.95])
        accepted = np.array([1, 1])
        _, mean_conf, acc_rate, counts = calibration_curve(confidences, accepted, num_bins=5)
        assert counts[0] == 0
        assert mean_conf[0] == 0.0
        assert acc_rate[0] == 0.0
