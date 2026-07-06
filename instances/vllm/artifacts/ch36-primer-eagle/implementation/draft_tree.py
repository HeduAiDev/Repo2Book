"""
PAPER: arXiv:2401.15077 EAGLE §3.1 (tree-structured draft via tree
attention, Fig.6/Appendix A.1 static k-ary tree) + §3.3 (verification via
recursive multi-round speculative sampling, consistent with SpecInfer) and
arXiv:2406.16858 EAGLE-2 §4.1 (value V_i = product of path confidences,
selective expansion of the top-k highest-value nodes) + §4.2 (reranking:
top-m by value with shallower-first tiebreak, then a tree attention mask so
each token only sees its ancestors) + §3.2 (well-calibrated: draft-model
confidence approximates the true acceptance rate, which is *why* V_i can be
approximated by a product of confidences instead of requiring a target-LLM
forward pass to know the true per-node acceptance rate p_j).

A minimal tree data structure over the toy AutoregressionHead/ToyTargetLLM
(feature_autoregression.py), small enough (<=3 layers, <=4 children) to
hand-trace: each node holds a drafted token, the feature that produced it,
its *local* confidence (draft model's own softmax probability for that
token, §3.2's c_j), and its *value* V_i (the path product, §4.1's Eq.).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from feature_autoregression import AutoregressionHead, ToyTargetLLM


@dataclass
class TreeNode:
    token_id: int
    feature: torch.Tensor
    confidence: float  # PAPER: EAGLE-2 §3.2 -- c_j, the draft model's own prob
    parent: "TreeNode | None"
    depth: int
    value: float = 0.0  # PAPER: EAGLE-2 §4.1 -- V_i, filled by compute_value
    children: list = field(default_factory=list)


# PAPER: EAGLE-2 §4.1 Eq. (V_i = prod_{t_j in Path(root,t_i)} p_j ~= prod c_j)
def compute_value(node: TreeNode) -> float:
    """
    Value = product of confidence scores along the path from root to
    `node`, approximating the true global acceptance rate
    (prod of true per-node acceptance rates p_j) by the draft model's own
    confidence (prod of c_j) -- justified by the §3.2 well-calibrated
    observation that confidence tracks acceptance rate closely. The root's
    confidence is fixed to 1.0 (see build_root), so V_root = 1.0, the
    maximum possible value.
    """
    v = 1.0
    cur: TreeNode | None = node
    while cur is not None:
        v *= cur.confidence
        cur = cur.parent
    node.value = v
    return v


# PAPER: EAGLE-2 §4.1 (a node's confidence is fixed at 1.0 for the tree root,
# the already-accepted token the draft continues from)
def build_root(token_id: int, feature: torch.Tensor) -> TreeNode:
    root = TreeNode(token_id=token_id, feature=feature, confidence=1.0, parent=None, depth=0)
    compute_value(root)
    return root


# PAPER: EAGLE §3.1/Appendix A.1 (top-k children per expanded node)
def expand_node(
    draft_head: AutoregressionHead,
    target_llm: ToyTargetLLM,
    node: TreeNode,
    branching_k: int,
) -> list[TreeNode]:
    """
    Expands one node into up to `branching_k` children: feeds
    (node.token_id's embedding, node.feature) through the draft head to get
    one next feature, reads its distribution via the shared LM Head, and
    keeps the top `branching_k` tokens by probability as sibling children
    ("we select the top-k tokens with the highest probabilities as child
    nodes", EAGLE Appendix A.1; EAGLE-2 reuses the same per-node expansion
    mechanics, §4.1).
    """
    token_embed = target_llm.embed_tokens(torch.tensor(node.token_id))
    next_feature = draft_head(token_embed, node.feature)
    probs = target_llm.next_token_distribution(next_feature)
    k = min(branching_k, probs.numel())
    top_probs, top_tokens = torch.topk(probs, k=k)
    children: list[TreeNode] = []
    for prob, tok in zip(top_probs.tolist(), top_tokens.tolist()):
        child = TreeNode(
            token_id=int(tok),
            feature=next_feature,
            confidence=float(prob),
            parent=node,
            depth=node.depth + 1,
        )
        compute_value(child)
        children.append(child)
    node.children.extend(children)
    return children


# PAPER: EAGLE §3.1/Appendix A.1 -- static tree: expand every node every layer
def build_static_tree(
    draft_head: AutoregressionHead,
    target_llm: ToyTargetLLM,
    root: TreeNode,
    depth: int,
    branching_k: int,
) -> list[TreeNode]:
    """
    Builds a full k-ary tree of the given depth: EAGLE-1's *static* tree
    structure (every node at every layer is expanded, in contrast with
    EAGLE-2's selective expansion in `expansion_phase` below). Returns all
    nodes, including `root`, in BFS order.
    """
    all_nodes = [root]
    frontier = [root]
    for _ in range(depth):
        next_frontier: list[TreeNode] = []
        for node in frontier:
            next_frontier.extend(expand_node(draft_head, target_llm, node, branching_k))
        all_nodes.extend(next_frontier)
        frontier = next_frontier
    return all_nodes


# PAPER: EAGLE-2 §4.1 Expansion Phase -- expand only the top-k highest-value nodes
def expansion_phase(
    draft_head: AutoregressionHead,
    target_llm: ToyTargetLLM,
    last_layer: list[TreeNode],
    k: int,
    branching_k: int,
) -> list[TreeNode]:
    """
    "We choose the top-k tokens with the highest global acceptance
    probabilities from the current layer for expansion" (§4.1). Only the k
    highest-value nodes of `last_layer` get expanded -- contrast with
    `build_static_tree`'s blanket expansion of every node.
    """
    ranked = sorted(last_layer, key=lambda n: n.value, reverse=True)[:k]
    new_nodes: list[TreeNode] = []
    for node in ranked:
        new_nodes.extend(expand_node(draft_head, target_llm, node, branching_k))
    return new_nodes


# PAPER: EAGLE-2 §4.2 (connectivity invariant of the reranked top-m selection)
def _assert_connected(selected: list[TreeNode]) -> None:
    """
    PAPER: EAGLE-2 §4.2 invariant -- "the value of a node is always less
    than or equal to that of its parent node" (confidences are in [0,1]),
    which together with the shallower-first tiebreak guarantees that a
    parent's value is never ranked below its child's, so top-m by
    (-value, depth) always selects a node's parent whenever it selects the
    node itself. Checked defensively here, not re-derived.
    """
    selected_ids = {id(n) for n in selected}
    for n in selected:
        if n.parent is not None:
            assert id(n.parent) in selected_ids, (
                f"reranking_phase produced a disconnected tree: node at "
                f"depth {n.depth} (token {n.token_id}) selected without its parent"
            )


# PAPER: EAGLE-2 §4.2 Reranking Phase -- top-m by value, shallower-first tiebreak
def reranking_phase(all_nodes: list[TreeNode], m: int) -> list[TreeNode]:
    """
    "We rerank all draft tokens and select the top m tokens with the
    highest values... For nodes with the same value, we prioritize
    selecting shallower nodes. This ensures that the top m tokens selected
    after reranking still form a connected tree." (§4.2)
    """
    ranked = sorted(all_nodes, key=lambda n: (-n.value, n.depth))
    selected = ranked[:m]
    _assert_connected(selected)
    return selected


# PAPER: EAGLE-2 §4.2 (attention mask s.t. each token sees only its ancestors)
def build_tree_attention_mask(nodes: list[TreeNode]) -> np.ndarray:
    """
    Flattens `nodes` (already reranked) into the tree-attention mask used at
    verification time: mask[i, j] = True iff node j lies on the path from
    the root to node i (i.e. j is an ancestor of i, or j == i) --
    "the attention mask must be adjusted according to the tree structure to
    ensure that each token can only see its ancestor nodes" (§4.2). This
    mirrors the flattened mask that replaces vanilla causal attention's full
    lower-triangular matrix once the draft is a tree instead of a chain.
    """
    n = len(nodes)
    index_of = {id(node): i for i, node in enumerate(nodes)}
    mask = np.zeros((n, n), dtype=bool)
    for i, node in enumerate(nodes):
        cur: TreeNode | None = node
        while cur is not None:
            j = index_of.get(id(cur))
            if j is None:
                break  # ancestor pruned by reranking -- ruled out by _assert_connected
            mask[i, j] = True
            cur = cur.parent
    return mask


# PAPER: Appendix A.2 Algorithm 1 (reconstructs the p_hat argument that
# algorithm's recursion needs from the single confidence scalar this toy
# tree stores per candidate -- glue, not itself a paper equation)
def _one_hot_like(token: int, confidence: float, vocab_size: int) -> np.ndarray:
    """
    Reconstructs a toy p_hat distribution for a sibling candidate from the
    single confidence scalar the tree stores (this reference tree only keeps
    each candidate's top-1 confidence, not its full vocabulary
    distribution): puts `confidence` mass on `token`, spreads the rest
    uniformly. Not a real softmax output -- just enough structure to
    exercise `accept_reject`/`residual_distribution` with the one number the
    tree already recorded.
    """
    p_hat = np.full(vocab_size, (1.0 - confidence) / max(vocab_size - 1, 1))
    p_hat[token] = confidence
    return p_hat


# PAPER: EAGLE §3.3 (recursive multi-round speculative sampling, consistent with SpecInfer)
def verify_tree(root: TreeNode, target_dist_fn, rng: np.random.Generator):
    """
    Depth-first tree verification (§3.3): at each node, gather its
    children as sibling candidates, obtain the target LLM's distribution
    `target_dist_fn(node)` for that tree position (standing in for indexing
    into the single tree-attention verification forward pass, §3.3), and run
    `speculative_sampling.multi_round_speculative_sampling`
    (Appendix A.2 Algorithm 1) among the siblings. Recurse into the accepted
    child's subtree; stop at the first node whose accepted token doesn't
    match any drafted child (a resampled/bonus token with no further tree to
    descend into).

    Returns the accepted path as a list of token ids (root's own token is
    excluded -- root is the already-accepted token this draft round started
    from).
    """
    from speculative_sampling import multi_round_speculative_sampling

    accepted_path: list[int] = []
    node = root
    siblings = node.children
    while siblings:
        candidate_tokens = [c.token_id for c in siblings]
        p_target = target_dist_fn(node)
        candidate_dists = [_one_hot_like(c.token_id, c.confidence, len(p_target)) for c in siblings]
        us = rng.uniform(0.0, 1.0, size=len(siblings))
        token, _rounds = multi_round_speculative_sampling(
            p_target, candidate_tokens, candidate_dists, us, rng
        )
        accepted_path.append(token)
        match = next((c for c in siblings if c.token_id == token), None)
        if match is None:
            break
        node = match
        siblings = node.children
    return accepted_path


# PAPER: EAGLE-2 §3.2 (well-calibrated: confidence vs. measured acceptance rate)
def calibration_curve(
    confidences: np.ndarray, accepted: np.ndarray, num_bins: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Buckets (confidence, accepted) pairs into `num_bins` equal-width bins
    over [0,1] and returns (bin_centers, mean_confidence_per_bin,
    empirical_acceptance_rate_per_bin, counts_per_bin) -- the same style of
    check as Fig.6 (confidence vs. measured acceptance rate), used to
    numerically support "there is a strong positive correlation between the
    draft model's confidence score and the acceptance rate of the token"
    (§3.2). Empty bins report 0.0 for both means.
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1]), 0, num_bins - 1)
    mean_conf = np.zeros(num_bins)
    acc_rate = np.zeros(num_bins)
    counts = np.zeros(num_bins, dtype=int)
    for b in range(num_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_conf[b] = float(confidences[mask].mean())
            acc_rate[b] = float(accepted[mask].mean())
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, mean_conf, acc_rate, counts
