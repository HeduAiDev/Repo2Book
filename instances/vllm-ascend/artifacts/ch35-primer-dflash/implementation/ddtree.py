"""
A small, paper-faithful reference implementation of DDTree's best-first
draft-tree construction.

PAPER: arXiv:2604.12989 "Accelerating Speculative Decoding with Block
Diffusion Draft Trees" (DDTree) §3-4, Eq.(1)-(8), Algorithm 1. DDTree builds
directly on top of a DFlash-style block-diffusion drafter, treated as a
black box: a single block-diffusion forward pass already produces, for each
of the L future positions, a full marginal distribution q_i(.|c,b) over the
vocabulary (paper.md §3 Eq.2) -- independent of the other positions' sampled
values, since that is exactly why the block can be generated in one forward
pass. Vanilla DFlash only samples and verifies *one* trajectory from these L
marginals; DDTree instead selects, under a fixed node budget B, the B
prefixes with the highest probability under the factorized distribution Q
(Eq.6-8) -- proved optimal for a tractable surrogate of the true expected
acceptance length (Proposition 1/2) -- via an O(B log B) best-first heap
search (Algorithm 1) rather than enumerating the exponentially many
candidate prefixes.

Not modeled here: the target-model tree-attention verification step itself
(§4.4, "ancestor-only attention mask") -- this module only reproduces the
tree *construction* (Eq.4-8, Algorithm 1), which is the part expressible as
a self-contained, paper-faithful numerical procedure independent of any
particular transformer implementation.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field


# PAPER: §4.2 Eq.(7) (prefix probability under the factorized distribution Q)
def prefix_log_prob(rank_tuple: tuple[int, ...], topk_log_probs: list[list[float]]) -> float:
    """
    log q(u|c,b) = sum_{i=1}^{|u|} log q_i(u_i|c,b)  (Eq.7, in log space).

    rank_tuple: 1-indexed ranks (rho_1, ..., rho_d) identifying the depth-d
        prefix (v_1^(rho_1), ..., v_d^(rho_d)) via §4.3's rank-space
        indexing (tokens identified by their probability rank at each
        depth, not by vocabulary id).
    topk_log_probs[depth][rank-1]: log q_{depth+1}(v_{depth+1}^{(rank)} | c, b),
        already sorted descending in rank within each depth (required by
        Algorithm 1's sibling/child expansion).
    """
    return sum(topk_log_probs[depth][rank - 1] for depth, rank in enumerate(rank_tuple))


# PAPER: §4.3 Algorithm 1 (heap entry: a rank-tuple ordered by its score sigma(rho))
@dataclass(order=True)
class _HeapItem:
    neg_score: float
    rank_tuple: tuple[int, ...] = field(compare=False)


# PAPER: §4.3 Algorithm 1 (best-first draft-tree construction)
def best_first_tree(
    topk_log_probs: list[list[float]], budget: int, top_k: int | None = None
) -> list[tuple[int, ...]]:
    """
    Algorithm 1: starting from a max-heap seeded with the depth-1 root
    (rank (1,)), repeatedly pop the highest-scoring rank-tuple, add it to
    the tree, and push its two possible extensions -- the next sibling at
    the same depth (rank_d + 1) and the first child one depth deeper (rank
    1) -- until the tree holds `budget` nodes or the heap is exhausted.

    Proposition 2 shows the optimal tree under node budget B is exactly the
    B highest-probability prefixes; Proposition 3 shows Algorithm 1 recovers
    exactly this set without enumerating all K^L candidates. This
    implementation recomputes each candidate's score by summing
    prefix_log_prob directly rather than the paper's incremental delta
    update (sigma(rho) - log q_d^(rho_d) + log q_d^(rho_d+1)) -- numerically
    identical, since only the rank at one depth differs, just simpler code.

    topk_log_probs: as in prefix_log_prob; topk_log_probs[i] must be sorted
        so that rank 1 has the highest probability at depth i+1.
    budget: node budget B.
    top_k: K, the number of top tokens tracked per depth (defaults to
        len(topk_log_probs[0])).

    Returns the popped rank-tuples in pop order (highest-probability first).
    """
    depths = len(topk_log_probs)
    k_at_depth = top_k if top_k is not None else len(topk_log_probs[0])

    # PAPER: §4.3 (sigma(rho), the log-probability score used to order the heap)
    def score(rho: tuple[int, ...]) -> float:
        return prefix_log_prob(rho, topk_log_probs)

    root = (1,)
    heap = [_HeapItem(-score(root), root)]
    tree: list[tuple[int, ...]] = []
    seen = {root}

    while len(tree) < budget and heap:
        item = heapq.heappop(heap)
        rho = item.rank_tuple
        tree.append(rho)

        depth = len(rho)
        last_rank = rho[-1]
        if last_rank + 1 <= k_at_depth:
            sibling = rho[:-1] + (last_rank + 1,)
            if sibling not in seen:
                seen.add(sibling)
                heapq.heappush(heap, _HeapItem(-score(sibling), sibling))
        if depth < depths:
            child = rho + (1,)
            if child not in seen:
                seen.add(child)
                heapq.heappush(heap, _HeapItem(-score(child), child))

    return tree


# PAPER: §4.2 Proposition 1 / Eq.(8) (expected acceptance length under Q is additive over tree nodes)
def expected_acceptance_length_surrogate(
    tree: list[tuple[int, ...]], topk_log_probs: list[list[float]]
) -> float:
    """
    E_{Y~Q}[alpha_T(Y)] = sum_{u in T} q(u|c,b)  (Eq.8).

    This is the surrogate objective that best_first_tree greedily (and, per
    Proposition 3, optimally) maximizes under a node budget.
    """
    return sum(math.exp(prefix_log_prob(rho, topk_log_probs)) for rho in tree)
