"""Driver: DDTree best-first draft-tree construction (Algorithm 1) on the
per-position marginals of one block-diffusion pass. Verifies:
  (1) best-first pop order == the B highest-probability prefixes (Prop 2/3),
  (2) the tree's expected-acceptance-length surrogate (Eq.8) beats vanilla
      DFlash's single greedy trajectory.
depths=3, top_k=2, budget=4. Dumps every number used in explainer.json.
"""
from __future__ import annotations
import itertools
import json
import math

from ddtree import (
    best_first_tree,
    expected_acceptance_length_surrogate,
    prefix_log_prob,
)

# Per-position top-2 marginals from one block-diffusion forward pass.
# depth 0 (pos1): [0.6, 0.4]; depth 1: [0.7, 0.3]; depth 2: [0.8, 0.2].
probs = [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]]
topk_log_probs = [[math.log(p) for p in row] for row in probs]
depths = 3
top_k = 2
budget = 4

tree = best_first_tree(topk_log_probs, budget=budget, top_k=top_k)

# per-pop record: prefix prob and running surrogate
pop_rows = []
running = 0.0
for rho in tree:
    p = round(math.exp(prefix_log_prob(rho, topk_log_probs)), 4)
    running = round(running + p, 4)
    pop_rows.append({"rank_tuple": list(rho), "depth": len(rho),
                     "prefix_prob": p, "running_surrogate": running})

tree_surrogate = round(expected_acceptance_length_surrogate(tree, topk_log_probs), 4)

# (1) brute-force top-`budget` prefixes over all rank tuples up to depth L
all_prefixes = []
for d in range(1, depths + 1):
    for combo in itertools.product(range(1, top_k + 1), repeat=d):
        all_prefixes.append((combo, math.exp(prefix_log_prob(combo, topk_log_probs))))
all_prefixes.sort(key=lambda x: -x[1])
brute_topB = sorted([list(c) for c, _ in all_prefixes[:budget]])
tree_sorted = sorted([list(t) for t in tree])
optimal_match = (brute_topB == tree_sorted)

# (2) vanilla single greedy trajectory = always rank 1 => chain (1,),(1,1),(1,1,1)
single_chain = [(1,), (1, 1), (1, 1, 1)]
single_surrogate = round(expected_acceptance_length_surrogate(single_chain, topk_log_probs), 4)

out = {
    "params": {"depths": depths, "top_k": top_k, "budget": budget, "marginals": probs},
    "pop_rows": pop_rows,
    "tree_surrogate": tree_surrogate,
    "brute_force_topB_prefixes": brute_topB,
    "best_first_matches_bruteforce": optimal_match,
    "num_nodes_returned": len(tree),
    "single_trajectory_chain_surrogate": single_surrogate,
}
print(json.dumps(out, indent=2))
assert optimal_match, "best-first tree must equal brute-force top-B"
with open("ddtree.json", "w") as f:
    json.dump(out, f, indent=2)
