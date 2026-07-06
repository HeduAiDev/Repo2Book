"""Driver: tree verification via multi-round speculative sampling (SpecInfer).

EAGLE arXiv:2401.15077 §3.3 + Appendix A.2 Algorithm 1: a tree draft offers
k sibling candidates at each node. Multi-round speculative sampling tries them
in order against the target distribution p; on rejecting candidate i it does
NOT immediately resample -- it adjusts p <- norm(max(0, p - p_hat_i)) and tries
the NEXT sibling against the adjusted p. Only if all k siblings are rejected
does it sample from the final residual. This is exactly why a *tree* accepts
more than a chain: a rejected first child gives the second child a second
chance instead of ending the round.

Part 1 walks the recursion at one node with k=3 siblings (explicit p_target,
per-sibling p_hat, and U(0,1) draws) so each round's ratio/verdict/adjusted p
is visible, with the first sibling rejected and the second accepted. Part 2
builds a small depth-2 branching-2 tree over the toy models and runs the
library verify_tree end-to-end to report the accepted-path length.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from speculative_sampling import (accept_reject,  # noqa: E402
                                  multi_round_speculative_sampling,
                                  residual_distribution)
from draft_tree import build_root, build_static_tree, verify_tree  # noqa: E402
from feature_autoregression import AutoregressionHead, ToyTargetLLM  # noqa: E402

R = 3


def rnd(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), R)


# ---- Part 1: multi-round recursion at one node (k=3 siblings, vocab 4) ----
p_target = np.array([0.40, 0.35, 0.15, 0.10])
sib_tokens = [2, 0, 1]  # three drafted sibling candidates
sib_dists = [
    np.array([0.10, 0.10, 0.70, 0.10]),  # p_hat for candidate token 2
    np.array([0.55, 0.20, 0.15, 0.10]),  # p_hat for candidate token 0
    np.array([0.20, 0.60, 0.10, 0.10]),  # p_hat for candidate token 1
]
us = [0.90, 0.20, 0.50]  # reject sibling 1, accept sibling 2

rounds = []
p = p_target.copy()
final_token, rounds_tried = None, None
for i, (t_i, p_hat_i, u_i) in enumerate(zip(sib_tokens, sib_dists, us), start=1):
    ratio = min(1.0, float(p[t_i] / p_hat_i[t_i]))
    accepted = accept_reject(p, p_hat_i, t_i, u_i)
    rounds.append({
        "round": i,
        "candidate_token": t_i,
        "p_target_t": rnd(p[t_i]),
        "p_hat_t": rnd(p_hat_i[t_i]),
        "accept_ratio": rnd(ratio),
        "u": u_i,
        "accepted": bool(accepted),
    })
    if accepted:
        final_token, rounds_tried = t_i, i
        break
    p = residual_distribution(p, p_hat_i)

# Cross-check against the library recursion.
lib_token, lib_rounds = multi_round_speculative_sampling(
    p_target.copy(), sib_tokens, sib_dists, us, np.random.default_rng(0)
)
assert (lib_token, lib_rounds) == (final_token, rounds_tried), (lib_token, lib_rounds)

# ---- Part 2: end-to-end tree verify on toy models (accepted-path length) ----
target = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=0)
draft = AutoregressionHead(hidden_dim=4, seed=2)
root_feat = target.forward_prefix(torch.tensor([1, 3, 2]))[-1]
root = build_root(token_id=4, feature=root_feat)
nodes = build_static_tree(draft, target, root, depth=2, branching_k=2)


def target_dist_fn(node):
    return target.next_token_distribution(node.feature).detach().numpy()


rng = np.random.default_rng(7)
accepted_path = verify_tree(root, target_dist_fn, rng)

out = {
    "part1_node": {
        "vocab_size": 4,
        "num_siblings": 3,
        "rounds": rounds,
        "accepted_token": final_token,
        "rounds_tried": rounds_tried,
    },
    "part2_tree": {
        "depth": 2,
        "branching_k": 2,
        "num_tree_nodes": len(nodes),
        "accepted_path": accepted_path,
        "accepted_path_len": len(accepted_path),
    },
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("tree_verify.json").write_text(json.dumps(out, indent=2))
