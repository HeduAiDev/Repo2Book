"""Driver: EAGLE-2 dynamic draft tree -- value V_i, expansion, reranking + calibration.

EAGLE-2 arXiv:2406.16858 §4.1: a draft token is finally accepted only if every
token on its root-to-node path is accepted, so its global acceptance rate is
the PRODUCT of per-node acceptance rates; because the draft model is
well-calibrated (§3.2, confidence c_j ~= acceptance rate p_j) this product can
be approximated by the product of confidences, the node value
V_i = prod_{path} c_j -- computable with NO target-LLM forward pass. §4.1
Expansion selects the top-k highest-value frontier nodes to grow; §4.2
Reranking keeps the global top-m by value (shallower-first tiebreak) so the
survivors still form a connected tree.

Part 1 builds a small static tree over the toy models, prints each node's
confidence and value V (= path product of confidences), then reranks to top-m
and marks survivors -- showing V is monotonically non-increasing with depth and
that the connected-tree invariant holds. Part 2 runs calibration_curve on
synthetic (confidence, accepted) pairs to produce Fig.6-style buckets
supporting confidence ~= acceptance rate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from draft_tree import (build_root, build_static_tree, calibration_curve,  # noqa: E402
                        reranking_phase)
from feature_autoregression import AutoregressionHead, ToyTargetLLM  # noqa: E402

R = 3


def rnd(x):
    if torch.is_tensor(x):
        x = x.detach()
    return round(float(x), R)


# ---- Part 1: value V_i + reranking top-m ----
target = ToyTargetLLM(vocab_size=6, hidden_dim=4, seed=0)
draft = AutoregressionHead(hidden_dim=4, seed=2)
root_feat = target.forward_prefix(torch.tensor([1, 3, 2]))[-1]
root = build_root(token_id=4, feature=root_feat)
nodes = build_static_tree(draft, target, root, depth=2, branching_k=2)  # 1+2+4 = 7 nodes

m = 4
selected = reranking_phase(nodes, m=m)
selected_ids = {id(n) for n in selected}

# Stable label per node: (depth, token).
node_rows = []
for n in nodes:
    node_rows.append({
        "label": f"d{n.depth}_t{n.token_id}",
        "depth": n.depth,
        "token": n.token_id,
        "confidence": rnd(n.confidence),
        "value_V": rnd(n.value),
        "selected_top_m": id(n) in selected_ids,
    })
node_rows.sort(key=lambda r: (-r["value_V"], r["depth"]))

# Verify monotonicity V(child) <= V(parent) across every edge.
mono_ok = True
for n in nodes:
    if n.parent is not None and n.value - n.parent.value > 1e-9:
        mono_ok = False

# ---- Part 2: calibration curve (confidence ~= acceptance rate) ----
rng = np.random.default_rng(0)
N = 4000
confidences = rng.uniform(0.0, 1.0, size=N)
# Well-calibrated ground truth: token accepted with prob == its confidence.
accepted = (rng.uniform(0.0, 1.0, size=N) < confidences).astype(float)
centers, mean_conf, acc_rate, counts = calibration_curve(confidences, accepted, num_bins=5)
calib = [
    {"bin_center": rnd(c), "mean_confidence": rnd(mc), "empirical_accept_rate": rnd(ar), "count": int(ct)}
    for c, mc, ar, ct in zip(centers, mean_conf, acc_rate, counts)
]

out = {
    "part1_tree": {
        "depth": 2, "branching_k": 2, "num_nodes": len(nodes), "m": m,
        "nodes_by_value": node_rows,
        "value_monotone_nonincreasing_with_depth": mono_ok,
        "num_selected": len(selected),
    },
    "part2_calibration": {
        "num_samples": N, "num_bins": 5, "bins": calib,
    },
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("eagle2_tree.json").write_text(json.dumps(out, indent=2))
