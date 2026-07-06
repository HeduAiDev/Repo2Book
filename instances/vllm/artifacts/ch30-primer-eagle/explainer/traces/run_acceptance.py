"""Driver: speculative-sampling acceptance criterion + expected accept length.

EAGLE arXiv:2401.15077 §2 (reused from Leviathan et al. 2023): a drafted token
t_hat is accepted with probability min(1, p(t_hat)/p_hat(t_hat)); on rejection
the position is resampled from the residual norm(max(0, p - p_hat)) and all
later drafts are discarded. The distribution-preservation PROOF is ch28 §28.5
-- not repeated here; this driver only walks one concrete chain to make the
accept/reject arithmetic and the expected accepted length concrete.

Setup: vocab of 4, a chain of gamma=3 greedy-drafted tokens. Per position we
show the drafted token, p_hat(t_hat), p(t_hat), the acceptance ratio
min(1, p/p_hat), a fixed U(0,1) draw u, and the accept/reject verdict. A
rejection at position 3 shows the residual distribution and the resample.
Expected accepted draft length = sum_{k=1..gamma} prod_{i<=k} alpha_i with
alpha_i = min(1, p_i(t_hat_i)/p_hat_i(t_hat_i)); +1 bonus token per target pass.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from speculative_sampling import accept_reject, residual_distribution  # noqa: E402

R = 3


def rnd(x):
    return round(float(x), R)


# Three positions: (p_target, p_hat_draft) over a vocab of 4.
positions = [
    {"p": np.array([0.10, 0.60, 0.20, 0.10]), "p_hat": np.array([0.05, 0.70, 0.15, 0.10])},
    {"p": np.array([0.50, 0.20, 0.20, 0.10]), "p_hat": np.array([0.55, 0.15, 0.20, 0.10])},
    {"p": np.array([0.20, 0.20, 0.10, 0.50]), "p_hat": np.array([0.10, 0.10, 0.05, 0.75])},
]
# Fixed U(0,1) draws: chosen so positions 1,2 accept and position 3 rejects.
us = [0.30, 0.50, 0.90]

rows = []
alphas = []
stopped = False
for i, (pos, u) in enumerate(zip(positions, us), start=1):
    p, p_hat = pos["p"], pos["p_hat"]
    t_hat = int(np.argmax(p_hat))
    ratio = min(1.0, float(p[t_hat] / p_hat[t_hat]))
    alphas.append(ratio)
    accepted = accept_reject(p, p_hat, t_hat, u)
    rows.append({
        "position": i,
        "draft_token": t_hat,
        "p_hat_t": rnd(p_hat[t_hat]),
        "p_t": rnd(p[t_hat]),
        "accept_ratio": rnd(ratio),
        "u": u,
        "accepted": bool(accepted),
    })
    if not accepted:
        stopped = True
        resid = residual_distribution(p, p_hat)
        resample_tok = int(np.argmax(resid))
        rejection = {
            "position": i,
            "residual_dist": [rnd(v) for v in resid],
            "resampled_token": resample_tok,
            "residual_mass_on_resampled": rnd(resid[resample_tok]),
        }
        break

# Expected accepted draft length E = sum_k prod_{i<=k} alpha_i (all gamma positions).
all_alphas = []
for pos in positions:
    t_hat = int(np.argmax(pos["p_hat"]))
    all_alphas.append(min(1.0, float(pos["p"][t_hat] / pos["p_hat"][t_hat])))
E_accept = 0.0
prefix = 1.0
prefix_products = []
for a in all_alphas:
    prefix *= a
    prefix_products.append(rnd(prefix))
    E_accept += prefix
out = {
    "vocab_size": 4,
    "gamma": 3,
    "rows": rows,
    "rejection": rejection,
    "alphas": [rnd(a) for a in all_alphas],
    "prefix_products": prefix_products,
    "expected_accepted_draft_tokens": rnd(E_accept),
    "expected_tokens_per_target_pass": rnd(E_accept + 1.0),
}
print(json.dumps(out, indent=2))
Path(__file__).with_name("acceptance.json").write_text(json.dumps(out, indent=2))
