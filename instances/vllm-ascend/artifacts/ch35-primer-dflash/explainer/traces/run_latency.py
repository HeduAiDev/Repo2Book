"""Driver: speculative speedup / latency model (Eq.1-3). Two levers on
eta = L_target / L, L = (T_draft + T_verify) / tau:
  (a) block-diffusion drafting cuts T_draft (Eq.3 vs Eq.2),
  (b) KV injection raises tau.
Runs the reference impl and dumps every number used in explainer.json.
"""
from __future__ import annotations
import json

from latency_model import per_token_latency, speedup, speedup_for_mode

# Toy units: target model's own per-token AR latency = 1.0 (the yardstick).
l_target = 1.0
t_verify = 1.0     # one parallel target verification pass over the block
t_step = 0.2       # one tiny AR draft-model forward
t_parallel = 0.5   # one 5-layer block-diffusion forward over the whole block
gamma = 8

scenarios = [
    # (label, mode, tau) — same gamma/t_verify/l_target throughout.
    ("AR draft, EAGLE-style tau",        "autoregressive", 3.0),
    ("diffusion draft, same tau",        "diffusion",      3.0),
    ("diffusion draft + KV-injection tau", "diffusion",    4.2),
]

rows = []
for label, mode, tau in scenarios:
    if mode == "autoregressive":
        t_draft = round(gamma * t_step, 4)
    else:
        t_draft = round(t_parallel, 4)
    L = round(per_token_latency(t_draft, t_verify, tau), 4)
    eta = round(speedup_for_mode(mode, gamma, t_step, t_parallel, t_verify, tau, l_target), 4)
    rows.append({"label": label, "mode": mode, "gamma": gamma, "tau": tau,
                 "T_draft": t_draft, "T_verify": t_verify, "L": L, "eta": eta})

# sanity: eta = l_target / L
for r in rows:
    assert abs(r["eta"] - round(l_target / r["L"], 4)) < 1e-3, r

out = {
    "params": {"l_target": l_target, "t_verify": t_verify, "t_step": t_step,
               "t_parallel": t_parallel, "gamma": gamma},
    "rows": rows,
}
print(json.dumps(out, indent=2))
with open("latency.json", "w") as f:
    json.dump(out, f, indent=2)
