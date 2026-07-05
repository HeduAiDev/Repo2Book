#!/usr/bin/env python3
"""Explainer driver for ch33 (speculative sampling primer).

Runs the paper-faithful reference implementation
(instances/.../ch33.../implementation/speculative_sampling.py + mtp_module.py)
on tiny hand-checkable toy distributions and dumps one JSON trace per
worked-example mechanism into this directory. Every number that ends up in an
explainer.json worked-example table is emitted here verbatim (rounded to the
displayed precision) so lint_explainer can find it.

Toy distributions over a 4-symbol vocab [A,B,C,D]:
    p (target M_p)      = [0.5, 0.3, 0.1, 0.1]
    q (approx  M_q)     = [0.4, 0.2, 0.3, 0.1]
chosen so beta = sum min(p,q) = 0.8 and the residual p' = [0.5,0.5,0,0] are
clean round numbers a reader can verify by hand.

Run:  cd implementation && PYTHONPATH=. python3 ../explainer/traces/run_specsample.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import speculative_sampling as ss

OUT = Path(__file__).resolve().parent
R = 3  # rounding for displayed numbers


def rnd(x):
    return round(float(x), R)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"wrote {name}")


VOCAB = ["A", "B", "C", "D"]
p = np.array([0.5, 0.3, 0.1, 0.1])
q = np.array([0.4, 0.2, 0.3, 0.1])


# ---------------------------------------------------------------- accept/reject
def trace_accept_reject():
    rng = np.random.default_rng(0)
    rows = []
    for i, tok in enumerate(VOCAB):
        qle = bool(q[i] <= p[i])
        accept_prob = 1.0 if qle else p[i] / q[i]
        rows.append({
            "token": tok,
            "p": rnd(p[i]),
            "q": rnd(q[i]),
            "q<=p": "yes" if qle else "no",
            "accept_min(1,p/q)": rnd(accept_prob),
        })
    beta = ss.acceptance_rate(p, q)
    # Monte-Carlo: draw x~q, accept with prob min(1,p/q); empirical accept freq -> beta.
    N = 400000
    accepts = 0
    for _ in range(N):
        _, acc = ss.propose_and_check(p, q, rng)
        accepts += acc
    emp_beta = accepts / N
    out = {
        "params": {"vocab": VOCAB, "p": [rnd(v) for v in p], "q": [rnd(v) for v in q]},
        "rows": rows,
        "beta_sum_min_pq": rnd(beta),
        "montecarlo": {"N": N, "empirical_accept_freq": rnd(emp_beta)},
    }
    dump("accept_reject.json", out)
    return out


# --------------------------------------------------------------------- residual
def trace_residual():
    rng = np.random.default_rng(1)
    pprime = ss.residual_distribution(p, q)
    norm_const = float(np.maximum(p - q, 0.0).sum())
    rows = []
    for i, tok in enumerate(VOCAB):
        rows.append({
            "token": tok,
            "p-q": rnd(p[i] - q[i]),
            "max(0,p-q)": rnd(max(0.0, p[i] - q[i])),
            "p'=norm": rnd(pprime[i]),
        })
    # Monte-Carlo: force a rejection each draw, resample from p', check empirical p'.
    N = 400000
    counts = np.zeros(len(VOCAB))
    for _ in range(N):
        x_new = int(rng.choice(len(p), p=pprime))
        counts[x_new] += 1
    emp = counts / N
    out = {
        "params": {"vocab": VOCAB, "p": [rnd(v) for v in p], "q": [rnd(v) for v in q]},
        "rows": rows,
        "norm_const": rnd(norm_const),
        "one_minus_beta": rnd(1.0 - ss.acceptance_rate(p, q)),
        "montecarlo": {"N": N, "empirical_pprime": [rnd(v) for v in emp]},
    }
    dump("residual.json", out)
    return out


# ------------------------------------------------------- distribution preserving
def trace_dist_preserving():
    rng = np.random.default_rng(2)
    beta = ss.acceptance_rate(p, q)
    pprime = ss.residual_distribution(p, q)
    # Monte-Carlo the full single-token speculative step; count final tokens.
    N = 400000
    counts = np.zeros(len(VOCAB))
    for _ in range(N):
        tok, _ = ss.speculative_sampling_step(p, q, rng)
        counts[tok] += 1
    emp = counts / N
    rows = []
    for i, tok in enumerate(VOCAB):
        acc_contrib = min(p[i], q[i])
        res_contrib = (1.0 - beta) * pprime[i]
        rows.append({
            "token": tok,
            "min(p,q)": rnd(acc_contrib),
            "(1-beta)p'": rnd(res_contrib),
            "sum=P(x)": rnd(acc_contrib + res_contrib),
            "p(x)_target": rnd(p[i]),
            "MC_freq": rnd(emp[i]),
        })
    out = {
        "params": {"vocab": VOCAB, "p": [rnd(v) for v in p], "q": [rnd(v) for v in q],
                   "beta": rnd(beta)},
        "rows": rows,
        "montecarlo": {"N": N, "empirical_freq": [rnd(v) for v in emp]},
    }
    dump("dist_preserving.json", out)
    return out


# ------------------------------------------------------------------- alpha=E(beta)
def trace_alpha():
    rng = np.random.default_rng(3)
    # two prefixes with different (p,q) -> different beta; alpha = mean(beta).
    pairs = [
        ("prefix1", np.array([0.5, 0.3, 0.1, 0.1]), np.array([0.4, 0.2, 0.3, 0.1])),
        ("prefix2", np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.25, 0.25, 0.25, 0.25])),
    ]
    rows = []
    betas = []
    N = 400000
    for name, pp, qq in pairs:
        beta = ss.acceptance_rate(pp, qq)
        dlk = ss.lukaszyk_karmowski_divergence(pp, qq)
        betas.append(beta)
        acc = 0
        for _ in range(N):
            _, a = ss.propose_and_check(pp, qq, rng)
            acc += a
        rows.append({
            "prefix": name,
            "beta_sum_min": rnd(beta),
            "D_LK=1-beta": rnd(dlk),
            "MC_accept_freq": rnd(acc / N),
        })
    alpha = float(np.mean(betas))
    mean_dlk = float(np.mean([1.0 - b for b in betas]))
    rows.append({
        "prefix": "alpha=E(beta)",
        "beta_sum_min": rnd(alpha),
        "D_LK=1-beta": rnd(mean_dlk),
        "MC_accept_freq": rnd((rows[0]["MC_accept_freq"] + rows[1]["MC_accept_freq"]) / 2),
    })
    out = {"rows": rows, "alpha": rnd(alpha),
           "check_1_minus_E_DLK": rnd(1.0 - mean_dlk)}
    dump("alpha.json", out)
    return out


# ----------------------------------------------------------- expected token count
def trace_expected_length():
    alpha = 0.8
    rows = []
    bound = 1.0 / (1.0 - alpha)
    for gamma in (1, 3, 5, 10):
        e = ss.expected_generated_tokens(alpha, gamma)
        geo = sum(alpha ** k for k in range(gamma + 1))  # cross-check by direct sum
        rows.append({
            "gamma": gamma,
            "E_tokens": rnd(e),
            "geometric_sum_check": rnd(geo),
            "cap_bound_1/(1-a)": rnd(bound),
        })
    out = {"alpha": alpha, "rows": rows}
    dump("expected_length.json", out)
    return out


# ---------------------------------------------------------------- walltime speedup
def trace_walltime():
    rows = []
    # Table 1 rows (c = 0): must reproduce paper's 3.69X / 6.86X.
    for alpha, gamma in ((0.8, 5), (0.9, 10)):
        f = ss.walltime_improvement_factor(alpha, gamma, 0.0)
        rows.append({
            "alpha": alpha, "gamma": gamma, "c": 0.0,
            "speedup": rnd(f), "note": "Table 1",
        })
    # c > 0 case: find optimal gamma and its factor; also Corollary 3.9 lower bound.
    alpha, c = 0.8, 0.05
    g_opt = ss.optimal_gamma(alpha, c)
    f_opt = ss.walltime_improvement_factor(alpha, g_opt, c)
    lower = (1.0 + alpha) / (1.0 + c)
    rows.append({
        "alpha": alpha, "gamma": g_opt, "c": c,
        "speedup": rnd(f_opt), "note": f"optimal gamma; Cor 3.9 lower bound {rnd(lower)}",
    })
    out = {"rows": rows, "cor39_lower_bound_a0.8_c0.05": rnd(lower)}
    dump("walltime.json", out)
    return out


# ------------------------------------------------ mtp causal-chain (figure numbers)
def trace_mtp():
    import torch
    import mtp_module as mm

    torch.manual_seed(0)
    depth, hidden, vocab, T, batch = 3, 8, 16, 6, 1
    model = mm.DeepSeekMTPPredictor(depth=depth, hidden_size=hidden, vocab_size=vocab)
    h_main = torch.randn(batch, T, hidden)
    token_ids = torch.randint(0, vocab, (batch, T))
    outs = model.forward(h_main, token_ids)
    rows = []
    for k, (h_k, logits_k) in enumerate(outs, start=1):
        rows.append({
            "depth_k": k,
            "valid_len": int(h_k.shape[1]),   # T - k, window shrinks by 1 per depth
            "token_offset": k,                 # consumes t_{i+k}
            "h_k_shape": list(h_k.shape),
            "logits_k_shape": list(logits_k.shape),
        })
    shared = (model.modules_by_depth[0].embed is model.modules_by_depth[1].embed
              and model.modules_by_depth[0].out_head is model.modules_by_depth[1].out_head)
    out = {
        "params": {"depth": depth, "hidden": hidden, "vocab": vocab, "T": T},
        "rows": rows,
        "shared_emb_and_head_across_depths": shared,
    }
    dump("mtp_causal_chain.json", out)
    return out


if __name__ == "__main__":
    print("cwd:", os.getcwd())
    trace_accept_reject()
    trace_residual()
    trace_dist_preserving()
    trace_alpha()
    trace_expected_length()
    trace_walltime()
    trace_mtp()
    print("done")
