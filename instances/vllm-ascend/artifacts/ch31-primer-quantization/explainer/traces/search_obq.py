"""Search for a small OBQ worked-example where second-order compensation
genuinely flips a rounding decision and reduces Eq.1 output error vs RTN."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from gptq import (hessian_from_activations, make_asymmetric_per_row_quantizer,
                  obq_pick_and_compensate, remove_hessian_row_col, reconstruction_error)


def obq_row(w_row, H, quant_fn):
    Hinv = np.linalg.inv(H)
    w = w_row.copy()
    idx = list(range(len(w_row)))
    final = np.zeros(len(w_row))
    for _ in range(len(w_row)):
        q_local, _, w_after = obq_pick_and_compensate(Hinv, w, quant_fn)
        final[idx[q_local]] = w_after[q_local]
        w = np.delete(w_after, q_local)
        Hinv = remove_hessian_row_col(Hinv, q_local)
        del idx[q_local]
    return final


best = None
for seed in range(3000):
    rng = np.random.default_rng(seed)
    d = 3
    T = 4
    n_bits = 2
    # strongly correlated activation channels -> big Hessian off-diagonals
    base = rng.normal(size=(1, T))
    X = base + 0.15 * rng.normal(size=(d, T)) + rng.uniform(0.6, 1.0, (d, 1))
    w_row = rng.uniform(-1, 1, d)
    H = hessian_from_activations(X)
    if np.linalg.cond(H) > 1e4:
        continue
    qfn = make_asymmetric_per_row_quantizer(w_row.reshape(1, -1), n_bits)
    W = w_row.reshape(1, -1)
    rtn = qfn(w_row)
    obq = obq_row(w_row, H, qfn)
    if np.allclose(rtn, obq):
        continue  # need a flipped decision
    e_rtn = reconstruction_error(W, rtn.reshape(1, -1), X)
    e_obq = reconstruction_error(W, obq.reshape(1, -1), X)
    if e_obq > 0:
        ratio = e_rtn / e_obq
        if 2.0 <= ratio <= 6.0:
            # prefer the most modest, believable win (closest to ~3x)
            score = abs(ratio - 3.0)
            if best is None or score < best[0]:
                best = (score, seed, w_row, X, e_rtn, e_obq, rtn, obq)

if best:
    ratio, seed, w_row, X, e_rtn, e_obq, rtn, obq = best
    print(f"seed={seed} ratio={ratio:.3f}")
    print("w_row=", np.round(w_row, 3).tolist())
    print("X=", np.round(X, 3).tolist())
    print("rtn=", np.round(rtn, 4).tolist(), "obq=", np.round(obq, 4).tolist())
    print(f"e_rtn={e_rtn:.5f} e_obq={e_obq:.5f}")
else:
    print("none found")
