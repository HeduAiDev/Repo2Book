#!/usr/bin/env python3
"""ch43 m1 worked example — faithful host-numpy reproduction of tutorial 06's
exp2 online-softmax recurrence (the *else* / non-causal branch of _attn_fwd_inner,
python/tutorials/06-fused-attention.py:L57-L77).

This is NOT the Triton kernel (that needs a GPU); it is a byte-for-byte numeric
mirror of the same scalar recurrence so the (m_i, l_i, acc) walk shown in the
chapter has real, reproducible numbers. We follow the kernel exactly:
  m_i init = -inf          (L158)
  l_i init = +1.0          (L159)  -> first alpha=exp2(-inf)=0 wipes it, see round 1
  acc init = 0             (L160)
  qk_scale = sm_scale * 1.44269504   (L162-163, 1/ln2)
  else-branch per K-block (L69-77):
    m_ij  = max(m_i, max(qk,1) * qk_scale)
    qk    = qk * qk_scale - m_ij
    p     = exp2(qk)
    l_ij  = sum(p)
    alpha = exp2(m_i - m_ij)
    l_i   = l_i*alpha + l_ij
    acc   = acc*alpha + p @ v
    m_i   = m_ij
epilogue (L184-186): m_i += log2(l_i); o = acc / l_i
Toy params kept tiny so a reader can hand-check: HEAD_DIM=4, BLOCK_N=2, two K/V
blocks (N_CTX=4). We trace query row 0. Block 1 is engineered to out-score block 0
so m_i strictly grows and alpha<1 rescaling is actually exercised (non-degenerate).
"""
import json
import math
import numpy as np

R = 4  # display rounding


def r(x):
    if np.isscalar(x) or (hasattr(x, "ndim") and x.ndim == 0):
        return round(float(x), R)
    return [round(float(v), R) for v in np.asarray(x).ravel()]


sm_scale = 0.5
qk_scale = sm_scale * 1.44269504  # kernel L162-163

# query row 0 (HEAD_DIM=4)
q = np.array([1.0, 0.0, 1.0, 0.0])

# two K/V blocks, BLOCK_N=2 each. K columns are keys.
K_blocks = [
    np.array([[1.0, 0.0, 0.0, 0.0],   # k0
              [0.0, 1.0, 0.0, 0.0]]),  # k1
    np.array([[2.0, 0.0, 2.0, 0.0],   # k2  (aligned with q -> big score)
              [0.0, 0.0, 0.0, 0.0]]),  # k3
]
V_blocks = [
    np.array([[1.0, 0.0, 0.0, 0.0],   # v0
              [0.0, 1.0, 0.0, 0.0]]),  # v1
    np.array([[0.0, 0.0, 1.0, 0.0],   # v2
              [0.0, 0.0, 0.0, 1.0]]),  # v3
]

m_i = -math.inf
l_i = 1.0
acc = np.zeros(4)

rounds = []
for b in range(2):
    Kb = K_blocks[b]        # (BLOCK_N, HEAD_DIM)
    Vb = V_blocks[b]        # (BLOCK_N, HEAD_DIM)
    qk = Kb @ q            # (BLOCK_N,)  == q . k_j
    rowmax = float(np.max(qk))
    m_ij = max(m_i, rowmax * qk_scale)
    qk_shift = qk * qk_scale - m_ij
    p = np.exp2(qk_shift)
    l_ij = float(np.sum(p))
    alpha = math.exp2(m_i - m_ij) if math.isfinite(m_i) else 0.0
    l_i = l_i * alpha + l_ij
    acc = acc * alpha + p @ Vb
    rounds.append({
        "round": b + 1,
        "block": b,
        "qk_raw": r(qk),
        "rowmax": r(rowmax),
        "m_ij": r(m_ij),
        "alpha": r(alpha),
        "p": r(p),
        "l_ij": r(l_ij),
        "l_i": r(l_i),
        "acc": r(acc),
    })
    m_i = m_ij

# epilogue
M = m_i + math.log2(l_i)
o = acc / l_i

out = {
    "params": {
        "HEAD_DIM": 4, "BLOCK_M": 2, "BLOCK_N": 2, "N_CTX": 4,
        "n_blocks": 2, "sm_scale": sm_scale,
        "qk_scale": round(qk_scale, 8), "log2e": 1.44269504,
        "m_i_init": "-inf", "l_i_init": 1.0,
    },
    "rounds": rounds,
    "epilogue": {"m_i_final": r(m_i), "l_i_final": r(l_i),
                 "M": r(M), "o": r(o)},
}
print(json.dumps(out, indent=2))
