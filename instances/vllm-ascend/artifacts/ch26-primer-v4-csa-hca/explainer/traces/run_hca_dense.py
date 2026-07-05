"""Driver: hca-heavy-compress-dense —— HCA 不重叠重压(Eq.22-23)+ 稠密 MQA(Eq.26)。
n=8, m'=4(真实发行版 m'=128,此处用 4 作可心算的缩小替身),恒等投影。
对比 CSA:HCA 块 i 只用自己的 m' 个 token(无 overlap 借用),之后对全部压缩块做稠密注意力。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from csa_compression import hca_compress_sequence
from shared_mqa_grouped_output import mqa_core_attention

n, m_prime, c = 8, 4, 2
H = np.array([[float(t + 1), float(t + 1)] for t in range(n)])
I2 = np.eye(2)
B = np.zeros((m_prime, c))

C_comp = hca_compress_sequence(H, I2, I2, B, m_prime)   # (2, 2)

# 记录每块的源 token 范围(证明无重叠)
rows = []
num_blocks = n // m_prime
for i in range(num_blocks):
    src_start = i * m_prime
    src_end = i * m_prime + m_prime - 1
    Z_block = H[src_start:src_end + 1, 0]
    e = np.exp(Z_block - np.max(Z_block))
    w = e / e.sum()
    rows.append({
        "block": i,
        "src_token_start": src_start,
        "src_token_end": src_end,
        "n_source_tokens": m_prime,
        "softmax_weights_ch0": [round(float(x), 3) for x in w],
        "C_comp_ch0": round(float(C_comp[i, 0]), 3),
    })

# 稠密 MQA:query 对全部压缩块(无 top-k)
q_heads = np.array([[1.0, 0.0], [0.0, 1.0]])
o = mqa_core_attention(q_heads, C_comp)
mqa_summary = {
    "num_kv_entries_attended": int(C_comp.shape[0]),   # 稠密=全部压缩块
    "o_head0_ch0": round(float(o[0, 0]), 3),
    "o_head1_ch0": round(float(o[1, 0]), 3),
}

out = {"params": {"n": n, "m_prime": m_prime, "c": c, "note_real_m_prime": 128},
       "rows": rows, "dense_mqa": mqa_summary}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "hca_dense.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
