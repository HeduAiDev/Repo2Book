"""Driver: kvcomp-hash-hamming-selection —— KVComp 落地:LSH hash + 汉明距离 top-k 选块,
must_select_blocks 强制并入首块(sink)与最近块(recent)。dim_in=8, hash_bits=8(1 字节)。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "implementation"))
from kvcomp_hash import (qr_orthogonal_random_weights, HashEncoder,
                         hamming_distance_packed, select_topk_blocks_by_hamming)

W = qr_orthogonal_random_weights(dim_in=8, hash_bits=8, seed=7)
enc = HashEncoder(hash_weights=W)

# query 与 6 个候选块代表向量(8 维);块3 特意构造成与 query 很接近
rng = np.random.default_rng(42)
query = rng.normal(size=8)
keys = rng.normal(size=(6, 8))
keys[3] = query + 0.01 * rng.normal(size=8)     # 与 query 近似 -> 汉明距离应最小

q_hash = enc.compute_hash(query)
k_hashes = enc.compute_hash(keys)

dists = hamming_distance_packed(q_hash[None, :], k_hashes)
TOP_K = 2
must = [0, -1]      # 首块(sink) + 最后一块(recent)
selected = select_topk_blocks_by_hamming(q_hash, k_hashes, top_k=TOP_K, must_select_blocks=must)
sel_set = set(selected.tolist())
num_blocks = k_hashes.shape[0]
forced = {0, num_blocks - 1}
# 汉明 top-k(未含强制项)
topk_only = set(np.argsort(dists)[:TOP_K].tolist())

rows = []
for b in range(num_blocks):
    rows.append({
        "block": b,
        "hamming_dist": int(dists[b]),
        "in_hamming_topk": 1 if b in topk_only else 0,
        "forced_must_select": 1 if b in forced else 0,
        "selected": 1 if b in sel_set else 0,
    })

out = {"params": {"dim_in": 8, "hash_bits": 8, "top_k": TOP_K, "num_blocks": num_blocks,
                  "must_select_blocks": must},
       "selected_blocks": selected.tolist(),
       "rows": rows}
print(json.dumps(out, indent=2, ensure_ascii=False))
with open(os.path.join(os.path.dirname(__file__), "kvcomp_hash.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
