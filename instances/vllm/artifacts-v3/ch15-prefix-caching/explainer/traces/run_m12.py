# ch15 m12 哈希粒度分离驱动：BlockHashListWithBlockSize 惰性重串——链尾即前缀
# 指纹（kv_cache_utils.py:L2245-L2314，docstring 图例：16 粒度第 2 个哈希直接当
# 32 粒度用）；resolve_block_hashes 细粒度查找保留原始列表（L2321-L2351）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.request import Request  # noqa: E402

kcu.init_none_hash(sha256)
HASHER16 = kcu.get_request_block_hasher(16, sha256)

# 真实哈希：64 token、hash_block_size=16 → 4 个细粒度哈希
req = Request("r", list(range(64)), block_hasher=HASHER16)
raw = list(req.block_hashes)   # h0@16, h1@32, h2@48, h3@64（每个链住整条前缀）

out = {"params": {"hash_block_size": 16, "prompt_tokens": 64,
                  "fine_hashes": 4,
                  "fine_boundaries_tokens": [16, 32, 48, 64]}}

# --- 1) 惰性重串：粗块视图直接复用块内最后一个细粒度哈希（零重算） ---
view32 = kcu.BlockHashListWithBlockSize(raw, 16, 32)
view64 = kcu.BlockHashListWithBlockSize(raw, 16, 64)
out["restring"] = {
    "raw_indices": [0, 1, 2, 3],
    "raw_covers_tokens": ["0-15", "16-31", "32-47", "48-63"],
    "view32_len": len(view32),
    "view32_uses_raw_indices": [1, 3],
    "view32_block0_covers_tokens": "0-31",
    "view32_block1_covers_tokens": "32-63",
    "view32_block0_is_raw_h1": view32[0] is raw[1],
    "view32_block1_is_raw_h3": view32[1] is raw[3],
    "view64_len": len(view64),
    "view64_uses_raw_indices": [3],
    "view64_block0_covers_tokens": "0-63",
    "view64_block0_is_raw_h3": view64[0] is raw[3],
    "recomputation_count": 0,
    "note": "粗块哈希 = 块内最后一个细粒度哈希——它已链住整条前缀，直接当粗块指纹用",
}

# --- 2) resolve_block_hashes：等粒度直用 / 细粒度保留原始 / 粗粒度包视图 ---
fine = kcu.resolve_block_hashes(raw, 16, 32,
                                supports_fine_grained_hash_lookup=True,
                                alignment_tokens=16)
coarse = kcu.resolve_block_hashes(raw, 16, 32,
                                  supports_fine_grained_hash_lookup=True,
                                  alignment_tokens=32)
same = kcu.resolve_block_hashes(raw, 16, 16)
out["resolve"] = {
    "fine_keeps_raw_list": fine is raw,
    "fine_alignment_tokens": 16,
    "coarse_is_view": coarse is not raw and isinstance(
        coarse, kcu.BlockHashListWithBlockSize),
    "coarse_alignment_tokens": 32,
    "same_granularity_reuse": same is raw,
    "note": "supports_fine_grained_hash_lookup 且 alignment<block_size → 保留原始"
            "细粒度列表供 phase 2 块内探测；否则包成粗视图",
}

# --- 3) 指纹同一性的可观察面：view32[0] 查表 == raw[1] 查表 ---
pool_hashes_equal = view32[0] == raw[1]
out["identity"] = {
    "view32_block0_equals_raw_h1": pool_hashes_equal,
    "tokens_fingerprinted_by_both": 32,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m12.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
