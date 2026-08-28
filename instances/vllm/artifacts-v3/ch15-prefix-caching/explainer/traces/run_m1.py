# ch15 m1 链式哈希驱动：hash_i = H(parent, 本块 tokens, extra_keys)（kv_cache_utils.py:L596-L623）
# + 请求侧增量 hasher 只算新满块（L705-L748）+ 构造尾/append 增量（request.py:L249-L265）
# + NONE_HASH 种子（L99-L114）。PYTHONHASHSEED=0 → NONE_HASH=sha256("0") 可复现。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.hashing import sha256  # noqa: E402
import implementation.kv_cache_utils as kcu  # noqa: E402
from implementation.request import Request  # noqa: E402

# NONE_HASH 是模块级全局，init_none_hash 之后才存在——须以属性访问
kcu.init_none_hash(sha256)
NONE_HASH = kcu.NONE_HASH
hash_block_tokens = kcu.hash_block_tokens
HASHER16 = kcu.get_request_block_hasher(16, sha256)

out = {"params": {
    "hash_block_size": 16,
    "hash_algo": "sha256",
    "seed": "PYTHONHASHSEED=0",
    "none_hash_hex_head": NONE_HASH.hex()[:12],
    "none_hash_len_bytes": len(NONE_HASH),
}}

# --- 1) 增量：哈希随 token 到达，只算新满块 ---
tokens = list(range(37))
req = Request("r", tokens, block_hasher=HASHER16)
events = [{"event": "构造（37 token）", "num_tokens": req.num_tokens,
           "满块数": len(req.block_hashes)}]
req.append_output_token_ids(list(range(37, 43)))  # 43 token 仍 2 满块
events.append({"event": "append 6 token（至 43）", "num_tokens": req.num_tokens,
               "满块数": len(req.block_hashes)})
req.append_output_token_ids(list(range(43, 50)))  # 50 token → 3 满块
events.append({"event": "append 7 token（至 50）", "num_tokens": req.num_tokens,
               "满块数": len(req.block_hashes)})
req.append_output_token_ids(list(range(50, 51)))  # 51 token → 第 4 满块（48-64 边界…不：51>64 才 4 块）
events.append({"event": "append 1 token（至 51）", "num_tokens": req.num_tokens,
               "满块数": len(req.block_hashes)})
out["events"] = events

# --- 2) 链式本体：逐块手算对照（parent 链式 + 首块 parent=NONE_HASH） ---
all_tokens = req.all_token_ids
manual = []
parent = None
for i in range(len(req.block_hashes)):
    blk = all_tokens[i * 16:(i + 1) * 16]
    h = hash_block_tokens(sha256, parent, blk)
    manual.append(h)
    parent = h
out["chain"] = {
    "hasher_equals_manual": req.block_hashes == manual,
    "block0_cover_tokens": "0-15",
    "block1_cover_tokens": "16-31",
    "block2_cover_tokens": "32-47",
    "h0_hex_head": req.block_hashes[0].hex()[:12],
    "h1_hex_head": req.block_hashes[1].hex()[:12],
    "h2_hex_head": req.block_hashes[2].hex()[:12],
    "h0_neq_h1": req.block_hashes[0] != req.block_hashes[1],
    "first_block_parent_is_none_hash_seed":
        req.block_hashes[0] == hash_block_tokens(sha256, None, all_tokens[0:16]),
    "same_block_twice_same_hash":
        hash_block_tokens(sha256, None, all_tokens[0:16])
        == hash_block_tokens(sha256, None, all_tokens[0:16]),
    "change_token_change_hash":
        hash_block_tokens(sha256, None, all_tokens[0:16])
        != hash_block_tokens(sha256, None, [99] + all_tokens[1:16]),
}

# --- 3) 指纹性质：第 0 块改一个 token → 第 1 块哈希也全变（链式传播） ---
same_tail = all_tokens[16:32]
b0_orig = hash_block_tokens(sha256, None, all_tokens[0:16])
b0_diff = hash_block_tokens(sha256, None, [99] + all_tokens[1:16])
out["fingerprint"] = {
    "h1_from_orig_b0_hex_head":
        hash_block_tokens(sha256, b0_orig, same_tail).hex()[:12],
    "h1_from_diff_b0_hex_head":
        hash_block_tokens(sha256, b0_diff, same_tail).hex()[:12],
    "tail_tokens_identical": True,
    "h1_differs_when_b0_differs":
        hash_block_tokens(sha256, b0_orig, same_tail)
        != hash_block_tokens(sha256, b0_diff, same_tail),
    "block1_tokens_range": "16-31",
}

# --- 4) 两请求共享前 32 token：前两个哈希逐一相等、第三个不同 ---
reqA = Request("a", list(range(50)), block_hasher=HASHER16)
reqB = Request("b", list(range(32)) + list(range(200, 218)), block_hasher=HASHER16)
out["two_requests"] = {
    "share_prefix_tokens": 32,
    "hash0_equal": reqA.block_hashes[0] == reqB.block_hashes[0],
    "hash1_equal": reqA.block_hashes[1] == reqB.block_hashes[1],
    "hash2_equal": reqA.block_hashes[2] == reqB.block_hashes[2],
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m1.json"),
          "w", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
