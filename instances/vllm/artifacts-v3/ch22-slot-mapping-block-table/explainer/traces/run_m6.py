# ch22 explainer 驱动脚本 · m6 读腿：flash_attn 穿 block_table 间接寻址（F7 回收）
# 同一请求先写腿落池（slot 直寻址）、再读腿穿表读回（block_table 间接寻址），
# 与「按块表行把物理块拼回逻辑序列」的稠密参照逐元素对拍——写直读间闭合。
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.flash_attn import (  # noqa: E402
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

CPU = torch.device("cpu")
BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, NUM_HEADS = 16, 1, 8, 2


class _AttentionLayer:
    _k_scale = torch.tensor(1.0)
    _v_scale = torch.tensor(1.0)
    _q_scale = torch.tensor(1.0)

    def __init__(self, kv_cache, num_heads=NUM_HEADS):
        self.kv_cache = kv_cache
        self.impl = FlashAttentionImpl(
            num_heads=num_heads, head_size=HEAD_DIM,
            scale=1.0 / math.sqrt(HEAD_DIM),
            num_kv_heads=NUM_KV_HEADS, alibi_slopes=None, sliding_window=None,
            kv_cache_dtype="auto")


torch.manual_seed(0)
pool = torch.zeros(32, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM)
layer = _AttentionLayer(pool)

# 请求 0：块表行 [3, 8, 6]——32 个历史 token 落块 3/8（上一拍写的），
# 本拍 2 个新 token（pos 32/33）落逻辑块 2 → 物理块 6（slot = 6*16+{0,1}）。
BLOCK_TABLE_ROW = [3, 8, 6]
block_table = torch.tensor([BLOCK_TABLE_ROW], dtype=torch.int32)
q = torch.randn(2, NUM_HEADS, HEAD_DIM)
k_new = torch.randn(2, NUM_KV_HEADS, HEAD_DIM)
v_new = torch.randn(2, NUM_KV_HEADS, HEAD_DIM)
slot_mapping = torch.tensor([6 * BLOCK_SIZE + 0, 6 * BLOCK_SIZE + 1],
                            dtype=torch.int64)

# 预置历史：块 3 与块 8 各 16 行真值（当作上一拍写腿落下的 KV）。
hist_k = torch.randn(32, NUM_KV_HEADS, HEAD_DIM)
hist_v = torch.randn(32, NUM_KV_HEADS, HEAD_DIM)
key_cache, value_cache = pool.transpose(1, 2).split(HEAD_DIM, dim=-1)
key_cache[3] = hist_k[:16].reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
key_cache[8] = hist_k[16:32].reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
value_cache[3] = hist_v[:16].reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
value_cache[8] = hist_v[16:32].reshape(BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)

# 写腿：本拍 2 个新 token 的 K/V 落进块 6 前两槽（slot 直寻址散写）。
layer.impl.do_kv_cache_update(layer, k_new, v_new, pool, slot_mapping)

# 读腿：attention 穿表间接寻址读历史 + 新写。
md = FlashAttentionMetadata(
    num_actual_tokens=2, max_query_len=2,
    query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    max_seq_len=34, seq_lens=torch.tensor([34], dtype=torch.int32),
    block_table=block_table, slot_mapping=slot_mapping,
    use_cascade=False, common_prefix_len=0, cu_prefix_query_lens=None,
    prefix_kv_lens=None, suffix_kv_lens=None,
)
out = torch.zeros(2, NUM_HEADS * HEAD_DIM)
layer.impl.forward(layer, q, k_new, v_new, pool, md, output=out)

# 稠密参照：按块表行把块 3/8/6 拼回逻辑序列直接算 attention。
k_hist = torch.cat(
    [key_cache[3][:16], key_cache[8][:16], key_cache[6][:2]], dim=0
).reshape(-1, HEAD_DIM)
v_hist = torch.cat(
    [value_cache[3][:16], value_cache[8][:16], value_cache[6][:2]], dim=0
).reshape(-1, HEAD_DIM)
scale = 1.0 / math.sqrt(HEAD_DIM)
per_query = []
for qi in range(2):
    q_i = q[qi].reshape(NUM_HEADS, HEAD_DIM)[0]
    logits = (k_hist @ q_i) * scale
    logits = logits[: 32 + qi + 1]  # causal
    p = torch.softmax(logits, dim=-1)
    expect = p @ v_hist[: 32 + qi + 1]
    got = out[qi].reshape(NUM_HEADS, HEAD_DIM)[0]
    per_query.append({
        "query": qi,
        "abs_pos": 32 + qi,
        "keys_visible": 32 + qi + 1,
        "max_abs_diff_vs_dense": float((got - expect).abs().max()),
        "out_head0_first4": [round(float(x), 6) for x in got[:4]],
        "expect_head0_first4": [round(float(x), 6) for x in expect[:4]],
    })

# 读腿穿表访问过的物理块（kernel 内景：每个逻辑块号现场查物理块）。
blocks_visited = []
for logical in range(3):
    phys = int(block_table[0, logical])
    rows = min(BLOCK_SIZE, 34 - logical * BLOCK_SIZE)
    blocks_visited.append({
        "logical_block": logical, "physical_block": phys,
        "rows_read": rows,
        "pos_range": [logical * BLOCK_SIZE,
                      logical * BLOCK_SIZE + rows - 1],
    })

out = {
    "mechanism": "m6 读腿：block_table 间接寻址（F7 回收）",
    "trace_source": "run（读腿 op 的 host 镜像：每请求穿 block_table 逐逻辑块 gather）",
    "params": {
        "block_size": BLOCK_SIZE, "seq_len": 34, "num_new_tokens": 2,
        "block_table_row": BLOCK_TABLE_ROW,
        "slots_for_new_tokens": [int(x) for x in slot_mapping],
        "anchor_read": "vllm/v1/attention/backends/flash_attn.py:L1041-L1066",
        "anchor_write": "csrc/libtorch_stable/cache_kernels.cu:L326-L333",
    },
    "blocks_visited_by_read": blocks_visited,
    "write_leg_landing": {
        "slot0": 96, "slot1": 97,
        "block6_row0_is_new_k": bool(torch.equal(
            key_cache[6, 0].reshape(-1), k_new[0].reshape(-1))),
        "block6_row1_is_new_k": bool(torch.equal(
            key_cache[6, 1].reshape(-1), k_new[1].reshape(-1))),
    },
    "per_query_vs_dense_reference": per_query,
    "max_abs_diff_overall": max(
        r["max_abs_diff_vs_dense"] for r in per_query),
}
path = os.path.join(os.path.dirname(__file__), "m6.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m6 trace written:", path)
