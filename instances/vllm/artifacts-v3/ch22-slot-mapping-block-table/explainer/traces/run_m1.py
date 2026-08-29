# ch22 explainer 驱动脚本 · m1 槽位恒等式与逆分解
# 参数与 dossier.theory[0] 的 worked-example 算术底座一致：
#   block_size=16、请求 prompt 100 token、第 100 个 token（pos=99）落在
#   第 99//16=6 个逻辑块的第 99%16=3 槽；块表行第 6 项=物理块 9 → slot=147。
# 运行：host python（精简版 BlockTable 的 CPU 镜像 = kernel 本体的逐行镜像，
# 恒等式/PAD 尾与 Triton kernel 逐字同构，见 implementation/block_table.py
# HOST SEAM 注）。写腿核验用 reshape_and_cache_flash 的 host 镜像
# （cache_kernels.cu kernel 本体的逐 token 镜像）。
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.block_table import BlockTable  # noqa: E402
from implementation.flash_attn import reshape_and_cache_flash  # noqa: E402

CPU = torch.device("cpu")
BLOCK_SIZE = 16
# 7 个逻辑块：行内第 6 项刻意放物理块 9（与 dossier 例子一致）。
ROW = [3, 8, 2, 7, 1, 5, 9]
NUM_TOKENS = 100

bt = BlockTable(
    block_size=BLOCK_SIZE,
    max_num_reqs=4,
    max_num_blocks_per_req=16,
    max_num_batched_tokens=128,
    pin_memory=False,
    device=CPU,
    kernel_block_size=BLOCK_SIZE,
    cp_kv_cache_interleave_size=1,
)
bt.add_row(ROW, row_idx=0)
bt.commit_block_table(1)

positions = torch.arange(NUM_TOKENS, dtype=torch.int64)
qsl = torch.tensor([0, NUM_TOKENS], dtype=torch.int32)
bt.compute_slot_mapping(1, qsl, positions)
slots = bt.slot_mapping.gpu

# 逐 token 记录：正向换算（kernel 侧）与逆向分解（写侧 CUDA kernel 的算法）
SAMPLED_POS = [0, 1, 15, 16, 17, 32, 80, 99]
per_token = []
for pos in SAMPLED_POS:
    logical = pos // BLOCK_SIZE
    off = pos % BLOCK_SIZE
    phys = int(bt.block_table.np[0, logical])
    slot = int(slots[pos])
    inv_block = slot // BLOCK_SIZE
    inv_off = slot % BLOCK_SIZE
    per_token.append({
        "pos": pos,
        "logical_block": logical,
        "offset_in_block": off,
        "row_entry_physical_block": phys,
        "slot": slot,
        "inverse_block": inv_block,
        "inverse_offset": inv_off,
        "round_trip_ok": bool(inv_block == phys and inv_off == off),
    })

# 写腿核验：把 slot_mapping 前缀喂给 reshape_and_cache_flash（kernel 镜像），
# 验证 K 行确实落在 逆分解出的 (物理块, 块内行) 上——正逆闭合的运行时证据。
# 池布局 [num_blocks, num_kv_heads, block_size, 2*head_dim]（FA 主流布局）。
NUM_BLOCKS_POOL, KV_HEADS, HEAD_DIM = 12, 1, 4
pool = torch.zeros(NUM_BLOCKS_POOL, KV_HEADS, BLOCK_SIZE, 2 * HEAD_DIM)
key_cache, value_cache = pool.transpose(1, 2).split(HEAD_DIM, dim=-1)
key = (torch.arange(NUM_TOKENS * KV_HEADS * HEAD_DIM, dtype=torch.float32)
       .reshape(NUM_TOKENS, KV_HEADS, HEAD_DIM))
value = key + 1000.0
reshape_and_cache_flash(key, value, key_cache, value_cache,
                        slots[:NUM_TOKENS], "auto", None, None)
write_leg = []
for r in per_token:
    pos = r["pos"]
    landed = bool(torch.equal(
        key_cache[r["row_entry_physical_block"], r["offset_in_block"]]
        .reshape(-1),
        key[pos].reshape(-1)))
    write_leg.append({
        "pos": pos,
        "slot": r["slot"],
        "landed_block": r["row_entry_physical_block"],
        "landed_row": r["offset_in_block"],
        "kv_landed_exact": landed,
    })

# 整行核对（100 个 token 全部服从恒等式）
all_ok = all(
    int(slots[p]) == int(bt.block_table.np[0, p // BLOCK_SIZE]) * BLOCK_SIZE
    + p % BLOCK_SIZE
    for p in range(NUM_TOKENS)
)

out = {
    "mechanism": "m1 槽位恒等式与逆分解",
    "trace_source": "run（精简版 host 镜像；kernel 本体 = block_table.py:L410-L442 与 cache_kernels.cu:L326-L333 的逐行同构镜像）",
    "params": {
        "block_size": BLOCK_SIZE,
        "kv_cache_block_size": int(bt.kv_cache_block_size),
        "blocks_per_kv_block": int(bt.blocks_per_kv_block),
        "num_tokens": NUM_TOKENS,
        "block_table_row": ROW,
        "num_row_entries": len(ROW),
        "max_num_batched_tokens": 128,
    },
    "per_token": per_token,
    "write_leg_verification": write_leg,
    "identity_holds_all_100_tokens": bool(all_ok),
    "slot_array_first_8": [int(x) for x in slots[:8]],
    "slot_array_sample_90_100": [int(x) for x in slots[90:100]],
    "dossier_example": {
        "pos": 99, "logical": 6, "offset": 3, "row_entry": 9,
        "slot": int(slots[99]),
        "check": f"{9}*{16}+{3}={int(slots[99])}",
    },
}
path = os.path.join(os.path.dirname(__file__), "m1.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m1 trace written:", path)
