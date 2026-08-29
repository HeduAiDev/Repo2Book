# ch22 explainer 驱动脚本 · m8 CP（上下文并行）分片烤进 kernel
# W=2、I=2、block_size=16 → virtual_block_size=32：两个 rank 各持一份
# BlockTable（行 [10, 20]），对同一串 positions 0..63 各自跑换算——
# I-token 交错归属（is_local）、本地偏移紧凑重排（local_block_offsets）、
# 非本秩槽位打 PAD。单卡（W=1、I=1）退化恒等对照。
# 注：dcp_world_size/dcp_rank/cp_kv_cache_interleave_size 的属性覆写是
# host 镜像观测位（kernel 侧是烤进的 constexpr，取值同源——测试同款手法）。
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.block_table import BlockTable  # noqa: E402

CPU = torch.device("cpu")
W, I, BS = 2, 2, 16
ROW = [10, 20]
VBS = BS * W  # 32


def make_rank_bt(rank):
    bt = BlockTable(
        block_size=BS, max_num_reqs=2, max_num_blocks_per_req=16,
        max_num_batched_tokens=64, pin_memory=False, device=CPU,
        kernel_block_size=BS, cp_kv_cache_interleave_size=I,
    )
    bt.dcp_world_size = W       # host 观测位：kernel 烘干值同源
    bt.dcp_rank = rank
    bt.add_row(ROW, row_idx=0)
    bt.commit_block_table(1)
    return bt


bt_r0 = make_rank_bt(0)
bt_r1 = make_rank_bt(1)

positions = torch.arange(64, dtype=torch.int64)
qsl = torch.tensor([0, 64], dtype=torch.int32)
bt_r0.compute_slot_mapping(1, qsl, positions)
bt_r1.compute_slot_mapping(1, qsl, positions)
s0 = [int(x) for x in bt_r0.slot_mapping.gpu]
s1 = [int(x) for x in bt_r1.slot_mapping.gpu]

# 逐 token 记账（kernel 内景的三个中间量逐一可验）。
per_token = []
for pos in range(32, 44):  # 第二个虚拟块（vbi=1 → 行内第 1 项=20）
    vbi = pos // VBS
    voff = pos - vbi * VBS
    owner = (voff // I) % W
    lbo = (voff // (W * I)) * I + voff % I
    per_token.append({
        "pos": pos,
        "vbi": vbi, "voff": voff,
        "stripe": voff // I, "owner_rank": owner,
        "is_local_r0": owner == 0, "is_local_r1": owner == 1,
        "local_block_offsets": lbo,
        "block_index": vbi + lbo // BS,
        "row_entry": ROW[vbi],
        "slot_r0": s0[pos], "slot_r1": s1[pos],
    })

# 覆盖核对：每 rank 每 32-token 虚拟块恰得 16 个本秩 slot（合起来无重无漏），
# 其余 16 个打 PAD；两 rank 的本地偏移各自恰好铺满 [0,16)。
r0_local = [p for p in range(64) if s0[p] != -1]
r1_local = [p for p in range(64) if s1[p] != -1]
lbo_r0 = sorted((p - (p // VBS) * VBS) // (W * I) * I + (p - (p // VBS) * VBS) % I
                for p in r0_local if 32 <= p < 64)
coverage = {
    "num_local_r0": len(r0_local), "num_local_r1": len(r1_local),
    "num_pad_r0": 64 - len(r0_local), "num_pad_r1": 64 - len(r1_local),
    "no_overlap": len(set(r0_local) & set(r1_local)) == 0,
    "union_covers_all": sorted(r0_local + r1_local) == list(range(64)),
    "r0_local_offsets_second_vblock": lbo_r0,
    "r0_local_offsets_is_0_to_15": lbo_r0 == list(range(16)),
    "same_slot_values_both_ranks": sorted(
        s0[p] for p in r0_local if 32 <= p < 64) == sorted(
        s1[p] for p in r1_local if 32 <= p < 64),
}

# 单卡退化对照：W=1、I=1 时三件全部恒等。
ROW_SINGLE = [10, 20, 11, 21]  # 单卡 64 token → 4 个逻辑块
bt_single = BlockTable(
    block_size=BS, max_num_reqs=2, max_num_blocks_per_req=16,
    max_num_batched_tokens=64, pin_memory=False, device=CPU,
    kernel_block_size=BS, cp_kv_cache_interleave_size=1,
)
assert bt_single.dcp_world_size == 1 and bt_single.dcp_rank == 0
bt_single.add_row(ROW_SINGLE, row_idx=0)
bt_single.commit_block_table(1)
bt_single.compute_slot_mapping(1, qsl, positions)
s_single = [int(x) for x in bt_single.slot_mapping.gpu]
single_degenerate = {
    "dcp_world_size": 1, "cp_interleave": 1,
    "virtual_block_size": BS * 1,
    "block_table_row": ROW_SINGLE,
    "slot_35": s_single[35],
    "slot_35_expect": ROW_SINGLE[35 // BS] * BS + 35 % BS,
    "identity_holds": all(
        s_single[p] == ROW_SINGLE[p // BS] * BS + p % BS for p in range(64)),
    "no_pad": all(x != -1 for x in s_single),
}

out = {
    "mechanism": "m8 CP 分片：I-token 交错归属 + 本地偏移重排 + 非本秩 PAD",
    "trace_source": "run（精简版 host 镜像；kernel CP 三件逐字 = block_table.py:L413-L428、L441）",
    "params": {
        "dcp_world_size": W, "cp_kv_cache_interleave_size": I,
        "block_size": BS, "kv_cache_block_size": BS,
        "virtual_block_size": VBS, "blocks_per_kv_block": 1,
        "block_table_row": ROW, "num_tokens": 64,
        "anchor": "vllm/v1/worker/block_table.py:L413-L428,L441",
    },
    "dossier_example_pos35": {
        "pos": 35, "vbi": 35 // 32, "voff": 3,
        "owner_rank": (3 // 2) % 2, "lbo": (3 // 4) * 2 + 3 % 2,
        "slot_rank1": s1[35], "slot_rank0": s0[35],
    },
    "per_token": per_token,
    "coverage": coverage,
    "single_gpu_degenerate": single_degenerate,
}
path = os.path.join(os.path.dirname(__file__), "m8.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m8 trace written:", path)
