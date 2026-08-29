# ch22 explainer 驱动脚本 · m2 Triton GPU 端换算的 kernel 组织
# 两拍场景（chunked prefill 拍 + 双 decode 拍），对拍 grid=(num_reqs+1,) 的
# 每 program 分工：前 num_reqs 个 program 各算一请求 token 区间的恒等式，
# 第 num_reqs+1 个 program 专职把 [num_tokens, max) 填 PAD_SLOT_ID。
# 持久 buffer 地址跨拍不变（CUDA graph 回放的地址前提）。
# do_not_specialize=["num_tokens","max_num_tokens"]（block_table.py:L379）：
# 两个每拍变化的标量不特化——本 trace 记录两拍各自的 num_tokens 供对照。
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.block_table import BlockTable  # noqa: E402

CPU = torch.device("cpu")
BLOCK_SIZE = 16
MAX_NUM_BATCHED_TOKENS = 128
# 两个请求的页表行：r0 覆盖逻辑块 0-7、r1 覆盖逻辑块 0-7（拍内最长 pos=119）。
ROW_R0 = [3, 8, 2, 7, 1, 5, 9, 4]
ROW_R1 = [7, 1, 5, 2, 9, 4, 6, 8]

bt = BlockTable(
    block_size=BLOCK_SIZE,
    max_num_reqs=8,
    max_num_blocks_per_req=16,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    pin_memory=False,
    device=CPU,
    kernel_block_size=BLOCK_SIZE,
    cp_kv_cache_interleave_size=1,
)
bt.add_row(ROW_R0, row_idx=0)
bt.add_row(ROW_R1, row_idx=1)
bt.commit_block_table(2)

addr_before = bt.slot_mapping.gpu.data_ptr()

# ── 拍 1（chunked prefill）：r0 首段 100 token（pos 0..99）、
#    r1 续算 20 token（num_computed=100 → pos 100..119）。
qsl1 = torch.tensor([0, 100, 120], dtype=torch.int32)
pos1 = torch.cat([torch.arange(0, 100), torch.arange(100, 120)]).to(torch.int64)
bt.compute_slot_mapping(2, qsl1, pos1)
slots1 = [int(x) for x in bt.slot_mapping.gpu]

# ── 拍 2（decode）：两请求各 1 token——r0 pos=100、r1 pos=120。
qsl2 = torch.tensor([0, 1, 2], dtype=torch.int32)
pos2 = torch.tensor([100, 120], dtype=torch.int64)
bt.compute_slot_mapping(2, qsl2, pos2)
slots2 = [int(x) for x in bt.slot_mapping.gpu]

addr_after = bt.slot_mapping.gpu.data_ptr()

def program_accounting(qsl, num_tokens, num_reqs):
    """按 kernel 的 grid=(num_reqs+1,) 逐 program 记账（host 镜像同构）。"""
    programs = []
    for req_idx in range(num_reqs):
        start = int(qsl[req_idx])
        end = int(qsl[req_idx + 1])
        programs.append({
            "program_id": req_idx,
            "role": f"request_{req_idx}",
            "interval": [start, end],
            "num_tokens_in_interval": end - start,
            "tile_loop_iters": -(-((end - start)) // 1024),  # ceil(len/1024)
        })
    programs.append({
        "program_id": num_reqs,
        "role": "PAD",
        "interval": [num_tokens, MAX_NUM_BATCHED_TOKENS],
        "num_tokens_in_interval": MAX_NUM_BATCHED_TOKENS - num_tokens,
        "tile_loop_iters": -(-(MAX_NUM_BATCHED_TOKENS - num_tokens) // 1024),
    })
    return programs

out = {
    "mechanism": "m2 Triton GPU 端换算的 kernel 组织",
    "trace_source": "run（精简版 host 镜像；kernel 本体逐字 = block_table.py:L379-L442）",
    "params": {
        "block_size": BLOCK_SIZE,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "tile_block_size_constexpr": 1024,
        "grid_size_formula": "num_reqs + 1",
        "row_r0": ROW_R0,
        "row_r1": ROW_R1,
        "do_not_specialize_args": ["num_tokens", "max_num_tokens"],
    },
    "step1_chunked_prefill": {
        "num_reqs": 2,
        "grid": 3,
        "num_tokens": 120,
        "query_start_loc": [0, 100, 120],
        "positions_head_tail": [0, 99, 100, 119],
        "programs": program_accounting(qsl1, 120, 2),
        "slot_mapping_full": slots1,
        "slot_checkpoints": {
            "slot_0": slots1[0], "slot_99": slots1[99],
            "slot_100": slots1[100], "slot_119": slots1[119],
            "slot_120_pad": slots1[120], "slot_127_pad": slots1[127],
        },
    },
    "step2_decode": {
        "num_reqs": 2,
        "grid": 3,
        "num_tokens": 2,
        "query_start_loc": [0, 1, 2],
        "positions": [100, 120],
        "programs": program_accounting(qsl2, 2, 2),
        "slot_mapping_full": slots2,
        "slot_checkpoints": {
            "slot_0": slots2[0], "slot_1": slots2[1],
            "slot_2_pad": slots2[2], "slot_127_pad": slots2[127],
        },
    },
    "persistent_buffer_address": {
        "data_ptr_step1": addr_before,
        "data_ptr_step2": addr_after,
        "address_unchanged": bool(addr_before == addr_after),
    },
    "pad_id": -1,
}
path = os.path.join(os.path.dirname(__file__), "m2.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("m2 trace written:", path)
