"""Driver for m9 (槽位换算恒等式 slot = block_table[req][pos//block_size] ×
block_size + pos%block_size) — host run against the ch13 companion.

HOST SEAM 说明：host 无 CUDA launch——compute_slot_mapping 在 CPU 设备走
kernel 本体的逐行镜像（同一恒等式、同一 PAD 尾、同一变量名）；CUDA 分支
（_compute_slot_mapping_kernel）逐字保留、容器内真跑。本 trace 数字与 kernel
单卡（TOTAL_CP_WORLD_SIZE=1）语义一致。

场景一：块表行 [3,1,7]、positions 0..47 → 三个物理段 48..63 / 16..31 /
112..127（位置递增而物理槽位中段反而更低——间接寻址的可视化）；
场景二：尾部 PAD（20 个真 token、max 64 → [20,64) 全 -1）；
场景三：两条 decode 各 16 token、行 [2] 与 [5] → 段 32..47 与 80..95；
逆运算：slot → (块号, 偏移)（读侧共用同一恒等式）。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.block_table import PAD_SLOT_ID, BlockTable  # noqa: E402

BLOCK_SIZE = 16


def make_bt(max_num_reqs=2, max_blocks_per_req=8, max_tokens=64):
    return BlockTable(
        block_size=BLOCK_SIZE,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_blocks_per_req,
        max_num_batched_tokens=max_tokens,
        pin_memory=False,
        device=torch.device("cpu"),  # HOST SEAM
        kernel_block_size=BLOCK_SIZE,
    )


def main():
    # 场景一：单请求、块表行 [3,1,7]
    bt = make_bt()
    row = [3, 1, 7]
    bt.append_row(row, 0)
    num_tokens = 48
    qs = torch.tensor([0, num_tokens, num_tokens], dtype=torch.int32)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    bt.compute_slot_mapping(1, qs, positions)
    slots = [int(s) for s in bt.slot_mapping.np[:num_tokens]]

    seg1 = slots[0:16]
    seg2 = slots[16:32]
    seg3 = slots[32:48]

    # 场景二：尾部 PAD（20 token 跨 2 块：行 [1,2]）
    bt2 = make_bt()
    bt2.append_row([1, 2], 0)
    n2 = 20
    qs2 = torch.tensor([0, n2, n2], dtype=torch.int32)
    pos2 = torch.arange(n2, dtype=torch.int64)
    bt2.compute_slot_mapping(1, qs2, pos2)
    tail = list(bt2.slot_mapping.np[n2:64])
    head2 = list(bt2.slot_mapping.np[:n2])

    # 场景三：两条 decode
    bt3 = make_bt()
    bt3.append_row([2], 0)
    bt3.append_row([5], 1)
    qs3 = torch.tensor([0, 16, 32, 32], dtype=torch.int32)
    pos3 = torch.cat(
        [torch.arange(16, dtype=torch.int64), torch.arange(16, dtype=torch.int64)]
    )
    bt3.compute_slot_mapping(2, qs3, pos3)
    two_reqs = {
        "row0_block": 2, "row0_slots": [int(bt3.slot_mapping.np[0]), int(bt3.slot_mapping.np[15])],
        "row1_block": 5, "row1_slots": [int(bt3.slot_mapping.np[16]), int(bt3.slot_mapping.np[31])],
    }

    # 逆运算：slot → (块号, 块内偏移)——读侧（attention 穿块表）共用同一恒等式
    inverse = [
        {"slot": 112, "block": 112 // BLOCK_SIZE, "offset": 112 % BLOCK_SIZE},
        {"slot": 96, "block": 96 // BLOCK_SIZE, "offset": 96 % BLOCK_SIZE},
        {"slot": 80, "block": 80 // BLOCK_SIZE, "offset": 80 % BLOCK_SIZE},
    ]

    # 断言：每个 slot 都等于 块表[pos//16]*16 + pos%16
    expected = [
        row[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE for pos in range(48)
    ]
    assert slots == expected
    assert seg1[0] == 48 and seg1[15] == 63
    assert seg2[0] == 16 and seg2[31 - 16] == 31
    assert seg3[0] == 112 and seg3[47 - 32] == 127
    assert all(t == PAD_SLOT_ID for t in tail) and PAD_SLOT_ID == -1
    assert head2 == [16 + p for p in range(16)] + [32 + p for p in range(4)]

    out = {
        "driver": "run_m9_slot_identity.py",
        "mechanism": "m9 槽位换算恒等式（block_table.py:L380-L442 kernel / L182-L211 派发；positions GPU 张量进 gpu_model_runner.py:L2188-L2201）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（HOST SEAM：CPU 镜像 = kernel 逐行对应；CUDA 分支容器内真跑）",
        "environment_note": "trace 取自 host CPU 镜像——恒等式/PAD 尾/变量名与 Triton kernel 单卡语义逐行一致；取证环境无 CUDA 与 pin 的差异仅在 launch 途径，数值无差",
        "config": {"block_size": BLOCK_SIZE, "pad_slot_id": PAD_SLOT_ID},
        "scenario1_row_3_1_7": {
            "block_table_row": row,
            "positions": "0..47",
            "segment_pos_0_15": {"block_entry": 3, "slots": f"{seg1[0]}..{seg1[15]}",
                                  "first": seg1[0], "last": seg1[15]},
            "segment_pos_16_31": {"block_entry": 1, "slots": f"{seg2[0]}..{seg2[31-16]}",
                                   "first": seg2[0], "last": seg2[15]},
            "segment_pos_32_47": {"block_entry": 7, "slots": f"{seg3[0]}..{seg3[47-32]}",
                                   "first": seg3[0], "last": seg3[15]},
            "sample": [
                {"pos": 0, "pos_div_16": 0, "pos_mod_16": 0, "block_entry": 3, "slot": slots[0]},
                {"pos": 15, "pos_div_16": 0, "pos_mod_16": 15, "block_entry": 3, "slot": slots[15]},
                {"pos": 16, "pos_div_16": 1, "pos_mod_16": 0, "block_entry": 1, "slot": slots[16]},
                {"pos": 31, "pos_div_16": 1, "pos_mod_16": 15, "block_entry": 1, "slot": slots[31]},
                {"pos": 32, "pos_div_16": 2, "pos_mod_16": 0, "block_entry": 7, "slot": slots[32]},
                {"pos": 47, "pos_div_16": 2, "pos_mod_16": 15, "block_entry": 7, "slot": slots[47]},
            ],
            "note": "位置 16..31 的物理槽位（16..31）反而低于位置 0..15 的（48..63）——间接寻址让物理位置与逻辑位置脱钩",
        },
        "scenario2_pad_tail": {
            "block_table_row": [1, 2],
            "real_tokens": n2,
            "max_num_batched_tokens": 64,
            "tail_range": f"[{n2}, 64)",
            "tail_all_pad": True,
            "pad_value": PAD_SLOT_ID,
            "real_slots_head": f"{head2[0]}..{head2[15]}（块 1）+ {head2[16]}..{head2[19]}（块 2）",
            "head_first": int(head2[0]),
            "head_last": int(head2[-1]),
            "note": "最后一个 program 专职 PAD 尾（L399-L408）——CUDA graph 捕获的是 max 形状，尾部每拍重填",
        },
        "scenario3_two_decode_reqs": {
            **two_reqs,
            "note": "grid=(num_reqs+1,)：每 program 处理一请求的 token 区间（query_start_loc 切段）；positions 是各请求自己的序列位置",
        },
        "inverse_read_side": {
            "samples": inverse,
            "note": "给定 slot：块号 = slot//16、偏移 = slot%16——写侧（slot_mapping）与读侧（块表）共用同一恒等式",
        },
    }

    dst = Path(__file__).resolve().parent / "m9_slot_identity.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out["scenario1_row_3_1_7"]["sample"], ensure_ascii=False, indent=1))
    print(json.dumps(out["scenario2_pad_tail"], ensure_ascii=False))
    print(json.dumps(two_reqs, ensure_ascii=False))


if __name__ == "__main__":
    main()
