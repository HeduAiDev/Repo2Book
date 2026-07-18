#!/usr/bin/env python3
"""ch03 — grid/mask 算术的纯控制流复算（host 直接跑，无需 NPU/CANN）。

本章 kind=skip_impl，无精简版：add_kernel 需 NPU 运行时才能真跑。
但 pid->block_start->offsets->mask 这一段是**纯整数算术**，与硬件无关，
可在 host 上逐字复算，用来核对手工推演表里的每个数字（97 / 128 / 896 等）。

复算的算式逐字对应源码（third_party/ascend/tutorials/01-vector-add.py）：
  L65 block_start = pid * BLOCK_SIZE
  L66 offsets     = block_start + arange(0, BLOCK_SIZE)
  L68 mask        = offsets < n_elements
  L85 grid        = cdiv(n_elements, BLOCK_SIZE)
"""
import json
import math


def cdiv(a, b):
    return (a + b - 1) // b  # == triton.cdiv


def trace(n_elements, BLOCK_SIZE, dump_all_pids=True):
    grid = cdiv(n_elements, BLOCK_SIZE)
    rows = []
    for pid in range(grid):
        block_start = pid * BLOCK_SIZE
        off_lo = block_start
        off_hi = block_start + BLOCK_SIZE - 1  # inclusive
        valid = sum(1 for o in range(off_lo, off_hi + 1) if o < n_elements)
        masked = BLOCK_SIZE - valid
        rows.append({
            "pid": pid,
            "block_start": block_start,
            "offsets_lo": off_lo,
            "offsets_hi": off_hi,
            "valid": valid,
            "masked": masked,
        })
    total_valid = sum(r["valid"] for r in rows)
    return {
        "n_elements": n_elements,
        "BLOCK_SIZE": BLOCK_SIZE,
        "grid": grid,
        "total_valid": total_valid,
        "rows": rows if dump_all_pids else [rows[0], rows[-1]],
    }


if __name__ == "__main__":
    # 教学小例：读者能心算跟上（n=10, BLOCK_SIZE=4）
    small = trace(10, 4)
    # 真实 tutorial 参数（源码 L93 size=98432, L86 BLOCK_SIZE=1024）
    real = trace(98432, 1024, dump_all_pids=False)
    out = {"small_n10_bs4": small, "real_n98432_bs1024": real}
    print(json.dumps(out, indent=2))
