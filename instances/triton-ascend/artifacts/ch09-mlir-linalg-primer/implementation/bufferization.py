"""ch09 —— bufferization：把不可变张量物化进内存，少分配少拷贝（m21/t7）。

对照两条策略跑同一段 tiled 计算：
  - `bufferize_naive`：每次 `insert_slice` 都当成产出一个新张量值（老老实实的
    tensor 函数式语义：整份拷贝 + 写入切片），对应「不做 destination-passing
    优化」的朴素路径；
  - `bufferize_dps`：预先分配一份 buffer（对应 `outs` 操作数）、每块 tile 直接
    原地写入——这正是 §7.4 destination-passing style 授权的优化（"a
    bufferization constraint with no observable impact on the functional
    semantics"，paper.md:L319-L325）。

两条路径数值上必须完全相同（legal by design）；分配次数不同——这里数的是
**CPU 上的分配次数**，不是任何性能数字，宿主无 NPU/CANN，不能也不去伪造真机
耗时。
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import numpy as np

from structured_op import StructuredOp
from tiling import extract_slice, image_of_domain, insert_slice, tile


# PAPER: [Linalg §3.4] bufferization 目标是尽可能少分配、少拷贝
# （paper.md:L369-L373）——两个计数字段就是把这句话变成一个可比较的数字。
class BufferizeReport(NamedTuple):
    """两条 bufferization 策略跑完之后的对照结果。"""

    output: np.ndarray
    OUT_OF_PLACE_ALLOC_COUNT: int  # naive 路径：insert_slice 每次都新分配
    DPS_ALLOC_COUNT: int  # DPS 路径：只在一开始分配一次


# PAPER: [Linalg §3.4]（paper.md:L369-L373，原文 read-after-write 冲突一段）——
# 每次写都新分配 buffer 永远安全但浪费；这里把"永远安全但浪费"字面地实现出来，
# 用分配计数把"浪费"变成一个可以数出来的量，不是一句空话。
def bufferize_naive(
    op: StructuredOp,
    ins: Dict[str, np.ndarray],
    out_shape: Tuple[int, ...],
    tile_sizes: Dict[str, int],
    init: float = 0.0,
) -> BufferizeReport:
    """朴素路径：每块 tile 算完之后，`tensor.insert_slice` 被当成「产出一个新的
    完整输出张量」——整份 `.copy()` 再写入这一块——严格对应张量不可变、写入即
    产出新值的函数式语义（bufferization 优化*之前*的样子）。
    """
    ins_shapes = {n: a.shape for n, a in ins.items()}
    global_domain = op.iteration_domain(ins_shapes, out_shape)
    out = np.full(out_shape, init, dtype=float)
    alloc_count = 1  # 初始这份 out 本身就是一次分配
    for local_domain in tile(op, global_domain, tile_sizes):
        local_ins = {
            name: extract_slice(ins[name], image_of_domain(local_domain, op.operand_maps[name]))
            for name in op.operand_names
        }
        out_ranges = image_of_domain(local_domain, op.result_map)
        local_out_shape = tuple(hi - lo for lo, hi in out_ranges)
        local_result = op.apply(local_ins, local_out_shape, init=init)
        new_out = out.copy()  # 朴素 insert_slice：整份重新分配
        alloc_count += 1
        insert_slice(new_out, out_ranges, local_result)
        out = new_out
    return BufferizeReport(output=out, OUT_OF_PLACE_ALLOC_COUNT=alloc_count, DPS_ALLOC_COUNT=0)


# PAPER: [Linalg §3.4] destination-passing style 把 `outs` 绑定成"就地 bufferize
# 的理想候选"（paper.md:L315-L325）——一次分配、原地写完所有 tile。
def bufferize_dps(
    op: StructuredOp,
    ins: Dict[str, np.ndarray],
    out_shape: Tuple[int, ...],
    tile_sizes: Dict[str, int],
    init: float = 0.0,
) -> BufferizeReport:
    """destination-passing 路径：`out` 只分配一次（对应 `outs` 操作数），每块
    tile 直接原地写入这份 buffer——这是 §7.4 说的「`outs` 不改变函数式语义，
    只是一条 bufferization 约束」在代码里的样子：数值上与 `bufferize_naive`
    完全一样，分配次数从 `len(tiles)+1` 降到 1。
    """
    ins_shapes = {n: a.shape for n, a in ins.items()}
    global_domain = op.iteration_domain(ins_shapes, out_shape)
    out = np.full(out_shape, init, dtype=float)  # 唯一一次分配：DPS 的 outs buffer
    for local_domain in tile(op, global_domain, tile_sizes):
        local_ins = {
            name: extract_slice(ins[name], image_of_domain(local_domain, op.operand_maps[name]))
            for name in op.operand_names
        }
        out_ranges = image_of_domain(local_domain, op.result_map)
        local_out_shape = tuple(hi - lo for lo, hi in out_ranges)
        local_result = op.apply(local_ins, local_out_shape, init=init)
        insert_slice(out, out_ranges, local_result)  # 原地写，不新分配
    return BufferizeReport(output=out, OUT_OF_PLACE_ALLOC_COUNT=0, DPS_ALLOC_COUNT=1)
