"""ch09 —— tiling：循环体里仍是同一个结构化算子（m16），像即子集（m15/t3）。

只对 **parallel** 迭代维开放 tiling（归约维每个 tile 都取满量程）——这是本参考
实现的一个刻意收窄：论文的 tile size 例子 `1x8x32x1x8` 覆盖全部 5 维，但要对归约
维分块还需要跨 tile 累加（partial reduction），那是另一层复杂度，dossier 的
worked example 不需要，这里就不实现，只如实注明（见各函数文档）。
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterator, Tuple

import numpy as np

from structured_op import Domain, IndexingMap, StructuredOp


# PAPER: [Linalg §3.1] "The derivation of dense subsets is obtained by computing
# the image of the iteration domain by the indexing function for each tensor."
# (paper.md:L280-L284) —— t3 的可执行化，本章「最该被读者带走的一句话」。
def image_of_domain(domain: Domain, indexing_map: IndexingMap) -> Tuple[Tuple[int, int], ...]:
    """某个操作数在给定迭代域下会被访问到的那一片：逐轴调用 `AffineExpr.image`。"""
    return tuple(expr.image(domain) for expr in indexing_map)


# PAPER: [Linalg §3.1] `tensor.extract_slice`（paper.md:L335）。
def extract_slice(tensor: np.ndarray, ranges: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    """读子集——对应 `tensor.extract_slice`。返回拷贝，不与原张量共享底层内存
    （tensor 值语义：切片本身也是一个独立的、不可变的张量值）。
    """
    return tensor[tuple(slice(lo, hi) for lo, hi in ranges)].copy()


# PAPER: [Linalg §3.1] `tensor.insert_slice`（paper.md:L335）。
def insert_slice(
    dst: np.ndarray, ranges: Tuple[Tuple[int, int], ...], value: np.ndarray
) -> None:
    """写回子集——对应 `tensor.insert_slice`。这里就地修改 `dst`；是否可以就地
    修改而不产生额外分配，是 §7.4 destination-passing style 要回答的问题，见
    `bufferization.py`。这个函数本身只是机械的写入，naive 与 DPS 两条路都靠它。
    """
    dst[tuple(slice(lo, hi) for lo, hi in ranges)] = value


# PAPER: [Linalg §3.1] "Tiling the operation introduces scf.for loops as well as
# subset operations ... The tiled form of the operation is itself a
# linalg.conv_1d_nwc_wcf operating on the tiled subsets." (paper.md:L333-L339)
def tile(
    op: StructuredOp, global_domain: Domain, tile_sizes: Dict[str, int]
) -> Iterator[Domain]:
    """按 `tile_sizes`（维名 -> tile 宽度）切分 `global_domain`，逐个 yield 出
    每块 tile 对应的局部迭代域（其余未列出的维——含全部归约维——保持满量程）。

    边界 tile（除不尽的最后一块）天然变短，对应论文说的「没有哪个静态张量类型对
    每次迭代都合法」；本参考实现用 NumPy 的动态形状直接承接这件事（不需要像
    MLIR 那样显式引入 `!tDyn` 类型再靠 canonicalization 收窄）。
    """
    for name in tile_sizes:
        d = op.dim_names.index(name)
        if op.iterator_types[d] != "parallel":
            raise ValueError(
                f"只允许对 parallel 维分块，{name!r} 是 "
                f"{op.iterator_types[d]!r}（本参考实现的范围收窄，见模块文档）"
            )
    tiled = []
    for name, size in tile_sizes.items():
        d = op.dim_names.index(name)
        lo, hi = global_domain[d]
        starts = range(lo, hi, size)
        tiled.append((d, [(s, min(s + size, hi)) for s in starts]))
    if not tiled:
        yield dict(global_domain)
        return
    dims_order = [d for d, _ in tiled]
    choices = [ranges for _, ranges in tiled]
    for combo in itertools.product(*choices):
        local = dict(global_domain)
        for d, rng in zip(dims_order, combo):
            local[d] = rng
        yield local


# PAPER: [Linalg §3.1] "The tiled form of the operation is itself a
# linalg.conv_1d_nwc_wcf operating on the tiled subsets." (paper.md:L333-L339)
def tile_and_run(
    op: StructuredOp,
    ins: Dict[str, np.ndarray],
    out_shape: Tuple[int, ...],
    tile_sizes: Dict[str, int],
    init: float = 0.0,
) -> np.ndarray:
    """m16 的端到端可执行版本：对每块 tile，用 `image_of_domain` 求出每个操作数
    该读哪一片（`extract_slice`），把切下来的小张量喂给**同一个** `op.apply`
    ——不是另写一份小算子，就是原来那个 `StructuredOp` 实例，只是操作数变小了
    ——再把局部结果 `insert_slice` 回全量输出。

    这里不追求省分配（每块 tile 都直接写回预分配好的 `out`），"省几次分配/拷贝"
    这个问题留给 `bufferization.py` 对照着讲。
    """
    ins_shapes = {n: a.shape for n, a in ins.items()}
    global_domain = op.iteration_domain(ins_shapes, out_shape)
    out = np.full(out_shape, init, dtype=float)
    for local_domain in tile(op, global_domain, tile_sizes):
        local_ins = {
            name: extract_slice(ins[name], image_of_domain(local_domain, op.operand_maps[name]))
            for name in op.operand_names
        }
        out_ranges = image_of_domain(local_domain, op.result_map)
        local_out_shape = tuple(hi - lo for lo, hi in out_ranges)
        local_result = op.apply(local_ins, local_out_shape, init=init)
        insert_slice(out, out_ranges, local_result)
    return out
