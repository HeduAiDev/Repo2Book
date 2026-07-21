"""ch09 —— named op 与 generic op：同一个东西的两种穿法（m19）。

[Linalg §3.3, 脚注 6]（paper.md:L296-L313）：算子体在 `linalg.generic` 形式下
显式打印，在「named」形式（如 `linalg.conv_xxx`）下则被省略——named 算子只是
generic 算子的声明式配置，语义完全归结到 generic。`to_generic` 在这里只是把
`is_named` 标志翻过来；索引映射、迭代器类型、算子体都是同一份对象，因为按论文
说的，它们本来就一直是同一份。
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict

from structured_op import AffineExpr, StructuredOp


# PAPER: [Linalg §3] 恒等索引表达式的小工厂，供下面几个具名算子拼索引映射用
# （paper.md:L248-L264）。
def _id(dim: int) -> AffineExpr:
    return AffineExpr(((dim, 1),))


# PAPER: [Linalg §3] `I[n, w+kw, c]` 里 `w+kw` 这种两维相加的索引表达式
# （paper.md:L248-L264）。
def _sum2(dim_a: int, dim_b: int) -> AffineExpr:
    return AffineExpr(((dim_a, 1), (dim_b, 1)))


# PAPER: [Linalg §3] 索引记法 O[n,w,f]=I[n,w+kw,c]·K[kw,c,f]，维序
# (n,w,f,kw,c)（paper.md:L248-L264）；论文全程用来演示 tiling/padding/向量化
# 的例子。
def make_conv_1d_nwc_wcf() -> StructuredOp:
    """构造 `linalg.conv_1d_nwc_wcf`：3 个迭代维是并行的输出维
    （batch/空间/输出通道），2 个是归约维（核宽、输入通道）。"""
    n, w, f, kw, c = range(5)
    i_map = (_id(n), _sum2(w, kw), _id(c))
    k_map = (_id(kw), _id(c), _id(f))
    o_map = (_id(n), _id(w), _id(f))
    return StructuredOp(
        name="conv_1d_nwc_wcf",
        dim_names=("n", "w", "f", "kw", "c"),
        iterator_types=("parallel", "parallel", "parallel", "reduction", "reduction"),
        operand_names=("I", "K"),
        operand_maps={"I": i_map, "K": k_map},
        result_map=o_map,
        body=lambda acc, i_val, k_val: acc + i_val * k_val,
        is_named=True,
    )


# PAPER: [Linalg §3.4] "linalg.matmul ... any other operation that reduces to
# linalg.generic" (paper.md:L323)；[Linalg §3.5] 用矩阵乘做渐进向量下降的
# 例子（paper.md:L375-L377）。维序 (i,j,k)，k 是归约（收缩）维。
def make_matmul() -> StructuredOp:
    """构造 `linalg.matmul`：论文两次都拿它当 destination-passing style 与
    向量化的例子；这里也是本参考实现 vectorization 情形 4（归约）的落点。"""
    i, j, k = range(3)
    a_map = (_id(i), _id(k))
    b_map = (_id(k), _id(j))
    c_map = (_id(i), _id(j))
    return StructuredOp(
        name="matmul",
        dim_names=("i", "j", "k"),
        iterator_types=("parallel", "parallel", "reduction"),
        operand_names=("A", "B"),
        operand_maps={"A": a_map, "B": b_map},
        result_map=c_map,
        body=lambda acc, a_val, b_val: acc + a_val * b_val,
        is_named=True,
        vectorizable_reduce="sum_of_products",
    )


# PAPER: [Linalg §3.3] 向量化表情形 (1)「逐点算子」的最小实例（paper.md:L361）。
# 两篇论文都只泛泛描述这一情形，没有点名一个具体的 linalg 逐点算子——这是本参考
# 实现自己挑的最简单实例，不是论文点名的算子，用来让「情形 1」可以被跑一遍。
def make_pointwise_add() -> StructuredOp:
    """两个同形张量逐元素相加：全部索引都是恒等映射，没有归约维。"""
    i, j = range(2)
    ident = (_id(i), _id(j))
    return StructuredOp(
        name="pointwise_add",
        dim_names=("i", "j"),
        iterator_types=("parallel", "parallel"),
        operand_names=("A", "B"),
        operand_maps={"A": ident, "B": ident},
        result_map=ident,
        body=lambda a_val, b_val: a_val + b_val,
        is_named=False,
    )


# 只登记论文真正点名过的两个 named op（conv_1d_nwc_wcf、matmul）；pointwise_add
# 不是论文命名的算子，不进这张「named op 名录」。
named_op_registry: Dict[str, Callable[[], StructuredOp]] = {
    "conv_1d_nwc_wcf": make_conv_1d_nwc_wcf,
    "matmul": make_matmul,
}

conv_1d_nwc_wcf = make_conv_1d_nwc_wcf
matmul = make_matmul


# PAPER: [Linalg §3.3, 脚注 6] named 形式只是省略了算子体的写法，语义仍归结到
# `linalg.generic`（paper.md:L296-L313）。
def to_generic(op: StructuredOp) -> StructuredOp:
    """把一个 named op「展开」成 generic 形式：索引映射/迭代器类型/算子体
    原样不动，只把 `is_named` 翻成 False——因为按论文的说法，这些内容从来
    就没有因为「穿哪件衣服」而变过。返回新对象，不修改传入的 `op`。
    """
    return dataclasses.replace(op, is_named=False)
