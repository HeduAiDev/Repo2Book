"""ch09 —— 向量化：数据搬运通用，计算体分五种情形（m18/§8.3）。

[Linalg §3.3]（paper.md:L353-L367）给出 5 种情形：(1) 逐点、(2) 低维广播、
(3) 置换、(4) 归约、(5) 滑窗。dossier 的范围只要求覆盖 (1) 与 (4)——本文件
就只实现这两种，(2)(3)(5) 遇到时显式 `NotImplementedError`，不假装支持。

"vector.transfer 的索引跟随 linalg 算子的索引表达式，这部分对所有 linalg 算子
通用"（[Linalg §3.3]，paper.md:L286-L288，L355）——这句「通用」指的是数据搬运，
不是计算体；下面 `vectorize` 的实现也照这条边界来：情形判断只看索引映射的形状，
计算体是否能被向量化下降，仍要看算子体本身（"视对算子体的进一步分析"，见情形 4）。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from structured_op import StructuredOp

_LETTERS = "abcdefghijklmnopqrstuvwxyz"


# PAPER: [Linalg §3.3] `vector.contract` 需要索引映射是迭代维的纯置换
# （paper.md:L286-L294）——单个操作数下标串的翻译规则。
def _einsum_subscript(imap) -> str:
    subs = []
    for expr in imap:
        d = expr.pure_dim()
        if d is None:
            raise NotImplementedError(
                "einsum 式向量化要求索引映射是迭代维的纯置换（无求和/无偏移）——"
                "遇到复合表达式（如卷积的 w+kw）不在此列，那是情形 (5) 滑窗，"
                "本参考实现不实现"
            )
        subs.append(_LETTERS[d])
    return "".join(subs)


# PAPER: [Linalg §3.3] 情形 (4)「有归约维……下降成一等的 vector.contract」
# （paper.md:L364），配套 [Linalg §3.3, Fig.6] 的 `indexing_maps`/`iterator_types`
# 属性对（paper.md:L292）——这里用它们拼出等价的 `np.einsum` 下标串。
def build_einsum_subscripts(op: StructuredOp) -> str:
    """把 `op` 的索引映射翻成 `np.einsum` 的下标字符串（如 `"ik,kj->ij"`）——
    只有当每个操作数与结果的索引映射都是迭代维的纯置换时才有意义（对应
    `vector.contract` 能处理的形状，不覆盖卷积那种滑窗耦合）。
    """
    ins_subs = ",".join(_einsum_subscript(op.operand_maps[n]) for n in op.operand_names)
    out_sub = _einsum_subscript(op.result_map)
    return f"{ins_subs}->{out_sub}"


# PAPER: [Linalg §3.3] 向量化配方：每个操作数一个 vector.transfer_read、向量形式
# 计算、vector.transfer_write 写回（paper.md:L353-L355）。
def vectorize(
    op: StructuredOp, ins: Dict[str, np.ndarray], out_shape: Tuple[int, ...]
) -> np.ndarray:
    """`StructuredOp.apply` 的向量化替代实现——数值上必须与 `apply` 逐元素相同
    （legal by design，[Linalg §3.6] paper.md:L385），但不再逐点跑 Python 循环。

    - 无归约维 → 情形 (1) 逐点：要求每个操作数的索引映射与结果完全一致（索引
      全为恒等），直接把整块数组喂给 `op.body`（NumPy 原生逐元素广播）。
    - 有归约维 → 情形 (4)：要求 `op.vectorizable_reduce == "sum_of_products"`
      （即算子体是乘加），用 `np.einsum` 一次性算出——这是"下降成一等的
      vector.contract"（paper.md:L364）在数组层面的对应物。算子体不是乘加时
      按论文原话"视对算子体的进一步分析"，本参考实现不猜，直接报
      `NotImplementedError`。
    """
    reduce_dims = [i for i, k in enumerate(op.iterator_types) if k == "reduction"]
    if not reduce_dims:
        for name in op.operand_names:
            if op.operand_maps[name] != op.result_map:
                raise NotImplementedError(
                    "情形 (1) 逐点向量化要求每个操作数的索引映射与结果一致——"
                    f"{name!r} 不满足；广播（情形 2）/置换（情形 3）需要"
                    "vector.broadcast/vector.transpose，本参考实现不实现"
                )
        result = op.body(*[ins[name] for name in op.operand_names])
        assert result.shape == tuple(out_shape)
        return result

    if op.vectorizable_reduce != "sum_of_products":
        raise NotImplementedError(
            "情形 (4) 归约向量化只覆盖乘加算子体（sum_of_products）——把任意算子体"
            "下降成 vector.contract 需要'视对算子体的进一步分析'"
            "（[Linalg §3.3] paper.md:L364），本参考实现不做这一步通用分析"
        )
    subs = build_einsum_subscripts(op)
    result = np.einsum(subs, *[ins[name] for name in op.operand_names])
    assert result.shape == tuple(out_shape)
    return result
