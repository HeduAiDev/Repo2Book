"""ch09 —— MLIR/Linalg 结构化算子（structured op）的可执行代数核心。

这不是 MLIR 的复刻：这里不建模 Op/Region/Block 的类层次（那是叙事与配图的活，
见 chapter narrative）。本文件只把 [Linalg §2.3.6]/[Linalg §3] 描述的「structured
op 自带索引表达式、迭代域隐式且可推导」这条数学核心变成可跑、可打断点的对象。

CPU / NumPy only（trace_source: cpu-numpy-reference）——宿主无昇腾 NPU/CANN，
不产生也不依赖任何真机数值。

术语诚实提示（对应 dossier open_question 1）：论文正文用的措辞是
"indexing function / indexing expressions"；`indexing_maps`/`iterator_types`
这对属性名只在两篇论文的 `vector.contract` 例子里出现过（[Linalg §3.3, Fig.6]，
paper.md:L290-L294）。本文件里 `AffineExpr`/`IndexingMap`/`iterator_types` 这几个
类型名和字段名是本参考实现自己的脚手架命名，用来把论文的数学写成代码；语义上
分别对应论文的「索引表达式」与「迭代器类型」，不代表论文本身用了这套类名。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[int, ...]
Domain = Dict[int, Tuple[int, int]]  # dim_index -> (lo, hi) 半开区间，绝对坐标


# PAPER: [Linalg §3] `linalg.conv_1d_nwc_wcf` 的索引记法 O[n,w,f]=I[n,w+kw,c]·K[kw,c,f]
# 里，`I` 的空间轴写成 `w+kw`——两个迭代维、系数皆为 1 相加（paper.md:L248-L264）。
@dataclass(frozen=True)
class AffineExpr:
    """一个操作数某一轴的索引表达式：`sum(coeff * point[dim]) + offset`。"""

    terms: Tuple[Tuple[int, int], ...]  # ((dim_index, coeff), ...)，同一维最多出现一次
    offset: int = 0

    # PAPER: [Linalg §3] 索引表达式的逐点求值（paper.md:L248-L264）。
    def eval(self, point: Point) -> int:
        return sum(coeff * point[d] for d, coeff in self.terms) + self.offset

    # PAPER: [Linalg §3] t2 反解迭代域时用到的「纯恒等」判定（paper.md:L266-L278）。
    def pure_dim(self) -> Optional[int]:
        """若这条表达式恰好是某个迭代维的恒等映射（系数 1、无偏移、不与其他维相加），
        返回该维下标，否则返回 None。——这是 t2 反解迭代域时唯一认可的「可读出边界」形态。
        """
        if self.offset == 0 and len(self.terms) == 1 and self.terms[0][1] == 1:
            return self.terms[0][0]
        return None

    # PAPER: [Linalg §3.1] "The derivation of dense subsets is obtained by computing
    # the image of the iteration domain by the indexing function for each tensor."
    # (paper.md:L280-L284)
    def image(self, domain: Domain) -> Tuple[int, int]:
        """给定每个迭代维的绝对区间 `domain[d]=(lo,hi)`，求这条仿射表达式在该迭代域上
        的像——即该操作数这一轴会被实际访问到的最小连续区间 `[lo, hi)`。

        因为 `terms` 里每一维都是独立的加性项（仿射、无交叉乘积），像的上下界可以
        逐项独立取到：系数为正就用该维的下/上端点，系数为负则反过来，最后求和。
        """
        lo = self.offset
        hi = self.offset
        for d, coeff in self.terms:
            dlo, dhi = domain[d]
            dmax = dhi - 1
            if coeff >= 0:
                lo += coeff * dlo
                hi += coeff * dmax
            else:
                lo += coeff * dmax
                hi += coeff * dlo
        return lo, hi + 1


IndexingMap = Tuple[AffineExpr, ...]  # 每个操作数一个：每轴一条 AffineExpr
IteratorKind = str  # "parallel" | "reduction" —— 逐字取自 [Linalg §3.3, Fig.6]


# PAPER: [Linalg §3] "The iteration domain is implicit in the operation description
# and is such that the iterators span the entire data of the operands."
# (paper.md:L266-L278) —— t2 的可执行化。
def derive_iteration_domain(
    dim_names: Sequence[str],
    operand_maps_and_shapes: Sequence[Tuple[IndexingMap, Tuple[int, ...]]],
) -> Domain:
    """从「迭代器必须扫过每个操作数的全部数据」这条规则反解迭代域边界。

    **受限的推导**：对每个迭代维，找一个操作数的某一轴，其索引表达式是该维的
    *纯* 恒等映射（系数 1、无偏移、不与别的维相加），用那一轴的尺寸当边界。
    这精确复现论文自己对 conv_1d_nwc_wcf 的推导（`n,w,f` 读 `O` 的形状、`kw,c`
    读 `K` 的形状），但**不是**通用的仿射集合求解器——论文注明稠密情形下的通用推导
    「可由连续施加 Fourier-Motzkin 消元过程得出」（paper.md:L278），本参考实现
    不尝试实现这一步。
    """
    bounds: Domain = {}
    for d in range(len(dim_names)):
        found: Optional[Tuple[int, int]] = None
        for imap, shape in operand_maps_and_shapes:
            for axis, expr in enumerate(imap):
                if expr.pure_dim() == d:
                    bound = (0, shape[axis])
                    if found is not None and found != bound:
                        raise ValueError(
                            f"维 {dim_names[d]!r}：不同操作数给出冲突的边界 "
                            f"{found} vs {bound}"
                        )
                    found = bound
        if found is None:
            raise ValueError(
                f"维 {dim_names[d]!r}：没有任何操作数的某一轴是它的纯恒等映射——"
                "通用推导需要 Fourier-Motzkin 消元，本参考实现不实现该步骤"
                "（见 paper.md:L278）"
            )
        bounds[d] = found
    return bounds


# PAPER: [Linalg §2.3.6] structured op 的三条可检验性质（paper.md:L236-L240）：
# ①同时作用于 tensor/memref；②能分解成作用在结构化子集上的自己的小号版本；
# ③自带独立性/归约等结构信息。
@dataclass
class StructuredOp:
    # PAPER: [Linalg §2.3.6]/[Linalg §3] 见上——本类落地这三条性质里的第③条。
    """一个 `linalg` structured op：索引表达式 + 隐式迭代域 + 标量算子体。

    本类只落地性质③（索引/迭代器结构）；性质②（分解/tiling）在 `tiling.py`，
    性质①（tensor 与 memref 两种容器）不在本参考实现里区分——所有操作数都是普通
    NumPy 数组，把它们当「不可变的 tensor 值」处理（`apply` 每次都返回新分配的
    数组），memref 的显式 layout 是 §6.3 的叙事/配图材料，不在这里建模。
    """

    name: str
    dim_names: Tuple[str, ...]
    iterator_types: Tuple[IteratorKind, ...]  # 与 dim_names 一一对应
    operand_names: Tuple[str, ...]
    operand_maps: Dict[str, IndexingMap]
    result_map: IndexingMap
    body: Callable[..., float]  # 归约维非空时签名 (acc,*ins)->acc，否则 (*ins)->out
    is_named: bool = True
    vectorizable_reduce: Optional[str] = None  # 仅供 vectorization.py 用，见该文件

    # PAPER: [Linalg §3] 迭代域「须扫过操作数全部数据」(paper.md:L270) 里的
    # 「操作数」在这里也含 `outs`——见下方文档。
    def iteration_domain(
        self, ins_shapes: Dict[str, Tuple[int, ...]], out_shape: Tuple[int, ...]
    ) -> Domain:
        """derive_iteration_domain 的封装：`operands` 既包括输入也包括 `outs`。

        这不是随手加的——论文自己给的例子里，`w` 的边界只能从 `O` 的形状读出
        （`I` 的 w 轴是 `w+kw`，不是纯恒等），`I`/`K` 都给不出 `w` 的边界。也就是说
        「迭代器扫过操作数全部数据」这句话里的「操作数」天然含 `outs`——这正是
        §7.4 destination-passing style 里 `outs` 作为一等操作数出现的原因之一。
        """
        pairs = [(self.operand_maps[n], ins_shapes[n]) for n in self.operand_names]
        pairs.append((self.result_map, out_shape))
        return derive_iteration_domain(self.dim_names, pairs)

    # PAPER: [Linalg §3] 索引记法 O[n,w,f]=I[n,w+kw,c]·K[kw,c,f] 的标量参考求值
    # （paper.md:L248-L264）——这是 ground truth，不是任何后端真实的执行方式，
    # 向量化后的替代实现见 vectorization.py。
    def apply(
        self,
        ins: Dict[str, np.ndarray],
        out_shape: Tuple[int, ...],
        init: float = 0.0,
    ) -> np.ndarray:
        """逐点走一遍迭代域、调用 `body`。`out_shape` 对应 `outs` 操作数——
        调用方须显式给出（模拟 destination-passing 里 `outs` 提供形状这件事），
        `init` 是归约的幺元（sum 用 0.0，见 `padding.neutral_element`）。
        """
        ins_shapes = {n: ins[n].shape for n in self.operand_names}
        domain = self.iteration_domain(ins_shapes, out_shape)
        out = np.full(out_shape, init, dtype=float)
        reduce_dims = [i for i, k in enumerate(self.iterator_types) if k == "reduction"]
        ranges = [range(*domain[d]) for d in range(len(self.dim_names))]
        for point in itertools.product(*ranges):
            out_idx = tuple(expr.eval(point) for expr in self.result_map)
            args = [
                ins[name][tuple(expr.eval(point) for expr in self.operand_maps[name])]
                for name in self.operand_names
            ]
            if reduce_dims:
                out[out_idx] = self.body(out[out_idx], *args)
            else:
                out[out_idx] = self.body(*args)
        return out
