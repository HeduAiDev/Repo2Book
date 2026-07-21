"""m18 —— 向量化：数据搬运通用，计算体分五种情形（[Linalg §3.3]，paper.md:L353-L367）。

参考实现覆盖论文自列 5 种情形里的 (1) 逐点 与 (4) 归约；(2)(3)(5) 显式拒绝
（不假装支持）。这里对三个算子各跑一遍，记录：
  - 迭代器类型里 parallel / reduction 各几个（这是「情形判断」的输入）；
  - 走哪条向量化路径（逐点整块运算 / einsum 收缩）；
  - 向量化结果与逐点参考求值 `StructuredOp.apply` 的最大偏差（legal by design：必须为 0）；
  - 不支持的情形抛出的异常类型与信息。
另外把「数据搬运部分对所有 linalg 算子通用」这句落到数字上：无论哪个算子，每个操作数
恰好一次 transfer_read、结果一次 transfer_write，读写的索引直接来自算子的索引表达式。
"""
from __future__ import annotations

import numpy as np

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf, make_matmul, make_pointwise_add  # noqa: E402
from vectorization import build_einsum_subscripts, vectorize  # noqa: E402


def transfer_counts(op):
    """数据搬运部分：每个操作数一次 transfer_read，结果一次 transfer_write。"""
    return {
        "transfer_read_count": len(op.operand_names),
        "transfer_write_count": 1,
        "read_index_exprs": {n: ["+".join(op.dim_names[d] for d, _ in e.terms)
                                 for e in op.operand_maps[n]] for n in op.operand_names},
        "write_index_expr": ["+".join(op.dim_names[d] for d, _ in e.terms) for e in op.result_map],
    }


def main() -> None:
    rows = []

    # ---- 情形 (1)：逐点 ----
    add = make_pointwise_add()
    A = np.arange(6, dtype=float).reshape(2, 3)      # 0..5
    B = np.arange(6, dtype=float).reshape(2, 3) * 10  # 0,10,...,50
    ref_add = add.apply({"A": A, "B": B}, out_shape=(2, 3))
    vec_add = vectorize(add, {"A": A, "B": B}, out_shape=(2, 3))
    rows.append({
        "case": 1,
        "case_name": "逐点（索引全为恒等）",
        "op": add.name,
        "n_parallel": add.iterator_types.count("parallel"),
        "n_reduction": add.iterator_types.count("reduction"),
        "vector_path": "整块逐元素运算（无归约）",
        "apply_first_row": ref_add[0].tolist(),
        "vectorized_first_row": vec_add[0].tolist(),
        "max_abs_diff": float(np.max(np.abs(ref_add - vec_add))),
        "transfers": transfer_counts(add),
    })

    # ---- 情形 (4)：归约（乘加 → 收缩） ----
    mm = make_matmul()
    M, Kdim, Nn = 2, 3, 4
    Am = np.arange(M * Kdim, dtype=float).reshape(M, Kdim)
    Bm = np.arange(Kdim * Nn, dtype=float).reshape(Kdim, Nn)
    ref_mm = mm.apply({"A": Am, "B": Bm}, out_shape=(M, Nn))
    vec_mm = vectorize(mm, {"A": Am, "B": Bm}, out_shape=(M, Nn))
    rows.append({
        "case": 4,
        "case_name": "有归约维（乘加 → 收缩）",
        "op": mm.name,
        "n_parallel": mm.iterator_types.count("parallel"),
        "n_reduction": mm.iterator_types.count("reduction"),
        "vector_path": f"einsum {build_einsum_subscripts(mm)}",
        "apply_first_row": ref_mm[0].tolist(),
        "vectorized_first_row": vec_mm[0].tolist(),
        "max_abs_diff": float(np.max(np.abs(ref_mm - vec_mm))),
        "transfers": transfer_counts(mm),
    })

    # ---- 情形 (5)：滑窗——本参考实现显式拒绝 ----
    conv = make_conv_1d_nwc_wcf()
    I = np.ones((1, 8, 2))
    K = np.ones((3, 2, 3))
    try:
        vectorize(conv, {"I": I, "K": K}, out_shape=(1, 6, 3))
        refused = {"raised": False}
    except NotImplementedError as exc:
        refused = {"raised": True, "error_type": "NotImplementedError", "message": str(exc)}
    rows.append({
        "case": 5,
        "case_name": "滑窗（如卷积）",
        "op": conv.name,
        "n_parallel": conv.iterator_types.count("parallel"),
        "n_reduction": conv.iterator_types.count("reduction"),
        "vector_path": "本参考实现不实现，显式拒绝",
        "apply_first_row": None,
        "vectorized_first_row": None,
        "max_abs_diff": None,
        "refusal": refused,
        "transfers": transfer_counts(conv),
    })

    assert rows[0]["max_abs_diff"] == 0.0 and rows[1]["max_abs_diff"] == 0.0
    assert refused["raised"]

    dump("m18", {
        "mechanism": "m18-vectorization-cases",
        "paper_ref": "[Linalg §3.3] paper.md:L353-L367",
        "paper_case_count": 5,
        "cases_covered_by_reference_impl": [1, 4],
        "rows": rows,
    })


if __name__ == "__main__":
    main()
