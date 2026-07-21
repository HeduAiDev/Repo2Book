"""m22 —— 多维向量算子的渐进下降（[Linalg §3.5]，paper.md:L375-L377）。

论文自列 5 步 (a)-(e)：(a) vector unrolling、(b) 把 transfer_read 里的转置物化、
(c) 生成 1 维 load 与广播、(d) 把 contract 降成外积（论文注明也可选内积或 LLVM 矩阵
intrinsic）、(e) 映射到 SIMD 的 fused multiply-add。

本驱动脚本只把其中**可以在 CPU/NumPy 上诚实数出来**的两步变成数字：
  - (a) unrolling 的「非 2 的幂拆成 2 的幂的组合」：论文原例 `vector<12xf32>` 拆成
    3 个 `vector<4xf32>`——这里就按这个例子算拆分数与余数；
  - (d)(e) contract 降外积：把矩阵乘的收缩维逐个展开成**秩 1 更新**（外积累加），
    记录每一步累加器的状态，并与 `StructuredOp.apply` / einsum 一条路对账。
(b)(c) 是 IR 形态上的改写（转置物化、1 维 load/广播），在纯 NumPy 数组层面没有可诚实
观测的对应量，本脚本不编造——留给正文与配图讲。

论文最终得到的向量 IR 作用在 `vector<8xf32>`（论文举的是 AVX2 宽度）——**这是论文的
CPU 例子**，本脚本沿用它只是为了对齐论文叙述，与昇腾 NPU 无对位关系。
"""
from __future__ import annotations

import numpy as np

from _common import dump  # noqa: E402
from named_ops import make_matmul  # noqa: E402
from vectorization import build_einsum_subscripts  # noqa: E402


def unroll_plan(total: int, target: int) -> dict:
    """(a) 把长度 `total` 的一维向量拆成若干个长度 `target` 的片段。"""
    full = total // target
    rest = total % target
    return {
        "source_vector_len": total,
        "target_vector_len": target,
        "n_full_pieces": full,
        "remainder": rest,
        "is_exact": rest == 0,
    }


def main() -> None:
    mm = make_matmul()
    M, Kdim, Nn = 2, 3, 8  # N=8 对齐论文最终的 vector<8xf32>（论文的 CPU/AVX2 例子）
    A = np.arange(1, M * Kdim + 1, dtype=float).reshape(M, Kdim)
    B = np.arange(1, Kdim * Nn + 1, dtype=float).reshape(Kdim, Nn)
    reference = mm.apply({"A": A, "B": B}, out_shape=(M, Nn))
    einsum_subs = build_einsum_subscripts(mm)
    contracted = np.einsum(einsum_subs, A, B)

    # (d)(e)：contract → 逐个收缩维的外积（秩 1 更新）累加
    acc = np.zeros((M, Nn))
    steps = []
    for k in range(Kdim):
        a_col = A[:, k]          # (c) 1 维 load：取出参与本次外积的一列
        b_row = B[k, :]          # (c) 1 维 load + 广播：本次外积的另一因子
        update = np.outer(a_col, b_row)
        acc = acc + update
        steps.append({
            "k": k,
            "a_col": a_col.tolist(),
            "b_row_first_2": b_row[:2].tolist(),
            "outer_product_shape": list(update.shape),
            "update_0_0": float(update[0, 0]),
            "acc_0_0_after": float(acc[0, 0]),
            "acc_1_0_after": float(acc[1, 0]),
            "fma_on_width_8_vectors_this_step": M,   # 每行一次「广播标量 × 向量 + 累加」
            "scalar_multiply_adds_this_step": M * Nn,
        })

    result = {
        "M": M, "K": Kdim, "N": Nn,
        "einsum_subscripts": einsum_subs,
        "n_rank1_updates": Kdim,
        "total_fma_on_width_8_vectors": M * Kdim,
        "total_scalar_multiply_adds": M * Nn * Kdim,
        "steps": steps,
        "final_acc_0_0": float(acc[0, 0]),
        "reference_0_0": float(reference[0, 0]),
        "max_abs_diff_vs_apply": float(np.max(np.abs(acc - reference))),
        "max_abs_diff_vs_einsum": float(np.max(np.abs(acc - contracted))),
    }
    assert result["max_abs_diff_vs_apply"] == 0.0
    assert result["max_abs_diff_vs_einsum"] == 0.0

    dump("m22", {
        "mechanism": "m22-progressive-vector-lowering",
        "paper_ref": "[Linalg §3.5] paper.md:L375-L377",
        "paper_step_count": 5,
        "unrolling_paper_example": unroll_plan(12, 4),
        "outer_product_lowering": result,
        "not_modeled": ["(b) 转置物化", "(c) 1 维 load/广播的 IR 形态"],
        "cpu_example_note": "vector<8xf32> 是论文举的 AVX2 宽度（CPU 例子），与昇腾 NPU 无对位关系",
    })


if __name__ == "__main__":
    main()
