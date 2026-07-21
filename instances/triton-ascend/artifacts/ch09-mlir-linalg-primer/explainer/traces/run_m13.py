"""m13 —— 算子自带索引表达式：O[n,w,f] = I[n,w+kw,c] · K[kw,c,f]（[Linalg §3]，paper.md:L248-L264）。

做两件事：
1. 用论文原始类型（O:1x988x64、I:1x990x32、K:3x32x64）核对形状关系 988 = 990 - 3 + 1；
2. 换一组读者能心算的小参数（N=1, W=8, C=2, F=3, KW=3 → out_w=6），把 O[0,0,0] 这一个
   输出格子的索引表达式**逐个迭代点**求值：每一步读哪两个元素、乘出多少、累加到多少。
   最后与 `StructuredOp.apply` 的结果对账。
"""
from __future__ import annotations

import itertools

import numpy as np

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402


def main() -> None:
    op = make_conv_1d_nwc_wcf()

    # ---- 1) 论文原始形状的关系核对 ----
    paper = {
        "I_shape": [1, 990, 32],
        "K_shape": [3, 32, 64],
        "O_shape": [1, 988, 64],
    }
    paper["out_w_from_shapes"] = paper["I_shape"][1] - paper["K_shape"][0] + 1  # 990 - 3 + 1
    paper["matches_O_1"] = paper["out_w_from_shapes"] == paper["O_shape"][1]

    # ---- 2) 小参数逐点求值 ----
    N, W_in, C, F, KW = 1, 8, 2, 3, 3
    out_w = W_in - KW + 1
    I = np.zeros((N, W_in, C))
    for w in range(W_in):
        for c in range(C):
            I[0, w, c] = w + 1 + 10 * c  # c=0 -> 1..8, c=1 -> 11..18
    K = np.zeros((KW, C, F))
    for kw in range(KW):
        for c in range(C):
            for f in range(F):
                K[kw, c, f] = (kw + 1) * (1 if c == 0 else 2) * (f + 1)

    n_i, w_i, f_i, kw_i, c_i = (op.dim_names.index(x) for x in ("n", "w", "f", "kw", "c"))
    target = {"n": 0, "w": 0, "f": 0}  # 盯住输出格子 O[0,0,0]

    steps = []
    acc = 0.0
    for kw, c in itertools.product(range(KW), range(C)):
        point = [0] * 5
        point[n_i], point[w_i], point[f_i] = target["n"], target["w"], target["f"]
        point[kw_i], point[c_i] = kw, c
        point = tuple(point)
        i_idx = tuple(e.eval(point) for e in op.operand_maps["I"])
        k_idx = tuple(e.eval(point) for e in op.operand_maps["K"])
        o_idx = tuple(e.eval(point) for e in op.result_map)
        prod = float(I[i_idx] * K[k_idx])
        acc = float(op.body(acc, I[i_idx], K[k_idx]))
        steps.append({
            "kw": kw,
            "c": c,
            "I_index": list(i_idx),          # = [n, w+kw, c]
            "I_value": float(I[i_idx]),
            "K_index": list(k_idx),          # = [kw, c, f]
            "K_value": float(K[k_idx]),
            "product": prod,
            "acc_after": acc,
            "O_index": list(o_idx),
        })

    out = op.apply({"I": I, "K": K}, out_shape=(N, out_w, F))
    small = {
        "N": N, "W_in": W_in, "C": C, "F": F, "KW": KW,
        "out_w": out_w,
        "out_w_formula": f"{W_in} - {KW} + 1 = {out_w}",
        "I_shape": list(I.shape),
        "K_shape": list(K.shape),
        "O_shape": list(out.shape),
        "iteration_points_per_output_cell": KW * C,
        "steps_for_O_0_0_0": steps,
        "acc_final": acc,
        "apply_O_0_0_0": float(out[0, 0, 0]),
        "match": bool(abs(acc - float(out[0, 0, 0])) < 1e-12),
        "abs_diff": abs(acc - float(out[0, 0, 0])),
    }
    assert small["match"], "逐点手工求值与 StructuredOp.apply 不一致"
    assert paper["matches_O_1"]

    dump("m13", {
        "mechanism": "m13-indexing-expression",
        "paper_ref": "[Linalg §3] paper.md:L248-L264",
        "paper_shapes": paper,
        "small_example": small,
    })


if __name__ == "__main__":
    main()
