"""m16 —— tiling 的不动点：循环体里仍是**同一个**结构化算子，只是张量更小
（[Linalg §3.1]，paper.md:L333-L339）。

小参数（N=1, W_in=8, C=2, F=3, KW=3 → out_w=6），跑两种 tile size：
  - tile_w=4：除不尽，切出 [0,4) 与 [4,6) 两块，**局部输出类型不同**（1x4x3 / 1x2x3）
    ——论文说的「没有哪个静态张量类型对每次迭代都合法」在这里是可以打印出来的；
  - tile_w=3：除得尽，两块都是满 tile。
每块 tile 记录：局部迭代域、各操作数切片形状、执行它的算子对象（名字 + Python id，
证明是同一个实例而非另写的小算子）、以及与「不切、一次算完」的参考结果的最大偏差。
"""
from __future__ import annotations

import numpy as np

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402
from tiling import extract_slice, image_of_domain, insert_slice, tile, tile_and_run  # noqa: E402


def make_data(N=1, W_in=8, C=2, F=3, KW=3):
    I = np.zeros((N, W_in, C))
    for w in range(W_in):
        for c in range(C):
            I[0, w, c] = w + 1 + 10 * c
    K = np.zeros((KW, C, F))
    for kw in range(KW):
        for c in range(C):
            for f in range(F):
                K[kw, c, f] = (kw + 1) * (1 if c == 0 else 2) * (f + 1)
    return I, K, (N, W_in - KW + 1, F)


def run_tile_size(op, I, K, out_shape, tile_w, reference):
    outer_op_id = id(op)  # 循环外那个算子对象；下面每块 tile 都要证明用的还是它
    w_i = op.dim_names.index("w")
    global_domain = op.iteration_domain({"I": I.shape, "K": K.shape}, out_shape)
    out = np.zeros(out_shape)
    rows = []
    for t, local in enumerate(tile(op, global_domain, {"w": tile_w})):
        local_ins = {n: extract_slice(
            {"I": I, "K": K}[n], image_of_domain(local, op.operand_maps[n]))
            for n in op.operand_names}
        o_ranges = image_of_domain(local, op.result_map)
        local_out_shape = tuple(hi - lo for lo, hi in o_ranges)
        body_op = op  # 循环体里执行的算子——不是另写的小算子，就是外面那一个
        local_result = body_op.apply(local_ins, local_out_shape)
        insert_slice(out, o_ranges, local_result)
        ref_slice = reference[tuple(slice(lo, hi) for lo, hi in o_ranges)]
        rows.append({
            "tile_index": t,
            "domain_w": list(local[w_i]),
            "tile_width_w": local[w_i][1] - local[w_i][0],
            "is_full_tile": (local[w_i][1] - local[w_i][0]) == tile_w,
            "I_slice_shape": list(local_ins["I"].shape),
            "K_slice_shape": list(local_ins["K"].shape),
            "local_out_shape": list(local_out_shape),
            "op_name_in_loop_body": body_op.name,
            "op_is_the_same_python_object": id(body_op) == outer_op_id,
            "iterator_types_in_loop_body": list(body_op.iterator_types),
            "local_max_abs_diff_vs_reference": float(np.max(np.abs(local_result - ref_slice))),
        })
    end_to_end = tile_and_run(op, {"I": I, "K": K}, out_shape, {"w": tile_w})
    distinct_shapes = sorted({tuple(r["local_out_shape"]) for r in rows})
    return {
        "tile_w": tile_w,
        "n_tiles": len(rows),
        "tiles": rows,
        "distinct_local_out_shapes": [list(s) for s in distinct_shapes],
        "n_distinct_local_out_shapes": len(distinct_shapes),
        "max_abs_diff_vs_reference": float(np.max(np.abs(end_to_end - reference))),
        "manual_loop_matches_tile_and_run": bool(np.array_equal(out, end_to_end)),
    }


def main() -> None:
    op = make_conv_1d_nwc_wcf()
    I, K, out_shape = make_data()
    reference = op.apply({"I": I, "K": K}, out_shape=out_shape)

    cases = [run_tile_size(op, I, K, out_shape, tw, reference) for tw in (4, 3)]
    for c in cases:
        assert c["max_abs_diff_vs_reference"] == 0.0
        assert c["manual_loop_matches_tile_and_run"]

    # 归约维不许分块（本参考实现的范围收窄，tiling.py 模块文档）
    try:
        list(tile(op, op.iteration_domain({"I": I.shape, "K": K.shape}, out_shape), {"c": 1}))
        reduction_tiling = {"raised": False}
    except ValueError as exc:
        reduction_tiling = {"raised": True, "error_type": "ValueError", "message": str(exc)}

    dump("m16", {
        "mechanism": "m16-tiling-same-op",
        "paper_ref": "[Linalg §3.1] paper.md:L333-L339",
        "params": {"N": 1, "W_in": 8, "C": 2, "F": 3, "KW": 3, "out_shape": list(out_shape)},
        "reference_O_0_0_0": float(reference[0, 0, 0]),
        "cases": cases,
        "reduction_dim_tiling_rejected": reduction_tiling,
    })


if __name__ == "__main__":
    main()
