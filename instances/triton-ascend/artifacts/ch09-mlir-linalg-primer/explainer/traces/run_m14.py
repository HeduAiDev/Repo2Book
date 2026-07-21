"""m14 —— 隐式迭代域：边界不是写出来的，是从「迭代器扫过操作数全部数据」反解出来的
（[Linalg §3]，paper.md:L266-L278）。

跑三件事：
1. 逐维扫描每个操作数（含 `outs`）的每一轴，记录哪些轴是该迭代维的**纯恒等**映射
   ——这是本参考实现唯一认可的「可读出边界」形态（structured_op.py 的
   `AffineExpr.pure_dim` / `derive_iteration_domain`）；
2. 在论文原始形状（I:1x990x32、K:3x32x64、O:1x988x64）上跑 `derive_iteration_domain`，
   对照论文给出的那 5 条不等式；
3. **反证**：把 `outs`（O）从操作数集合里拿掉，`w` 维立刻没有任何纯恒等来源——
   参考实现显式抛 ValueError（不去猜）。这正是「迭代器扫过操作数全部数据」里的
   「操作数」天然含 outs 的可运行证据。
"""
from __future__ import annotations

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402
from structured_op import derive_iteration_domain  # noqa: E402


def scan_pure_identity_sources(op, shapes):
    """逐维列出「哪个操作数的哪一轴是该迭代维的纯恒等映射」。"""
    per_dim = {name: [] for name in op.dim_names}
    all_maps = [(n, op.operand_maps[n]) for n in op.operand_names] + [("O", op.result_map)]
    axis_expr_shape = []
    for operand, imap in all_maps:
        for axis, expr in enumerate(imap):
            d = expr.pure_dim()
            terms = "+".join(op.dim_names[di] for di, _ in expr.terms)
            axis_expr_shape.append({
                "operand": operand,
                "axis": axis,
                "expr": terms,
                "is_pure_identity": d is not None,
                "extent": shapes[operand][axis],
            })
            if d is not None:
                per_dim[op.dim_names[d]].append({
                    "operand": operand, "axis": axis, "extent": shapes[operand][axis]
                })
    return per_dim, axis_expr_shape


# 论文自己写出的那 5 条不等式各自的边界来源（paper.md:L274-L278，逐字对应
# `0 <= n < O.0`, `0 <= w < O.1`, `0 <= f < O.2`, `0 <= kw < K.0`, `0 <= c < K.1`）。
PAPER_BOUND_SOURCE = {"n": "O.0", "w": "O.1", "f": "O.2", "kw": "K.0", "c": "K.1"}


def run_case(op, label, shapes):
    ins_shapes = {"I": tuple(shapes["I"]), "K": tuple(shapes["K"])}
    per_dim, axis_table = scan_pure_identity_sources(op, shapes)
    domain = op.iteration_domain(ins_shapes, tuple(shapes["O"]))
    dims = []
    total = 1
    for d, name in enumerate(op.dim_names):
        lo, hi = domain[d]
        trip = hi - lo
        total *= trip
        dims.append({
            "dim": name,
            "iterator_type": op.iterator_types[d],
            "bound_lo": lo,
            "bound_hi": hi,
            "trip_count": trip,
            "paper_bound_source": PAPER_BOUND_SOURCE[name],
            "all_pure_identity_sources": [f"{s['operand']}.{s['axis']}" for s in per_dim[name]],
            "n_pure_identity_sources": len(per_dim[name]),
            "all_sources_agree": len({s["extent"] for s in per_dim[name]}) == 1,
        })
    return {
        "label": label,
        "shapes": shapes,
        "axis_table": axis_table,
        "dims": dims,
        "total_iteration_points": total,
    }


def main() -> None:
    op = make_conv_1d_nwc_wcf()

    paper_case = run_case(op, "论文原始形状", {
        "I": [1, 990, 32], "K": [3, 32, 64], "O": [1, 988, 64]})
    small_case = run_case(op, "小参数（可心算）", {
        "I": [1, 8, 2], "K": [3, 2, 3], "O": [1, 6, 3]})

    # 反证：拿掉 outs（O），w 维无处可读
    try:
        derive_iteration_domain(
            op.dim_names,
            [(op.operand_maps["I"], (1, 990, 32)), (op.operand_maps["K"], (3, 32, 64))],
        )
        without_outs = {"raised": False}
    except ValueError as exc:
        without_outs = {"raised": True, "error_type": "ValueError", "message": str(exc)}
    assert without_outs["raised"], "拿掉 outs 后本应无法反解 w 维"

    dump("m14", {
        "mechanism": "m14-implicit-iteration-domain",
        "paper_ref": "[Linalg §3] paper.md:L266-L278",
        "cases": [paper_case, small_case],
        "derive_without_outs": without_outs,
    })


if __name__ == "__main__":
    main()
