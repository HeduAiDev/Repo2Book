"""m15 —— 求像即子集：「这块循环碰哪片数据」是一次代数计算
（[Linalg §3.1]，paper.md:L280-L284）。

对同一个 `conv_1d_nwc_wcf`，取若干个切过的迭代域，逐张量调用 `image_of_domain`
（= 逐轴对索引表达式求像），记录：
  - O 的像 = tile 本身（索引全是纯恒等）；
  - I 的像在 w 轴上**比 tile 宽**（索引是 `w+kw` 的耦合）——宽出来的部分就是滑窗算子的 halo；
  - K 的像与空间 tile 无关（K 只依赖 kw/c/f）。
两组参数：论文形状 + 论文 tile size `1x8x32x1x8` 的空间维（w tile = 8）；以及读者能
心算的小参数（out_w=6、tile_w=4，故意除不尽，最后一块只有 2）。
"""
from __future__ import annotations

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402
from tiling import image_of_domain, tile  # noqa: E402


def probe(op, label, shapes, tile_w):
    ins_shapes = {"I": tuple(shapes["I"]), "K": tuple(shapes["K"])}
    global_domain = op.iteration_domain(ins_shapes, tuple(shapes["O"]))
    w_i = op.dim_names.index("w")
    kw_i = op.dim_names.index("kw")
    kw_extent = global_domain[kw_i][1] - global_domain[kw_i][0]
    rows = []
    for t, local in enumerate(tile(op, global_domain, {"w": tile_w})):
        w_lo, w_hi = local[w_i]
        i_img = image_of_domain(local, op.operand_maps["I"])
        k_img = image_of_domain(local, op.operand_maps["K"])
        o_img = image_of_domain(local, op.result_map)
        rows.append({
            "tile_index": t,
            "domain_w": [w_lo, w_hi],
            "tile_width_w": w_hi - w_lo,
            "kernel_width_kw": kw_extent,
            "I_image": [list(r) for r in i_img],
            "I_image_w": list(i_img[1]),
            "I_image_width_w": i_img[1][1] - i_img[1][0],
            "halo_extra": (i_img[1][1] - i_img[1][0]) - (w_hi - w_lo),
            "K_image": [list(r) for r in k_img],
            "K_image_equals_full_K": [list(r) for r in k_img] == [[0, s] for s in shapes["K"]],
            "O_image": [list(r) for r in o_img],
            "O_image_w": list(o_img[1]),
            "O_image_equals_tile": list(o_img[1]) == [w_lo, w_hi],
        })
        # 像宽 = tile 宽 + 核宽 - 1（滑窗耦合 w+kw 的直接后果）
        assert rows[-1]["I_image_width_w"] == (w_hi - w_lo) + kw_extent - 1
    # tile 数可能很多（论文形状按 8 切 w 有 124 块）——trace 只留首 3 块与最后一块，
    # 另附全量统计；「像宽 = tile 宽 + 核宽 - 1」已在上面对**每一块**逐块 assert 过。
    sampled = rows[:3] + rows[-1:] if len(rows) > 4 else rows
    widths = sorted({r["tile_width_w"] for r in rows})
    return {
        "label": label,
        "shapes": shapes,
        "tile_w": tile_w,
        "n_tiles": len(rows),
        "distinct_tile_widths": widths,
        "n_full_tiles": sum(1 for r in rows if r["tile_width_w"] == tile_w),
        "all_tiles_satisfy_image_width_rule": True,  # 上面逐块 assert 通过才会走到这里
        "tiles_sampled": sampled,
    }


def main() -> None:
    op = make_conv_1d_nwc_wcf()
    paper = probe(op, "论文形状 + 论文 tile size 的 w 维（8）",
                  {"I": [1, 990, 32], "K": [3, 32, 64], "O": [1, 988, 64]}, tile_w=8)
    small = probe(op, "小参数（out_w=6，tile_w=4，除不尽）",
                  {"I": [1, 8, 2], "K": [3, 2, 3], "O": [1, 6, 3]}, tile_w=4)
    dump("m15", {
        "mechanism": "m15-subset-by-image",
        "paper_ref": "[Linalg §3.1] paper.md:L280-L284",
        "cases": [paper, small],
        "note": "kw/c 是归约维，本参考实现不对它们分块（tiling.py 的范围收窄），故 K 的像始终是完整 K",
    })


if __name__ == "__main__":
    main()
