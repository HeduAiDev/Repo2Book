"""m21 —— bufferization：把不可变张量物化进内存，少分配少拷贝
（[Linalg §3.4]，paper.md:L369-L373；destination-passing style 见 paper.md:L315-L325）。

同一段 tiled 卷积，跑两条 bufferization 策略：
  - naive：每次 `insert_slice` 都当成「产出一个新的完整输出张量」（整份 copy 再写入），
    严格对应张量不可变的函数式语义——永远安全，但每块 tile 多一次整份分配；
  - DPS：`outs` 只分配一次，每块 tile 原地写入。
数值必须完全一致（legal by design），差别只在分配次数。跑三种 tile size，让
「naive 分配次数 = tile 数 + 1、DPS 恒为 1」这条关系有三组数据可比。

计数是 CPU 上的结构性计数，不是性能数字（host 无昇腾 NPU/CANN）。
"""
from __future__ import annotations

import numpy as np

from _common import dump  # noqa: E402
from bufferization import bufferize_dps, bufferize_naive  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402
from tiling import tile  # noqa: E402


def main() -> None:
    op = make_conv_1d_nwc_wcf()
    N, W_in, C, F, KW = 1, 8, 2, 3, 3
    out_w = W_in - KW + 1
    I = np.zeros((N, W_in, C))
    for w in range(W_in):
        for c in range(C):
            I[0, w, c] = w + 1 + 10 * c
    K = np.zeros((KW, C, F))
    for kw in range(KW):
        for c in range(C):
            for f in range(F):
                K[kw, c, f] = (kw + 1) * (1 if c == 0 else 2) * (f + 1)
    out_shape = (N, out_w, F)
    reference = op.apply({"I": I, "K": K}, out_shape=out_shape)

    rows = []
    for tile_w in (1, 2, 4):
        n_tiles = len(list(tile(op, op.iteration_domain(
            {"I": I.shape, "K": K.shape}, out_shape), {"w": tile_w})))
        naive = bufferize_naive(op, {"I": I, "K": K}, out_shape, {"w": tile_w})
        dps = bufferize_dps(op, {"I": I, "K": K}, out_shape, {"w": tile_w})
        rows.append({
            "tile_w": tile_w,
            "n_tiles": n_tiles,
            "naive_alloc_count": naive.OUT_OF_PLACE_ALLOC_COUNT,
            "naive_alloc_equals_n_tiles_plus_1": naive.OUT_OF_PLACE_ALLOC_COUNT == n_tiles + 1,
            "dps_alloc_count": dps.DPS_ALLOC_COUNT,
            "alloc_saved": naive.OUT_OF_PLACE_ALLOC_COUNT - dps.DPS_ALLOC_COUNT,
            "naive_vs_dps_max_abs_diff": float(np.max(np.abs(naive.output - dps.output))),
            "dps_vs_untiled_reference_max_abs_diff": float(np.max(np.abs(dps.output - reference))),
            "O_0_0_0_naive": float(naive.output[0, 0, 0]),
            "O_0_0_0_dps": float(dps.output[0, 0, 0]),
        })
        assert rows[-1]["naive_vs_dps_max_abs_diff"] == 0.0
        assert rows[-1]["dps_vs_untiled_reference_max_abs_diff"] == 0.0
        assert rows[-1]["naive_alloc_equals_n_tiles_plus_1"]

    dump("m21", {
        "mechanism": "m21-bufferization",
        "paper_ref": "[Linalg §3.4] paper.md:L369-L373",
        "params": {"N": N, "W_in": W_in, "C": C, "F": F, "KW": KW, "out_shape": list(out_shape)},
        "reference_O_0_0_0": float(reference[0, 0, 0]),
        "rows": rows,
        "counting_note": "CPU 上的分配次数，非性能数字；host 无昇腾 NPU/CANN",
    })


if __name__ == "__main__":
    main()
