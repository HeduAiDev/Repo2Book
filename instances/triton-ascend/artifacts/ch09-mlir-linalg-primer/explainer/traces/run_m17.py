"""m17 —— padding 的幺元条件（[Linalg §3.2]，paper.md:L341-L351）。

场景：conv 的归约维（输入通道 c，真实宽度 4）被手工切成 `[0,3)` 与 `[3,4)` 两段。
第二段只有 1 个真实通道，要补到静态尺寸 3 才能和第一段用同一份「满 tile」代码算。
  - 补对幺元（sum 的幺元 = 0）：两段部分和相加 == 不分段的参考结果；
  - 补错幺元（补 1，且 K 侧补位是非零残留值 3）：结果偏离参考值。
两条路的数字都打出来，「补错幺元 → 结果错」因此是可运行的反例，不是断言。

第二段：packing 的**拷贝次数**对照（[Linalg §3.2] 的「外提 pad、存进更高维 packed
张量」）——不外提时每块 tile 各 pad 一次，外提后一次性打包成一个多出一维的 packed
张量。这里只数结构性的次数与形状，不涉及任何性能数字。
"""
from __future__ import annotations

import numpy as np

from _common import dump  # noqa: E402
from named_ops import make_conv_1d_nwc_wcf  # noqa: E402
from padding import NEUTRAL_ELEMENTS, neutral_element, pad_to_static  # noqa: E402
from tiling import extract_slice, image_of_domain, tile  # noqa: E402


def main() -> None:
    op = make_conv_1d_nwc_wcf()
    N, W_in, C, F, KW = 1, 8, 4, 2, 3
    out_w = W_in - KW + 1
    I = np.zeros((N, W_in, C))
    for w in range(W_in):
        for c in range(C):
            I[0, w, c] = w + 1 + 10 * c
    K = np.ones((KW, C, F))  # 全 1 核，便于读者心算
    out_shape = (N, out_w, F)

    reference = op.apply({"I": I, "K": K}, out_shape=out_shape)

    split = 3
    I_a, K_a = I[:, :, :split], K[:, :split, :]
    I_b, K_b = I[:, :, split:], K[:, split:, :]  # 只有 C-split=1 个真实通道
    partial_a = op.apply({"I": I_a, "K": K_a}, out_shape=out_shape)

    rows = []
    for label, i_pad, k_pad in [
        ("正确幺元（sum → 0）", neutral_element("sum"), 999.0),
        ("错误幺元（补 1）", 1.0, 3.0),
    ]:
        I_bp = pad_to_static(I_b, target_shape=(N, W_in, split), neutral=i_pad)
        K_bp = pad_to_static(K_b, target_shape=(KW, split, F), neutral=k_pad)
        partial_b = op.apply({"I": I_bp, "K": K_bp}, out_shape=out_shape)
        combined = partial_a + partial_b
        rows.append({
            "strategy": label,
            "I_pad_value": i_pad,
            "K_pad_value": k_pad,
            "padded_channels": split - I_b.shape[2],
            "partial_b_O_0_0_0": float(partial_b[0, 0, 0]),
            "combined_O_0_0_0": float(combined[0, 0, 0]),
            "reference_O_0_0_0": float(reference[0, 0, 0]),
            "max_abs_diff_vs_reference": float(np.max(np.abs(combined - reference))),
            "matches_reference": bool(np.allclose(combined, reference)),
        })
    assert rows[0]["matches_reference"] and not rows[1]["matches_reference"]
    assert rows[1]["max_abs_diff_vs_reference"] > 0

    # ---- packing：pad 外提前后的拷贝次数与形状 ----
    global_domain = op.iteration_domain({"I": I.shape, "K": K.shape}, out_shape)
    tile_w = 4
    tiles = list(tile(op, global_domain, {"w": tile_w}))
    i_images = [image_of_domain(t, op.operand_maps["I"]) for t in tiles]
    widths = [img[1][1] - img[1][0] for img in i_images]
    target_w = max(widths)
    padded = [pad_to_static(extract_slice(I, img), (N, target_w, C), neutral_element("sum"))
              for img in i_images]
    packed = np.stack(padded)  # 多出一维的 packed 张量：所有 tile 连续排布
    packing = {
        "tile_w": tile_w,
        "n_tiles": len(tiles),
        "I_image_widths": widths,
        "padded_static_width": target_w,
        "pad_calls_inside_loop": len(tiles),
        "packed_tensor_allocations_after_hoisting": 1,
        "packed_shape": list(packed.shape),
        "packed_is_contiguous": bool(packed.flags["C_CONTIGUOUS"]),
    }

    dump("m17", {
        "mechanism": "m17-padding-packing",
        "paper_ref": "[Linalg §3.2] paper.md:L341-L351",
        "params": {"N": N, "W_in": W_in, "C": C, "F": F, "KW": KW, "out_w": out_w,
                   "reduction_split_at_c": split, "real_channels_in_segment_b": int(I_b.shape[2])},
        "neutral_table": {k: (str(v) if not np.isfinite(v) else v)
                          for k, v in NEUTRAL_ELEMENTS.items()},
        "reference_O_0_0_0": float(reference[0, 0, 0]),
        "partial_a_O_0_0_0": float(partial_a[0, 0, 0]),
        "strategies": rows,
        "packing": packing,
    })


if __name__ == "__main__":
    main()
