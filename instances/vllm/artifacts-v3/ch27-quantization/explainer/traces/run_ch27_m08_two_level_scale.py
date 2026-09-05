"""ch27-m08 驱动脚本 —— e8m0 块缩放(2 的幂)与 FP4(e2m1) 两级缩放的格式账。

跑法(host, 纯 CPU numpy): python run_ch27_m08_two_level_scale.py
输出: ch27_m08_two_level_scale.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m08(figure-only)。e8m0/e2m1/NVFP4 是 OCP
Microscaling(MX)/NVIDIA Blackwell 的硬件格式约定(不在论文包);vLLM 落地
对照:vllm/model_executor/layers/quantization/input_quant_fp8.py:L243-L244
(ue8m0:scales_raw = exp2(ceil(log2(scales_raw))) 一行即 e8m0 数学)、
modelopt.py:L1151-L1162(e2m1 两值一字节打包)+ L1178-L1190(每 16 输入
元素一个 e4m3 块 scale + 全局 fp32)+ L1013(group_size=16)。

三件:
① e8m0 = 8 位纯指数(无尾数)→ scale 只能是 2 的幂;取整 exp2∘ceil∘log2,
   代价:scale 平均多花 2^{E[1-frac]} = 2/ln2 ≈ 1.4427 倍(数值验证)。
② e2m1 格点 = ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}(16 个,含 ±0);min 非零 0.5
   与 max 6 之比 12——单级 scale 撑不住块内动态范围。
③ 两级缩放的必要性:块 A(16 值全 ~0.1)+ 块 B(16 值含 6.0)。单级全局
   scale(把全张量 amax 映到 6):块 A 全部坍缩到 0/0.5;两级(块 scale
   e4m3 × 全局 fp32):块 A 自带小尺子,误差降一个量级。
"""
import json
from pathlib import Path

import numpy as np


def r(v, nd=6):
    return round(float(v), nd)


def e2m1_grid():
    """e2m1 全部正格点:1 指数位(bias 1)+ 1 尾数位。

    e=0 次正规:m/2 × 2^(1-bias) = m/2 → {0.5}(m=1);
    e>=1 正规:(1+m/2) × 2^(e-1) → {1, 1.5, 2, 3, 4, 6}。
    编码共 16(±{0, 0.5, 1, 1.5, 2, 3, 4, 6},±0 同值)。
    """
    vals = set()
    for e in range(4):
        for m in range(2):
            if e == 0:
                if m == 0:
                    continue
                v = (m / 2.0) * 2.0**0  # bias=1: 次正规 = m/2 × 2^0 = 0.5
            else:
                v = (1.0 + m / 2.0) * 2.0 ** (e - 1)
            vals.add(v)
    return sorted(vals)


def e4m3_positive():
    vals = []
    for e in range(16):
        for m in range(8):
            if e == 15 and m == 7:
                continue
            if e == 0:
                if m == 0:
                    continue
                v = (m / 8.0) * 2.0**-6
            else:
                v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
            vals.append(v)
    return sorted(vals)


def quantize_to(x, grid_pos):
    """对称:round-to-nearest 到 ±grid。"""
    g = np.asarray(sorted({-v for v in grid_pos} | {0.0} | set(grid_pos)))
    idx = np.argmin(np.abs(g - x))
    return float(g[idx])


def main():
    out = {}

    # ---- ① e8m0:2 的幂 + ceil 取整的代价 ----
    # 抽样假设:u = frac(log2 s) 在 [0,1) 均匀(对数均匀)——理论因子
    # E[2^{1-u}] = ∫_0^1 2^{1-u} du = 1/ln2 ≈ 1.4427 的口径。
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 1.0, 1_000_000)
    raw = 2.0 ** (rng.integers(-4, 3, 1_000_000) + u)  # log 均匀 -> u 均匀
    ceiled = 2.0 ** np.ceil(np.log2(raw))
    factor = ceiled / raw
    examples_raw = [0.013, 0.02, 0.0037, 1.2, 3.0]
    out["e8m0"] = {
        "bit_layout": "8 位纯指数(无尾数),bias 127 → 值域 2^-127 .. 2^127",
        "representable_values": "只能是 2 的幂(尾数位为 0)",
        "rounding_rule": "exp2(ceil(log2(·))) —— 向上取到最近 2 的幂(不溢出优先)",
        "vllm_line": "vllm/model_executor/layers/quantization/input_quant_fp8.py:L243-L244 use_ue8m0 分支",
        "examples": [
            {"raw": r(v, 4), "ceiled": r(2.0 ** np.ceil(np.log2(v)), 4)}
            for v in examples_raw
        ],
        "mean_scale_overhead_factor": r(factor.mean(), 4),
        "theoretical_factor": r(1.0 / np.log(2.0), 4),
        "overhead_formula": "E[2^{1-u}] (u = frac(log2 s) 在 [0,1) 均匀) = 1/ln2 ≈ 1.4427:平均多花 ~44% scale,换不溢出保证",
        "n_samples": 1000000,
    }

    # ---- ② e2m1 格点 ----
    g2 = e2m1_grid()
    out["e2m1"] = {
        "positive_grid": [r(v, 3) for v in g2],
        "positive_count": len(g2),
        "total_with_sign_and_zero": 2 * len(g2) + 1,
        "max": 6.0,
        "min_nonzero": 0.5,
        "in_grid_dynamic_range_ratio": r(6.0 / 0.5, 1),
        "note": "e2m1:1 指数位(bias 1)+ 1 尾数位;格点间倍距:0.5,1 间距 0.5;2,3 间距 1;4,6 间距 2——段间倍增;16 个格点(含 ±0)就是全部可用精度",
    }

    # ---- ③ 两级缩放的必要性 ----
    E4M3 = e4m3_positive()

    def e4m3_round(x):
        g = np.asarray(E4M3)
        return float(g[np.argmin(np.abs(g - x))])

    rng = np.random.default_rng(3)
    block_a = np.round(rng.uniform(0.08, 0.12, 16), 3)  # 16 值全 ~0.1
    block_b = np.round(rng.uniform(0.5, 6.0, 15), 3)
    block_b[0] = 6.0  # 张量 amax 定尺者
    tensor_amax = 6.0

    # 单级:全局 scale 把全张量 amax 映到 e2m1 max=6
    s_single = tensor_amax / 6.0
    qa_single = np.array([quantize_to(v / s_single, g2) for v in block_a])
    qa_single_real = qa_single * s_single
    qb_single = np.array([quantize_to(v / s_single, g2) for v in block_b])
    qb_single_real = qb_single * s_single
    err_single_a = np.abs(block_a - qa_single_real)

    # 两级:每块 scale 把块 amax 映到 6,块 scale 本身存 e4m3(再被全局 fp32 归一)
    sa_raw = float(np.abs(block_a).max()) / 6.0
    sa_e4m3 = e4m3_round(sa_raw)
    qa_two = np.array([quantize_to(v / sa_e4m3, g2) for v in block_a])
    qa_two_real = qa_two * sa_e4m3
    err_two_a = np.abs(block_a - qa_two_real)

    out["two_level_necessity"] = {
        "block_a": [r(v, 3) for v in block_a],
        "block_b_amax": 6.0,
        "block_a_amax": r(float(np.abs(block_a).max()), 3),
        "single_level": {
            "global_scale_tensor_amax_over_6": r(s_single, 3),
            "block_a_quantized": [r(v, 2) for v in qa_single],
            "block_a_dequant": [r(v, 4) for v in qa_single_real],
            "block_a_abs_err": [r(v, 4) for v in err_single_a],
            "block_a_mean_abs_err": r(err_single_a.mean(), 4),
            "block_a_values_collapsed_to": sorted({r(v, 2) for v in qa_single}),
            "block_b_mean_abs_err": r(np.abs(block_b - qb_single_real).mean(), 4),
        },
        "two_level": {
            "block_a_scale_raw_amax_over_6": r(sa_raw, 5),
            "block_a_scale_stored_e4m3": r(sa_e4m3, 6),
            "block_a_quantized": [r(v, 2) for v in qa_two],
            "block_a_dequant": [r(v, 4) for v in qa_two_real],
            "block_a_abs_err": [r(v, 4) for v in err_two_a],
            "block_a_mean_abs_err": r(err_two_a.mean(), 4),
            "block_a_values_used": sorted({r(v, 2) for v in qa_two}),
        },
        "err_improvement_factor": r(err_single_a.mean() / err_two_a.mean(), 1),
        "nvfp4_layout_vllm": {
            "weight_pack": "modelopt.py:L1151-L1162:两个 e2m1 打包进一个 uint8(input 维 //2)",
            "block_scale": "modelopt.py:L1178-L1190:每 group_size=16(L1013)个输入元素一个 e4m3 块 scale",
            "global_scale": "input_scale/weight_scale_2 两个 fp32 per-tensor 标量",
            "alpha_precompute": "modelopt.py:L1216-L1219:装载期预计算 alpha = input_global_scale × weight_global_scale(两级乘积)",
            "quantkey": "quant_utils.py:L148-L156:ScaleDesc(1,16) 的 scale + kStaticTensorScale 的 scale2 双层描述",
        },
        "claim": "e2m1 块内动态范围只有 6/0.5=12:单级全局 scale 下块 A(全 ~0.1,张量 amax=6)的值映射到 0.0833-0.125,离最近格点 0.5 还差 4-6 倍,几乎全部坍缩到 0;两级给块 A 自带 0.02 的小尺子,16 值全落在 4-6 的格点上",
    }

    p = Path(__file__).with_name("ch27_m08_two_level_scale.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
