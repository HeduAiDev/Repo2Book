"""ch27-m04 驱动脚本 —— GPTQ 二阶补偿全推导的可示教数值轨迹。

跑法(host, 纯 CPU numpy): python run_ch27_m04_gptq.py
输出: ch27_m04_gptq.json(与本脚本同目录)

素材对应 dossier 机制 ch27-m04,论文 arXiv:2210.17323 §3 Eq.1-Eq.3(OBQ)、
§4 Step 1-3 + Algorithm 1(GPTQ)、§5 Setup/Baselines(网格与 RTN 对照)。

手推件:1×4 权重行 w=[0.2,-0.5,0.9,0.0] + 3 条校准样本 X(列点积全为 0/1,
H=2X^TX 可手算)。block_size=2 让 lazy batch 的「块内即时 / 块末总账」结构
在 4 列上完整出现(两个块)。补偿真的移动了未量化列(非平凡分支:块末更新
改变第 3/4 列的取整决策,与 RTN 码不同)。
对照件:8×12 合成层(corr=0.95)上 GPTQ vs RTN 记分板(§5 Table 3 的
可跑对应物)、Cholesky 路径与朴素逐列路径逐位对账、B 不变式、复杂度账。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "implementation"))
from gptq import (  # noqa: E402
    dequantize_with_grid,
    gptq_naive_inverse_updates,
    gptq_quantize,
    hessian_update_flops,
    inverse_hessian_cholesky,
    layer_hessian,
    layer_output_error,
    obq_quantize_row,
    quantize_with_grid,
    rtn_quantize,
    row_grid_params,
)


def r(v, nd=5):
    return round(float(v), nd)


def toy_xy(seed=3, d_row=8, d_col=12, n_samples=48, corr=0.95):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((d_row, d_col))
    base = rng.standard_normal((n_samples, 1))
    noise = rng.standard_normal((n_samples, d_col))
    x = corr * base + np.sqrt(1 - corr**2) * noise
    return w, x


def main():
    out = {}

    # ================= ① 手推件:1×4 行 + 手算 Hessian =================
    # 4 条 0/1 校准样本(特征 1/2 相关、特征 4 与全体相关;H 正定、可手算);
    # 注意样本数 < 特征数时 H=2X^TX 奇异——n_samples=d_col 是手推的下限。
    w_row = np.array([[0.45, -0.2, -0.05, 0.15]])
    X = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    H = layer_hessian(X)

    out["hand_setup"] = {
        "w_row": [r(v) for v in w_row[0]],
        "X_calibration": [[int(v) for v in row] for row in X],
        "n_samples": 4,
        "d_col": 4,
        "H_2XTX": [[r(v, 2) for v in row] for row in H],
        "H_formula": "H = 2 X^T X,只依赖层输入 X、与权重无关(§4 Step 1 全行同序的合法性来源)",
        "lambda_damp": r(0.01 * np.mean(np.diag(H)), 5),
        "lambda_formula": "dampening = 1% 平均对角元(§4 Step 3)",
        "grid_scale": r(0.65 / 15),
        "grid_zp": -3,
        "grid_formula": "per-row min-max 网格(§5 Setup):scale=(0.45-(-0.2))/15=0.65/15, zp=-8-round(-0.2/scale)=-8-(-5)=-3;过程开始前固定",
    }

    # ---- OBQ 贪心单行(§3 Eq.2-Eq.3):动态挑列 ----
    obq_trace = []
    q_obq, w_hat_obq = obq_quantize_row(w_row[0], H, num_bits=4, trace=obq_trace)
    out["obq_greedy"] = {
        "steps": [
            {"step": i + 1, "picked_col": int(p), "cost": r(c, 6)}
            for i, (p, c) in enumerate(obq_trace)
        ],
        "greedy_order": [int(p) for p, _ in obq_trace],
        "codes": [int(v) for v in q_obq],
        "w_hat": [r(v) for v in w_hat_obq],
        "note": "贪心序 = 每步选 (quant(w)-w)^2/[H^-1]_qq 最小者——不是 0,1,2,3 顺序;§3 OBQ 无 dampening,H 正定可逆",
    }

    # ---- GPTQ 固定列序 + lazy batch(block_size=2):逐列快照 ----
    # 与 gptq_quantize 逐行同构(网格在 j%gs==0 时取当前最新权重;gs=4 全行一组,
    # 即只在 j=0 取一次原始权重——「网格在过程开始前固定」的 §3 假设)。
    U = inverse_hessian_cholesky(X, damp=0.01)
    Wc = w_row.astype(float).copy()
    gs = 4
    E = np.zeros((1, 2))
    i1, i2 = 0, 2
    steps = []
    for j in range(4):
        if j % gs == 0:
            scale, zp = row_grid_params(Wc[:, j : j + gs], 4)
            s0, z0 = float(scale[0]), int(zp[0])
        w_before = float(Wc[0, j])
        q = int(quantize_with_grid(Wc[:, j], s0, z0, 4)[0])
        q_real = float(dequantize_with_grid(np.array([q]), s0, z0)[0])
        e = (Wc[0, j] - q_real) / U[j, j]
        E[0, j - i1] = e
        Wc[:, j:i2] -= np.outer([e], U[j, j:i2])
        Wc[0, j] = q_real
        steps.append(
            {
                "block": "A" if i1 == 0 else "B",
                "col": j,
                "w_before_quant": r(w_before),
                "q": q,
                "w_hat_j": r(q_real),
                "err_div_Ujj": r(e),
                "U_jj": r(U[j, j]),
                "W_after_inblock_update": [r(v) for v in Wc[0]],
            }
        )
        if j == i2 - 1:
            E_block = [r(v) for v in E[0]]
            Wc[:, i2:] -= E @ U[i1:i2, i2:]
            steps.append(
                {
                    "block": "A" if i1 == 0 else "B",
                    "col": j,
                    "stage": "块末 lazy batch 总账",
                    "E_block": E_block,
                    "W_after_block_end_update": [r(v) for v in Wc[0]],
                }
            )
            i1, i2 = i2, min(i2 + 2, 4)
            E = np.zeros((1, 2))

    Q_g, W_hat_g, err_gptq_hand = gptq_quantize(w_row, X, num_bits=4, block_size=2)
    out["gptq_fixed_order_trace"] = {
        "U_upper_cholesky": [[r(v) for v in row] for row in U],
        "U_formula": "Cholesky(H^-1)^T 上三角(Algorithm 1 前置行;U^T U = H^-1)",
        "diagonal_U": [r(U[j, j]) for j in range(4)],
        "steps": steps,
        "final_codes": [int(v) for v in Q_g[0]],
        "final_W_hat": [r(v) for v in W_hat_g[0]],
        "note": "block_size=2:col 0 的误差只即时补偿到 col 1(块内);col 2/3 在块 A 末尾由 E@U 一次性总账——col 2 因此换码(见 RTN 对照)",
    }

    # ---- RTN 对照(同一副网格直接取整,无补偿) ----
    q_rtn, W_hat_rtn = rtn_quantize(w_row, num_bits=4)
    err_rtn_hand = layer_output_error(w_row, W_hat_rtn, X)
    out["hand_scoreboard"] = {
        "gptq_codes": [int(v) for v in Q_g[0]],
        "rtn_codes": [int(v) for v in q_rtn[0]],
        "rtn_w_hat": [r(v) for v in W_hat_rtn[0]],
        "codes_differ_at": [j for j in range(4) if int(Q_g[0][j]) != int(q_rtn[0][j])],
        "err_gptq_layer_output": r(err_gptq_hand, 6),
        "err_rtn_layer_output": r(err_rtn_hand, 6),
        "formula": "‖WX − ŴX‖²_F(Eq.1 记分板;同一副网格,差别只在 round 的顺序与找补)",
    }

    # ================= ② 记分板:8×12 合成层(corr=0.95)=================
    w2, x2 = toy_xy()
    _, _, err_gptq_3bit = gptq_quantize(w2, x2, num_bits=3, block_size=8)
    _, w_hat_rtn2 = rtn_quantize(w2, num_bits=3)
    err_rtn_3bit = layer_output_error(w2, w_hat_rtn2, x2)
    _, _, err_gptq_4bit = gptq_quantize(w2, x2, num_bits=4, block_size=8)
    _, w_hat_rtn2b = rtn_quantize(w2, num_bits=4)
    err_rtn_4bit = layer_output_error(w2, w_hat_rtn2b, x2)
    out["scoreboard_8x12"] = {
        "shape": "8x12, n_samples=48, corr=0.95(特征强相关 -> H 各向异性)",
        "err_rtn_3bit": r(err_rtn_3bit, 4),
        "err_gptq_3bit": r(err_gptq_3bit, 4),
        "ratio_rtn_over_gptq_3bit": r(err_rtn_3bit / err_gptq_3bit, 2),
        "err_rtn_4bit": r(err_rtn_4bit, 4),
        "err_gptq_4bit": r(err_gptq_4bit, 4),
        "ratio_rtn_over_gptq_4bit": r(err_rtn_4bit / err_gptq_4bit, 2),
        "paper_table3_opt175b_wiki2_ppl": {
            "full_fp16": 8.34,
            "rtn_3bit": "7.3e3(崩溃)",
            "gptq_3bit": 8.68,
            "rtn_4bit": 10.54,
            "gptq_4bit": 8.37,
            "source": "arXiv:2210.17323 §5 Table 3(paper.md 表体 L195-L301;同 bit 同网格,差别只在 round 方式)",
        },
    }

    # ================= ③ Step 2/3 只改执行方式不改数学 =================
    q_a, w_a, _ = gptq_quantize(w2, x2, num_bits=4, block_size=8)
    q_b, w_b = gptq_naive_inverse_updates(w2, x2, num_bits=4)
    q_c, _, err_c = gptq_quantize(w2, x2, num_bits=4, block_size=3)
    out["equivalence_checks"] = {
        "cholesky_lazy_batch_vs_naive_per_column_codes_equal": bool(
            np.array_equal(q_a, q_b)
        ),
        "w_hat_max_abs_diff": r(np.abs(w_a - w_b).max(), 8),
        "block_size_8_vs_3_codes_equal": bool(np.array_equal(q_a, q_c)),
        "err_block8_vs_block3_equal": bool(err_c == gptq_quantize(w2, x2, num_bits=4, block_size=8)[2]),
        "claim": "Step 2(lazy batch)与 Step 3(Cholesky)不改变量化结果,只改执行方式——逐位对账",
    }

    # ---- 贪心序 vs 固定序误差相近(§4 Step 1 "similar") ----
    w3, x3 = toy_xy(seed=7, d_row=1, d_col=10, n_samples=24)
    h3 = layer_hessian(x3)
    t3 = []
    _, w_hat_obq3 = obq_quantize_row(w3[0], h3, num_bits=3, trace=t3)
    err_obq3 = layer_output_error(w3, w_hat_obq3[None, :], x3)
    _, _, err_gptq3 = gptq_quantize(w3, x3, num_bits=3, block_size=8)
    out["greedy_vs_fixed_order"] = {
        "obq_greedy_order": [int(p) for p, _ in t3],
        "err_obq_greedy": r(err_obq3, 4),
        "err_gptq_fixed_order": r(err_gptq3, 4),
        "ratio": r(max(err_obq3, err_gptq3) / min(err_obq3, err_gptq3), 2),
        "claim": "贪心挑列与固定列序最终误差相近(§4 Step 1;GPTQ 舍弃贪心换全行同序——H 只认输入,全行共享同一次 H^-1 更新)",
    }

    # ================= ④ 复杂度账(§4 Step 1) =================
    obq_f, gptq_f = hessian_update_flops(4096, 4096)
    obq_s, gptq_s = hessian_update_flops(8, 12)
    out["complexity"] = {
        "flops_4096x4096": {
            "obq_d_row_d_col_cubed": int(obq_f),
            "gptq_max_d_row_d_col_sq_d_col_cubed": int(gptq_f),
            "speedup_min_drow_dcol": int(obq_f / gptq_f),
        },
        "flops_8x12_toy": {"obq": int(obq_s), "gptq": int(gptq_s), "speedup": int(obq_s / gptq_s)},
        "formula": "OBQ O(d_row·d_col^3) -> GPTQ O(max{d_row·d_col^2, d_col^3}),提速 min{d_row,d_col}(H^-1 更新从每权重一次降到每列一次)",
        "paper_175b_runtime": "175B 单卡 A100 约 4 GPU 小时(arXiv:2210.17323 Abstract/§5 Table 2)",
    }

    p = Path(__file__).with_name("ch27_m04_gptq.json")
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
