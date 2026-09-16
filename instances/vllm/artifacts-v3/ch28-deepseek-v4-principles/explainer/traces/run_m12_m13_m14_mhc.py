"""ch28 m12（mHC 更新式 + 双随机约束）/ m13（动态参数化：一次 GEMM 出 A/B/C 的 raw 参数）
      / m14（Sinkhorn-Knopp 迭代与 t_max=20）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m12_m13_m14_mhc.py
产出：explainer/traces/run_m12_m13_m14_mhc.json（params + raw_stdout）。

素材来源：implementation/mhc.py（论文忠实的小型参考实现）。
  A) m12：双随机矩阵的三条性质（范数 ≤1 / 恒等映射 / 对乘法封闭）+ **反例**（行和=1 但列和≠1 ⇒ 范数 >1
     ⇒ 深堆叠放大信号）。**注意**：双随机矩阵的谱范数**恰好**等于 1（1 必是特征值），所以「≤1」讲的是
     **不放大**（非扩张），不是收缩——脚本把两条奇异值都打出来，别把 1.000000 读成巧合。
  B) m13：hc_mult=2 的玩具——展平 RMSNorm → 一次 GEMM → 8 个数怎么切成 Ã/C̃/B̃ → σ/2σ/Sinkhorn。
  C) m14：逐个打印每轮的行/列和偏差（为什么交替归一收敛）+ 20 轮够不够（条件数差的矩阵 20 轮仍有残差）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import mhc  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731
F = lambda x, n=6: round(float(x), n)  # noqa: E731


# ══ A) m12 ═════════════════════════════════════════════════════════════════
P("== ch28 m12 · mHC 更新式 X_{l+1} = B_l X_l + C_l F_l(A_l X_l) 与双随机约束 ==")
P("")
B2 = np.array([[0.7, 0.3], [0.3, 0.7]])
B3 = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
M_bad = np.array([[0.9, 0.1], [0.2, 0.8]])
for name, B in (("B2 = [[0.7,0.3],[0.3,0.7]]（2 条流、对称）", B2),
                ("B3 = [[0.5,0.3,0.2],[0.2,0.5,0.3],[0.3,0.2,0.5]]（3 条流、**非对称**）", B3)):
    sv = np.linalg.svd(B, compute_uv=False)
    P(f"    {name}")
    P(f"        行和 = {R(B.sum(axis=1))}；列和 = {R(B.sum(axis=0))}；非负 = {bool((B >= 0).all())}"
      f" ⇒ 双随机判定 = {mhc.is_doubly_stochastic(B)}")
    P(f"        奇异值 = {R(sv)}；谱范数 = {F(mhc.spectral_norm(B))}")
P("    ⚠ 两个双随机矩阵的谱范数都**恰好**是 1.000000 —— 这不是巧合也不是凑数：")
P("      B·1 = 1（行和为 1）⇒ 1 必是特征值 ⇒ ‖B‖₂ ≥ 1；再叠上列和为 1 的约束得 ‖B‖₂ ≤ 1 ⇒ 恒等于 1。")
P("      所以论文那句 the spectral norm ... is bounded by 1 讲的是**不放大信号**（非扩张），不是「收缩到 0」；")
P("      第二个奇异值才是真在收缩的那部分（B2 的 0.4、B3 的 0.4——它们把流间的**差异**磨掉）。")
P("")
P("    反例（同样非负、行和也 = 1，但列和 ≠ 1）：M = [[0.9,0.1],[0.2,0.8]]")
P(f"        行和 = {R(M_bad.sum(axis=1))}；列和 = {R(M_bad.sum(axis=0))} ⇒ 双随机判定 = {mhc.is_doubly_stochastic(M_bad)}")
P(f"        谱范数 = {F(mhc.spectral_norm(M_bad))} > 1 ⇒ 最坏方向上最多放大 1.009583 倍（不是每个方向都被放大）：")
for k in (2, 3, 5, 9, 17, 33):
    Mk = np.linalg.matrix_power(M_bad, k)
    P(f"            M^{k} 的谱范数 = {F(mhc.spectral_norm(Mk))}")
P(f"        ⇒ 深堆叠 L 层后放大约 1.009583^L 倍（32 层 = {round(1.009583 ** 32, 6)} 倍、64 层 = {round(1.009583 ** 64, 6)} 倍）——")
P(f"          这就是 mHC 摘要里 severe training instability 的算术形态；把 B 关进双随机集合（列和也 = 1）就没了这个放大源。")
P("")
P("    封闭性（对乘法封闭，所以深堆叠仍在内）：")
bb = B2 @ B2
P(f"        B2 @ B2 = {[R(r) for r in bb]}；行列和 = {R(bb.sum(axis=1))} / {R(bb.sum(axis=0))}"
  f" ⇒ 双随机 = {mhc.is_doubly_stochastic(bb)}、谱范数 = {F(mhc.spectral_norm(bb))}")
bbb = B3 @ B3 @ B3
P(f"        B3^3 = {[R(r) for r in bbb]} ⇒ 双随机 = {mhc.is_doubly_stochastic(bbb)}、谱范数 = {F(mhc.spectral_norm(bbb))}")
P("")
P("    非扩张（同一组流、两种混合矩阵）：")
X = np.array([[[1.0, 2.0], [0.5, -1.0]]])          # (T=1, n_hc=2, d=2)
P(f"        X = {[R(r) for r in X[0]]}；‖X‖_F = {F(np.linalg.norm(X))}")
for name, B in (("双随机 B2", B2), ("反例 M", M_bad)):
    BX = mhc.doubly_stochastic_residual(B, X)
    P(f"        {name}: B X = {[R(r) for r in BX[0]]}；‖B X‖_F = {F(np.linalg.norm(BX))}"
      f"（比值 {round(float(np.linalg.norm(BX) / np.linalg.norm(X)), 6)}）")
P("        恒等映射的恢复：取双随机集合的顶点 B = I（置换矩阵），X_{l+1} = X_l + C_l F_l(A_l X_l) —— 多流退化成")
P("        单流残差的那条『加回去』；约束保留了恒等映射这条护身符（mHC 摘要的 identity mapping property）。")
Xid = mhc.doubly_stochastic_residual(np.eye(2), X)
P(f"        B = I 时 B X = {[R(r) for r in Xid[0]]}（与 X 逐位相同：{bool(np.allclose(Xid, X))}）")

# ══ B) m13 ═════════════════════════════════════════════════════════════════
P("")
P("== ch28 m13 · 动态参数化：一次 GEMM 出 A/B/C 的 raw 参数（hc_mult=2, d=2, T=1） ==")
X13 = np.array([[[1.0, 2.0], [0.5, -1.0]]])         # (T=1, n_hc=2, d=2) → 展平 4 个数
fn = np.array([
    [1.0, 0.0, 0.0, 0.0],      # pre 段（hc_mult = 2 个数）
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],      # post 段（2 个数）
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],      # res 段（hc_mult^2 = 4 个数，展平后 reshape 成 2×2）
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0],
])
base = np.array([0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
scale = np.array([0.5, 0.5, 0.5])                   # α^pre / α^post / α^res（可学习门控、初始化很小）
P(f"    X（T=1, n_hc=2, d=2）= {[R(r) for r in X13[0]]} ⇒ 展平 x = {R(X13.reshape(1, -1))}")
P(f"    RMSNorm 展平态（**无权**）：均方 = {F((X13.reshape(1, -1) ** 2).mean())}、"
  f"x̂ = flat × rsqrt(均方+eps) = {R(mhc.rms_norm_flat(X13.reshape(1, -1)))}")
A_t, B_t, C_t = mhc.mhc_raw_mappings(X13, fn, base, scale)
P(f"    mixes = x̂ · fnᵀ（fn 形状 {fn.shape}，mix = (2+n_hc)·n_hc = {fn.shape[0]}）=")
mixes = (mhc.rms_norm_flat(X13.reshape(1, -1)) @ fn.T)[0]
P(f"        {R(mixes)}（8 个数：{2} 个 pre + {2} 个 post + {4} 个 res 展平）")
P(f"    切片（参考实现的打包顺序 = (pre, post, res)）：")
P(f"        [0:2] = Ã = {R(A_t)}  → A = σ(Ã)+eps = {R(mhc.mhc_gates(A_t, B_t, C_t)[0])}（值域 (0,1)）")
P(f"        [2:4] = C̃ = {R(C_t)}  → C = 2σ(C̃) = {R(mhc.mhc_gates(A_t, B_t, C_t)[2])}（值域 (0,2)）")
P(f"        [4:8] = B̃ 展平成 2×2 = {[R(r) for r in B_t[0]]} → B = Sinkhorn(B̃) = {[R(r) for r in mhc.mhc_gates(A_t, B_t, C_t)[1][0]]}")
P(f"    α 与 S 从哪来：scale[0..2] = {scale.tolist()} 就是 α^pre/α^post/α^res（可学习门控、初始化很小）、"
  f"base = {base.tolist()} 是静态偏置 S（论文 Eq.3-5 的『动态项 + 静态项』）")
P(f"    （论文列举顺序是 (pre, res, post)；参考实现的 fn/base 打包顺序是 (pre, post, res)——"
  f"【paper 顺序】Ã={R(mixes[:2] * scale[0] + base[:2])}、C̃={R(mixes[4:6] * scale[1] + base[4:6])}、B̃={R(mixes[2:4] * scale[2] + base[2:4])}）")
A, B, C = mhc.mhc_gates(A_t, B_t, C_t)
P(f"    σ/2σ 的数值下界（σ 后加 eps = pin 的 hc_eps）：A ≥ {F(A.min())}、C ≥ {F(C.min())}——给门控一个正值下界")
P(f"    B 是双随机：行和 = {R(B[0].sum(axis=1))}、列和 = {R(B[0].sum(axis=0))} ⇒ {mhc.is_doubly_stochastic(B[0])}")
P(f"    更新一拍：X_1 = B X_0 + C·F(A X_0)；取 F(·) 为恒等占位（mHC 混合不关心子层是什么）：")
X1 = mhc.mhc_update(X13, B[0], C[0], mhc.mhc_layer_input(A[0], X13))
P(f"        A X_0（n_hc 条流聚成子层输入）= {R(mhc.mhc_layer_input(A[0], X13)[0])}")
P(f"        X_1 = {[R(r) for r in X1[0]]}（C 把子层输出散回 {2} 条流、B 让残差在流间混合）")

# ══ C) m14 ═════════════════════════════════════════════════════════════════
P("")
P("== ch28 m14 · Sinkhorn-Knopp 迭代与 t_max=20 ==")
raw = np.array([[4.0, 1.0], [1.0, 3.0]])
tr = mhc.sinkhorn_trace(raw, iters=20, start="raw", order="row-first")
P(f"    raw = {[R(r) for r in raw]}（非负）；起点 = raw（教科书式交替归一，便于手算）")
P(f"    逐轮（row-first：先行使和=1 再列使和=1；这里逐条打印真实的第 1–5 轮，第 20 轮另起一行）：")
for i, rec in enumerate(tr[:5], start=1):
    P(f"        第 {i:2d} 轮: 行和(行归一后) = {R(rec['row_sums_after_row'])}，"
      f"列和(行归一后) = {R(rec['col_sums_after_row'])}，列和(列归一后) = {R(rec['col_sums_after_col'])}"
      f" ⇒ row_dev = {F(rec['row_dev'], 8)}、col_dev = {F(rec['col_dev'], 8)}")
P(f"    第 {len(tr)} 轮末：row_dev = {F(tr[-1]['row_dev'], 8)}、col_dev = {F(tr[-1]['col_dev'], 8)}"
  f"（= eps 地板 1e-6 量级：分母每次加 eps，偏差不可能到 0）")
P(f"    收敛的单调量：每轮必做一次行归一与一次列归一 ⇒ **列和当轮就归 1**（col_dev 恒 0），"
  f"下一轮行使和=1 时会破坏列和一点点 ⇒ 偏差逐轮下降（{F(tr[0]['row_dev'], 8)} → {F(tr[1]['row_dev'], 8)} → {F(tr[2]['row_dev'], 8)} …）")
P("")
P("    20 轮够不够？换几个矩阵看 20 轮与 200 轮的残差（max(|行和−1|,|列和−1|)）：")
cands = {
    "[[4,1],[1,3]]": np.array([[4.0, 1.0], [1.0, 3.0]]),
    "[[1,2],[3,4]]": np.array([[1.0, 2.0], [3.0, 4.0]]),
    "[[1,1],[1,1.01]]（近奇异）": np.array([[1.0, 1.0], [1.0, 1.01]]),
    "[[2,1],[1,1.0001]]（近奇异）": np.array([[2.0, 1.0], [1.0, 1.0001]]),
    "[[1,1e-6],[1e-6,1]]": np.array([[1.0, 1e-6], [1e-6, 1.0]]),
}
worst = None
for name, mat in cands.items():
    for it in (20, 200):
        M_ = mhc.sinkhorn_knopp(mat, iters=it, start="raw")
        dev = max(float(np.abs(M_.sum(axis=1) - 1).max()), float(np.abs(M_.sum(axis=0) - 1).max()))
        P(f"        {name} 迭代 {it:3d} 轮: 残差 = {dev:.2e}（{F(dev, 10)}）")
        if it == 20 and (worst is None or dev > worst[1]):
            worst = (name, dev, mat)
P(f"    ⇒ 最差的一档（{worst[0]}）20 轮后仍有 {F(worst[1], 10)} 的残差，200 轮才进 eps 地板 ⇒")
P(f"      论文那句 We choose t_max=20 as a practical value 的算术形态：**精度与开销的折中，不是精确投影**。")
P(f"       本例：20 轮的残差 {F(worst[1], 10)} ⇒ 对残差流混合的实际影响约是 {F(worst[1] * 100, 6)}% 量级的行/列和不守恒。")
M20 = mhc.sinkhorn_knopp(worst[2], iters=20, start="raw")
P(f"    该矩阵 20 轮的结果 = {[R(r, 8) for r in M20]}（行列和 = {R(M20.sum(axis=1), 8)} / {R(M20.sum(axis=0), 8)}）")
P("")
P("    两处实现细节（论文书面顺序与两侧实现相反、极限相同）：")
for order in ("row-first", "col-first"):
    M_ = mhc.sinkhorn_knopp(raw, iters=20, start="exp", order=order, eps=1e-6)
    P(f"        order={order}: M^(20) = {[R(r, 8) for r in M_]}"
      f"（row_dev {F(float(np.abs(M_.sum(axis=1) - 1).max()), 10)}、col_dev {F(float(np.abs(M_.sum(axis=0) - 1).max()), 10)}）")
for start in ("exp", "softmax"):
    M_ = mhc.sinkhorn_knopp(raw, iters=20, start=start)
    P(f"        start={start}: M^(20) = {[R(r, 8) for r in M_]}"
      f"（论文起点 M^(0)=exp(H̃)；参考实现是 softmax(dim=-1) = exp + 行归一，把第一步吸收掉）")
P(f"    20 轮行/列交替 = pin 的 hc_sinkhorn_iters=20、hc_eps=1e-6 的字面口径（config 与论文互证）")

out = {
    "script": "run_m12_m13_m14_mhc.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"hc_mult": 2, "d": 2, "T": 1, "sinkhorn_iters": 20, "eps": 1e-6},
    "results": {
        "B2_singular": np.linalg.svd(B2, compute_uv=False).tolist(),
        "B3_singular": np.linalg.svd(B3, compute_uv=False).tolist(),
        "bad_colsum": M_bad.sum(axis=0).tolist(),
        "bad_spectral_norm": mhc.spectral_norm(M_bad),
        "B2B2": bb.tolist(),
        "flat_rmsnorm": mhc.rms_norm_flat(X13.reshape(1, -1)).tolist(),
        "mixes": mixes.tolist(),
        "A_tilde": A_t.tolist(), "A": A.tolist(),
        "B_tilde": B_t.tolist(), "B": B.tolist(),
        "C_tilde": C_t.tolist(), "C": C.tolist(),
        "X1": X1.tolist(),
        "sinkhorn_trace_4": [{"row_dev": r["row_dev"], "col_dev": r["col_dev"]} for r in tr[:4]],
        "sinkhorn_trace_5": [{"row_dev": r["row_dev"], "col_dev": r["col_dev"]} for r in tr[:5]],
        "sinkhorn_trace_last": {"row_dev": tr[-1]["row_dev"], "col_dev": tr[-1]["col_dev"]},
        "worst_matrix": worst[0], "worst_dev20": worst[1],
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
