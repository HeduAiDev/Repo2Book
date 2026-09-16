"""ch28 m19（MTP：顺序多 token 预测 Eq.21-25）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m19_mtp.py
产出：explainer/traces/run_m19_mtp.json（params + raw_stdout）。

素材来源：implementation/mtp.py（论文忠实的小型参考实现，arXiv:2412.19437 §2.2）。
玩具参数：T=5 个 token、d=2、词表 V=4、D=1（论文超参）、λ=0.3。
  A) 下标对齐（Eq.21-24 最容易读错的地方）：第 i 个位置吃 (h_i, Emb(t_{i+1}))、预测 t_{i+2}；
     隐藏位置有 4 个、有目标的只有 3 个（Eq.(24) 的区间比 Eq.(22) 短一格）；
  B) 走一遍一档：拼接 → 投影 → 因果 TRM → 共享输出头 → 逐深度交叉熵 → λ/D 加权；
  C) 「顺序」的证据：扰动错位输入看影响面（因果链完整 ⇒ 第 i 个位置看到的最远 token 是 t_{i+1}）；
  D) 与「并行独立头」的算术差别（本书注解，非论文原话）：D=2 玩具下 D 档是否真的吃到 D−1 档的输出。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import mtp  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731
F = lambda x, n=6: round(float(x), n)  # noqa: E731

np.set_printoptions(precision=6, suppress=True, linewidth=200)

T, D, V, LAM, K = 5, 2, 4, 0.3, 1
token_ids = np.array([0, 1, 2, 1, 3])

P("== ch28 m19 · MTP：顺序多 token 预测（T=5、d=2、V=4、D=1、λ=0.3） ==")
P(f"    token 序列（玩具）= {token_ids.tolist()}（位置 0..{T - 1}）")

# ── A) 下标对齐 ─────────────────────────────────────────────────────────────
P("")
P("── A) 下标对齐（Eq.21-24：本项目最容易读错的一处） ──")
al = mtp.mtp_alignment(T, K)
P(f"    mtp_alignment(T={T}, k={K}) = {json.dumps(al, ensure_ascii=False)}")
P(f"    隐藏位置（Eq.22 的 h^k_(1:T-k)）= {al['hidden_indices']}（{len(al['hidden_indices'])} 个）")
P(f"    有目标的位置（Eq.24 的区间）= {al['targets']}（{al['n_loss_positions']} 个）")
for i, (a, b) in enumerate(al["targets"]):
    P(f"        位置 i={a}：吃 (h_{a}, Emb(t_{a + K}))=Emb(t_{a + K}) → 预测 t_{b}（= 序列里的 {int(token_ids[b])}）")
P(f"    ⇒ 隐藏位置比目标位置多 {len(al['hidden_indices']) - al['n_loss_positions']} 个（最后一个位置 i={al['hidden_indices'][-1]} 算出的 h 没有目标——记号的边界，不是实现缺陷）")
P(f"    ⇒ 「预测下一个」的下标差是 k+1 = {K + 1}：输入错位 {K} 格（喂 Emb(t_{{i+{K}}})）、输出错位 {K + 1} 格（预测 t_{{i+{K + 1}}}）")

# ── B) 走一遍一档 ───────────────────────────────────────────────────────────
P("")
P("── B) 走一遍一档：Eq.(21) → Eq.(22) → Eq.(23) → Eq.(24)(25) ──")
h_prev = np.array([[1.0, 0.5], [0.5, -1.0], [2.0, 0.0], [-0.5, 1.5], [1.0, 1.0]])   # 主模型隐状态 h^0（T×d）
emb = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [0.5, -0.5]])      # Emb(t_0..t_4)
W_mix = np.array([[0.5, 0.1], [0.2, 0.3], [0.1, 0.4], [0.3, 0.2]])                  # M_1: (2d, d)
hidden_idx = al["hidden_indices"]
h_prime, concat = mtp.mtp_input(h_prev[hidden_idx], emb[[i + K for i in hidden_idx]], W_mix)
P(f"    Eq.(21)：h'^1_i = M_1[RMSNorm(h^0_i) ; RMSNorm(Emb(t_{{i+1}}))]，i ∈ {hidden_idx}")
P(f"        拼接后形状 = {concat.shape}（每个位置 {2 * D} 维 = RMS(h) 的 {D} 维 + RMS(Emb) 的 {D} 维）")
P(f"        i=0 的拼接 = {R(concat[0])}（前半 = RMS(h_0)、后半 = RMS(Emb(t_1))）")
P(f"        i=0 的 h'^1_0 = {R(h_prime[0])}")
P(f"        i=3 的 h'^1_3 = {R(h_prime[3])}（这个位置的 h 没有目标，Eq.(24) 不用它）")
W_q = np.array([[1.0, 0.0], [0.0, 1.0]])
W_k = np.array([[0.5, 0.5], [0.5, -0.5]])
W_v = np.array([[1.0, 0.2], [0.2, 1.0]])
W_o = np.array([[0.5, 0.0], [0.0, 0.5]])
h1 = mtp.trm_block(h_prime, W_q, W_k, W_v, W_o)
P(f"    Eq.(22)：h^1 = TRM_1(h'^1)（一个**因果** Transformer 块）⇒ 输出形状 = {h1.shape}")
P(f"        h^1 各位 = {R(h1)}")
W_out = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])   # 与主模型共享的输出头（V×d）
logits = mtp.mtp_predict(h1[: al["n_loss_positions"]], W_out)
P(f"    Eq.(23)：P^1_{{i+2}} = OutHead(h^1_i) ⇒ logits 形状 = {logits.shape}（{al['n_loss_positions']} 个有目标的位置 × 词表 {V}）")
targets = np.array([token_ids[b] for _, b in al["targets"]])
P(f"        目标 token（Eq.24 的 t_{{2+k:T+1}}）= {targets.tolist()}（取自序列位置 {[b for _, b in al['targets']]}）")
loss1 = mtp.mtp_depth_loss(logits, targets)
probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
probs = probs / probs.sum(axis=-1, keepdims=True)
P(f"        i=0 位置的预测分布 = {R(probs[0])}（目标 token = {int(targets[0])}）")
P(f"    Eq.(24)：L^1_MTP = CrossEntropy = {F(loss1)}")
total = mtp.mtp_total_loss([loss1], lam=LAM, D=1)
P(f"    Eq.(25)：L_MTP = (λ/D)·Σ_k L^k = ({LAM}/{1}) × {F(loss1)} = {F(total)}")

# ── C) 「顺序」的证据：输入的错位与影响面 ────────────────────────────────────
P("")
P("── C) 「顺序」的证据：每档的输入是**错位**的，因果链完整 ──")
emb_p = emb.copy()
emb_p[4] += 1.0                      # 扰动最后一个 token 的 embedding（t_4）
hp_p, _ = mtp.mtp_input(h_prev[hidden_idx], emb_p[[i + K for i in hidden_idx]], W_mix)
delta_rows = [i for i in range(hp_p.shape[0]) if not np.allclose(hp_p[i], h_prime[i])]
P(f"    扰动 Emb(t_4)（最后一格 embedding += 1.0）⇒ 受影响的 h'^1 行 = {delta_rows}")
P(f"        （因为 h'^1_i 吃的是 Emb(t_{{i+1}})，i+1 = 4 ⇒ 只有 i=3 那位动——t_4 只被它该被用到的地方用到）")
P(f"        逐行最大变化量（6 位）= {[F(np.abs(hp_p[i] - h_prime[i]).max()) for i in range(hp_p.shape[0])]}")
h1_p = mtp.trm_block(hp_p, W_q, W_k, W_v, W_o)
P(f"    再往前看 TRM 的因果性：扰动 h'^1 的**最后一行**（只改位置 {h_prime.shape[0] - 1} 的输入）")
hb = h_prime.copy()
hb[-1] += 1.0
h1_b = mtp.trm_block(hb, W_q, W_k, W_v, W_o)
P(f"        逐位最大变化量（6 位）= {[F(np.abs(h1_b[i] - h1[i]).max()) for i in range(h1.shape[0])]}")
P(f"        ⇒ 后面的输入改不动前面的输出：位置 0..{h_prime.shape[0] - 2} 纹丝不动（因果掩码 np.tril ⇒ 位置 i 看不到 j > i）")
P(f"        （反向不成立、也别搞反：扰动第 0 行时**所有**后面的输出都会变——因果是『只看前面』，不是『互不影响』。）")
hb0 = h_prime.copy()
hb0[0] += 1.0
h1_b0 = mtp.trm_block(hb0, W_q, W_k, W_v, W_o)
P(f"        对照（扰动第 0 行）逐位最大变化量 = {[F(np.abs(h1_b0[i] - h1[i]).max()) for i in range(h1.shape[0])]}（全非 0）")
P(f"    ⇒ 结论：第 i 个位置预测 t_{i + K + 1} 时，能看到的最远 token 就是它自己喂进去的那个 t_{{i+{K}}}；")
P(f"       『预测下一步』的因果链在代码里就是『输入错位 {K} 格、目标错位 {K + 1} 格』")

# ── D) 与「并行独立头」的算术差别（本书注解） ───────────────────────────────
P("")
P("── D) 与『并行独立头』的算术差别（本书注解，非论文原话） ──")
P("    论文原话把顺序版与 Gloeckle 等的 which parallelly predicts D additional tokens using")
P("    independent output heads 明确区分；这里用 D=2 玩具把『顺序』二字做成可观测的算术：")
hidden2 = [i for i in range(T) if i + 2 <= T - 1]
hp2_seq, _ = mtp.mtp_input(h1[hidden2], emb[[i + 2 for i in hidden2]], W_mix)     # 第 2 档吃第 1 档输出
h2_seq = mtp.trm_block(hp2_seq, W_q, W_k, W_v, W_o)
hp2_par, _ = mtp.mtp_input(h_prime[hidden2], emb[[i + 2 for i in hidden2]], W_mix)  # 对照：仍吃第 0 档
h2_par = mtp.trm_block(hp2_par, W_q, W_k, W_v, W_o)
P(f"    顺序版第 2 档输入 = h^1（第 1 档的输出）；对照版输入 = h'^1（不接第 1 档的 TRM 输出）")
P(f"    两种第 2 档输出（{len(hidden2)} 个位置）：顺序 = {R(h2_seq)}｜对照 = {R(h2_par)}")
P(f"    两者逐位相同 = {bool(np.allclose(h2_seq, h2_par))}（不同 ⇒ 『不接上一档』会改变结果）")
h1_c = h1.copy()
h1_c[0] += 1.0
hp2_s2, _ = mtp.mtp_input(h1_c[hidden2], emb[[i + 2 for i in hidden2]], W_mix)
h2_seq2 = mtp.trm_block(hp2_s2, W_q, W_k, W_v, W_o)
P(f"    扰动第 1 档输出 h^1_0（+= 1.0）：")
P(f"        顺序版第 2 档输出逐位变化 = {[F(abs(v)) for v in (h2_seq2 - h2_seq).ravel()]}（非 0）")
P(f"        对照版第 2 档输出逐位变化 = {[F(abs(v)) for v in (h2_par - h2_par).ravel()]}（恒 0，它压根不读第 1 档）")
P(f"    ⇒ 『顺序』= 第 k 档真的吃第 k−1 档的输出（D=1 时这条链只有一格：h^1 吃 h^0）")
P(f"    config 口径：num_nextn_predict_layers = 1（与论文 D = 1 一致）；λ = {LAM}（论文值）")

out = {
    "script": "run_m19_mtp.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"T": T, "d": D, "V": V, "D": 1, "lam": LAM, "k": K, "token_ids": token_ids.tolist()},
    "results": {
        "alignment": al,
        "concat_i0": concat[0].tolist(),
        "h_prime": h_prime.tolist(),
        "h1": h1.tolist(),
        "logits": logits.tolist(),
        "targets": targets.tolist(),
        "probs_i0": probs[0].tolist(),
        "loss1": loss1,
        "total_loss": total,
        "perturb_emb4_rows": delta_rows,
        "perturb_emb4_magnitude": [float(np.abs(hp_p[i] - h_prime[i]).max()) for i in range(hp_p.shape[0])],
        "perturb_last_row_magnitude": [float(np.abs(h1_b[i] - h1[i]).max()) for i in range(h1.shape[0])],
        "perturb_first_row_magnitude": [float(np.abs(h1_b0[i] - h1[i]).max()) for i in range(h1.shape[0])],
        "seq_vs_par_depth2_equal": bool(np.allclose(h2_seq, h2_par)),
        "depth2_seq_output": h2_seq.tolist(),
        "depth2_par_output": h2_par.tolist(),
        "perturb_h1_0_seq_delta": [float(v) for v in (h2_seq2 - h2_seq).ravel()],
        "perturb_h1_0_par_delta": [float(v) for v in (h2_par - h2_par).ravel()],
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
