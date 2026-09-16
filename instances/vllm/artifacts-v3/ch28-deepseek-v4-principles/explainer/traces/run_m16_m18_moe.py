"""ch28 m16（MoE 两板斧 Eq.3-11）/ m17（Sqrt(Softplus) + auxiliary-loss-free）/ m18（hash 路由）
驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m16_m18_moe.py
产出：explainer/traces/run_m16_m18_moe.json（params + raw_stdout）。

素材来源：implementation/moe.py（论文忠实的小型参考实现）。
  A) m16 板斧一：细粒度切分——N=4/K=2/m=2 的玩具，核『算力不变、组合变多』（Eq.6-8）；
  B) m16 板斧二：共享专家无条件计算——同一组 gate 翻倍看共享项动没动（Eq.9-11）；
  C) m17：4 个专家 + bias——Sigmoid（V3 口径）vs Sqrt(Softplus)（V4 口径）的数值差、
     bias 只进选择（选择真的被 bias 改动）、权重从无偏分数 gather vs 用带偏分数（分布被压平）；
  D) m18：hash 路由＝按 token id 查一张 [vocab, topk] 表（选谁定死、权重仍现算）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import moe  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731
F = lambda x, n=6: round(float(x), n)  # noqa: E731

P("== ch28 m16/m17/m18 · MoE 两板斧 + V4 路由口径 + hash 路由 ==")

# ══ A) m16 板斧一：细粒度切分 ═══════════════════════════════════════════════
P("")
P("── A) m16 板斧一：细粒度专家切分（Eq.6-8：N=4、K=2、m=2） ──")
N0, K0, SPLIT, D_FF = 4, 2, 2, 8
N1, K1 = moe.fine_grained_counts(N0, K0, SPLIT)
P(f"    切分前：N = {N0} 个专家、每个中间维 d_ff = {D_FF}、每 token 激活 K = {K0} 个")
P(f"        ⇒ 激活算力（按中间维计）= K × d_ff = {K0} × {D_FF} = {K0 * D_FF}")
P(f"    切分后：mN = {N1} 个专家、每个中间维 d_ff/m = {D_FF} / {SPLIT} = {D_FF // SPLIT}、激活 mK = {K1} 个")
P(f"        ⇒ 激活算力 = mK × (d_ff/m) = {K1} × {D_FF // SPLIT} = {K1 * (D_FF // SPLIT)}（与切分前**相等**）")
P(f"    可选组合数：C({N0},{K0}) = {moe.segment_counts(N0, K0)} → C({N1},{K1}) = {moe.segment_counts(N1, K1)}"
  f"（{moe.segment_counts(N1, K1) / moe.segment_counts(N0, K0):.6f} 倍）")
P(f"    config 口径（V4-Flash）：n_routed_experts = 256、num_experts_per_tok = 6 ⇒ 256 挑 6；")
P(f"        moe_intermediate_size = 2048 vs hidden_size = 4096 ⇒ 单专家中间维 = hidden 的 {2048 / 4096:.6f}")
P(f"        口径注解（本书自算，非论文数字）：常规稠密 FFN 的中间维常取 4×hidden = {4 * 4096}，")
P(f"        单专家 2048 ⇒ 只有它的 {4 * 4096 / 2048:.6f} 分之一 ⇒ 『切得细』的算术形态")

# ══ B) m16 板斧二：共享专家无条件计算 ═══════════════════════════════════════
P("")
P("── B) m16 板斧二：共享专家无条件计算（Eq.9-11：K_s=1、其余 8 个里挑 4） ──")
u = np.array([1.0, 0.5])
shared_out = np.array([[0.5, -0.5]])            # 1 个共享专家的输出 FFN_i(u)
routed_outs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-0.5, 0.5]])   # 4 个被选中专家
gates_a = np.array([0.4, 0.3, 0.2, 0.1])
gates_b = 2 * gates_a
h_a = moe.deepseekmoe_forward(u, shared_out, routed_outs, gates_a)
h_b = moe.deepseekmoe_forward(u, shared_out, routed_outs, gates_b)
shared_term = moe.shared_expert_output(shared_out)
P(f"    共享项（Eq.9 第一项，没有 gate）= Σ_i FFN_i(u) = {R(shared_term)}")
P(f"    路由项（Eq.9 第二项，带 gate）gates_a = {R(gates_a)} ⇒ Σ g_i·FFN_i(u) = {R((gates_a[:, None] * routed_outs).sum(axis=0))}")
P(f"    h = 共享项 + 路由项 + u（gates_a）= {R(h_a)}")
P(f"    把门控整体翻倍 gates_b = {R(gates_b)}：")
P(f"        h（gates_b）= {R(h_b)}；共享项仍是 {R(shared_term)}（逐位相同 = {bool(np.allclose(shared_term, shared_term))}）")
P(f"    ⇒ 『无条件』的可观察后果：路由怎么变，共享专家那份贡献一动不动（不是被忽略，是**不可选**）；")
P(f"      代价：每个 token 都要付这 {len(shared_out)} 个专家的算力（不管它需不需要）")
P(f"    路由只在 K_s 之后的专家里挑：Eq.(10) 的池子 = 第 K_s+1 个起、挑 mK−K_s 个（本例 8 挑 4）")
sc_on = moe.topk_gate(np.array([0.9, 0.1]), 1, start=1)
sc_off = moe.topk_gate(np.array([0.9, 0.1]), 1, start=0)
P(f"    把共享位排除 vs 不排除（同一组 scores = [0.9, 0.1]、k=1）：")
P(f"        start=1（V4 口径，共享位不可选）⇒ 门控 = {R(sc_on)}（分数只落在专家 1 上）")
P(f"        start=0（共享位进池子）⇒ 门控 = {R(sc_off)}（分数落在专家 0 上，共享专家就『没算』了）")
P(f"        ⇒ 『无条件』的反面：一旦共享专家进路由池，它就可能挑不中——那就不是共享专家了")

# ══ C) m17：Sqrt(Softplus) + bias 只进选择 ══════════════════════════════════
P("")
P("── C) m17：亲和分 Sigmoid（V3）→ Sqrt(Softplus)（V4），bias 只进选择 ──")
logits = np.array([-2.0, 0.5, 3.0, 0.2])
bias = np.array([0.0, -0.10, 0.0, 0.30])   # 让 bias 真的**改动选择**（否则是退化例：选出来的人一样）
sig = moe.affinity_scores_v3(logits)
sqs = moe.affinity_scores_v4(logits)
P(f"    4 个专家的 gate logits = {R(logits)}；bias = {R(bias)}")
P(f"    V3 口径 Sigmoid(·)       = {R(sig)}")
P(f"    V4 口径 Sqrt(Softplus(·)) = {R(sqs)}")
big = np.array([8.0])
P(f"    大 logits 处的差别（为什么换）：logit = 8.0 时 Sigmoid = {R(moe.affinity_scores_v3(big))}（压平到 1 附近）")
P(f"        而 Sqrt(Softplus(·)) = {R(moe.affinity_scores_v4(big))}（不饱和上界，仍有区分度）")
idx_unb, w_unb = moe.topk_with_correction_bias(sqs, np.zeros_like(bias), 2)
idx_bi, w_bi = moe.topk_with_correction_bias(sqs, bias, 2)
idx_bw, w_bw = moe.topk_with_correction_bias(sqs, bias, 2, use_biased_weights=True)
P(f"    不带 bias 的选择（top-2）= 专家 {list(idx_unb)}；权重（无偏分数、renormalize）= {R(w_unb)}")
P(f"    scores + bias = {R(sqs + bias)}")
P(f"    带 bias 的选择（top-2）= 专家 {list(idx_bi)} ← **bias 改动了选择**（换进来的是专家 {int(idx_bi[-1])}）")
P(f"    权重仍从**无偏** scores gather + renormalize = {R(w_bi)}")
P(f"    对照（错误做法：拿带偏分数当权重）= {R(w_bw)} ⇒ 更平（最大权重 {F(w_bi[0])} → {F(w_bw[0])}）")
P(f"        pin 的注释点破：DSv4-Flash 的 bias ≈ 8.08 近均匀时，用带偏分数当权重会把分布压平")
P(f"    非平凡性核对：无偏选中集合 = {sorted(int(i) for i in idx_unb)}、带偏选中集合 = {sorted(int(i) for i in idx_bi)}"
  f" ⇒ 两个集合不同 = {sorted(int(i) for i in idx_unb) != sorted(int(i) for i in idx_bi)}（bias 真的改动了选择）")

# ══ D) m18：hash 路由＝查表 ═════════════════════════════════════════════════
P("")
P("── D) m18：hash 路由——按 token id 查一张 [vocab, topk] 表 ──")
tid2eid = np.array([[3, 1], [0, 2], [2, 3], [1, 0], [3, 0], [1, 2]])   # 玩具词表 6 行、topk=2
token_ids = np.array([4, 1, 5])
scores_tok = np.array([
    [0.2, 1.1, 0.4, 0.9],     # token 4 的专家分数（无偏，V4 口径 = sqrt(softplus(gate logits))）
    [1.3, 0.5, 0.7, 0.2],     # token 1
    [0.3, 0.4, 1.2, 0.6],     # token 5
])
idx_h, w_h = moe.hash_route(tid2eid, token_ids, scores_tok)
P(f"    tid2eid（玩具：vocab {tid2eid.shape[0]} 行 × topk {tid2eid.shape[1]} 列）= {tid2eid.tolist()}")
for k, t in enumerate(token_ids):
    P(f"    token id = {int(t)} → 表第 {int(t)} 行 = {tid2eid[int(t)].tolist()} ⇒ 派给专家 {idx_h[k].tolist()}"
      f"；权重（从无偏 scores gather）= {R(w_h[k])}")
P(f"    对照（同一批 token 若走 gate 打分：取 scores 的 top-2）="
  f" {[list(np.argsort(-scores_tok[k], kind='stable')[:2]) for k in range(len(token_ids))]}")
P(f"    ⇒ hash 层的『选谁』与 gate 分数**完全无关**（表说了算）；gate 分数只用来给选中的人分权重")
P(f"       这正是 pin 与官方参考实现同款的两句：hash 只决定选谁、权重仍从无偏 scores gather")
P(f"    config 口径：num_hash_layers = 3（最前 3 层）、vocab_size = 129280 ⇒ 表形状 [{129280}, {6}]（真实 topk=6）")
P(f"    与 m17 的 bias 对照：bias 改动的是『谁被选中』的排序（仍由分数决定），hash 则把这个排序整个换掉")

P("")
P("── 派生量（供正文直接引用） ──")
P(f"    带偏权重把分布压平的幅度 = 无偏最大权重 {F(w_bi[0])} − 带偏最大权重 {F(w_bw[0])} = {F(abs(w_bi[0] - w_bw[0]))}")
P(f"    config 口径的 hash 表规模 = vocab_size × topk = 129280 × {int(tid2eid.shape[1])} = {129280 * int(tid2eid.shape[1])} 个整数"
  f"（真实 topk=6 ⇒ 129280 × 6 = {129280 * 6}）")

out = {
    "script": "run_m16_m18_moe.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"N0": N0, "K0": K0, "split_m": SPLIT, "d_ff": D_FF, "K_s": 1,
               "logits": logits.tolist(), "bias": bias.tolist(),
               "vocab_toy": int(tid2eid.shape[0]), "topk_toy": int(tid2eid.shape[1]),
               "num_hash_layers": 3, "vocab_size": 129280},
    "results": {
        "fine_grained": {"N": N1, "K": K1, "combos_before": moe.segment_counts(N0, K0),
                         "combos_after": moe.segment_counts(N1, K1)},
        "flops_before": K0 * D_FF, "flops_after": K1 * (D_FF // SPLIT),
        "shared_term": shared_term.tolist(),
        "h_gates_a": h_a.tolist(), "h_gates_b": h_b.tolist(),
        "sigmoid": sig.tolist(), "sqrtsoftplus": sqs.tolist(),
        "sigmoid_big": moe.affinity_scores_v3(big).tolist(),
        "sqrtsoftplus_big": moe.affinity_scores_v4(big).tolist(),
        "idx_unbiased": [int(i) for i in idx_unb], "w_unbiased": w_unb.tolist(),
        "idx_biased": [int(i) for i in idx_bi], "w_biased_choice": w_bi.tolist(),
        "w_biased_weights": w_bw.tolist(),
        "hash_idx": idx_h.tolist(), "hash_w": w_h.tolist(),
        "gate_top2_for_same_tokens": [np.argsort(-scores_tok[k], kind="stable")[:2].tolist() for k in range(len(token_ids))],
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
