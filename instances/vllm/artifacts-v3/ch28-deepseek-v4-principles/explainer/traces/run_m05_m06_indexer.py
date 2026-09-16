"""ch28 m05（索引器打分 Eq.13-16）/ m06（块级 top-k 与因果 Eq.17）驱动脚本。

跑法：/d/Env/Miniconda/python explainer/traces/run_m05_m06_indexer.py
产出：explainer/traces/run_m05_m06_indexer.json（params + raw_stdout）。

素材来源：implementation/indexer.py（论文忠实的小型参考实现）。
玩具参数：n_h^I = 2 个索引头、c^I = 2、4 个候选压缩块、w = [0.7, 0.3]、m = 4。
本脚本两件事：
  A) m05：打分 I_{t,s} = Σ_h w_h·ReLU(q^I_h·K^IComp_s) → top-2（含 ReLU 把负相关打成 0 的那一格）；
  B) m06：候选数 = 已完成的块数（论文口径 t//m vs 代码口径 (t+1)//m 差一格）、
     短上下文全选快路径、−1 哨兵。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import indexer as ix  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


R = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

P("== ch28 m05/m06 · 索引器（n_h^I=2, c^I=2, 4 个候选块, w=[0.7, 0.3], m=4） ==")

# ── A) 打分（m05） ─────────────────────────────────────────────────────────
P("")
P("── A) 打分：I_{t,s} = Σ_h w_h · ReLU(q^I_{t,h} · K^IComp_s)  （Eq.16） ──")
q_I = np.array([[1.0, 0.0], [0.0, 1.0]])          # 两个索引头；取两行为单位向量，q·k 直接读成 K 的两列（透明起见）
K_icomp = np.array([[2.0, 0.5], [1.0, 0.0], [-0.5, -2.0], [1.5, 0.25]])   # 4 个压缩块的键
w = np.array([0.7, 0.3])
P(f"    q^I（{q_I.shape[0]} 头 × {q_I.shape[1]} 维）= {[R(r) for r in q_I]}")
P(f"    K^IComp（4 块 × 2 维）= {[R(r) for r in K_icomp]}")
P(f"    w^I = {w.tolist()}")
dots = np.einsum("hc,sc->hs", q_I, K_icomp)
P(f"    逐头内积 q^I_h · K^IComp_s（{dots.shape[0]} 头 × {dots.shape[1]} 块）=")
for h in range(dots.shape[0]):
    P(f"        头 {h}: {R(dots[h])}")
relu = np.maximum(dots, 0.0)
for h in range(relu.shape[0]):
    P(f"        ReLU 后 头 {h}: {R(relu[h])} ← 负相关被打成 0")
I = ix.index_scores(q_I, K_icomp, w)
P(f"    I_{'{t,s}'} = Σ_h w_h·ReLU(·) = {R(I)}")
P(f"    手算核对：块 0 = 0.7×2.0 + 0.3×0.5 = {round(0.7 * 2.0 + 0.3 * 0.5, 6)}；"
  f"块 1 = 0.7×1.0 + 0.3×0.0 = {round(0.7 * 1.0 + 0.3 * 0.0, 6)}；"
  f"块 2 = 0.7×0 + 0.3×0 = {0.0}；块 3 = 0.7×1.5 + 0.3×0.25 = {round(0.7 * 1.5 + 0.3 * 0.25, 6)}")
idx, valid = ix.select_topk_compressed(I, topk=2)
P(f"    Top-2（Eq.17）= 块 {idx[0].tolist()}，valid = {valid[0].tolist()} ⇒ 选中块 0 与块 3（块 1 的 0.7 输给块 3 的 1.125）")
I_norelu = (w[:, None] * dots).sum(axis=0)
P(f"    对照（去掉 ReLU 的同一组内积）= {R(I_norelu)}：块 2 会拿到 {round(float(I_norelu[2]), 6)}（负分）")
P(f"    本例里块 2 无论有没有 ReLU 都排最后（排序未翻转）——ReLU 的真正作用在规模上：64 个索引头、512 个候选块，"
  f"没有 ReLU 时『多个头都投反对票』的块会累积大额负分，ReLU 让每个块只被**正相关**推动")
# 两个实现层缩放（论文 Eq.16 未写）
scale_s = ix.index_scores(q_I, K_icomp, w, softmax_scale=2 ** -0.5, weights_scaling=2 ** -0.5)
P(f"    实现层补充的两个缩放（论文未写）：softmax_scale = c^I^-0.5 = {round(2 ** -0.5, 6)}、"
  f"weights_scaling = n_h^I^-0.5 = {round(2 ** -0.5, 6)}")
P(f"    带缩放的 I = {R(scale_s)} ⇒ 与不带缩放的 top-2 次序一致：{bool(np.array_equal(np.argsort(-scale_s)[:2], np.argsort(-I)[:2]))}"
  f"（两个缩放都是**常数因子**，不改变选择结果——它们是实现层为数值/量化服务的，不是机制）")
# 短上下文全选快路径
P("")
P("── 短上下文快路径（候选 ≤ topk ⇒ 一个不落地全选，pin 的 _fill_short_context_topk_indices） ──")
for msl, m, topk in ((8, 4, 512), (2048, 4, 512), (4096, 4, 512)):
    buf, _ = ix.select_topk_compressed(np.zeros(msl // m), topk=4, causal_threshold=msl // m)
    P(f"    max_seq_len={msl}, m={m}, topk={topk}: 候选 = {msl} // {m} = {msl // m} ≤ {topk} -> "
      f"{ix.short_context_select_all(msl, m, topk)}；本例 topk=4 的缓冲 = {buf[0].tolist()}（−1 是哨兵位）")

# ── B) 因果与 −1 哨兵（m06） ────────────────────────────────────────────────
P("")
P("── B) 候选数 = 已完成的块数（唯一实现方式：候选集合本身就是因果的） ──")
M = 4
P(f"    论文 Eq.16 措辞：s < Floor(t/m)（严格在自己所在块之前）；pin/参考实现：num_compressed = (t+1)//m")
P(f"    m = {M}；t = 0..11（三个块）")
for t in range(12):
    paper = ix.causal_candidate_count(t, M, "paper")
    impl = ix.causal_candidate_count(t, M, "impl")
    vis = list(range(impl))
    P(f"    t={t:2d}: 论文 {t} // {M} = {paper}｜代码 ({t}+1) // {M} = {impl}｜可见块 {vis}"
      f"｜差一格={impl - paper}｜滑窗（n_win=3 口径）[{max(t - 3 + 1, 0)}, {t}]")
P(f"    差一格只出现在**块尾 token**（t = {M - 1}, {2 * M - 1}, {3 * M - 1}）：代码允许块尾看自己那块，仍严格因果（不含未来 token）")
P(f"    t={2}（第 3 个 token）时可见压缩块 = {ix.causal_candidate_count(2, M, 'impl')} 个——还没攒满一块，只能靠滑窗")
P("")
P("── 因果约束怎么进 top-k：越界位置置 −∞ + 候选不足填 −1 哨兵 ──")
I_long = np.array([0.5, 2.0, 1.0, 0.0, 3.0, 0.2, 0.1, 4.0])
P(f"    8 个候选块的分数 I = {R(I_long)}")
for thresh, k in ((8, 2), (6, 2), (8, 4)):
    idx2, v2 = ix.select_topk_compressed(I_long, topk=k, causal_threshold=thresh)
    P(f"    causal_threshold={thresh}（只看前 {thresh} 块）、topk={k} -> 选中 {idx2[0].tolist()}（valid {v2[0].tolist()}）")
idx3, v3 = ix.select_topk_compressed(np.array([0.5, 2.0]), topk=5, causal_threshold=2)
P(f"    候选只有 2 个而 topk=5：选中 {idx3[0].tolist()}（后 3 个位置填 −1 哨兵）")
P(f"    ⇒ 短上下文里『全选』与『−1 哨兵』是同一件事的两面：候选不足时排序无意义，全填进去")

P("")
P(f"── 派生量：1M 上下文下候选墙 vs 预算 ──")
P(f"    候选 = 1000000 // 4 = {1000000 // 4} 条；每 query 实看 = min(512, 候选) = {min(512, 1000000 // 4)} 条")
P(f"    压缩比 = 候选 / 实看 = {1000000 // 4} / {min(512, 1000000 // 4)} = {(1000000 // 4) / min(512, 1000000 // 4):.5f}"
  f"（≈ {(1000000 // 4) / 512:.2f}，即压缩把候选降 4 倍、稀疏再把实看数封在 512）")

out = {
    "script": "run_m05_m06_indexer.py",
    "raw_stdout": "\n".join(LINES),
    "params": {"n_index_heads": 2, "index_head_dim": 2, "n_candidate_blocks": 4, "w": w.tolist(), "m": M},
    "results": {
        "dots": dots.tolist(),
        "relu": relu.tolist(),
        "I": I.tolist(),
        "top2": idx.tolist(),
        "I_scaled": scale_s.tolist(),
        "causal_paper": [ix.causal_candidate_count(t, M, "paper") for t in range(12)],
        "causal_impl": [ix.causal_candidate_count(t, M, "impl") for t in range(12)],
    },
}
with open(Path(__file__).with_suffix(".json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\n[written] {Path(__file__).with_suffix('.json').name}")
