"""ch33 m06+m07 驱动脚本（并行之病：多模态碰撞 + DSpark 两阶段生成）。

跑法：python explainer/traces/run_m06_m07_collision.py
产出：explainer/traces/run_m06_m07_collision.json

素材来源：implementation/dspark.py（multimodal_collision_demo / make_toy_backbone /
MarkovHead / ConfidenceHead / sample_sequential / dspark_draft /
anchor_as_first_inputs）。
m06：'of course'/'no problem' 双 mode——并行逐位边缘化拼出 'of problem'，
DSpark 序列头修正为 'of course'（论文 §3.1 开篇例子的可运行版）。
m07：玩具骨干（V=6,d=8,2 层）γ=3 逐步手推 prev 链；骨干前向调用计数=1
（T_draft 与 γ 无关的结构性证据）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import dspark  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731
V6 = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731

VOCAB = ["<anchor>", "of", "no", "course", "problem"]
OF, NO, COURSE, PROBLEM = 1, 2, 3, 4

# ══ m06 · 多模态碰撞 ════════════════════════════════════════════════════════
P("== ch33 m06 · 并行之病：多模态碰撞（对前驱边缘化）与 DSpark 序列修正 ==")
demo = dspark.multimodal_collision_demo(greedy=True)

P("")
P("-- [m06·双 mode 上下文] 位 1 分布与位 2 条件分布 --")
P(f"    p(x_1) = of {F(demo['p_x1'][OF])} / no {F(demo['p_x1'][NO])}（上下文两 mode 都容）")
P(f"    x_1='of'  行: course {F(demo['cond_x2'][OF][COURSE])} / problem {F(demo['cond_x2'][OF][PROBLEM])}")
P(f"    x_1='no'  行: course {F(demo['cond_x2'][NO][COURSE])} / problem {F(demo['cond_x2'][NO][PROBLEM])}")

P("")
P("-- [m06·边缘化] 并行 drafter 位 2 只能学边缘 p(x_2)=Σ p(x_1)p(x_2|x_1) --")
marg = demo["marginal_x2"]
P(f"    course  = 0.6*0.6 + 0.4*0.02 = {F(marg[COURSE])}")
P(f"    problem = 0.6*0.3 + 0.4*0.95 = {F(marg[PROBLEM])}")
P(f"    逐位独立 argmax：位 1 -> 'of'（0.6），位 2 -> 'problem'（{F(marg[PROBLEM])} > {F(marg[COURSE])}）")
P(f"    并行块 = ['of', 'problem']——跨 mode 拼接（另一位 'no course' 同理）")

P("")
P("-- [m06·连贯率] 并行独立采样落在两个真 mode 上的概率 --")
P(f"    p(x_1)*p(x_2) 的联合落在 ('of','course') + ('no','problem') = "
      f"0.6*{F(marg[COURSE])} + 0.4*{F(marg[PROBLEM])} = {F(demo['parallel_coherent_prob'])}"
      f"——并行独立采样约 {(1 - demo['parallel_coherent_prob']) * 100:.1f}% 的块是跨 mode 的")

P("")
P("-- [m06·DSpark 修正] 位 1 采出 'of' 后，rank-1 Markov 偏置抬 'course' 压 'problem' --")
P(f"    W_1 行（of +1 / no -1 / anchor 0）、W_2 列（course +3 / problem -3）——B=W_1W_2")
steps = demo["steps"]
for s in steps:
    P(f"    位 {s['k']}: prev='{VOCAB[s['prev']]}'  bias = {V6(s['bias'])}")
p1 = demo["dspark_probs"][1]
P(f"    位 1 加偏置后 p(x_2) = of {F(p1[OF])} / no {F(p1[NO])} / course {F(p1[COURSE])} / "
      f"problem {F(p1[PROBLEM])}（course 从边缘 {F(marg[COURSE])} 抬到 {F(p1[COURSE])}）")
P(f"    DSpark 块 = {['' + VOCAB[t] for t in demo['dspark_block']]}")
# 两个分支各自的偏置后分布（'of' 分支已采；'no' 分支单独算——Markov 偏置按 prev 行查表）
from spec_decode import softmax_lastdim  # noqa: E402
markov_demo = dspark.MarkovHead(demo["markov_w1"], demo["markov_w2"])
U2 = demo["base_logits"][1]
p_of_branch = softmax_lastdim(U2 + markov_demo.bias(markov_demo.embed([OF]))[0])
p_no_branch = softmax_lastdim(U2 + markov_demo.bias(markov_demo.embed([NO]))[0])
P(f"    'no' 分支（未走到、单独算）：p(x_2) course {F(p_no_branch[COURSE])} / "
      f"problem {F(p_no_branch[PROBLEM])}（镜像偏置压 course 抬 problem）")
ds_coherent = (demo["dspark_probs"][0][OF] * p_of_branch[COURSE]
               + demo["dspark_probs"][0][NO] * p_no_branch[PROBLEM])
P(f"    DSpark 连贯率（随机采样口径）= 0.6*{F(p_of_branch[COURSE])} + 0.4*{F(p_no_branch[PROBLEM])}"
      f" = {F(ds_coherent)}"
      f"（并行 {F(demo['parallel_coherent_prob'])} -> 半自回归 {F(ds_coherent)}）")

# ══ m07 · DSpark 两阶段生成 ═════════════════════════════════════════════════
P("")
P("== ch33 m07 · DSpark 两阶段：并行骨干出 U_k，序列头按 p_k∝softmax(U_k+B) 左到右采样 ==")

backbone = dspark.make_toy_backbone(vocab_size=6, d=8, num_layers=2, num_target_layers=2, seed=3)
rng_mk = np.random.default_rng(4)
markov = dspark.MarkovHead(rng_mk.normal(0, 0.5, (6, 4)), rng_mk.normal(0, 0.5, (4, 6)))
rng_ch = np.random.default_rng(5)
conf_head = dspark.ConfidenceHead(rng_ch.normal(0, 0.5, 12))  # d(8)+r(4)=12 维

ctx_rng = np.random.default_rng(6)
concat_hiddens = ctx_rng.normal(0, 1.0, (3, 16))  # 3 个上下文 token × 2 层 target 隐状态拼接
context = backbone.precompute_context(concat_hiddens)

# 前向调用计数（T_draft 与 gamma 无关的结构性证据）
calls = {"n": 0}
fwd_orig = backbone.forward


def fwd_counted(*a, **kw):
    calls["n"] += 1
    return fwd_orig(*a, **kw)


backbone.forward = fwd_counted

ANCHOR, GAMMA = 2, 3
out3 = dspark.dspark_draft(backbone, context, ANCHOR, GAMMA, markov, conf_head, greedy=True)
P(f"[m07·玩具参数] V=6、d=8、骨干 2 层、Markov r=4、anchor={ANCHOR}、γ={GAMMA}（全部 seed 固定）")
P(f"    输入串（anchor-as-first）= {dspark.anchor_as_first_inputs(ANCHOR, GAMMA)}"
      f"（anchor + {GAMMA - 1} 个 MASK，{-1} 为 mask 槽约定）")
P(f"    并行阶段一次前向：backbone.forward 调用数 = {calls['n']}，"
      f"U 形状 = {list(out3['base_logits'].shape)}（γ 行 × V 列）")

P("")
P("-- [m07·序列阶段] 逐步手推 prev 链（greedy）--")
for s in out3["steps"]:
    logits = s["logits"]
    order = np.argsort(-logits)[:2]
    P(f"    位 {s['k']}: prev={s['prev']}  U_k top2 = "
          f"(token {int(order[0])}: {F(logits[order[0]])}, token {int(order[1])}: {F(logits[order[1]])})"
          f"  bias top2 = (token {int(np.argmax(s['bias']))}: {F(np.max(s['bias']))}, "
          f"token {int(np.argmin(s['bias']))}: {F(np.min(s['bias']))})"
          f"  -> 采出 token {s['token']}，prev 链更新为 {s['token']}")
P(f"    draft_tokens = {out3['draft_tokens']}（串行依赖的全部实现 = 循环里一次 prev 赋值）")
P(f"    逐位置信度 c_k = {[F(c) for c in out3['confidences']]}（Eq.7：σ(w·[h_k;W_1[x_(k-1)]])）")

P("")
P("-- [m07·gamma 无关] γ=5 再跑一次：前向调用数不变 --")
calls["n"] = 0
out5 = dspark.dspark_draft(backbone, context, ANCHOR, 5, markov, conf_head, greedy=True)
P(f"    γ=5：backbone.forward 调用数 = {calls['n']}，U 形状 = {list(out5['base_logits'].shape)}"
      f"——并行阶段仍是单次前向（T_draft 近独立于 γ）")

P("")
P("-- [m07·probabilistic 模式] 序列采样记 q(x)（vLLM probabilistic 的 draft_logits 逐位写入）--")
U = out3["base_logits"]
rng_s = np.random.default_rng(9)
toks, dists, steps_p = dspark.sample_sequential(U, ANCHOR, markov, rng=rng_s, greedy=False)
P(f"    seed=9 采样：draft_tokens = {toks}")
for k, p_k in enumerate(dists):
    top = int(np.argmax(p_k))
    P(f"    位 {k}: q(x) top1 = token {top}（{F(p_k[top])}），这是验证时概率比测试用的 p_d 行")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {
        "m06": {"vocab": VOCAB, "p_x1_of": 0.6, "p_x1_no": 0.4},
        "m07": {"V": 6, "d": 8, "layers": 2, "markov_rank": 4, "anchor": ANCHOR,
                "gamma": GAMMA, "seeds": "backbone=3, markov=4, conf=5, ctx=6, sample=9"},
    },
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
