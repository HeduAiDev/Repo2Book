"""ch33 m02+m03 驱动脚本（拒绝采样验证规则 + 无损定理）。

跑法：python explainer/traces/run_m02_m03_verify.py
产出：explainer/traces/run_m02_m03_verify.json

素材来源：implementation/spec_decode.py（accepted / sample_recovered / verify_block /
verify_block_greedy / empirical_output_distribution / acceptance_rate_sum_min /
analytical_acceptance_rate / generate_stream）。
手算组：词表 {A,B}={0,1}，p_t=(0.7,0.3)、p_d=(0.5,0.5)（与论文 App.A 反例同一组数字，
m02 手算 ↔ m03 无损证明 ↔ m11 置信标签三方闭环）。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import spec_decode as sd  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731
V3 = lambda a, n=6: [round(float(v), n) for v in np.asarray(a).ravel()]  # noqa: E731
A_NAME, B_NAME = "A", "B"

p_t = np.array([0.7, 0.3])
p_d = np.array([0.5, 0.5])

# ══ m02 · 验证规则 ══════════════════════════════════════════════════════════
P("== ch33 m02 · 拒绝采样验证规则（词表 {A,B}，p_t=(0.7,0.3)，p_d=(0.5,0.5)）==")

P("")
P("-- [m02·判据] 逐 token 接受概率 min(1, p_t(x)/p_d(x)) --")
P(f"    x=A: p_t/p_d = 0.7/0.5 = {F(0.7/0.5)} -> min(1,1.4) = 1.0（必收）")
P(f"    x=B: p_t/p_d = 0.3/0.5 = {F(0.3/0.5)}（以 0.6 的概率接受）")

P("")
P("-- [m02·残差] r(x)=max(0, p_t-p_d)；拒绝时的去向 --")
res = sd.residual_distribution(p_t, p_d)
P(f"    residual = ({V3(res)[0]}, {V3(res)[1]})；拒绝总质量 = {F(res.sum())}")
P(f"    = 1/2*||p_t-p_d||_1 = 1/2*(0.2+0.2) = {F(0.5*np.abs(p_t - p_d).sum())}")
norm_res = res / res.sum()
P(f"    norm(residual) = ({V3(norm_res)[0]}, {V3(norm_res)[1]})（恢复 token 恒为 A）")

P("")
P("-- [m02·verify_block 实跑] gamma=3、draft=[A,A,A]、target 三行各异（seed 搜出 接受,接受,拒绝 的位型）--")
P("    （位 0/1 的 token A 比值 1.4/1.3>1 必收、位 2 比值 0.4/0.5=0.8 才是掷骰位——三件事一次全见）")
draft_tokens = [0, 0, 0]
draft_probs = np.tile(p_d, (3, 1))
target_rows = np.array([[0.7, 0.3], [0.65, 0.35], [0.4, 0.6]])
bonus_row = np.array([0.7, 0.3])
target_probs = np.vstack([target_rows, bonus_row])
story = None
for seed in range(200):
    rng = np.random.default_rng(seed)
    _, _, info = sd.verify_block(draft_tokens, draft_probs, target_probs, rng)
    flags = [s["accepted"] for s in info["steps"]]
    if flags == [True, True, False]:
        story = (seed, info)
        break
assert story is not None
seed, info = story
P(f"    seed={seed}（位 0/1 接受、位 2 拒绝：接受段+首拒截断+残差恢复一次全见）")
for s in info["steps"]:
    tok = A_NAME if s["token"] == 0 else B_NAME
    P(f"    位 {s['k']}: token={tok}  p_t(x)={F(target_rows[s['k']][s['token']])}  p_d(x)=0.5  "
      f"u={F(s['u'])}  p_t/p_d={F(s['ratio'])}  "
      f"min(1,ratio)={F(s['accept_prob'])}  accepted={s['accepted']}")
res_row2 = sd.residual_distribution(target_rows[2], p_d)
P(f"    首拒位 {info['rejected_at']}：残差 = ({V3(res_row2)[0]}, {V3(res_row2)[1]})，"
      f"norm 后 = ({F(res_row2[0]/res_row2.sum())}, {F(res_row2[1]/res_row2.sum())})")
P(f"    recovered token = {'A' if info['recovered'] == 0 else 'B'}（Gumbel-max 从 norm(residual) 采出）")
P(f"    emitted（发射）= A, A, {'A' if info['recovered'] == 0 else 'B'}；"
      f"accepted 计数 = {info['n_accepted']}；其后 draft 位全部丢弃（首拒即停）")

P("")
P("-- [m02·全收+bonus] seed 搜出三位全接受的位型 --")
for seed in range(200):
    rng = np.random.default_rng(seed)
    _, _, info = sd.verify_block(draft_tokens, draft_probs, target_probs, rng)
    if info["rejected_at"] is None:
        break
P(f"    seed={seed}：位 0/1/2 全接受 -> 追加 bonus（从 target 第 4 行 (0.7,0.3) 采）= "
      f"{'A' if info['bonus'] == 0 else 'B'}；本周期发射 4 个 token（3 接受 + 1 bonus）")

P("")
P("-- [m02·对数空间等价] log p(x) > log u + log q(x)（V2 kernel 姿态）逐点对拍 --")
tok, row = 0, target_rows[2]  # x=A, p_t(A)=0.4, p_d(A)=0.5
log_p, log_q = np.log(row[tok]), np.log(p_d[tok])
for u in (0.5, 0.7, 0.9):
    ok_prob = sd.accepted(tok, row, p_d, u)
    ok_log = sd.accepted_logspace(tok, np.log(row), np.log(p_d), u)
    P(f"    u={u}: u*q = {F(u*0.5)} vs p = 0.4 -> {ok_prob}；"
      f"log 形态 {F(log_p)} > log(u)+log(q) = {F(np.log(u)+log_q)} -> {ok_log}；相等={ok_prob == ok_log}")

P("")
P("-- [m02·残差 Gumbel-max 免归一化] 三词表非退化例 p_t=(0.5,0.4,0.1) vs p_d=(0.2,0.1,0.7) --")
p3_t, p3_d = np.array([0.5, 0.4, 0.1]), np.array([0.2, 0.1, 0.7])
r3 = sd.residual_distribution(p3_t, p3_d)
P(f"    residual = ({V3(r3)[0]}, {V3(r3)[1]}, {V3(r3)[2]})，sum = {F(r3.sum())}，"
      f"norm = ({F(r3[0]/r3.sum())}, {F(r3[1]/r3.sum())}, {F(r3[2]/r3.sum())})")
for E in ([1.7, 0.9, 0.3], [0.5, 2.2, 1.1]):
    E = np.array(E)
    scores = r3 / E
    P(f"    Exp 抽样 E={E.tolist()} -> scores r/E = ({V3(scores)[0]}, {V3(scores)[1]}, {V3(scores)[2]})"
          f" -> argmax = token {int(np.argmax(scores))}（分母 Z 全程没算）")
rng = np.random.default_rng(21)
counts = np.zeros(3)
for _ in range(10000):
    counts[sd.sample_recovered(p3_t, p3_d, rng)] += 1
P(f"    MC 10000 次 sample_recovered 频率 = ({F(counts[0]/10000)}, {F(counts[1]/10000)}, {F(counts[2]/10000)})"
      f"（norm(residual) = (0.5, 0.5, 0.0) 对拍 ✓）")

P("")
P("-- [m02·单位置无损·经验] n=200000（seed=11）--")
emp = sd.empirical_output_distribution(p_t, p_d, 200000, np.random.default_rng(11))
P(f"    empirical P(y=A) = {F(emp[0])}, P(y=B) = {F(emp[1])}（对照 p_t = (0.7, 0.3)）")
acc_rate = sd.acceptance_rate_sum_min(p_t, p_d)
P(f"    经验接受率对照：sum min(p_t,p_d) = {F(acc_rate)} = 1 - 1/2*||p_t-p_d||_1 = "
      f"{F(sd.analytical_acceptance_rate(p_t, p_d))}")

# ══ m03 · 无损定理 ══════════════════════════════════════════════════════════
P("")
P("== ch33 m03 · 无损定理两情形直证（P(y=x) = min(q,p)(x) + max(0,p-q)(x) = p(x)）==")

P("")
P("-- [m03·逐 token 对账] 三词表例 p=(0.5,0.4,0.1) vs q=(0.2,0.1,0.7) --")
for x in range(3):
    m = min(p3_t[x], p3_d[x]); pos = max(0.0, p3_t[x] - p3_d[x])
    P(f"    x={x}: min(q,p) = {F(m)}, max(0,p-q) = {F(pos)}, 和 = {F(m+pos)} = p({x}) = {F(p3_t[x])} ✓")
P(f"    Σmin = {F(sd.acceptance_rate_sum_min(p3_t, p3_d))}；拒绝质量 Σmax(0,q-p) = "
      f"{F(np.maximum(p3_d - p3_t, 0).sum())}；残差质量 Σmax(0,p-q) = {F(r3.sum())}；两者相等 = "
      f"{'YES' if abs(np.maximum(p3_d-p3_t,0).sum() - r3.sum()) < 1e-12 else 'NO'}"
      f"（= 1/2*||p-q||_1 = {F(0.5*np.abs(p3_t-p3_d).sum())}——拒绝质量恰好被残差质量对冲）")
P(f"    (A,B) 组同账：Σmin = {F(acc_rate)}，拒绝质量 = {F(np.maximum(p_d-p_t,0).sum())}，"
      f"1/2*||·||_1 = {F(0.5*np.abs(p_t-p_d).sum())}")

P("")
P("-- [m03·greedy 特例] one-hot q：接受 ⟺ draft==target argmax --")
onehot = np.array([1.0, 0.0])
P(f"    Σmin(p_t, one-hot(argmax q)) = {F(sd.acceptance_rate_sum_min(p_t, onehot))}"
      f"（vs 全 q 的 {F(acc_rate)}：greedy 接受率更低，但输出分布仍无损——见下）")

# 整流生成流：greedy draft 流 ≡ target greedy 流
def target_cond(prev):  # 4 词表 bigram 玩具 target
    rows = {
        0: [0.60, 0.25, 0.10, 0.05],
        1: [0.10, 0.55, 0.25, 0.10],
        2: [0.05, 0.20, 0.60, 0.15],
        3: [0.25, 0.10, 0.15, 0.50],
    }
    return np.array(rows[prev])


def greedy_drafter(prev):
    p = target_cond(prev)
    g = int(np.argmax(p))
    return [g] * 3, None


def target_greedy_rollout(anchor, n):
    out, prev = [], anchor
    for _ in range(n):
        t = int(np.argmax(target_cond(prev)))
        out.append(t)
        prev = t
    return out


n_stream = 400
stream, stats = sd.generate_stream(target_cond, greedy_drafter, anchor=0,
                                   n_tokens=n_stream, rng=np.random.default_rng(3), greedy=True)
ref = target_greedy_rollout(0, n_stream)
P(f"    greedy draft 整流流（{n_stream} token）与 target greedy rollout 逐位相同 = "
      f"{'YES' if stream == ref else 'NO'}")

P("")
P("-- [m03·概率化 draft 整流流] bigram 条件分布对拍（n=4000，seed=7）--")
P("    （drafter 逐位从自己的 q_k 采样——q_k=0.8*target+0.2*均匀；声明的 q 就是采样的 q，")
P("      这是无损前提『draft 分布声明与实际采样一致』的玩具版）")
rng_d = np.random.default_rng(7)


def noisy_drafter(prev):
    toks, qs = [], []
    walk = int(prev)
    for _ in range(3):
        q = 0.8 * target_cond(walk) + 0.2 * np.array([0.25, 0.25, 0.25, 0.25])
        x = int(rng_d.choice(4, p=q))
        toks.append(x)
        qs.append(q)
        walk = x
    return toks, np.asarray(qs)


stream2, stats2 = sd.generate_stream(target_cond, noisy_drafter, anchor=0,
                                     n_tokens=4000, rng=np.random.default_rng(7))
rows_ok = []
for prev in range(4):
    idx = [i for i in range(len(stream2) - 1) if stream2[i] == prev]
    if len(idx) < 50:
        continue
    nxt = np.array([stream2[i + 1] for i in idx])
    emp_cond = np.array([(nxt == v).mean() for v in range(4)])
    err = float(np.abs(emp_cond - target_cond(prev)).max())
    rows_ok.append((prev, V3(emp_cond), F(err)))
for prev, emp_row, err in rows_ok:
    P(f"    prev={prev}: 经验条件分布 {emp_row} vs target 行 {V3(target_cond(prev))}（max|差| = {err}）")
P(f"    每 token 接受的『+1』：mean_emitted_per_cycle = {F(stats2['mean_emitted_per_cycle'])}，"
      f"mean_accepted_per_cycle = {F(stats2['mean_accepted_per_cycle'])}（差值 = 恒 1：拒绝位恢复或全收 bonus）")

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {
        "vocab": "{A=0,B=1}", "p_t": [0.7, 0.3], "p_d": [0.5, 0.5], "gamma": 3,
        "n_unit_position": 200000, "n_stream_greedy": n_stream, "n_stream_prob": 4000,
        "p3_t": [0.5, 0.4, 0.1], "p3_d": [0.2, 0.1, 0.7],
    },
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
