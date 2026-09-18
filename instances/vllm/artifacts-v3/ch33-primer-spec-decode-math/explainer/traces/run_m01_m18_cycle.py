"""ch33 m01+m18 驱动脚本（投机周期与延迟账 + 一次前向验证的布局算术）。

跑法：python explainer/traces/run_m01_m18_cycle.py
产出：explainer/traces/run_m01_m18_cycle.json（params + raw_stdout）。

素材来源：
- implementation/spec_decode.py（latency_per_token / speedup_vs_plain / Eq.(1)）
- implementation/acceptance_length.py（恒 α 闭式 + 蒙特卡洛对拍）
- m18 布局算术：vllm/v1/worker/gpu_model_runner.py:L2851-L2924 的数值 docstring
  （cu=[4,104,107,207,209]、num_draft=[3,0,2,0,1]）——本脚本用 numpy 原样复算
  docstring 的三步 repeat+arange 算术并逐值断言相等。
"""
import json
import sys
from pathlib import Path

import numpy as np

CH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CH / "implementation"))
import spec_decode  # noqa: E402
import acceptance_length as al  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LINES = []


def P(s=""):
    LINES.append(str(s))
    print(s)


F = lambda x, n=6: round(float(x), n)  # noqa: E731

# ══ m01 · 周期与延迟账 L=(T_draft+T_verify)/tau ══════════════════════════════
P("== ch33 m01 · 投机周期与延迟账：L=(T_draft+T_verify)/tau（Eq.1） ==")
T_DRAFT, T_VERIFY = 1.0, 8.0
P(f"[m01·参数] T_draft={F(T_DRAFT)} ms, T_verify={F(T_VERIFY)} ms（玩具单位：ms；比例可换算）")

P("")
P("-- [m01·基线] 无投机：每 token 一次 target 前向，L_base = T_verify --")
L_base = F(spec_decode.latency_per_token(0.0, T_VERIFY, 1.0))
P(f"    L_base = 8.0/1 = {L_base} ms/token")

P("")
P("-- [m01·好 drafter] tau=3：一次周期出 3 个 token --")
tau_good = 3.0
L_good = F(spec_decode.latency_per_token(T_DRAFT, T_VERIFY, tau_good))
sp_good = F(spec_decode.speedup_vs_plain(T_DRAFT, T_VERIFY, tau_good))
P(f"    L = (1.0+8.0)/3 = {F((T_DRAFT + T_VERIFY) / tau_good)} ms/token（= trace 值 {L_good}）")
P(f"    speedup = 8.0/3 = {F(T_VERIFY / ((T_DRAFT + T_VERIFY) / tau_good))}（= trace 值 {sp_good}）")

P("")
P("-- [m01·差 drafter] tau=1.2：接受率掉下来 --")
tau_bad = 1.2
L_bad = F(spec_decode.latency_per_token(T_DRAFT, T_VERIFY, tau_bad))
sp_bad = F(spec_decode.speedup_vs_plain(T_DRAFT, T_VERIFY, tau_bad))
P(f"    L = (1.0+8.0)/1.2 = {F((T_DRAFT + T_VERIFY) / tau_bad)} ms/token（= trace 值 {L_bad}）")
P(f"    speedup = 8.0/7.5 = {F(T_VERIFY / ((T_DRAFT + T_VERIFY) / tau_bad))}（= trace 值 {sp_bad}）")

P("")
P("-- [m01·盈亏平衡] speedup=1 的 tau：(T_draft+T_verify)/T_verify --")
tau_be = (T_DRAFT + T_VERIFY) / T_VERIFY
P(f"    tau_break_even = (1.0+8.0)/8.0 = {F(tau_be)}（tau 低于此值纯亏：L > 8.0）")

P("")
P("-- [m01·论文量级] tau=3.54509（alpha=0.92、gamma=3 恒定接受率闭式）--")
ALPHA, GAMMA = 0.92, 3
tau_92 = al.expected_accepted_length_constant(ALPHA, GAMMA)
P(f"    closed form (1-alpha^(gamma+1))/(1-alpha) = (1-0.92^4)/0.08 = {F(tau_92)}")
mc = al.simulate_accepted_length([ALPHA] * GAMMA, 30000, np.random.default_rng(5))
P(f"    Monte Carlo 30000 周期（seed=5）: mean tau = {F(mc)}")
L_92 = F(spec_decode.latency_per_token(T_DRAFT, T_VERIFY, tau_92))
sp_92 = F(spec_decode.speedup_vs_plain(T_DRAFT, T_VERIFY, tau_92))
P(f"    L = 9.0/3.54509 = {F((T_DRAFT + T_VERIFY) / tau_92)} ms/token（= trace 值 {L_92}）")
P(f"    speedup = {F(T_VERIFY / ((T_DRAFT + T_VERIFY) / tau_92))}（= trace 值 {sp_92}）")

P("")
P("-- [m01·三杠杆] 三行延迟账并排 --")
P(f"    基线                : tau=—      L={F(spec_decode.latency_per_token(0.0, T_VERIFY, 1.0))} ms/token  speedup=1.0")
P(f"    draft 更准（tau 3→3.54509）: L={L_92} ms/token  speedup={sp_92}")
P(f"    draft 更差（tau 3→1.2） : L={L_bad} ms/token  speedup={sp_bad}")

# ══ m18 · 一次前向验证的布局算术 ════════════════════════════════════════════
P("")
P("== ch33 m18 · 一次前向验证的布局算术（gpu_model_runner.py:L2851-L2924 docstring 复算）==")
cu_num_scheduled = np.array([4, 104, 107, 207, 209], dtype=np.int32)
num_draft = np.array([3, 0, 2, 0, 1], dtype=np.int32)
P(f"[m18·输入] cu_num_scheduled_tokens = {cu_num_scheduled.tolist()}")
P(f"[m18·输入] num_draft_tokens        = {num_draft.tolist()}")

num_sampled = num_draft + 1
P(f"[m18·step0] num_sampled = num_draft+1 = {num_sampled.tolist()}")

# Step 1: cumsum + per-segment arange
cu_num_sampled = np.cumsum(num_sampled)
total_sampled = int(cu_num_sampled[-1])
arange_s = np.concatenate([np.arange(n) for n in num_sampled])
P(f"[m18·step1] cu_num_sampled_tokens = {cu_num_sampled.tolist()}")
P(f"[m18·step1] _arange_scratch[:{total_sampled}] = {arange_s.tolist()}")

# Step 2+3: logits_indices
logits_indices = np.repeat(cu_num_scheduled - num_sampled, num_sampled) + arange_s
P(f"[m18·step2] repeat(cu_sched-num_sampled, num_sampled) = {np.repeat(cu_num_scheduled - num_sampled, num_sampled).tolist()}")
P(f"[m18·step3] logits_indices = step2 + arange = {logits_indices.tolist()}")

bonus_logits_indices = (cu_num_sampled - 1).tolist()
P(f"[m18·bonus] bonus_logits_indices = cu_num_sampled-1 = {bonus_logits_indices}")

# draft/target logits indices
cu_num_draft = np.cumsum(num_draft)
arange_d = np.concatenate([np.arange(n) for n in num_draft])
target_logits_indices = np.repeat(cu_num_sampled - num_sampled, num_draft) + arange_d
P(f"[m18·draft] cu_num_draft_tokens = {cu_num_draft.tolist()}")
P(f"[m18·draft] _arange_scratch[:{int(cu_num_draft[-1])}] = {arange_d.tolist()}")
P(f"[m18·draft] target_logits_indices = {target_logits_indices.tolist()}")

# draft_token_ids: input_ids[logits_indices][target_logits_indices + 1]
input_ids = np.arange(int(cu_num_scheduled[-1]))  # 用位置号当 token id（docstring 同款示意）
draft_token_ids = input_ids[logits_indices][target_logits_indices + 1]
P(f"[m18·draft_token_ids] input_ids[logits_indices][target_logits_indices+1] = {draft_token_ids.tolist()}")

assert logits_indices.tolist() == [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
assert bonus_logits_indices == [3, 4, 7, 8, 10]
assert target_logits_indices.tolist() == [0, 1, 2, 5, 6, 9]
assert draft_token_ids.tolist() == [1, 2, 3, 105, 106, 208]
P("[m18·断言] 四组输出与 docstring 逐值相等 ✓")

P("")
P("-- [m18·逐请求读法] 5 个请求的 draft 位与 bonus 位在展平 logits 里的交错 --")
start = 0
for r in range(5):
    ns = int(num_sampled[r]); nd = int(num_draft[r])
    seg = logits_indices[start:start + ns].tolist()
    tseg = [seg[j] for j in range(ns) if j < nd]
    bseg = seg[nd] if nd < ns else seg[-1]
    P(f"    req{r}: 采样位 {seg} = target 位 {tseg} + bonus 位 [{bseg}]")
    start += ns

# ══ 落盘 ═══════════════════════════════════════════════════════════════════
out = {
    "params": {
        "m01": {"T_draft_ms": 1.0, "T_verify_ms": 8.0, "tau_good": 3.0, "tau_bad": 1.2,
                "alpha": ALPHA, "gamma": GAMMA, "mc_cycles": 30000, "mc_seed": 5},
        "m18": {"cu_num_scheduled": cu_num_scheduled.tolist(), "num_draft": num_draft.tolist()},
    },
    "raw_stdout": "\n".join(LINES),
}
jf = Path(__file__).with_suffix(".json")
jf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print(f"\n[written] {jf}")
