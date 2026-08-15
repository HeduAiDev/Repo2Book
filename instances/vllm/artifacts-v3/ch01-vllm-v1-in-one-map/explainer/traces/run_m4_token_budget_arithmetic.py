#!/usr/bin/env python3
"""ch01 m4（调度只认 token 数）算术机械复核脚本——纯标准库、host 直跑。

注意：这不是对真实 vLLM 代码的 trace（ch01 无精简版 companion，
trace_source=manual）。本脚本只是把 explainer.json m4.worked_example.table
里的每一个数字机械重算一遍，保证零手算、零编造。

常量与公式锚点（vLLM v0.27.1, 6e448d0ea）：
  - token 预算类默认 2048           vllm/config/scheduler.py:L42
  - 钳制 num_new_tokens = min(num_new_tokens, token_budget)
      vllm/v1/core/sched/scheduler.py:L523（RUNNING 侧）/ L913（WAITING 侧）
  - decode 每拍每请求恰 1 个新 token
      vllm/v1/core/sched/scheduler.py:L516-L520
"""
import json

TOKEN_BUDGET = 2048   # vllm/config/scheduler.py:L42（类默认）
N_DECODE_REQS = 256   # 场景 A：256 个 decode 请求
PROMPT_TOKENS = 8192  # 场景 B：1 个 8K prompt 新请求

out = {}

# 场景 A：一拍 256 个 decode 请求、每请求 1 个新 token
a_beat_tokens = N_DECODE_REQS * 1
out["scenario_A"] = {
    "reqs": N_DECODE_REQS,
    "tokens_per_req_per_beat": 1,
    "beat_tokens": a_beat_tokens,
    "within_budget": a_beat_tokens <= TOKEN_BUDGET,
    "budget_remaining_after": TOKEN_BUDGET - a_beat_tokens,
}

# 场景 B：8192-token prompt 在 2048 预算下逐拍切块（chunked prefill）
pending = PROMPT_TOKENS
beats = []
while pending > 0:
    scheduled = min(pending, TOKEN_BUDGET)
    pending -= scheduled
    beats.append({"scheduled": scheduled, "pending_after": pending})
out["scenario_B"] = {
    "prompt_tokens": PROMPT_TOKENS,
    "token_budget": TOKEN_BUDGET,
    "beats": beats,
    "num_beats": len(beats),
    "pending_sequence": [PROMPT_TOKENS] + [b["pending_after"] for b in beats],
    "computed_tokens_sequence": [
        PROMPT_TOKENS - b["pending_after"] for b in beats
    ],
}

out["comparison"] = {
    # 同按 1 拍计：B 被钳制后一拍的 token 数是 A 一拍的几倍
    "B_beat_vs_A_beat": TOKEN_BUDGET // a_beat_tokens,
    # 全程：B 一个请求的总 token 是 A 一拍总量的几倍
    "B_total_vs_A_beat": PROMPT_TOKENS // a_beat_tokens,
    # ceil(8192/2048)：prefill 完成所需拍数
    "beats_to_finish_prefill": -(-PROMPT_TOKENS // TOKEN_BUDGET),
}

print(json.dumps(out, indent=2))
