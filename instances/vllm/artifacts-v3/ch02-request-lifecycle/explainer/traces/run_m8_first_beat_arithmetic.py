#!/usr/bin/env python3
"""m8 机械复核：256-token prompt 请求首拍/decode 拍的块账与拍数（纯算术，非 vLLM 代码 trace）。

本章 chapter_kind=l0_dynamic_walkthrough_no_companion、无精简版 companion，host 无 CUDA、
真 trace 须进容器加载完整引擎——超出「素材优先取 deepread 卡、不重复挖」边界。
本脚本只对 explainer.json m8 表格里的算术做机械复核；输入常量全部来自源码（file:Lxxx 见
explainer.json m8 params，均对 pin v0.27.1 逐行核实），行为锚点取 deepread/engine-loop.json
data_flow（「一个 256-token prompt 的请求，首次被调度的一拍」，行号已核验）。
"""
import json
import math

BLOCK_SIZE = 16        # vllm/config/cache.py:L47 DEFAULT_BLOCK_SIZE
TOKEN_BUDGET = 2048    # vllm/config/scheduler.py:L42 DEFAULT_MAX_NUM_BATCHED_TOKENS
PROMPT_TOKENS = 256    # deepread/engine-loop.json data_flow 样例请求
LONG_PREFILL_THRESHOLD = 0  # vllm/config/scheduler.py:L70 默认 0（不钳制）


def blocks_for(total_tokens: int) -> int:
    """每 token 一个 KV 槽、每块 BLOCK_SIZE 槽 → 覆盖 total_tokens 需 ceil(total/bs) 块。"""
    return math.ceil(total_tokens / BLOCK_SIZE)


def main() -> None:
    out = {
        "constants": {
            "block_size": BLOCK_SIZE,
            "token_budget": TOKEN_BUDGET,
            "prompt_tokens": PROMPT_TOKENS,
            "long_prefill_token_threshold": LONG_PREFILL_THRESHOLD,
        },
        "beats": [],
    }

    # 拍1 prefill：num_new_tokens = 256 - 0 = 256；阈值 0 不钳制；256 <= 2048 预算不钳制。
    sched1 = PROMPT_TOKENS if LONG_PREFILL_THRESHOLD == 0 else min(
        PROMPT_TOKENS, LONG_PREFILL_THRESHOLD)
    clamped1 = min(sched1, TOKEN_BUDGET)
    assert clamped1 == 256, clamped1
    blocks1 = blocks_for(256)
    out["beats"].append({
        "beat": 1, "phase": "prefill",
        "num_new_tokens": f"{PROMPT_TOKENS} - 0 = {PROMPT_TOKENS}",
        "budget_check": f"min({PROMPT_TOKENS}, {TOKEN_BUDGET}) = {clamped1} (不钳制)",
        "kv_blocks_after": blocks1,
        "kv_slots_after": blocks1 * BLOCK_SIZE,
        "tokens_computed_after": 256,
        "sampled_tokens": 1,  # prefill 尾 token 物化 logits → 首个输出 token
        "status": "WAITING -> RUNNING",
    })

    # 拍2 decode：总 token 257（prompt 256 + 首输出 token 回喂）；num_new_tokens = 1。
    total2 = 257
    blocks2 = blocks_for(total2)
    out["beats"].append({
        "beat": 2, "phase": "decode",
        "num_new_tokens": f"{total2} + 0 - 256 = 1",
        "kv_blocks_after": blocks2,
        "kv_blocks_new": blocks2 - blocks1,
        "kv_slots_after": blocks2 * BLOCK_SIZE,
        "tail_waste": blocks2 * BLOCK_SIZE - total2,
        "sampled_tokens": 1,
        "tokens_computed_after": total2,
    })
    assert blocks2 == 17 and blocks2 * BLOCK_SIZE == 272
    assert blocks2 * BLOCK_SIZE - total2 == 15 <= BLOCK_SIZE - 1

    # 拍3 decode：总 token 258；不新增块（258 <= 272 槽）。
    total3 = 258
    blocks3 = blocks_for(total3)
    out["beats"].append({
        "beat": 3, "phase": "decode",
        "num_new_tokens": f"{total3} - 257 = 1",
        "kv_blocks_after": blocks3,
        "kv_blocks_new": blocks3 - blocks2,
        "sampled_tokens": 1,
        "tokens_computed_after": total3,
    })
    assert blocks3 == blocks2 == 17

    # 拍数守恒：生成 n 个 token = 1 prefill 拍（含首 token）+ (n-1) decode 拍 = n 拍。
    out["beats_for_n_tokens"] = "1 + (n - 1) = n"
    # 块上界：任意时刻块数 = ceil(当前总 token / 16)，尾部浪费 <= 15 = block_size - 1。
    out["max_tail_waste"] = BLOCK_SIZE - 1

    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
