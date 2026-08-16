#!/usr/bin/env python3
"""m9 机械复核：回程消息数 = 拍数（每拍每客户端 1 条 EngineCoreOutputs），与批内请求数无关。

纯算术（非 vLLM 代码 trace）。行为锚点（对 pin v0.27.1 现核）：
- vllm/v1/core/sched/scheduler.py:L1885-L1887  should_emit_output = bool(new_token_ids or ...)
- vllm/v1/core/sched/scheduler.py:L2014-L2017  每客户端每拍恰组一条 EngineCoreOutputs
- vllm/v1/engine/core.py:L1435-L1442           busy loop 逐 (client_index, outputs) 入 output_queue
- vllm/v1/engine/__init__.py:L184-L215/L230-L258  每请求一条 EngineCoreOutput 装进 outputs 列表
场景参数（教学小值）：单引擎、单前台（client_index=0）、批内 4 个同拍 decode 请求、持续 3 拍。
"""
import json

K_REQUESTS = 4   # 批内同拍 decode 请求数（教学选取）
BEATS = 3        # 持续拍数
LONG_RUN_BEATS = 100


def main() -> None:
    out = {"params": {"k_requests": K_REQUESTS, "beats": BEATS}, "beats": []}
    actual_cum = 0
    naive_cum = 0
    for beat in range(1, BEATS + 1):
        actual_cum += 1                      # 每拍每客户端 1 条聚合消息
        naive_cum += K_REQUESTS              # 朴素：逐请求各发一条
        out["beats"].append({
            "beat": beat,
            "new_tokens_in_batch": f"{K_REQUESTS} x 1 = {K_REQUESTS}",
            "v1_messages_this_beat": 1,
            "engine_core_outputs_inside": K_REQUESTS,   # 每请求一条，装在同一条消息里
            "naive_messages_this_beat": K_REQUESTS,
            "cum_v1": actual_cum,
            "cum_naive": naive_cum,
        })
    out["total_v1"] = actual_cum
    out["total_naive"] = naive_cum
    out["compression_ratio"] = f"1/{K_REQUESTS}（消息数压缩到批内请求数分之一）"
    out["long_run"] = {
        "beats": LONG_RUN_BEATS,
        "v1_messages": LONG_RUN_BEATS,
        "naive_messages": LONG_RUN_BEATS * K_REQUESTS,
        "ratio": K_REQUESTS,
    }
    # 1:1 定理：prompt <= 单拍预算的请求，生成 n token = n 拍 = 恰出现在 n 条回程消息里。
    out["per_request_rule"] = "n tokens -> n beats -> n messages (prompt <= budget)"
    assert actual_cum == BEATS == 3 and naive_cum == 12
    assert LONG_RUN_BEATS * K_REQUESTS == 400
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
