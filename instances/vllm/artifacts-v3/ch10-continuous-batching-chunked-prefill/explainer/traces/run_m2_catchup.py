"""Driver for m2 (无 prefill/decode 相位：num_computed_tokens 追赶
num_tokens_with_spec —— 追赶公式三特例) — host run against the ch10
subtract-only scheduler companion (pin vLLM v0.27.1).

One request's whole life under the catch-up formula: an 8192-token prompt
with a 2048 token budget takes 4 prefill chunks (2048 each — special case 1
"new prompt" via the WAITING-side cut, then special case 3 "continuing
chunk" via the RUNNING-side clamp), then decodes exactly 1 token per beat
(special case 2) once outputs start arriving. Every formula component is
recorded per beat: num_tokens_with_spec + num_output_placeholders -
num_computed_tokens, then the three clamps (threshold / token_budget /
max_model_len headroom) and the scheduled result.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402

BUDGET = 2048
MODEL_LEN = 16384
PROMPT = 8192


def main():
    out = {
        "driver": "run_m2_catchup.py",
        "mechanism": "m2 追赶公式 num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens（scheduler.py:L516-L532；字段 request.py:L271-L277）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {
            "max_num_batched_tokens": BUDGET,
            "max_num_scheduled_tokens_fallback": BUDGET,
            "max_model_len": MODEL_LEN,
            "long_prefill_token_threshold": 0,
            "enable_chunked_prefill": True,
            "block_size": 16,
            "num_gpu_blocks": "1<<30 (非约束)",
        },
        "request": {"req_id": "r1", "prompt_len": PROMPT, "max_tokens": 32},
        "beats": [],
    }

    config = SchedulerConfig(max_num_batched_tokens=BUDGET)
    sched = Scheduler(config, max_model_len=MODEL_LEN, num_gpu_blocks=1 << 30, block_size=16)
    req = Request(
        request_id="r1",
        prompt_token_ids=list(range(PROMPT)),
        sampling_params=SamplingParams(max_tokens=32),
    )
    sched.add_request(req)

    def beat(label, special_case):
        nts_before = req.num_tokens_with_spec
        ph = req.num_output_placeholders
        computed_before = req.num_computed_tokens
        raw_gap = nts_before + ph - computed_before
        o = sched.schedule()
        scheduled = o.num_scheduled_tokens.get("r1")
        rec = {
            "beat": label,
            "special_case": special_case,
            "num_tokens_with_spec": nts_before,
            "num_output_placeholders": ph,
            "num_computed_tokens_before": computed_before,
            "raw_gap": raw_gap,
            "clamps": {
                "long_prefill_token_threshold": 0,
                "threshold_active": False,
                "token_budget": BUDGET,
                "budget_active": raw_gap > BUDGET,
                "max_model_len_headroom": MODEL_LEN - computed_before - 1,
                "max_len_active": False,
            },
            "scheduled": scheduled,
            "num_computed_tokens_after": req.num_computed_tokens,
            "is_prefill_chunk": req.is_prefill_chunk,
            "num_in_flight_tokens": req.num_in_flight_tokens,
        }
        out["beats"].append(rec)
        return o

    beat(1, "特例一：新 prompt（WAITING 侧 num_new = num_tokens - 0，被 min(token_budget) 切）")
    beat(2, "特例三：续 chunk（RUNNING 侧同一公式，差 6144 被预算钳到 2048）")
    beat(3, "特例三：续 chunk（差 4096 → 2048）")
    beat(4, "特例三：末 chunk（差 2048 → 2048；拍后追平，is_prefill_chunk 翻 False）")
    # ⑤ 拍回填：模拟采样各回 1 个 token
    req.append_output_token_ids(7)
    beat(5, "特例二：decode（num_tokens_with_spec 8193 - computed 8192 = 1）")
    req.append_output_token_ids(8)
    beat(6, "特例二：decode（差恒 1）")
    req.append_output_token_ids(9)
    beat(7, "特例二：decode（差恒 1）")

    out["chunk_sizes"] = [b["scheduled"] for b in out["beats"]]
    out["computed_progression"] = [b["num_computed_tokens_after"] for b in out["beats"]]
    out["num_beats_to_finish_prefill"] = 4
    out["prompt_to_budget_ratio"] = PROMPT // BUDGET

    dest = Path(__file__).with_name("m2_catchup.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], b["special_case"][:12], "gap", b["raw_gap"], "->", b["scheduled"],
              "computed", b["num_computed_tokens_after"])


if __name__ == "__main__":
    main()
