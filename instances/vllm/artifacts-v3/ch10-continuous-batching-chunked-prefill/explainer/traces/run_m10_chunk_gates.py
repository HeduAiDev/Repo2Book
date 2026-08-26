"""Driver for m10 (chunked prefill 切块三闸：long_prefill_token_threshold 钳制 →
enable_chunked_prefill 开关（关且超预算整拍 break）→ min(token_budget)) — host
run against the ch10 subtract-only scheduler companion (pin vLLM v0.27.1).

One 70-token prompt through three gate configurations:
(a) budget 32, threshold 0, chunked ON  -> chunks [32, 32, 6]  (budget gate)
(b) budget 2048, threshold 16, chunked ON -> chunks [16, 16, 16, 16, 6]
    (threshold gate binds before the budget gate — 4+1 beats)
(c) budget 32, threshold 0, chunked OFF -> num_new_tokens 70 > 32 every beat:
    the WAITING loop breaks before admission, the request stays WAITING
    forever (empty beat, TTFT unbounded under this config).
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


def make_request(req_id="r1", prompt_len=70, max_tokens=16):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def run_gates(tag, budget, threshold, chunked, n_beats):
    config = SchedulerConfig(
        max_num_batched_tokens=budget,
        long_prefill_token_threshold=threshold,
        enable_chunked_prefill=chunked,
    )
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)
    req = make_request()
    sched.add_request(req)
    beats = []
    for i in range(1, n_beats + 1):
        gap = req.num_tokens - req.num_computed_tokens
        o = sched.schedule()
        scheduled = o.num_scheduled_tokens.get("r1")
        over_budget_break = (not chunked) and gap > budget
        beats.append({
            "beat": i,
            "raw_gap": gap,
            "threshold": threshold,
            "after_threshold": min(gap, threshold) if 0 < threshold < gap else gap,
            "enable_chunked_prefill": chunked,
            "gap_over_budget": gap > budget,
            "chunked_off_break": over_budget_break,
            "budget": budget,
            "final_chunk": scheduled,
            "computed_after": req.num_computed_tokens,
            "is_prefill_chunk": req.is_prefill_chunk,
            "status": req.status.name,
        })
        if scheduled is None:
            break
    return {
        "config": {
            "max_num_batched_tokens": budget,
            "long_prefill_token_threshold": threshold,
            "enable_chunked_prefill": chunked,
        },
        "prompt_len": 70,
        "beats": beats,
        "chunk_sizes": [b["final_chunk"] for b in beats],
    }


def main():
    out = {
        "driver": "run_m10_chunk_gates.py",
        "mechanism": "m10 chunked prefill 切块三闸（scheduler.py:L874-L914；config vllm/config/scheduler.py:L70-L80）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "scenario_a_budget_gate": run_gates("a", budget=32, threshold=0, chunked=True, n_beats=3),
        "scenario_b_threshold_gate": run_gates("b", budget=2048, threshold=16, chunked=True, n_beats=5),
        "scenario_c_chunked_off": run_gates("c", budget=32, threshold=0, chunked=False, n_beats=3),
    }
    dest = Path(__file__).with_name("m10_chunk_gates.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for k in ("scenario_a_budget_gate", "scenario_b_threshold_gate", "scenario_c_chunked_off"):
        print(k, out[k]["chunk_sizes"])


if __name__ == "__main__":
    main()
