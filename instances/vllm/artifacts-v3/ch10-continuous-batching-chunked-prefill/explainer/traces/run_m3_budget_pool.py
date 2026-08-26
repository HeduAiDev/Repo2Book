"""Driver for m3 (单一 token 预算池：token_budget = max_num_scheduled_tokens 跨
RUNNING/WAITING 两阶段分账 + 守恒断言) — host run against the ch10 subtract-only
scheduler companion (pin vLLM v0.27.1).

Scenario B is the finger-count version (budget 4): a 2-token prompt and an
8-token prompt share one pool. The 8-token prompt is chunked [2, 3, 3] purely
by the remaining budget, then turns decode. Every beat records the pool
balance: spend per request, budget_left = budget - total, and the two
conservation invariants that schedule() asserts (sum <= budget, budget >= 0).

Scenario A (budget 32) reprises the m1 mixed batch from the pool's viewpoint:
RUNNING eats 3 (three decodes), WAITING spends 29 on the new chunk, beat
total lands exactly on 32.
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


def make_request(req_id, prompt_len, max_tokens=16):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def make_scheduler(budget):
    config = SchedulerConfig(max_num_batched_tokens=budget)
    return Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)


def run_scenario(sched, reqs, script, label, budget):
    """script: list of (beat_note, [req_ids to append 1 output before beat])"""
    beats = []
    for note, append_ids in script:
        for rid in append_ids:
            reqs[rid].append_output_token_ids(1)
        o = sched.schedule()
        beats.append({
            "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": o.total_num_scheduled_tokens,
            "budget": budget,
            "budget_left": budget - o.total_num_scheduled_tokens,
            "sum_equals_total_assert": sum(o.num_scheduled_tokens.values()) == o.total_num_scheduled_tokens,
            "sum_le_budget_assert": o.total_num_scheduled_tokens <= budget,
            "conservation_asserts_passed": True,  # schedule() returned => asserts held
            "ledger_after": {
                rid: {
                    "num_tokens": r.num_tokens,
                    "num_computed_tokens": r.num_computed_tokens,
                    "is_prefill_chunk": r.is_prefill_chunk,
                }
                for rid, r in reqs.items()
            },
        })
    return beats


def main():
    out = {
        "driver": "run_m3_budget_pool.py",
        "mechanism": "m3 单一 token 预算池跨两阶段分账 + 守恒断言（scheduler.py:L459/L523/L636-L637/L913/L1073/L1108-L1113）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "scenario_B": {
            "budget": 4,
            "requests": {"r1": {"prompt_len": 2}, "r2": {"prompt_len": 8}},
            "beats": None,
        },
        "scenario_A": {
            "budget": 32,
            "requests": {"r1": {"prompt_len": 8}, "r2": {"prompt_len": 8}, "r3": {"prompt_len": 8}, "r4": {"prompt_len": 64, "arrives": "before beat 2"}},
            "beats": None,
        },
    }

    # ---- scenario B: budget 4, prompts 2 and 8 -----------------------------
    sched = make_scheduler(4)
    reqs = {"r1": make_request("r1", 2), "r2": make_request("r2", 8)}
    for r in reqs.values():
        sched.add_request(r)
    out["scenario_B"]["beats"] = run_scenario(
        sched, reqs,
        [
            ("beat 1: r1 全量 2；r2 剩 2 → 首 chunk 2（8 被预算 2 截断）", []),
            ("beat 2: r1 decode 1；r2 续 chunk 3（差 6 被剩 3 截）", ["r1"]),
            ("beat 3: r1 decode 1；r2 续 chunk 3（差 3 恰等于剩 3）", ["r1"]),
            ("beat 4: r1 decode 1；r2 转 decode 恰 1", ["r1", "r2"]),
            ("beat 5: 两人 decode 各 1", ["r1", "r2"]),
        ],
        "B", 4,
    )
    out["scenario_B"]["r2_chunk_sizes"] = [
        b["num_scheduled_tokens"].get("r2") for b in out["scenario_B"]["beats"]
    ]

    # ---- scenario A: budget 32, mixed batch from the pool's viewpoint ------
    sched = make_scheduler(32)
    reqs = {
        "r1": make_request("r1", 8), "r2": make_request("r2", 8), "r3": make_request("r3", 8),
    }
    for r in reqs.values():
        sched.add_request(r)
    script = [
        ("beat 1: 三个 8-token 新 prompt 同拍全量进批", []),
        ("beat 2: RUNNING 三个 decode 各 1（花 3），WAITING r4 首 chunk 29 —— 池恰打满", ["r1", "r2", "r3"]),
        ("beat 3: decode 3 + r4 续 chunk 29 —— 再次恰打满", ["r1", "r2", "r3"]),
        ("beat 4: decode 3 + r4 尾 chunk 6 —— 池大量闲置", ["r1", "r2", "r3"]),
        ("beat 5: 四请求全 decode —— 池只用 4", ["r1", "r2", "r3", "r4"]),
    ]
    out["scenario_A"]["beats"] = []
    for i, (note, append_ids) in enumerate(script):
        if i == 1:
            reqs["r4"] = make_request("r4", 64)
            sched.add_request(reqs["r4"])
        out["scenario_A"]["beats"].extend(run_scenario(sched, reqs, [(note, append_ids)], "A", 32))

    dest = Path(__file__).with_name("m3_budget_pool.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for tag in ("scenario_B", "scenario_A"):
        print(tag)
        for b in out[tag]["beats"]:
            print(" ", b["num_scheduled_tokens"], "total", b["total"], "left", b["budget_left"])


if __name__ == "__main__":
    main()
