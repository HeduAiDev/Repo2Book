"""Driver for m15 (乐观推进：_update_after_schedule 在 GPU 未算时
num_computed_tokens += n + is_prefill_chunk 标记) — host run against the ch10
subtract-only scheduler companion (pin vLLM v0.27.1).

Part A — chunk lifecycle: budget 16, a 40-token prompt -> chunks [16, 16, 8].
Immediately after each schedule() returns (no execute_model exists anywhere
in this companion — the whole advance happens inside schedule()), the ledger
already shows computed += n and num_in_flight_tokens += n. is_prefill_chunk
stays True until the final chunk closes the gap, at which point the request
leaves _inflight_prefills.

Part B — set swap, not clear: pool 2 blocks, two 16-token prompts; beat 2's
block contention preempts r2. The SchedulerOutput holds the OLD
reset_preempted_req_ids set object; _update_after_schedule then REPLACES the
scheduler's field with a fresh empty set — so the output keeps {r2} while
the scheduler already shows {} (scheduler.py:L1361-L1365 NOTE: clearing in
place would also mutate the output).
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


def main():
    out = {
        "driver": "run_m15_optimistic.py",
        "mechanism": "m15 _update_after_schedule 乐观推进（scheduler.py:L1317-L1343）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "part_a_chunk_lifecycle": None,
        "part_b_set_swap": None,
    }

    # ---- part A: optimistic advance across the chunk lifecycle --------------
    config = SchedulerConfig(max_num_batched_tokens=16)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)
    r = make_request("r1", 40)
    sched.add_request(r)
    beats = []
    for i in range(1, 4):
        computed_before = r.num_computed_tokens
        in_flight_before = r.num_in_flight_tokens
        o = sched.schedule()
        beats.append({
            "beat": i,
            "scheduled": o.num_scheduled_tokens.get("r1"),
            "computed_before": computed_before,
            "computed_after_schedule_returns": r.num_computed_tokens,
            "in_flight_before": in_flight_before,
            "in_flight_after": r.num_in_flight_tokens,
            "gpu_has_run": False,
            "is_prefill_chunk": r.is_prefill_chunk,
            "in_inflight_prefills": r in sched._inflight_prefills,
            "note": "schedule() 刚返回、没有任何执行调用——账本已 +n（注释第 2 条：让 prefill 下一拍立即再可调度）",
        })
    out["part_a_chunk_lifecycle"] = {
        "config": {"max_num_batched_tokens": 16, "prompt_len": 40},
        "beats": beats,
        "chunk_sizes": [b["scheduled"] for b in beats],
        "computed_progression": [b["computed_after_schedule_returns"] for b in beats],
        "drain_note": "同步引擎里 ⑤ 拍 update_from_output 会消耗 in-flight（ch9/ch11 范围）；本精简版不含 ⑤ 拍，in_flight 只增——验证的是 _update_after_schedule 的记账方向",
    }

    # ---- part B: preempted/finished sets are swapped, not cleared -----------
    config = SchedulerConfig(max_num_batched_tokens=2048)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=2, block_size=16)
    r1 = make_request("a1", 16)
    r2 = make_request("a2", 16)
    sched.add_request(r1)
    sched.add_request(r2)
    o1 = sched.schedule()  # admit both, pool now empty
    r1.append_output_token_ids(1)
    r2.append_output_token_ids(1)
    set_before = sched.reset_preempted_req_ids
    o2 = sched.schedule()  # r1 needs 2nd block -> preempt r2
    out["part_b_set_swap"] = {
        "config": {"num_gpu_blocks": 2, "block_size": 16, "requests": {"a1": {"prompt_len": 16}, "a2": {"prompt_len": 16}}},
        "beat1_scheduled": dict(o1.num_scheduled_tokens),
        "beat2_scheduled": dict(o2.num_scheduled_tokens),
        "beat2_preempted_on_output": sorted(o2.preempted_req_ids),
        "output_set_is_scheduler_set_object": o2.preempted_req_ids is set_before,
        "scheduler_set_after": sorted(sched.reset_preempted_req_ids),
        "output_set_still_holds": sorted(o2.preempted_req_ids),
        "note": "换新不 clear（L1361-L1365）：若就地 clear()，已发出的 SchedulerOutput 会跟着变空——worker 就收不到抢占通知了",
    }

    dest = Path(__file__).with_name("m15_optimistic.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in beats:
        print("A", b["beat"], b["scheduled"], "computed", b["computed_before"], "->", b["computed_after_schedule_returns"],
              "chunk", b["is_prefill_chunk"])
    pb = out["part_b_set_swap"]
    print("B beat2", pb["beat2_scheduled"], "preempted on output", pb["beat2_preempted_on_output"],
          "same object", pb["output_set_is_scheduler_set_object"], "scheduler now", pb["scheduler_set_after"])


if __name__ == "__main__":
    main()
