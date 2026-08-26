"""Driver for m11 (WAITING 侧 allocate_slots 准入：None → break，绝不触发抢占；
full_sequence_must_fit 全序列准入门) — host run against the ch10 subtract-only
scheduler companion (pin vLLM v0.27.1).

Pool = 4 blocks x 16 = 64 token-slots. r1/r2 (16-token prompts, 1 block
each) in place; r3 is a 48-token prompt (3 blocks full sequence).

(a) full-ISL gate ON (default scheduler_reserve_full_isl=True): beats 2-3,
    r3's 48-token sequence needs 3 blocks > 2 free -> None -> break. r3
    stays WAITING, r1/r2 untouched: zero preemptions, zero wasted compute.

(b) full-ISL gate OFF, budget 64: beat 1 r3's FIRST CHUNK (min(48, 32)=32)
    needs only 2 blocks <= 2 free -> admitted — but the pool is now 100%
    committed. Beat 2: the 16-token continuation needs 1 more block, 0 free
    -> None -> preempt the FCFS tail, which is r3 itself -> its 32 computed
    tokens are DISCARDED (recompute-only: num_computed_tokens back to 0).
    "首 chunk 装得下 ≠ 整条装得下" (WC4, issue #39734 shape).
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


def make_request(req_id, prompt_len, max_tokens=8):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def run(tag, full_isl, budget):
    config = SchedulerConfig(
        max_num_batched_tokens=budget,
        scheduler_reserve_full_isl=full_isl,
    )
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=4, block_size=16)
    reqs = {}
    for rid in ("r1", "r2", "r3"):
        reqs[rid] = make_request(rid, 16 if rid != "r3" else 48)
    sched.add_request(reqs["r1"])
    sched.add_request(reqs["r2"])
    sched.add_request(reqs["r3"])  # r3 queued from the start

    calls = []
    orig_alloc = sched.kv_cache_manager.allocate_slots

    def rec_alloc(request, num_new_tokens, **kw):
        res = orig_alloc(request, num_new_tokens, **kw)
        calls.append({
            "req": request.request_id,
            "ask_tokens": num_new_tokens,
            "ok": res is not None,
            "full_sequence_must_fit": kw.get("full_sequence_must_fit", False),
        })
        return res

    sched.kv_cache_manager.allocate_slots = rec_alloc

    beats = []
    n = 3
    for i in range(1, n + 1):
        calls.clear()
        free_before = sched.kv_cache_manager.num_free_blocks
        o = sched.schedule()
        beats.append({
            "beat": i,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "allocate_calls": list(calls),
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "running_len": len(sched.running),
            "running_ids": [r.request_id for r in sched.running],
            "r3": {
                "status": reqs["r3"].status.name,
                "computed": reqs["r3"].num_computed_tokens,
                "num_preemptions": reqs["r3"].num_preemptions,
                "is_prefill_chunk": reqs["r3"].is_prefill_chunk,
            },
            "r1_r2_status": [reqs["r1"].status.name, reqs["r2"].status.name],
        })
    return {
        "config": {
            "scheduler_reserve_full_isl": full_isl,
            "num_gpu_blocks": 4, "block_size": 16,
            "pool_token_capacity": 64,
            "max_num_batched_tokens": budget,
        },
        "requests": {"r1": {"prompt_len": 16}, "r2": {"prompt_len": 16}, "r3": {"prompt_len": 48}},
        "beats": beats,
        "wasted_prefill_tokens": 32 if not full_isl else 0,
    }


def main():
    out = {
        "driver": "run_m11_admission_gate.py",
        "mechanism": "m11 WAITING 侧准入 None→break + full_sequence_must_fit 整序列门（scheduler.py:L965-L994；config/scheduler.py:L130-L134）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "scenario_gate_on": run("on", full_isl=True, budget=2048),
        "scenario_gate_off": run("off", full_isl=False, budget=64),
    }
    dest = Path(__file__).with_name("m11_admission_gate.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for tag in ("scenario_gate_on", "scenario_gate_off"):
        print(tag)
        for b in out[tag]["beats"]:
            print(" ", b["beat"], b["num_scheduled_tokens"], "preempted", b["preempted_req_ids"],
                  "r3", b["r3"]["status"], b["r3"]["computed"])


if __name__ == "__main__":
    main()
