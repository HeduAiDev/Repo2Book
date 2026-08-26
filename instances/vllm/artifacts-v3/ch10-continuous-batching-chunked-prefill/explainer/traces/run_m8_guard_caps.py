"""Driver for m8 (WAITING 守卫：本拍抢占过（preempted_reqs 非空）+ PAUSED 就整拍
不收新；max_num_running_reqs 上限) — host run against the ch10 subtract-only
scheduler companion (pin vLLM v0.27.1).

Sub-scenario (a) the guard: pool 3 blocks; r1 (16 tok, 1 block) and r2
(32 tok, 2 blocks) admitted in beat 1. r3 (16 tok) queued. Beat 2: r1's
decode needs a block, pool empty -> preempt tail r2 (frees 2) -> r1 retry OK
(free left 1). preempted_reqs=[r2] -> the WAITING stage is SKIPPED whole:
r3 is never even peeked (allocate-call log shows no r3 call), although 1
block is free and 2047 budget tokens remain. Beats 3-4: guard open, but the
preempted r2 sits at the waiting HEAD (prepend) and its 3-block full-ISL
need (> 1 free) breaks the loop before r3 is reached. Beat 5: r1 retired
manually (ch9/ch11 scope), r2 re-admitted as resumed.
Sub-scenario (b) the cap: max_num_seqs=2 with three 16-token prompts —
r3 stays waiting while len(running) is pinned at the cap, admitted only
after r1 retires.
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


def main():
    out = {
        "driver": "run_m8_guard_caps.py",
        "mechanism": "m8 WAITING 守卫 preempted_reqs/PAUSED + max_num_running_reqs（scheduler.py:L683-L692）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "scenario_a_guard": None,
        "scenario_b_cap": None,
    }

    # ---- scenario (a): preemption beat closes the whole WAITING stage -------
    config = SchedulerConfig(max_num_batched_tokens=2048)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=3, block_size=16)
    r1 = make_request("r1", 16)
    r2 = make_request("r2", 32)
    r3 = make_request("r3", 16)
    sched.add_request(r1)
    sched.add_request(r2)

    calls = []
    orig_alloc = sched.kv_cache_manager.allocate_slots

    def rec_alloc(request, num_new_tokens, **kw):
        res = orig_alloc(request, num_new_tokens, **kw)
        calls.append({"req": request.request_id, "ask_tokens": num_new_tokens, "ok": res is not None})
        return res

    sched.kv_cache_manager.allocate_slots = rec_alloc

    beats_a = []

    def beatA(label, note):
        calls.clear()
        o = sched.schedule()
        beats_a.append({
            "beat": label,
            "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": o.total_num_scheduled_tokens,
            "budget": 2048,
            "budget_left": 2048 - o.total_num_scheduled_tokens,
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "new_admissions": [d.req_id for d in o.scheduled_new_reqs],
            "resumed_admissions": sorted(o.scheduled_cached_reqs.resumed_req_ids),
            "allocate_calls": list(calls),
            "r3_ever_asked_this_beat": any(c["req"] == "r3" for c in calls),
            "waiting_order": [r.request_id for r in sched.waiting],
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
        })

    beatA(1, "beat 1: r1（1 块）+ r2（2 块）进批——池 3 块全占；r3 排队")
    sched.add_request(r3)
    r1.append_output_token_ids(1)
    r2.append_output_token_ids(1)
    beatA(2, "beat 2: r1 需第 2 块 → None → 抢 r2 → 重试 OK；preempted_reqs=[r2] → 守卫关闸：r3 连被 peek 都没有（allocate 无 r3 调用）")
    r1.append_output_token_ids(1)
    beatA(3, "beat 3: 守卫开，但被抢的 r2 在 waiting 队头（prepend）——r2 需 3 块 > 空闲 1 → None → break，r3 排在后面没轮到")
    r1.append_output_token_ids(1)
    beatA(4, "beat 4: 同拍 3——r2 需 3 块始终拿不到，r3 陪等")
    sched.running.remove(r1)
    sched.kv_cache_manager.free(r1)
    sched.requests.pop("r1", None)
    sched.finished_req_ids.add("r1")
    beatA(5, "beat 5: r1 退场（手工模拟 ⑤ 拍完成）→ 空闲 3 → r2 以 resumed 重入（33 token 整段重算）")
    out["scenario_a_guard"] = {
        "config": {"num_gpu_blocks": 3, "block_size": 16, "budget": 2048},
        "requests": {"r1": {"prompt_len": 16}, "r2": {"prompt_len": 32}, "r3": {"prompt_len": 16, "queued_before": "beat 2"}},
        "manual_retire_note": "r1 于拍 4 后手工退场（running.remove + kv free + finished_req_ids——模拟 ch9/ch11 的 ⑤ 拍完成路径）",
        "beats": beats_a,
    }

    # ---- scenario (b): max_num_running_reqs cap ------------------------------
    config = SchedulerConfig(max_num_batched_tokens=2048, max_num_seqs=2)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)
    c1 = make_request("r1", 16)
    c2 = make_request("r2", 16)
    c3 = make_request("r3", 16)
    for r in (c1, c2, c3):
        sched.add_request(r)
    beats_b = []

    def beatB(label, note):
        o = sched.schedule()
        beats_b.append({
            "beat": label,
            "note": note,
            "num_running": len(sched.running),
            "max_num_running_reqs": 2,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "new_admissions": [d.req_id for d in o.scheduled_new_reqs],
            "waiting_ids": [r.request_id for r in sched.waiting],
        })

    beatB(1, "beat 1: 收 r1、r2 后 num_running=2 ≥ cap → break，r3 留 waiting")
    c1.append_output_token_ids(1)
    c2.append_output_token_ids(1)
    beatB(2, "beat 2: r1/r2 各 decode 1；num_running 仍 2 ≥ cap → r3 继续等")
    sched.running.remove(c1)
    sched.kv_cache_manager.free(c1)
    sched.requests.pop("r1", None)
    sched.finished_req_ids.add("r1")
    c2.append_output_token_ids(1)
    beatB(3, "beat 3: r1 退场后 num_running=1 < cap → r3 首次进批（16 全量）")
    out["scenario_b_cap"] = {
        "config": {"max_num_seqs": 2, "budget": 2048},
        "manual_retire_note": "r1 于拍 2 后手工退场（同上，模拟 ⑤ 拍完成路径）",
        "beats": beats_b,
    }

    dest = Path(__file__).with_name("m8_guard_caps.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in beats_a:
        print("a", b["beat"], b["num_scheduled_tokens"], "preempted", b["preempted_req_ids"],
              "r3_asked", b["r3_ever_asked_this_beat"])
    for b in beats_b:
        print("b", b["beat"], b["num_running"], b["new_admissions"])


if __name__ == "__main__":
    main()
