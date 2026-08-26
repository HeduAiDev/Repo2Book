"""Driver for m1 (RUNNING 侧抢占重试环：allocate_slots None → while True 抢 FCFS
队尾 → 重试；抢到自己仍 None 则整拍放弃) — host run against the ch11 subtract-only
scheduler companion (pin vLLM v0.27.1, scheduler.py:L575-L630).

Scenario A (preempt-tail-then-retry): pool 4 blocks, r1/r2/r3 16-token prompts
(distinct token ids to avoid hash aliasing). Beat 1 admits all three (1 block
each, free 1). Beat 2: r1's decode needs its 2nd block and takes the last free
one; r2's decode then gets None -> preempt the FCFS tail (r3, youngest) ->
retry succeeds on r3's freed block. Allocate calls recorded via proxy:
[(r1,1)->OK, (r2,1)->None, (r2,1)->OK].
Beat 3: guard re-opened, r3 (PREEMPTED, waiting head) is re-attempted but its
resume needs 1 new block > 0 free -> None -> break. WAITING side never preempts.

Scenario B (self-preemption gives up the whole beat): pool 1, single r1.
Beat B-1 admits it (free 0). Beat B-2: decode needs a 2nd block -> None ->
preempt self (the only running request) -> `preempted_req == request` -> break:
the whole beat is abandoned even though its own freed block would arithmetically
suffice — resume goes through the WAITING admission path next beat (full-ISL
gate + watermark live there). Beat B-3: re-admitted as resumed with a 16-token
prefix re-hit + 1 new token.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.kv_cache_manager import get_request_block_hasher  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


def make_request(req_id, prompt, base, max_tokens=64):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        block_hasher=get_request_block_hasher(16),
    )


def step(sched, tokens_by_req):
    out = sched.schedule()
    req_ids = list(out.num_scheduled_tokens)
    from implementation.output import ModelRunnerOutput
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req.get(rid, []) for rid in req_ids],
    )
    outputs = sched.update_from_output(out, mro)
    emitted = {o.request_id: list(o.new_token_ids)
               for eco in outputs.values() for o in eco.outputs}
    return out, emitted


def run_scenario(sched, requests, script):
    """script: list of (label, note, tokens_by_req)."""
    calls = []
    orig = sched.kv_cache_manager.allocate_slots

    def rec(request, num_new_tokens, **kw):
        res = orig(request, num_new_tokens, **kw)
        calls.append({"req": request.request_id,
                      "ask": num_new_tokens, "ok": res is not None})
        return res

    sched.kv_cache_manager.allocate_slots = rec
    beats = []
    for label, note, tokens in script:
        calls.clear()
        free_before = sched.kv_cache_manager.num_free_blocks
        out, emitted = step(sched, tokens)
        beats.append({
            "beat": label, "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "allocate_calls": list(calls),
            "num_scheduled_tokens": dict(out.num_scheduled_tokens),
            "preempted_req_ids": sorted(out.preempted_req_ids),
            "running_order": [r.request_id for r in sched.running],
            "waiting_order": [r.request_id for r in sched.waiting],
            "resumed_req_ids": sorted(out.scheduled_cached_reqs.resumed_req_ids),
            "emitted": emitted,
            "req_state": {r.request_id: {
                "num_tokens": r.num_tokens,
                "num_computed_tokens": r.num_computed_tokens,
                "num_preemptions": r.num_preemptions,
                "status": r.status.name,
                "held_blocks": len(sched.kv_cache_manager._blocks.get(r.request_id, [])),
            } for r in requests.values()},
        })
    return beats


def main():
    out = {
        "driver": "run_m1_preempt_ring.py",
        "mechanism": "m1 RUNNING 侧抢占重试环（scheduler.py:L575-L630）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "scenario_A": {
            "config": {"num_gpu_blocks": 4, "block_size": 16,
                       "prompts": {"r1": 16, "r2": 16, "r3": 16},
                       "note": "三请求 16-token prompt（token id 互不重叠），beat 2 r2 触发抢占"},
        },
        "scenario_B": {
            "config": {"num_gpu_blocks": 1, "block_size": 16,
                       "prompts": {"r1": 16},
                       "note": "单请求占满唯一块：decode 需第 2 块 → 抢到自己 → 整拍放弃；"
                               "下一拍经 WAITING 准入恢复（前缀重命中 16 + 补 1）"},
        },
    }

    # ---- Scenario A ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    reqs = {"r1": make_request("r1", 16, 0), "r2": make_request("r2", 16, 100),
            "r3": make_request("r3", 16, 200)}
    for r in reqs.values():
        sched.add_request(r)
    out["scenario_A"]["beats"] = run_scenario(sched, reqs, [
        ("A-1", "首拍：三请求全量 16 各进批（各 1 块，空闲 1）", {"r1": [21], "r2": [22], "r3": [23]}),
        ("A-2", "r1 领走最后 1 空闲块；r2 差 1 → None → 抢队尾 r3 → 原样重试成功", {"r1": [24], "r2": [25]}),
        ("A-3", "守卫开：r1/r2 照常 decode；r3 恢复重命中 16 只差 1 新块 > 空闲 0 → None 只 break（WAITING 侧绝不抢占）", {"r1": [26], "r2": [27]}),
    ])
    assert out["scenario_A"]["beats"][1]["allocate_calls"] == [
        {"req": "r1", "ask": 1, "ok": True},
        {"req": "r2", "ask": 1, "ok": False},
        {"req": "r2", "ask": 1, "ok": True},
    ], out["scenario_A"]["beats"][1]["allocate_calls"]
    assert out["scenario_A"]["beats"][1]["preempted_req_ids"] == ["r3"]

    # ---- Scenario B ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=1, block_size=16)
    reqs = {"r1": make_request("r1", 16, 0)}
    for r in reqs.values():
        sched.add_request(r)
    out["scenario_B"]["beats"] = run_scenario(sched, reqs, [
        ("B-1", "单请求首拍占满唯一块（空闲 0）", {"r1": [5]}),
        ("B-2", "decode 需第 2 块 → None → 抢到自己（唯一在场者）→ break 整拍放弃", {}),
        ("B-3", "下一拍 WAITING 准入：重命中自己的 16 token 前缀 + 补 1（resumed）", {"r1": [6]}),
    ])
    b2 = out["scenario_B"]["beats"][1]
    assert b2["num_scheduled_tokens"] == {} and b2["preempted_req_ids"] == ["r1"]
    assert out["scenario_B"]["beats"][2]["resumed_req_ids"] == ["r1"]
    assert out["scenario_B"]["beats"][2]["num_scheduled_tokens"] == {"r1": 1}

    dest = Path(__file__).with_name("m1_preempt_ring.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for sc in ("scenario_A", "scenario_B"):
        for b in out[sc]["beats"]:
            print(sc, b["beat"], "sched", b["num_scheduled_tokens"],
                  "preempted", b["preempted_req_ids"],
                  "calls", [(c["req"], c["ask"], c["ok"]) for c in b["allocate_calls"]])


if __name__ == "__main__":
    main()
