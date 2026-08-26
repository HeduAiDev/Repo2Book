"""Driver for m2 (抢谁：FCFS self.running.pop() 队尾=最年轻 vs PRIORITY
max((priority, arrival_time)) + 本拍已领回滚) — host run, pin vLLM v0.27.1,
scheduler.py:L588-L615. The精简版 deleted the PRIORITY branch (dossier.delete
第 1 条), so this trace runs the FCFS half only; the PRIORITY contrast is
quoted from real source anchors (see explainer.json m2.quantified/caveats).

Pool 5 blocks. r1 = 32-token prompt (2 blocks, the BIGGEST consumer), r2/r3 =
16-token prompts (1 block each, identical size — the only distinguishing
factor between r2 and r3 is admission order). Beat 1 admits r1,r2,r3 in FCFS
order (running = [r1,r2,r3], free 1). Beat 2: r1's decode takes the last free
block with its 3rd block (free 0); r2's decode then needs its 2nd block ->
None -> self.running.pop() picks the TAIL = r3 (youngest, admitted last) —
NOT the biggest consumer r1 (who just took a block) and NOT the middle r2
(who triggered the failure). r3's block goes to r2 on retry.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.kv_cache_manager import get_request_block_hasher  # noqa: E402
from implementation.output import ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


def make_request(req_id, prompt_len, base, arrival):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
        arrival_time=arrival,
        block_hasher=get_request_block_hasher(16),
    )


def step(sched, tokens_by_req):
    out = sched.schedule()
    req_ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req.get(rid, []) for rid in req_ids],
    )
    outputs = sched.update_from_output(out, mro)
    emitted = {o.request_id: list(o.new_token_ids)
               for eco in outputs.values() for o in eco.outputs}
    return out, emitted


def main():
    out = {
        "driver": "run_m2_victim_choice.py",
        "mechanism": "m2 抢谁的选择：FCFS self.running.pop() 队尾（scheduler.py:L588-L615，精简版保留 FCFS 半边）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版；PRIORITY 分支随 dossier.delete 第 1 条删（真实源码 L588-L613 见 caveats）",
        "config": {
            "num_gpu_blocks": 5, "block_size": 16,
            "prompts": {"r1": 32, "r2": 16, "r3": 16},
            "arrival_order": ["r1", "r2", "r3"],
            "note": "r1 是最大占用者（2 块）；r2/r3 同为 1 块——二者唯一差别是入列顺序（r3 最年轻）",
        },
        "beats": [],
    }

    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=5, block_size=16)
    t0 = 1000.0
    reqs = {
        "r1": make_request("r1", 32, 0, t0),
        "r2": make_request("r2", 16, 100, t0 + 1.0),
        "r3": make_request("r3", 16, 200, t0 + 2.0),
    }
    for rid in ("r1", "r2", "r3"):
        sched.add_request(reqs[rid])

    calls = []
    orig = sched.kv_cache_manager.allocate_slots

    def rec(request, num_new_tokens, **kw):
        res = orig(request, num_new_tokens, **kw)
        calls.append({"req": request.request_id, "ask": num_new_tokens,
                      "ok": res is not None})
        return res

    sched.kv_cache_manager.allocate_slots = rec
    victim_events = []
    orig_preempt = sched._preempt_request

    def rec_preempt(request, timestamp, drop_stale_output=False):
        victim_events.append({"victim": request.request_id,
                              "arrival_time": request.arrival_time})
        return orig_preempt(request, timestamp, drop_stale_output)

    sched._preempt_request = rec_preempt

    def beat(label, note, tokens):
        calls.clear()
        free_before = sched.kv_cache_manager.num_free_blocks
        o, emitted = step(sched, tokens)
        out["beats"].append({
            "beat": label, "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "allocate_calls": list(calls),
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "running_order": [r.request_id for r in sched.running],
            "waiting_order": [r.request_id for r in sched.waiting],
            "blocks_held": {rid: len(sched.kv_cache_manager._blocks.get(rid, []))
                            for rid in reqs},
            "num_preemptions": {rid: reqs[rid].num_preemptions for rid in reqs},
            "emitted": emitted,
        })

    beat("1", "首拍 FCFS 序准入：r1(2 块)+r2(1 块)+r3(1 块)，空闲 1", {"r1": [31], "r2": [32], "r3": [33]})
    beat("2", "r1 decode 领走最后 1 空闲块（第 3 块）；r2 需第 2 块 → None → running.pop() 抢队尾 r3（最年轻，非触发者 r2、非最大占用者 r1）→ r2 重试成功", {"r1": [34]})

    assert victim_events == [{"victim": "r3", "arrival_time": t0 + 2.0}], victim_events
    b2 = out["beats"][1]
    assert b2["preempted_req_ids"] == ["r3"]
    assert b2["running_order"] == ["r1", "r2"]
    assert b2["blocks_held"] == {"r1": 3, "r2": 2, "r3": 0}
    assert b2["num_preemptions"] == {"r1": 0, "r2": 0, "r3": 1}
    out["victim_choice"] = {
        "victim": "r3",
        "why": "FCFS 下 running.pop() 恒取队尾=最晚入列者；与本拍谁触发分配失败无关、与持有块数无关",
        "victim_arrival_time": victim_events[0]["arrival_time"],
        "arrival_times": {"r1": t0, "r2": t0 + 1.0, "r3": t0 + 2.0},
        "r1_blocks_before_after": [2, 3],
        "r2_blocks_before_after": [1, 2],
        "r3_blocks_before_after": [1, 0],
        "priority_branch_note": "PRIORITY 策略换 max(running, key=(priority, arrival_time)) 并回滚被抢者本拍已领 token/块/预算——真实源码 scheduler.py:L588-L613，精简版已删不可运行",
    }

    dest = Path(__file__).with_name("m2_victim_choice.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"], "victim_events",
              victim_events, "blocks", b["blocks_held"], "preempts", b["num_preemptions"])


if __name__ == "__main__":
    main()
