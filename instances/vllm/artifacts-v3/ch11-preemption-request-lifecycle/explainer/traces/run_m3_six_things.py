"""Driver for m3 (_preempt_request 六件事：free 块（哈希保留）/ status=PREEMPTED /
num_computed_tokens=0 / 清 spec_token_ids / stale 标记（=in_flight，assign 不
累加）/ num_preemptions+1 + 回 waiting 队头) — host run, pin vLLM v0.27.1,
scheduler.py:L1274-L1315.

Pool 2 blocks. Beat 1 admits r1 (16-token, 1 block) and r2 (16-token, 1
block); a waiting request w1 is enqueued to show that the victim is prepended
at the HEAD of the waiting queue, ahead of w1. Before beat 2: r2.spec_token_ids
= [9, 9] (to verify '清 spec') and r2.num_in_flight_tokens = 2 (async-depth-2
simulation so the stale assign is visible with a non-trivial value; in the
sync engine this is 0 at schedule time and stale self-neutralizes).
Beat 2: r1's decode needs its 2nd block -> None -> preempt the tail r2.
A wrapper around _preempt_request snapshots every ledger field immediately
before and after the call.
"""
import json
import sys
import time
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


def make_request(req_id, prompt_len, base):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
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
    sched.update_from_output(out, mro)
    return out


def main():
    out = {
        "driver": "run_m3_six_things.py",
        "mechanism": "m3 _preempt_request 六件事（scheduler.py:L1274-L1315）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版（六件事全保留）",
        "config": {
            "num_gpu_blocks": 2, "block_size": 16,
            "prompts": {"r1": 16, "r2": 16, "w1": 16},
            "pre_state_setup": [
                "beat 1 后 r2.spec_token_ids=[9,9]（验证『清 spec』）",
                "beat 2 调度前 r2.num_in_flight_tokens=2（模拟 async 深度 2 的在途份额；同步引擎此处恒 0 → stale 自中和为 0）",
            ],
        },
        "beats": [],
    }

    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=2, block_size=16)
    r1 = make_request("r1", 16, 0)
    r2 = make_request("r2", 16, 100)
    w1 = make_request("w1", 16, 300)
    sched.add_request(r1)
    sched.add_request(r2)

    snapshots = []
    orig_preempt = sched._preempt_request

    def snap(req):
        return {
            "request_id": req.request_id,
            "status": req.status.name,
            "num_computed_tokens": req.num_computed_tokens,
            "spec_token_ids": list(req.spec_token_ids),
            "num_stale_output_tokens": req.num_stale_output_tokens,
            "num_in_flight_tokens": req.num_in_flight_tokens,
            "num_output_placeholders": req.num_output_placeholders,
            "num_preemptions": req.num_preemptions,
            "held_blocks": len(sched.kv_cache_manager._blocks.get(req.request_id, [])),
            "cached_block_hashes_size": len(sched.kv_cache_manager.cached_block_hashes),
            "waiting_order": [r.request_id for r in sched.waiting],
            "free_blocks": sched.kv_cache_manager.num_free_blocks,
        }

    def rec_preempt(request, timestamp, drop_stale_output=False):
        before = snap(request)
        orig_preempt(request, timestamp, drop_stale_output)
        after = snap(request)
        snapshots.append({"before": before, "after": after})
        return None

    sched._preempt_request = rec_preempt

    def beat(label, note, tokens):
        free_before = sched.kv_cache_manager.num_free_blocks
        o = step(sched, tokens)
        out["beats"].append({
            "beat": label, "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "running_order": [r.request_id for r in sched.running],
            "waiting_order": [r.request_id for r in sched.waiting],
            "r2_snapshot": snap(r2),
        })

    beat("1", "首拍 r1/r2 各 16 进批（各 1 块，空闲 0）", {"r1": [1], "r2": [2]})
    r2.spec_token_ids = [9, 9]
    r2.num_in_flight_tokens = 2  # async 深度 2 模拟（同步引擎此处为 0）
    sched.add_request(w1)  # 排在 r2 后面到达：victim 应回到 w1 之前（队头）
    beat("2", "r1 decode 需第 2 块 → None → 抢队尾 r2（_preempt_request 被包装快照）", {"r1": [3]})

    assert len(snapshots) == 1
    snap0 = snapshots[0]
    out["preempt_snapshot"] = snap0
    b, a = snap0["before"], snap0["after"]
    assert a["status"] == "PREEMPTED" and b["status"] == "RUNNING"
    assert a["num_computed_tokens"] == 0 and b["num_computed_tokens"] == 16
    assert a["spec_token_ids"] == [] and b["spec_token_ids"] == [9, 9]
    assert a["num_stale_output_tokens"] == 2 and b["num_stale_output_tokens"] == 0
    assert a["num_in_flight_tokens"] == 2  # 不动（stale 是 assign 不是搬走）
    assert a["num_preemptions"] == 1 and b["num_preemptions"] == 0
    assert a["held_blocks"] == 0 and b["held_blocks"] == 1
    assert a["free_blocks"] == 1 and b["free_blocks"] == 0
    assert b["cached_block_hashes_size"] == a["cached_block_hashes_size"] == 2
    assert b["waiting_order"] == ["w1"] and a["waiting_order"] == ["r2", "w1"]

    dest = Path(__file__).with_name("m3_six_things.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    print("six things before->after:")
    for k in ("status", "num_computed_tokens", "spec_token_ids",
              "num_stale_output_tokens", "num_preemptions", "held_blocks"):
        print(" ", k, b[k], "->", a[k])
    print("  waiting", b["waiting_order"], "->", a["waiting_order"])
    print("  cached_hashes", b["cached_block_hashes_size"], "->", a["cached_block_hashes_size"])


if __name__ == "__main__":
    main()
