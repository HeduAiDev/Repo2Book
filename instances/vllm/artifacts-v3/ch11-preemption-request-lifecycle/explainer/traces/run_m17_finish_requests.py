"""Driver for m17 (finish_requests 外部死法：断连 abort → FINISHED_ABORTED ·
三队列摘除（running/waiting/skipped）· 幂等 no-op · str/list/None 三形) —
host run, pin vLLM v0.27.1 (scheduler.py:L2237-L2298 + L1954-L1962).

Cases:
  A abort a RUNNING request (str form): two-pass removal — collected from
    running, status set, _free_request (blocks back to pool, del requests,
    finished_req_ids registered). Second call on the same id returns [] and
    a ghost id also returns [] — idempotent no-ops.
  B abort a WAITING request (list form): removed from waiting + skipped.
  C abort a PREEMPTED request: a self-preempted request sits in waiting —
    removed from there too.
  D request_ids=None (shutdown drain form): finishes everything.
(E: abort during execution — the update_from_output idempotent `continue`
 half of the contract — is exercised in traces/m11_hotloop.json beat 4.)
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
from implementation.request import Request, RequestStatus, SamplingParams  # noqa: E402
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
        "driver": "run_m17_finish_requests.py",
        "mechanism": "m17 finish_requests 外部死法（scheduler.py:L2237-L2298）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（两遍法/三队列摘除/幂等原样保留；REMOTE_KVS 延迟释放分支随 connector 删）",
        "cases": [],
    }

    def case(name, note, record):
        out["cases"].append({"case": name, "note": note, **record})

    # ---- A: RUNNING abort, str form, idempotent ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r1 = make_request("r1", 16, 0)
    sched.add_request(r1)
    step(sched, {"r1": [1]})
    assert r1 in sched.running
    ret = sched.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
    case("A RUNNING abort（str 形）", "断连 → FINISHED_ABORTED：两遍法摘 running + 置态 + free", {
        "returned": [r.request_id for r in ret],
        "r1_status": r1.status.name,
        "r1_finished_reason": r1.get_finished_reason().name,
        "r1_in_running": r1 in sched.running,
        "r1_in_requests": "r1" in sched.requests,
        "free_blocks": sched.kv_cache_manager.num_free_blocks,
        "finished_req_ids": sorted(sched.finished_req_ids),
    })
    ret2 = sched.finish_requests("r1", RequestStatus.FINISHED_ABORTED)
    ret3 = sched.finish_requests("ghost", RequestStatus.FINISHED_ABORTED)
    case("A' 幂等", "对已完成请求/未知 id：no-op（abort 双投递成立的前提）", {
        "second_call_returned": [r.request_id for r in ret2],
        "ghost_call_returned": [r.request_id for r in ret3],
    })

    # ---- B: WAITING abort, list form ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r2 = make_request("r2", 16, 100)
    sched.add_request(r2)
    ret = sched.finish_requests(["r2"], RequestStatus.FINISHED_ABORTED)
    case("B WAITING abort（list 形）", "从 waiting（与 skipped）摘除 + 置态 free", {
        "returned": [r.request_id for r in ret],
        "r2_in_waiting": r2 in sched.waiting,
        "r2_in_requests": "r2" in sched.requests,
    })

    # ---- C: PREEMPTED abort ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=1, block_size=16)
    r3 = make_request("r3", 16, 200)
    sched.add_request(r3)
    step(sched, {"r3": [7]})
    step(sched, {})  # decode 需第 2 块 → 自我被抢 → PREEMPTED 落 waiting
    assert r3.status == RequestStatus.PREEMPTED and r3 in sched.waiting
    ret = sched.finish_requests(["r3"], RequestStatus.FINISHED_ABORTED)
    case("C PREEMPTED abort", "被抢者住在 waiting：同样从 waiting 摘除 + free", {
        "r3_status_before": "PREEMPTED",
        "returned": [r.request_id for r in ret],
        "r3_in_waiting": r3 in sched.waiting,
        "r3_in_requests": "r3" in sched.requests,
        "r3_num_preemptions": r3.num_preemptions,
    })

    # ---- D: None = finish all ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r4 = make_request("r4", 16, 300)
    sched.add_request(r4)
    ret = sched.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    case("D None=全量 finish", "shutdown drain 形态：request_ids=None → requests.keys() 全收", {
        "returned": [r.request_id for r in ret],
        "num_unfinished": sched.get_num_unfinished_requests(),
    })

    a = out["cases"][0]
    assert a["returned"] == ["r1"] and a["r1_status"] == "FINISHED_ABORTED"
    assert a["r1_finished_reason"] == "ABORT"
    assert a["free_blocks"] == 4 and a["finished_req_ids"] == ["r1"]
    assert out["cases"][1]["second_call_returned"] == []
    assert out["cases"][2]["r2_in_waiting"] is False
    assert out["cases"][3]["returned"] == ["r3"]
    assert out["cases"][4]["returned"] == ["r4"]

    dest = Path(__file__).with_name("m17_finish_requests.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for c in out["cases"]:
        print(c["case"], {k: v for k, v in c.items() if k not in ("case", "note")})


if __name__ == "__main__":
    main()
