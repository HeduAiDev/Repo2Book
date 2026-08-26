"""Driver for m6 (waiting/skipped_waiting 双队列防队头阻塞：三阻塞态分流 ·
_try_promote 失败跳过 · step_skipped_waiting 收集与步末重排 · stale 在途者
推迟) — host run, pin vLLM v0.27.1 (scheduler.py:L687-L722 + L2050-L2074 +
L1099-L1101).

Cast (pool effectively unbounded):
  older : status WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR (blocked) — sits in
          skipped_waiting. In the精简版 _try_promote always returns False
          (its three promote branches belong to deleted subsystems), which is
          exactly the 'still waiting' semantics.
  newer : status PREEMPTED with num_stale_output_tokens=1 (async sim) —
          routed to the plain waiting queue (PREEMPTED is not blocked).
  ready : a normal WAITING request.
Beat 1: FCFS picks skipped_waiting first -> peek `older`: blocked & promote
fails -> pop into step_skipped_waiting; then waiting head `newer`: stale>0
-> deferred into step_skipped_waiting too; then `ready` is admitted. At step
end step_skipped_waiting=[newer, older] (prepend collects, later-skipped at
front) is prepended back via extendleft, which reverses again -> final
skipped_waiting order [older, newer] = this beat's skip order -> next beat
retries `older` first. No one is dropped, no one starves.
Beat 2: `newer`'s stale is drained (async: by update_from_output; here set to
0 to emulate) -> `older` still blocked -> skipped again; `newer` admitted as
a RESUMED request (PREEMPTED -> scheduled_resumed_reqs).
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
        "driver": "run_m6_dual_queue.py",
        "mechanism": "m6 双队列防队头阻塞（scheduler.py:L687-L722 + L2050-L2074 + L1099-L1101）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版（_try_promote 坍缩 return False=『仍在等』；三阻塞态清单原样保留）",
        "cast": {
            "older": "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR（阻塞态，路由进 skipped_waiting）",
            "newer": "PREEMPTED + num_stale_output_tokens=1（async 模拟：在途未排干 → 推迟恢复）",
            "ready": "普通 WAITING（可立即调度）",
        },
        "beats": [],
    }

    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=64, block_size=16)
    older = make_request("older", 16, 0)
    older.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
    newer = make_request("newer", 16, 100)
    newer.status = RequestStatus.PREEMPTED
    newer.num_stale_output_tokens = 1  # async 模拟：被抢时在途 1 份未排干
    ready = make_request("ready", 16, 200)
    sched._enqueue_waiting_request(older)  # → skipped_waiting
    sched._enqueue_waiting_request(newer)  # PREEMPTED 非阻塞 → waiting
    sched.add_request(ready)               # → waiting（同时登记 requests 账本）
    sched.requests["older"] = older        # 手工入队的两位也须登记（add_request 的另一半）
    sched.requests["newer"] = newer

    out["initial_queues"] = {
        "skipped_waiting": [r.request_id for r in sched.skipped_waiting],
        "waiting": [r.request_id for r in sched.waiting],
        "note": "单队列世界里 older 卡在队头会让 newer/ready 全体饿死；双队列把它隔离",
    }

    def beat(label, note, tokens):
        o = step(sched, tokens)
        out["beats"].append({
            "beat": label, "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "skipped_waiting_after": [r.request_id for r in sched.skipped_waiting],
            "waiting_after": [r.request_id for r in sched.waiting],
            "resumed_req_ids": sorted(o.scheduled_cached_reqs.resumed_req_ids),
            "older_status": older.status.name,
            "newer_status": newer.status.name,
        })

    beat("1", "skipped 优先 → older 阻塞跳过；waiting 队头 newer stale>0 推迟；ready 正常准入——三个都没丢", {"ready": [1]})
    assert out["beats"][0]["skipped_waiting_after"] == ["older", "newer"]
    assert out["beats"][0]["num_scheduled_tokens"] == {"ready": 16}

    newer.num_stale_output_tokens = 0  # async 下由 update_from_output 每拍冲销；此处置 0 模拟已排干
    beat("2", "stale 已排干：older 仍阻塞再跳过（每拍恰 peek 一次）；newer 以 resumed 恢复", {"ready": [2], "newer": [9]})
    assert out["beats"][1]["skipped_waiting_after"] == ["older"]
    assert out["beats"][1]["resumed_req_ids"] == ["newer"]
    assert "newer" in out["beats"][1]["num_scheduled_tokens"]

    dest = Path(__file__).with_name("m6_dual_queue.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    print("initial:", out["initial_queues"]["skipped_waiting"], out["initial_queues"]["waiting"])
    for b in out["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"],
              "skipped", b["skipped_waiting_after"], "waiting", b["waiting_after"],
              "resumed", b["resumed_req_ids"])


if __name__ == "__main__":
    main()
