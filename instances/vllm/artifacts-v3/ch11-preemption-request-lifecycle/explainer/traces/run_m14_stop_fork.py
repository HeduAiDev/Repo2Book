"""Driver for m14 (停止分流与 finish_reason 时序：finish_reason 必须在
_handle_stopped_request 之前抓取；status_before_stop 分流 stopped_running /
stopped_preempted；remove_all 批量摘除) — host run, pin vLLM v0.27.1
(scheduler.py:L1895-L1907 + L1946-L1951).

Scenario A (normal stop from RUNNING): r1 (eos=99) stops at beat 2 —
finish_reason captured BEFORE _handle_stopped_request; the handler returns
True (non-streaming) -> _free_request; status_before_stop=RUNNING ->
removed from running via remove_all; next beat's SchedulerOutput carries
finished_req_ids={r1} to tell the worker to clear cached state.
Scenario B (rare: stopped while PREEMPTED — the async overlap): rB's output
is still in flight when it is preempted; the in-flight token turns out to be
the EOS. The request is in the waiting queue at update time, yet the output
still carries finish_reason (captured before handle), status_before_stop=
PREEMPTED -> stopped_preempted branch -> removed from waiting (and
skipped_waiting) + freed.
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


def make_request(req_id, prompt_len, base, eos):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=64, eos_token_id=eos),
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
    emitted = {o.request_id: {"new_token_ids": list(o.new_token_ids),
                              "finish_reason": o.finish_reason.name if o.finish_reason is not None else None}
               for eco in outputs.values() for o in eco.outputs}
    return out, emitted


def main():
    out = {
        "driver": "run_m14_stop_fork.py",
        "mechanism": "m14 停止分流与 finish_reason 时序（scheduler.py:L1895-L1907 + L1946-L1951）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（先抓 reason 再 handle + 双分流 + remove_all 摘除原样保留；_handle_stopped_request 坍缩 return True——流式续段见 m15 manual）",
        "scenario_A": {"config": {"num_gpu_blocks": 4, "eos": 99}, "beats": []},
        "scenario_B": {"config": {"num_gpu_blocks": 4, "eos": 42},
                       "note": "async 重叠模拟：本拍已调度、输出回来前被抢 → 在途 token 恰是 EOS"},
    }

    # ---- Scenario A ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r1 = make_request("r1", 16, 0, eos=99)
    sched.add_request(r1)

    def beatA(label, note, tokens):
        o, emitted = step(sched, tokens)
        out["scenario_A"]["beats"].append({
            "beat": label, "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "emitted": emitted,
            "r1_status": r1.status.name,
            "r1_in_requests": "r1" in sched.requests,
            "r1_in_running": r1 in sched.running,
            "finished_req_ids_next_output": sorted(o.finished_req_ids),
        })

    beatA("A-1", "prefill 首 token=1，未停", {"r1": [1]})
    beatA("A-2", "decode 命中 EOS=99：先抓 finish_reason=STOP 再 handle（True→free）→ stopped_running 摘除 + del requests", {"r1": [99]})
    beatA("A-3", "下一拍 SchedulerOutput.finished_req_ids={r1} 通告 worker 清缓存", {})

    a2, a3 = out["scenario_A"]["beats"][1], out["scenario_A"]["beats"][2]
    assert a2["emitted"]["r1"]["finish_reason"] == "STOP"
    assert a2["r1_in_requests"] is False and a2["r1_in_running"] is False
    assert a3["finished_req_ids_next_output"] == ["r1"]

    # ---- Scenario B ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    rB = make_request("rB", 16, 100, eos=42)
    sched.add_request(rB)
    step(sched, {"rB": [1]})  # prefill
    outB = sched.schedule()   # rB decode 已调度（在途 1）
    sched.running.remove(rB)
    sched._preempt_request(rB, time.monotonic())  # 输出回来前被抢
    assert rB in sched.waiting
    mro = ModelRunnerOutput(req_ids=["rB"], req_id_to_index={"rB": 0},
                            sampled_token_ids=[[42]])
    outputs = sched.update_from_output(outB, mro)
    emitted = {o.request_id: {"new_token_ids": list(o.new_token_ids),
                              "finish_reason": o.finish_reason.name if o.finish_reason is not None else None}
               for eco in outputs.values() for o in eco.outputs}
    out["scenario_B"]["beats"] = [{
        "beat": "B-1",
        "note": "被抢当拍完成：status_before_stop=PREEMPTED → stopped_preempted 分支——从 waiting/skipped 双队列摘除 + 终点 free；输出仍带 finish_reason=STOP（先抓再 handle）",
        "emitted": emitted,
        "rB_status": rB.status.name,
        "rB_in_waiting": rB in sched.waiting,
        "rB_in_skipped": rB in sched.skipped_waiting,
        "rB_in_requests": "rB" in sched.requests,
        "free_blocks": sched.kv_cache_manager.num_free_blocks,
    }]
    b1 = out["scenario_B"]["beats"][0]
    assert b1["emitted"]["rB"]["finish_reason"] == "STOP"
    assert b1["rB_status"] == "FINISHED_STOPPED"
    assert b1["rB_in_waiting"] is False and b1["rB_in_requests"] is False
    out["timing_note"] = ("finish_reason 必须在 _handle_stopped_request 之前抓取"
                          "（真实源码 L1897-L1899 注释：流式会话的 handle 可能把 status 改回 WAITING）；"
                          "精简版 handle 恒 True，时序约束的『为什么』见 m15 manual 素材")

    dest = Path(__file__).with_name("m14_stop_fork.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["scenario_A"]["beats"]:
        print("A", b["beat"], b["emitted"], "in_requests", b["r1_in_requests"],
              "fin_ids", b["finished_req_ids_next_output"])
    print("B", b1["emitted"], b1["rB_status"], "in_waiting", b1["rB_in_waiting"])


if __name__ == "__main__":
    main()
