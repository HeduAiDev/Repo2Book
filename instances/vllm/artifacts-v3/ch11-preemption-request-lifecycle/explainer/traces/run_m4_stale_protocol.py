"""Driver for m4 (stale 在途输出协议：抢占时 num_stale_output_tokens ←
num_in_flight_tokens（assign 不累加）；update_from_output 每拍按本拍调度数
锁步 drain、stale 仍送达；排空前恢复被推迟；drop-mode 整段丢弃；同步版自
中和) — host run, pin vLLM v0.27.1 (request.py:L150-L162, scheduler.py:
L1297-L1308 / L1737-L1743 / L713-L722 / L1757-L1759).

Phases:
  P1 assign (async sim): after a schedule() that leaves r1 with in_flight=1,
  the driver sets in_flight=2 (simulating a second in-flight step of an async
  batch queue), removes r1 from running and calls _preempt_request ->
  stale=2, placeholders=0 (assign, not accumulate).
  P2 drain#1 + deliver: update_from_output returns r1's in-flight token 42 —
  stale output IS delivered — and drains stale 2->1, in_flight 2->1.
  P3 defer: next schedule() defers r1's resume (stale>0, not drop): r1 lands
  in skipped_waiting, nothing scheduled for it.
  P4 drain#2 (second in-flight step arrives): a second update_from_output on
  the same scheduler_output (emulating the 2nd in-flight step of the async
  pipeline returning) drains stale 1->0 and delivers 43.
  P5 resume: next schedule() admits r1 as resumed (prefix re-hit 16 + 3 new).
  P6 drop-mode: a fresh request preempted with drop_stale_output=True — its
  in-flight output is discarded entirely (not delivered, not appended).
  P7 sync self-neutralize: a natural synchronous preemption happens AFTER the
  previous beat's output has been accounted -> in_flight=0 -> stale=0.
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
    outputs = sched.update_from_output(out, mro)
    emitted = {o.request_id: list(o.new_token_ids)
               for eco in outputs.values() for o in eco.outputs}
    return out, emitted


def rstate(r):
    return {"num_tokens": r.num_tokens,
            "num_in_flight_tokens": r.num_in_flight_tokens,
            "num_stale_output_tokens": r.num_stale_output_tokens,
            "drop_stale_output": r.drop_stale_output,
            "num_output_placeholders": r.num_output_placeholders,
            "status": r.status.name,
            "output_token_ids": list(r.output_token_ids)}


def main():
    out = {
        "driver": "run_m4_stale_protocol.py",
        "mechanism": "m4 stale 在途输出协议（request.py:L150-L162 + scheduler.py:L1297-L1308/L1737-L1743/L713-L722/L1757-L1759）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版（stale 协议全保留）",
        "phases": [],
    }

    def phase(label, note, record):
        out["phases"].append({"phase": label, "note": note, **record})

    # ---------- P0: setup — r1 prefill + one decode ----------
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r1 = make_request("r1", 16, 0)
    sched.add_request(r1)
    step(sched, {"r1": [1]})  # prefill：1 块 + 首 token
    phase("P0", "r1 prefill 完成（16 token 1 块）+ 首 token 已出，in_flight 已回账归零",
          {"r1": rstate(r1)})

    # ---------- P1: schedule then preempt with async-simulated depth 2 ----------
    out2 = sched.schedule()  # r1 decode 1 调度（乐观推进 in_flight 0→1）
    r1.num_in_flight_tokens = 2  # 模拟 async 批队列深度 2（第 2 个在途步）；同步真实引擎恒 1
    sched.running.remove(r1)
    sched._preempt_request(r1, time.monotonic())
    phase("P1", "调度后输出回来前被抢（async 模拟：in_flight 置 2）→ stale ← in_flight（assign）",
          {"r1": rstate(r1),
           "waiting_order": [r.request_id for r in sched.waiting]})

    # ---------- P2: first in-flight output arrives — delivered + drain#1 ----------
    mro = ModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0},
                            sampled_token_ids=[[42]])
    outputs = sched.update_from_output(out2, mro)
    emitted = {o.request_id: list(o.new_token_ids)
               for eco in outputs.values() for o in eco.outputs}
    phase("P2", "第 1 个在途输出到达：42 仍送达（不丢弃）+ stale 2→1、in_flight 2→1 锁步冲销",
          {"r1": rstate(r1), "emitted": emitted})

    # ---------- P3: resume deferred while stale > 0 ----------
    o3, emitted3 = step(sched, {})
    phase("P3", "下一拍 schedule：r1 在 waiting 头但 stale=1>0 → 推迟恢复（落 skipped_waiting，本拍不调度它）",
          {"r1": rstate(r1), "num_scheduled_tokens": dict(o3.num_scheduled_tokens),
           "skipped_waiting": [r.request_id for r in sched.skipped_waiting],
           "waiting": [r.request_id for r in sched.waiting],
           "emitted": emitted3})

    # ---------- P4: second in-flight output arrives — delivered + drain#2 ----------
    mro = ModelRunnerOutput(req_ids=["r1"], req_id_to_index={"r1": 0},
                            sampled_token_ids=[[43]])
    outputs = sched.update_from_output(out2, mro)  # 同一 out 二次回账=模拟第 2 个在途步返回
    emitted = {o.request_id: list(o.new_token_ids)
               for eco in outputs.values() for o in eco.outputs}
    phase("P4", "第 2 个在途输出到达：43 送达 + stale 1→0、in_flight 1→0——排空",
          {"r1": rstate(r1), "emitted": emitted})

    # ---------- P5: resume now possible ----------
    o5, emitted5 = step(sched, {"r1": [44]})
    phase("P5", "stale 排空后下一拍：r1 以 resumed 恢复（重命中自己的 16 token 前缀 + 补 3）",
          {"r1": rstate(r1), "num_scheduled_tokens": dict(o5.num_scheduled_tokens),
           "resumed_req_ids": sorted(o5.scheduled_cached_reqs.resumed_req_ids),
           "emitted": emitted5})

    # ---------- P6: drop-mode ----------
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r2 = make_request("r2", 16, 100)
    sched.add_request(r2)
    step(sched, {"r2": [1]})
    outd = sched.schedule()
    sched.running.remove(r2)
    sched._preempt_request(r2, time.monotonic(), drop_stale_output=True)
    mro = ModelRunnerOutput(req_ids=["r2"], req_id_to_index={"r2": 0},
                            sampled_token_ids=[[42]])
    outputs = sched.update_from_output(outd, mro)
    emitted = [o for eco in outputs.values() for o in eco.outputs]
    phase("P6", "drop-mode（drop_stale_output=True）：在途输出整段丢弃——不外送也不入账",
          {"r2": rstate(r2), "emitted_count": len(emitted)})

    # ---------- P7: sync self-neutralize ----------
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=1, block_size=16)
    r3 = make_request("r3", 16, 200)
    sched.add_request(r3)
    step(sched, {"r3": [7]})  # prefill：in_flight 已被 update 归零
    step(sched, {})  # decode 需第 2 块 → 自我被抢：stale = in_flight = 0
    phase("P7", "同步版自中和：抢占发生在上拍输出已回账之后 → in_flight=0 → stale=0（协议只在 async/PP 咬合）",
          {"r3": rstate(r3)})

    # assertions
    assert out["phases"][1]["r1"]["num_stale_output_tokens"] == 2
    assert out["phases"][2]["emitted"] == {"r1": [42]}
    assert out["phases"][3]["skipped_waiting"] == ["r1"]
    assert out["phases"][4]["r1"]["num_stale_output_tokens"] == 0
    assert out["phases"][5]["resumed_req_ids"] == ["r1"]
    assert out["phases"][6]["emitted_count"] == 0
    assert 42 not in out["phases"][6]["r2"]["output_token_ids"]
    assert out["phases"][7]["r3"]["num_stale_output_tokens"] == 0

    dest = Path(__file__).with_name("m4_stale_protocol.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for p in out["phases"]:
        r = p.get("r1") or p.get("r2") or p.get("r3")
        print(p["phase"], "in_flight", r["num_in_flight_tokens"],
              "stale", r["num_stale_output_tokens"],
              "drop", r["drop_stale_output"], "status", r["status"])


if __name__ == "__main__":
    main()
