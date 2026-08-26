"""Driver for m6 (num_new_tokens==0 continue-not-break：不严格 FCFS，卡住的请求
不阻塞后面的) — host run against the ch10 subtract-only scheduler companion
(pin vLLM v0.27.1).

The driver simulates a delayed ⑤-拍 output return for r1 (the real sync
engine always appends the sampled token before the next schedule; the four
real zero-gap triggers are listed in the source comment at scheduler.py
L558-L567: PP>1 prompt-not-finished, async max-len reached, encoder budget
exhausted, mamba block alignment). With r1's gap at 0 the loop does
`req_index += 1; continue` — r2 (later in FCFS order) still gets its token
in the same beat. Scenario B: no outputs at all -> an empty schedule() that
still returns cleanly (heartbeat idles).
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


def make_scheduler():
    config = SchedulerConfig(max_num_batched_tokens=32)
    return Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)


def main():
    out = {
        "driver": "run_m6_zero_continue.py",
        "mechanism": "m6 num_new_tokens==0 → continue 而非 break（scheduler.py:L557-L573 woosuk 注）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "zero_gap_trigger_note": "驱动侧模拟 r1 的 ⑤ 拍采样回填延迟；真实同步引擎的四种零差距触发见源码注释 L558-L567（PP 未完/async 到顶/encoder 预算尽/mamba 对齐）",
        "scenario_A": {"requests": {"r1": {"prompt_len": 6}, "r2": {"prompt_len": 8}}, "beats": []},
        "scenario_B": {"requests": {"r1": {"prompt_len": 6}, "r2": {"prompt_len": 8}}, "beats": []},
    }

    # ---- scenario A: r1 stalled at gap 0, r2 keeps decoding -----------------
    sched = make_scheduler()
    r1 = make_request("r1", 6)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    sched.add_request(r2)

    def beatA(label, note, append_r1=False, append_r2=False):
        if append_r1:
            r1.append_output_token_ids(1)
        if append_r2:
            r2.append_output_token_ids(1)

        def gap(r):
            return r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens

        r1_gap_before, r2_gap_before = gap(r1), gap(r2)
        o = sched.schedule()
        out["scenario_A"]["beats"].append({
            "beat": label,
            "note": note,
            "r1": {"num_tokens": r1.num_tokens, "num_computed_tokens": r1.num_computed_tokens,
                   "gap_before_beat": r1_gap_before, "gap_after_beat": gap(r1)},
            "r2": {"num_tokens": r2.num_tokens, "num_computed_tokens": r2.num_computed_tokens,
                   "gap_before_beat": r2_gap_before, "gap_after_beat": gap(r2)},
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": o.total_num_scheduled_tokens,
        })

    beatA(1, "beat 1: r1 领 6、r2 领 8（WAITING 全量收新）")
    beatA(2, "beat 2: r1 差距 0 → continue；r2 差距 1 → 照常进批", append_r2=True)
    beatA(3, "beat 3: r1 仍卡（差距 0 → continue）；r2 继续每拍 1", append_r2=True)
    beatA(4, "beat 4: r1 的输出到了（差距 1）——两人同拍各 1", append_r1=True, append_r2=True)

    # ---- scenario B: all gaps 0 -> empty beat -------------------------------
    sched = make_scheduler()
    r1 = make_request("r1", 6)
    r2 = make_request("r2", 8)
    sched.add_request(r1)
    sched.add_request(r2)
    o = sched.schedule()  # admit both
    out["scenario_B"]["admission_beat"] = dict(o.num_scheduled_tokens)
    for i in (2, 3):
        o = sched.schedule()  # no outputs appended anywhere
        out["scenario_B"]["beats"].append({
            "beat": i,
            "r1_gap": 0, "r2_gap": 0,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": o.total_num_scheduled_tokens,
            "returned_cleanly": True,
            "note": "全员差距 0：continue 走完循环，空拍照样返回（守恒断言全过）",
        })

    dest = Path(__file__).with_name("m6_zero_continue.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["scenario_A"]["beats"]:
        print("A", b["beat"], b["num_scheduled_tokens"], "r1 gap", b["r1"]["gap_before_beat"],
              "r2 gap", b["r2"]["gap_before_beat"])
    for b in out["scenario_B"]["beats"]:
        print("B", b["beat"], b["num_scheduled_tokens"])


if __name__ == "__main__":
    main()
