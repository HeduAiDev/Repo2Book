"""Driver for m16 (下一拍续 chunk / decode 闭环：is_prefill_chunk 请求继续在
RUNNING 阶段被切；decode 请求差恒 1 —— 连续批处理的稳态) — host run against
the ch10 subtract-only scheduler companion (pin vLLM v0.27.1).

The chapter's centerpiece steady state, budget 32: r1/r2 (16-token prompts)
decode from beat 2 on; r3 (64-token prompt) is admitted in beat 2 and burns
down as chunks 30 -> 30 -> 4, then joins the decode club. Batch composition
per beat: {r1:1, r2:1, r3:30} = 32, again 32, then {1,1,4} = 6, then the
pure-decode steady state {1,1,1} = 3, forever. Every row records each
request's formula components (num_tokens / computed / gap) so the "one
formula, three shapes" closure is checkable by hand.
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

BUDGET = 32


def make_request(req_id, prompt_len, max_tokens=16):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def main():
    out = {
        "driver": "run_m16_steady_state.py",
        "mechanism": "m16 续 chunk / decode 闭环——连续批处理稳态（scheduler.py:L516-L520 + L1335-L1337）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {
            "max_num_batched_tokens": BUDGET, "max_model_len": 4096, "block_size": 16,
            "enable_chunked_prefill": True, "long_prefill_token_threshold": 0,
        },
        "requests": {
            "r1": {"prompt_len": 16}, "r2": {"prompt_len": 16},
            "r3": {"prompt_len": 64, "arrives": "before beat 2"},
        },
        "beats": [],
    }

    config = SchedulerConfig(max_num_batched_tokens=BUDGET)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)
    reqs = {"r1": make_request("r1", 16), "r2": make_request("r2", 16)}
    for r in reqs.values():
        sched.add_request(r)

    def rid_ledger(rid):
        r = reqs[rid]
        return {
            "num_tokens": r.num_tokens,
            "computed": r.num_computed_tokens,
            "gap": r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens,
            "is_prefill_chunk": r.is_prefill_chunk,
        }

    def beat(label, note):
        o = sched.schedule()
        out["beats"].append({
            "beat": label,
            "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": o.total_num_scheduled_tokens,
            "budget": BUDGET,
            "budget_left": BUDGET - o.total_num_scheduled_tokens,
            "ledger_after": {rid: rid_ledger(rid) for rid in ("r1", "r2", "r3") if rid in reqs},
        })

    beat(1, "beat 1: r1/r2 全量 16 各进批（预算恰用完 32），r3 还在 waiting")
    for rid in ("r1", "r2"):
        reqs[rid].append_output_token_ids(1)
    reqs["r3"] = make_request("r3", 64)
    sched.add_request(reqs["r3"])
    beat(2, "beat 2: r1/r2 各 decode 1（RUNNING 先吃 2），r3 首 chunk 30 进批——混合批 2+30=32")
    for rid in ("r1", "r2"):
        reqs[rid].append_output_token_ids(1)
    beat(3, "beat 3: r3 续 chunk（差 34 → 截 30）——批又是 2+30=32")
    for rid in ("r1", "r2"):
        reqs[rid].append_output_token_ids(1)
    beat(4, "beat 4: r3 尾 chunk 4 收官（64/64），is_prefill_chunk 翻 False——批 2+4=6")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(1)
    beat(5, "beat 5: 三人全 decode 各 1——纯 decode 稳态 3 token")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(1)
    beat(6, "beat 6: 稳态不变——差恒 1 的闭环")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(1)
    beat(7, "beat 7: 稳态不变")

    out["batch_totals"] = [b["total"] for b in out["beats"]]
    out["r3_chunk_sizes"] = [b["num_scheduled_tokens"].get("r3") for b in out["beats"]]

    dest = Path(__file__).with_name("m16_steady_state.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], b["num_scheduled_tokens"], "total", b["total"])


if __name__ == "__main__":
    main()
