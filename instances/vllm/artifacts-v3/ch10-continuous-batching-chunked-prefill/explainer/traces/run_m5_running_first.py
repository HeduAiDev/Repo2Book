"""Driver for m5 (RUNNING 先行：在途 decode 每请求 1 token 先吃预算 —— TPOT 优先
的两阶段顺序) — host run against the ch10 subtract-only scheduler companion
(pin vLLM v0.27.1).

Budget 16. Beat 1 admits three 4-token prompts. Before beat 2 a 20-token
prompt arrives. Beat 2: the RUNNING loop pays the three decodes first (3
tokens), the WAITING newcomer gets only the remaining 13 — its first chunk
is cut from 20 to 13. Beat 3 pays 3 again, the newcomer's tail chunk 7
closes its prefill. Beat 4: pure decode, all four get 1. The per-request
allocation ORDER inside the beat is recorded via an allocate_slots call
proxy (who asked for slots in which order).
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

BUDGET = 16


def make_request(req_id, prompt_len, max_tokens=16):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def main():
    out = {
        "driver": "run_m5_running_first.py",
        "mechanism": "m5 RUNNING 先行（scheduler.py:L483-L485 'First, schedule the RUNNING requests'）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {"max_num_batched_tokens": BUDGET, "max_model_len": 4096, "block_size": 16},
        "requests": {
            "r1": {"prompt_len": 4}, "r2": {"prompt_len": 4}, "r3": {"prompt_len": 4},
            "r4": {"prompt_len": 20, "arrives": "before beat 2"},
        },
        "beats": [],
    }

    config = SchedulerConfig(max_num_batched_tokens=BUDGET)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)
    reqs = {}
    for rid in ("r1", "r2", "r3"):
        reqs[rid] = make_request(rid, 4)
        sched.add_request(reqs[rid])

    calls = []
    orig_alloc = sched.kv_cache_manager.allocate_slots

    def rec_alloc(request, num_new_tokens, **kw):
        res = orig_alloc(request, num_new_tokens, **kw)
        calls.append({
            "req": request.request_id,
            "ask_tokens": num_new_tokens,
            "ok": res is not None,
            "full_sequence_must_fit": kw.get("full_sequence_must_fit", False),
        })
        return res

    sched.kv_cache_manager.allocate_slots = rec_alloc

    def beat(label, note):
        calls.clear()
        o = sched.schedule()
        total = o.total_num_scheduled_tokens
        out["beats"].append({
            "beat": label,
            "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "total": total,
            "budget": BUDGET,
            "budget_left": BUDGET - total,
            "allocation_order": list(calls),
            "decode_reqs_served": [rid for rid, n in o.num_scheduled_tokens.items() if n == 1],
            "r4_scheduled": o.num_scheduled_tokens.get("r4"),
            "r4_num_tokens": reqs["r4"].num_tokens if "r4" in reqs else None,
            "r4_computed_after": reqs["r4"].num_computed_tokens if "r4" in reqs else None,
            "r4_is_prefill_chunk": reqs["r4"].is_prefill_chunk if "r4" in reqs else None,
        })

    beat(1, "beat 1: running 空——r1/r2/r3 全量 4 各进批（WAITING 收新）")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(1)
    reqs["r4"] = make_request("r4", 20)
    sched.add_request(reqs["r4"])
    beat(2, "beat 2: RUNNING 先付 3 个 decode 各 1，WAITING 的 r4 只领到剩余 13——首 chunk 20 被截成 13")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(1)
    beat(3, "beat 3: 又先付 3 个 decode，r4 领尾 chunk 7，prefill 收官")
    for rid in ("r1", "r2", "r3", "r4"):
        reqs[rid].append_output_token_ids(1)
    beat(4, "beat 4: r4 转 decode——四人各恰 1")
    for rid in ("r1", "r2", "r3", "r4"):
        reqs[rid].append_output_token_ids(1)
    beat(5, "beat 5: 纯 decode 稳态")

    b2 = out["beats"][1]
    out["r4_first_chunk_cut"] = {"raw_need": 20, "paid_to_decodes": 3, "chunk": 13,
                                 "cut_fraction_pct": 65, "cut_fraction": 0.65}
    out["r4_chunk_sizes"] = [b["r4_scheduled"] for b in out["beats"]]

    dest = Path(__file__).with_name("m5_running_first.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], b["num_scheduled_tokens"], "order:", [(c["req"], c["ask_tokens"]) for c in b["allocation_order"]])


if __name__ == "__main__":
    main()
