"""Driver for m1 (迭代级调度契约：schedule() 每拍产出 {req_id: num_tokens}) — host
run against the ch10 subtract-only scheduler companion (pin vLLM v0.27.1).

Mixed-batch scenario at finger-count scale: budget 32 tokens, three 8-token
prompts admitted in beat 1, one 64-token prompt arrives before beat 2.
Five beats recorded: the batch is a {req_id: num_tokens} dictionary whose
token total (not request count) is the bounded quantity — 3 reqs / 24 tokens,
then 4 reqs / 32 tokens (3 decodes x1 + first chunk 29), 4/32 again (chunk
continues), 4/9 (chunk tail 6), 4/4 (pure decode steady state).

Also records the wire payload split (m14 evidence): for each new request the
full prompt_token_ids length vs for cached requests only the per-step diff.
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


def make_scheduler():
    config = SchedulerConfig(max_num_batched_tokens=BUDGET)
    return Scheduler(config, max_model_len=4096, num_gpu_blocks=1 << 30, block_size=16)


def main():
    out = {
        "driver": "run_m1_contract.py",
        "mechanism": "m1 迭代级调度契约：schedule() 每拍产出 {req_id: num_tokens}（interface.py:L54-L67）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {
            "max_num_batched_tokens": BUDGET,
            "max_num_scheduled_tokens_fallback": BUDGET,
            "max_num_seqs": 128,
            "max_model_len": 4096,
            "block_size": 16,
            "enable_chunked_prefill": True,
            "long_prefill_token_threshold": 0,
        },
        "requests": {
            "r1": {"prompt_len": 8, "arrives": "before beat 1"},
            "r2": {"prompt_len": 8, "arrives": "before beat 1"},
            "r3": {"prompt_len": 8, "arrives": "before beat 1"},
            "r4": {"prompt_len": 64, "arrives": "before beat 2"},
        },
        "beats": [],
    }

    sched = make_scheduler()
    reqs = {}
    for rid, plen in [("r1", 8), ("r2", 8), ("r3", 8)]:
        reqs[rid] = make_request(rid, plen)
        sched.add_request(reqs[rid])

    def beat(label, note):
        out_beat = sched.schedule()
        rec = {
            "beat": label,
            "note": note,
            "num_scheduled_tokens": dict(out_beat.num_scheduled_tokens),
            "total_num_scheduled_tokens": out_beat.total_num_scheduled_tokens,
            "requests_in_batch": len(out_beat.num_scheduled_tokens),
            "budget": BUDGET,
            "budget_left": BUDGET - out_beat.total_num_scheduled_tokens,
            "running_len": len(sched.running),
            "waiting_ids": [r.request_id for r in sched.waiting],
            "new_req_ids": [d.req_id for d in out_beat.scheduled_new_reqs],
            "conservation_assert_passed": True,  # schedule() returned => asserts held
        }
        # per-request ledger after the beat (乐观推进已发生)
        rec["ledger_after"] = {
            rid: {
                "num_tokens": r.num_tokens,
                "num_computed_tokens": r.num_computed_tokens,
                "is_prefill_chunk": r.is_prefill_chunk,
                "status": r.status.name,
            }
            for rid, r in reqs.items()
        }
        # m14 wire payload evidence: new = full, cached = diff
        rec["wire_payload"] = {
            "new_reqs": [
                {
                    "req_id": d.req_id,
                    "prompt_token_ids_len": len(d.prompt_token_ids),
                    "block_ids_len": sum(len(x) for x in d.block_ids),
                    "carries_sampling_params": d.sampling_params is not None,
                }
                for d in out_beat.scheduled_new_reqs
            ],
            "cached_reqs": {
                "req_ids": list(out_beat.scheduled_cached_reqs.req_ids),
                "num_scheduled_per_cached": [
                    out_beat.num_scheduled_tokens[i]
                    for i in out_beat.scheduled_cached_reqs.req_ids
                ],
                "new_block_ids_lens": [
                    None if nb is None else [len(x) for x in nb]
                    for nb in out_beat.scheduled_cached_reqs.new_block_ids
                ],
                "all_token_ids_keys": list(
                    out_beat.scheduled_cached_reqs.all_token_ids.keys()
                ),
                "resumed_req_ids": sorted(out_beat.scheduled_cached_reqs.resumed_req_ids),
            },
        }
        out["beats"].append(rec)
        return out_beat

    beat(1, "beat 1: running 空，三个 8-token prompt 同拍全量进批（WAITING 阶段收新）")
    # ⑤ 拍回填：为 r1-r3 各模拟采样回 1 个 token；r4 到达
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(101)
    reqs["r4"] = make_request("r4", 64)
    sched.add_request(reqs["r4"])
    beat(2, "beat 2: 三个 decode 各领 1，r4 首 chunk 被 min(token_budget)=29 截断")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(102)
    beat(3, "beat 3: r4 续 chunk（差 35 预算只余 29）")
    for rid in ("r1", "r2", "r3"):
        reqs[rid].append_output_token_ids(103)
    beat(4, "beat 4: r4 尾 chunk 6，prefill 收官")
    for rid in ("r1", "r2", "r3", "r4"):
        reqs[rid].append_output_token_ids(104)
    beat(5, "beat 5: 四个请求全 decode，各恰 1 token——连续批处理稳态")

    out["batch_shape_summary"] = [
        {"beat": b["beat"], "reqs": b["requests_in_batch"], "tokens": b["total_num_scheduled_tokens"]}
        for b in out["beats"]
    ]
    out["r4_sizes_across_life"] = [
        b["num_scheduled_tokens"].get("r4") for b in out["beats"]
    ]

    dest = Path(__file__).with_name("m1_contract.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], b["num_scheduled_tokens"], "total", b["total_num_scheduled_tokens"])


if __name__ == "__main__":
    main()
