"""Driver for m7 (RUNNING 侧 allocate_slots 抢占重试环：None→抢 FCFS 队尾→重试；
本拍抢占集合成为阶段二守卫条件) — host run against the ch10 subtract-only
scheduler companion (pin vLLM v0.27.1).

Pool = 2 blocks x 16 tokens = 32 token-slots, full-ISL admission on.
Beat 1 admits r1 and r2 (16-token prompts, one block each, pool empty).
Beat 2 both decode one token: r1's allocate_slots(1) needs a 2nd block,
pool is empty -> None -> preempt the FCFS tail (r2, the youngest) -> retry
succeeds on r2's freed block. The whole retry loop is recorded with an
allocate_slots call proxy: [(r1,1)->None, (r1,1)->OK].
Beats 3-4: r2 (PREEMPTED, num_computed_tokens=0, waiting head) is re-attempted
each beat but the full-ISL gate refuses (needs 2 blocks > 0 free) — the
WAITING side never preempts. Beat 5: r1 manually retired (ch9/ch11 scope:
running.remove + kv free), r2 re-admitted as a RESUMED request, re-running
all 17 tokens (16 prompt + its 1 previous output now part of the input) —
recompute-only preemption, v1 has no swap.
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


def make_request(req_id, prompt_len, max_tokens=6):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


def main():
    out = {
        "driver": "run_m7_preempt_retry.py",
        "mechanism": "m7 RUNNING 侧 allocate_slots 抢占重试环（scheduler.py:L576-L629）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch10 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {
            "num_gpu_blocks": 2, "block_size": 16, "pool_token_capacity": 32,
            "max_num_batched_tokens": 2048, "max_model_len": 4096,
            "scheduler_reserve_full_isl": True,
        },
        "requests": {"r1": {"prompt_len": 16}, "r2": {"prompt_len": 16}},
        "beats": [],
    }

    config = SchedulerConfig(max_num_batched_tokens=2048)
    sched = Scheduler(config, max_model_len=4096, num_gpu_blocks=2, block_size=16)
    r1 = make_request("r1", 16)
    r2 = make_request("r2", 16)
    sched.add_request(r1)
    sched.add_request(r2)

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

    def held_blocks(r):
        return len(sched.kv_cache_manager._blocks.get(r.request_id, []))

    def beat(label, note):
        calls.clear()
        free_before = sched.kv_cache_manager.num_free_blocks
        o = sched.schedule()
        out["beats"].append({
            "beat": label,
            "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "allocate_calls": list(calls),
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "waiting_order": [r.request_id for r in sched.waiting],
            "running_order": [r.request_id for r in sched.running],
            "r1": {"num_tokens": r1.num_tokens, "computed": r1.num_computed_tokens,
                   "held_blocks": held_blocks(r1), "num_preemptions": r1.num_preemptions,
                   "status": r1.status.name},
            "r2": {"num_tokens": r2.num_tokens, "computed": r2.num_computed_tokens,
                   "held_blocks": held_blocks(r2), "num_preemptions": r2.num_preemptions,
                   "status": r2.status.name},
            "resumed_req_ids": sorted(o.scheduled_cached_reqs.resumed_req_ids),
            "new_req_ids": [d.req_id for d in o.scheduled_new_reqs],
        })

    beat(1, "beat 1: r1/r2 全量 16 各进批，各持 1 块，池空")
    r1.append_output_token_ids(1)
    r2.append_output_token_ids(1)
    beat(2, "beat 2: r1 差 1 但需第 2 块 → None → 抢 FCFS 队尾 r2 → 重试 OK；本拍抢占过 → 阶段二关闸")
    r1.append_output_token_ids(1)
    beat(3, "beat 3: r1 块内增长（18≤32）无需新块；r2 重入被整序列门拒（需 2 块 > 空闲 0）——WAITING 绝不抢占")
    r1.append_output_token_ids(1)
    beat(4, "beat 4: 同拍 3——r2 继续等（重算需求 2 块一直得不到）")
    # 模拟 r1 完成（⑤ 拍生命周期归 ch9/ch11：手工 running.remove + 块归还）
    sched.running.remove(r1)
    sched.kv_cache_manager.free(r1)
    sched.requests.pop("r1", None)
    sched.finished_req_ids.add("r1")
    out["manual_retire_note"] = "r1 于拍 4 后手工退场（running.remove + kv free + finished_req_ids 登记——模拟 ch9/ch11 的 ⑤ 拍完成路径）"
    beat(5, "beat 5: r1 退场块回池（空闲 2）→ r2 以 resumed 重入，整段重算 17 token（16 prompt + 1 旧输出成了输入）")

    dest = Path(__file__).with_name("m7_preempt_retry.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"], "preempted", b["preempted_req_ids"],
              "calls", [(c["req"], c["ask_tokens"], c["ok"]) for c in b["allocate_calls"]])


if __name__ == "__main__":
    main()
