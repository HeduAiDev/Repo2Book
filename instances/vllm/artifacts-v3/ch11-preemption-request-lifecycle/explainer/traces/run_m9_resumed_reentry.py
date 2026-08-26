"""Driver for m9 (resumed 回流协议：PREEMPTED→RUNNING · scheduled_resumed_reqs ·
resumed_req_ids → worker 侧块表『整体替换而非追加』) — host run, pin vLLM
v0.27.1 (scheduler.py:L1055-L1075 + L1447 + output.py:L115-L121).

Pool 3, r1 = 48-token prompt (3 blocks).
  beat 1: prefill, 3 blocks held, free 0; first token out (num_tokens 49).
  beat 2: decode needs the 4th block -> self-preemption, blocks freed.
  beat 3 (resume): re-hits its own 3 blocks (cap = 48) -> num_computed 48,
      1 token scheduled; r1 lands in scheduled_resumed_reqs ->
      resumed_req_ids = {r1}; its new_block_ids entry is the FULL block
      table [-1,-1,-1,3] (3 re-hit placeholders + 1 new) — the worker
      REPLACES the whole table for resumed requests (output.py:L118-L121).
  beat 4 (ordinary decode): 0 new blocks needed; r1's new_block_ids entry
      is the EMPTY list [] — the worker APPENDS (nothing) for non-resumed.
Same request, two adjacent beats: one whole-table replacement, one append.
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


def cached_view(o):
    """Scheduled cached (non-new) requests: req_id -> (resumed?, new_block_ids)."""
    cd = o.scheduled_cached_reqs
    return {
        rid: {
            "resumed": rid in cd.resumed_req_ids,
            "new_block_ids": list(nb[0]) if nb else [],
            "num_computed_tokens": nct,
        }
        for rid, nb, nct in zip(cd.req_ids, cd.new_block_ids, cd.num_computed_tokens)
    }


def main():
    out = {
        "driver": "run_m9_resumed_reentry.py",
        "mechanism": "m9 resumed 回流协议：resumed_req_ids → worker 块表整体替换（scheduler.py:L1055-L1075 + L1447 + output.py:L115-L121）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（resumed 分流与 CachedRequestData 原样保留）",
        "config": {"num_gpu_blocks": 3, "block_size": 16, "prompt": 48},
        "beats": [],
    }

    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=3, block_size=16)
    r1 = make_request("r1", 48, 0)
    sched.add_request(r1)

    def beat(label, note, tokens):
        o = step(sched, tokens)
        out["beats"].append({
            "beat": label, "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "r1_status": r1.status.name,
            "r1_num_tokens": r1.num_tokens,
            "r1_num_computed_tokens": r1.num_computed_tokens,
            "r1_held_blocks": list(sched.kv_cache_manager._blocks.get("r1", [])),
            "free_blocks": sched.kv_cache_manager.num_free_blocks,
            "cached_reqs": cached_view(o),
            "new_reqs": [d.req_id for d in o.scheduled_new_reqs],
        })

    beat("1", "prefill 48 → 恰占 3 块（空闲 0）；首 token 已出（num_tokens=49）", {"r1": [7]})
    beat("2", "decode 需第 4 块 → 池尽 → 自我被抢：块归池、computed 清零、回 waiting 队头", {})
    beat("3", "恢复拍：重命中 3 块（cap=48）+ 补 1 → PREEMPTED→RUNNING；new_block_ids=整表 4 项（3 命中占位 -1 + 1 新块）——worker 整体替换", {"r1": [8]})
    beat("4", "普通 decode：0 新块 → new_block_ids=[] 空（追加语义：什么都不追加）——同一 r1 相邻两拍，一替换一追加", {"r1": [9]})

    b3, b4 = out["beats"][2], out["beats"][3]
    assert b3["cached_reqs"]["r1"]["resumed"] is True
    assert b3["cached_reqs"]["r1"]["new_block_ids"] == [-1, -1, -1, 3]
    assert b3["r1_status"] == "RUNNING" and b3["r1_num_computed_tokens"] == 49
    assert b4["cached_reqs"]["r1"]["resumed"] is False
    assert b4["cached_reqs"]["r1"]["new_block_ids"] == []
    out["protocol_summary"] = {
        "beat3_resumed_new_block_ids": [-1, -1, -1, 3],
        "beat3_semantics": "resumed → worker 用 new_block_ids 整体替换块表（output.py:L118-L121 注释原话 'used as the request block IDs instead of appending'）",
        "beat4_running_new_block_ids": [],
        "beat4_semantics": "非 resumed → append（本拍 0 新块即什么都不追加）",
    }

    dest = Path(__file__).with_name("m9_resumed_reentry.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"], "cached",
              b["cached_reqs"], "held", b["r1_held_blocks"], "free", b["free_blocks"])


if __name__ == "__main__":
    main()
