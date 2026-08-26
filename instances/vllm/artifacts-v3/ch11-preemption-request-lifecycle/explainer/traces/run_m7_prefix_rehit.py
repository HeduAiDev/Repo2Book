"""Driver for m7 (前缀恢复：free 不清哈希 → get_computed_blocks 重命中自己的
前缀——『重算』=重载+补算；cap=num_tokens-1 全命中也重算最后一个 token 且按块
对齐) — host run, pin vLLM v0.27.1 (block_pool.py:L719-L742 + kv_cache_
manager.py:L229-L259 + scheduler.py:L744-L766). This is the F2 伏笔埋点.

Scenario A (self re-hit after preemption): pool 4, r1 = 64-token prompt
(4 blocks). Beat 1 prefill takes all 4 blocks; beat 2 decode needs the 5th
-> self-preemption frees the blocks BUT the hashes stay in the table.
Beat 3: re-admission re-hits all 4 of its own blocks (cap = 65-1 = 64 ->
4 blocks) -> num_computed=64, only 1 token scheduled. A no-prefix-cache
world would re-run all 65 tokens (the counterfactual is recorded as a field).
Scenario B (cross-request + cap effect): rA (64-token, max_tokens=1) runs
to completion and is freed — its 4 block hashes stay; rB with the SAME
64-token prompt arrives: cap = 64-1 = 63 -> only 3 blocks may hit = 48
tokens, so the ENTIRE 4th block (16 tokens) is recomputed even though its
hash is present in the table — 'full hit still recomputes the last token',
block-aligned down.
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


def make_request(req_id, prompt_len, base, max_tokens=64):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
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


def kvview(sched, rid):
    return {
        "free_blocks": sched.kv_cache_manager.num_free_blocks,
        "cached_block_hashes_size": len(sched.kv_cache_manager.cached_block_hashes),
        "held_blocks": sched.kv_cache_manager._blocks.get(rid, []),
    }


def main():
    out = {
        "driver": "run_m7_prefix_rehit.py",
        "mechanism": "m7 前缀恢复：free 不清哈希 → get_computed_blocks 重命中（F2 埋点，block_pool.py:L719-L742 + kv_cache_manager.py:L229-L259）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/：满块哈希表登记于 allocate 的 cache 提交点、free 不清、get_computed_blocks 沿 block_hashes 连续命中计数（无 LRU 驱逐——『块被逐出=全量重算』的最坏情况归 ch15，本精简版哈希不逐出）",
        "scenario_A": {
            "config": {"num_gpu_blocks": 4, "block_size": 16, "prompt": 64},
            "beats": [],
        },
        "scenario_B": {
            "config": {"num_gpu_blocks": 8, "block_size": 16,
                       "prompts": {"rA": 64, "rB": 64},
                       "note": "rA 与 rB 的 prompt token 完全相同（range(0,64)）——链式哈希逐块相同"},
            "beats": [],
        },
    }

    # ---- Scenario A: self re-hit ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=4, block_size=16)
    r1 = make_request("r1", 64, 0)
    sched.add_request(r1)

    def beatA(label, note, tokens):
        o = step(sched, tokens)
        out["scenario_A"]["beats"].append({
            "beat": label, "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "resumed_req_ids": sorted(o.scheduled_cached_reqs.resumed_req_ids),
            "r1_num_tokens": r1.num_tokens,
            "r1_num_computed_tokens": r1.num_computed_tokens,
            "r1_block_hashes_len": len(r1.block_hashes),
            "r1_status": r1.status.name,
            "r1_num_preemptions": r1.num_preemptions,
            "kv": kvview(sched, "r1"),
        })

    beatA("A-1", "prefill 64 → 恰占 4 块（空闲 0）；首 token 已出（num_tokens=65）", {"r1": [7]})
    beatA("A-2", "decode 需第 5 块 → 池尽 → 自我被抢：4 块归还池，但 cached 哈希 4 条全部留表", {})
    beatA("A-3", "下一拍恢复：重命中自己的 4 块=64 token（cap=64），只补 1 token —— resumed", {"r1": [8]})

    a2 = out["scenario_A"]["beats"][1]
    a3 = out["scenario_A"]["beats"][2]
    assert a2["preempted_req_ids"] == ["r1"]
    assert a2["kv"]["free_blocks"] == 4 and a2["kv"]["cached_block_hashes_size"] == 4
    assert a3["num_scheduled_tokens"] == {"r1": 1}
    assert a3["r1_num_computed_tokens"] == 65  # 64 命中 + 1 新算（乐观推进后）
    assert a3["resumed_req_ids"] == ["r1"]
    out["scenario_A"]["recompute_accounting"] = {
        "with_prefix_cache_tokens_recomputed": 1,
        "without_prefix_cache_tokens_recomputed": 65,
        "ratio_note": "无前缀缓存的世界里这是 65 token 的全量重算（O(prompt+output)）",
    }

    # ---- Scenario B: cross-request hit + cap ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=8, block_size=16)
    rA = make_request("rA", 64, 0, max_tokens=1)
    sched.add_request(rA)
    step(sched, {"rA": [3]})  # 首拍 1 个输出 token 即 max_tokens 封顶 → 真完成并 free
    out["scenario_B"]["beats"].append({
        "beat": "B-1", "note": "rA 跑完（max_tokens=1 长度封顶）→ 终点 free：4 块归池，哈希 4 条留表",
        "rA_status": rA.status.name,
        "rA_finished_reason": rA.get_finished_reason().name,
        "rA_in_requests": "rA" in sched.requests,
        "kv": kvview(sched, "rA"),
    })
    assert rA.get_finished_reason().name == "LENGTH"

    rB = make_request("rB", 64, 0)
    sched.add_request(rB)
    o = step(sched, {"rB": [5]})
    out["scenario_B"]["beats"].append({
        "beat": "B-2", "note": "rB 同 prompt 到达：cap=num_tokens-1=63 → 命中按块对齐向下取 3 块=48；第 4 块哈希虽在表里也不命中 → 整块 16 token 重算",
        "num_scheduled_tokens": dict(o.num_scheduled_tokens),
        "rB_num_computed_tokens": rB.num_computed_tokens,
        "rB_block_hashes_len": len(rB.block_hashes),
        "kv": kvview(sched, "rB"),
    })
    b2 = out["scenario_B"]["beats"][1]
    assert b2["num_scheduled_tokens"] == {"rB": 16}
    assert b2["rB_num_computed_tokens"] == 64  # 48 命中 + 16 重算
    assert b2["kv"]["cached_block_hashes_size"] == 4  # rB 的哈希=rA 的哈希（同内容）
    out["scenario_B"]["hit_accounting"] = {
        "hashes_in_table": 4,
        "max_hit_blocks": 3,
        "hit_tokens": 48,
        "recomputed_block": "第 4 块整块 16 token（哈希在表、被 cap 挡下）",
        "why_cap": "kv_cache_manager.py:L253-L259 NOTE：全命中也必须重算最后一个 token 才有 logits；块对齐使代价放大到整块",
    }

    dest = Path(__file__).with_name("m7_prefix_rehit.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["scenario_A"]["beats"]:
        print("A", b["beat"], "sched", b["num_scheduled_tokens"], "kv", b["kv"])
    for b in out["scenario_B"]["beats"]:
        print("B", b["beat"], "sched", b.get("num_scheduled_tokens"), "kv", b["kv"])
    print("A accounting:", out["scenario_A"]["recompute_accounting"])
    print("B accounting:", out["scenario_B"]["hit_accounting"])


if __name__ == "__main__":
    main()
