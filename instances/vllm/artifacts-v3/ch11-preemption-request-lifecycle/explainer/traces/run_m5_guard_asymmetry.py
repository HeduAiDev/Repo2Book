"""Driver for m5 (抢占拍不收新守卫的 why：preempted_reqs 非空 = 内存紧张信号 →
整拍关闸；以及 RUNNING 抢 / WAITING 只 break 的不对称) — host run, pin
vLLM v0.27.1 (scheduler.py:L683-L692 + L987-L994).

Scenario A (the guard is load-bearing): pool 8 blocks. r1 = 16-token (1
block), r2 = 112-token (7 blocks, the tail). Beat 1 admits both (free 0).
Beat 2: r1's decode needs its 2nd block -> None -> preempt the tail r2 (7
blocks return, free 7) -> r1 retry OK (free 6). The guard then skips the
WAITING stage for the REST of this beat even though 6 free blocks would be
enough for r2's resume (it re-hits 112 and needs only 1 new block) —
re-admitting now would re-create the same contention (thrash). Beat 3: guard
open (preempted set was swapped fresh), r2 resumes with a 112-token prefix
re-hit + 1 new token.
Scenario B (asymmetry): pool 2. r1 = 32-token (2 blocks) admitted at beat 1
while a 16-token `victim` waiting request is refused by the full-ISL gate
(1 block needed, 0 free) -> WAITING-side None only BREAKS: nobody in running
is preempted by a waiting request. Beat 2: r1's own decode growth exhausts
the pool -> r1 preempts ITSELF (the RUNNING-side signal does preempt).
`victim` keeps num_preemptions=0 through it all.
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


def main():
    out = {
        "driver": "run_m5_guard_asymmetry.py",
        "mechanism": "m5 抢占拍不收新守卫 + RUNNING 抢 / WAITING 只 break 的不对称（scheduler.py:L683-L692 + L987-L994）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/ 只做减法精简版（守卫 `if not preempted_reqs and UNPAUSED` 原样保留）",
        "scenario_A": {
            "config": {"num_gpu_blocks": 8, "block_size": 16,
                       "prompts": {"r1": 16, "r2": 112},
                       "note": "守卫 load-bearing 场景：抢占后池里 6 空闲块足够 r2 恢复（重命中 112 只需 1 新块），但守卫整拍关闸"},
            "beats": [],
        },
        "scenario_B": {
            "config": {"num_gpu_blocks": 2, "block_size": 16,
                       "prompts": {"r1": 32, "victim": 16},
                       "note": "不对称场景：WAITING 侧 None 只 break（victim 毫发无损）；RUNNING 侧 None 才触发抢占（r1 自我牺牲）"},
            "beats": [],
        },
    }

    # ---- Scenario A ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=8, block_size=16)
    r1 = make_request("r1", 16, 0)
    r2 = make_request("r2", 112, 100)
    sched.add_request(r1)
    sched.add_request(r2)

    def beatA(label, note, tokens):
        free_before = sched.kv_cache_manager.num_free_blocks
        o = step(sched, tokens)
        out["scenario_A"]["beats"].append({
            "beat": label, "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "guard_open_next_beat_note": "守卫只看本拍局部变量 preempted_reqs",
            "r1_status": r1.status.name, "r2_status": r2.status.name,
            "r2_num_preemptions": r2.num_preemptions,
            "r2_num_computed_tokens": r2.num_computed_tokens,
            "r2_held_blocks": len(sched.kv_cache_manager._blocks.get("r2", [])),
            "resumed_req_ids": sorted(o.scheduled_cached_reqs.resumed_req_ids),
        })

    beatA("A-1", "首拍双准入：r1(1 块)+r2(7 块)，空闲 0", {"r1": [1], "r2": [2]})
    beatA("A-2", "r1 decode 需第 2 块 → 抢队尾 r2（7 块归池，空闲 7→6）→ r1 重试成功；守卫关闸：r2 虽只差 1 新块可恢复，本拍不收", {"r1": [3]})
    beatA("A-3", "下一拍守卫开：r2 重命中自己的 112 token 前缀 + 补 1 —— resumed 恢复（resumed_req_ids 含 r2）", {"r1": [4], "r2": [5]})

    a2 = out["scenario_A"]["beats"][1]
    a3 = out["scenario_A"]["beats"][2]
    assert a2["preempted_req_ids"] == ["r2"] and "r2" not in a2["num_scheduled_tokens"]
    assert a2["free_blocks_after"] == 6
    assert a3["resumed_req_ids"] == ["r2"] and a3["num_scheduled_tokens"] == {"r1": 1, "r2": 1}
    assert a3["r2_num_computed_tokens"] == 113  # 112 命中 + 1 新算（乐观推进后）

    # ---- Scenario B ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=2, block_size=16)
    r1 = make_request("r1", 32, 0)
    victim = make_request("victim", 16, 300)
    sched.add_request(r1)
    sched.add_request(victim)

    def beatB(label, note, tokens):
        free_before = sched.kv_cache_manager.num_free_blocks
        o = step(sched, tokens)
        out["scenario_B"]["beats"].append({
            "beat": label, "note": note,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "preempted_req_ids": sorted(o.preempted_req_ids),
            "r1_status": r1.status.name,
            "victim_status": victim.status.name,
            "r1_num_preemptions": r1.num_preemptions,
            "victim_num_preemptions": victim.num_preemptions,
            "victim_in_waiting": victim in sched.waiting,
        })

    beatB("B-1", "r1 准入占 2 块；victim 整序列需 1 块 > 空闲 0 → WAITING 侧 None → break：在场请求无人被抢", {"r1": [6]})
    beatB("B-2", "r1 decode 需第 3 块 → 池尽 → RUNNING 侧信号触发抢占：无他人可抢 → r1 自我被抢；victim 仍 WAITING、0 次被抢", {})

    b1, b2 = out["scenario_B"]["beats"]
    assert b1["num_scheduled_tokens"] == {"r1": 32}
    assert b1["victim_status"] == "WAITING" and b1["victim_num_preemptions"] == 0
    assert b2["preempted_req_ids"] == ["r1"] and b2["num_scheduled_tokens"] == {}
    assert b2["victim_status"] == "WAITING" and b2["victim_num_preemptions"] == 0

    dest = Path(__file__).with_name("m5_guard_asymmetry.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for sc in ("scenario_A", "scenario_B"):
        for b in out[sc]["beats"]:
            print(sc, b["beat"], "sched", b["num_scheduled_tokens"],
                  "preempted", b["preempted_req_ids"],
                  "free", b["free_blocks_before"], "->", b["free_blocks_after"])


if __name__ == "__main__":
    main()
