"""Driver for m11 (decode 稳态长块：每 block_size 个 token 多要一块；池尽
None → 抢占环) — host run against the ch13 companion.

两条 30-token 请求（各占 2 块、块内余量 2）在 pool 8 块（可用 7）上长大：
  token 31/32 块内免账 → 33 越界才要第 3 块 → 49 才要第 4 块……
  节奏表 + 池干后的抢占环（FCFS 弹队尾最新者、块全还、computed 归零——
  外部行为 ch11 的块侧内景）。
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402

BLOCK_SIZE = 16
NUM_BLOCKS = 8
LAYER = "model.layers.0.self_attn.attn"


def make_scheduler() -> Scheduler:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(size=NUM_BLOCKS * spec.page_size_bytes, shared_by=[LAYER])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)],
    )
    return Scheduler(
        kv_cache_config=config,
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def main():
    sched = make_scheduler()
    beats = []

    def held(req_id):
        single = sched.kv_cache_manager.coordinator.single_type_managers[0]
        return len(single.req_to_blocks.get(req_id, ()))

    def beat(label, req, num_new, note):
        before_held = held(req.request_id)
        res = sched.allocate_slots_for_running(req, num_new)
        if res is None:
            got = None
        else:
            got = res.get_block_ids()[0]
        beats.append({
            "beat": label,
            "request": req.request_id,
            "total_tokens": req.num_computed_tokens + num_new,
            "cdiv_by_16": -(-(req.num_computed_tokens + num_new) // BLOCK_SIZE),
            "held_before": before_held,
            "new_blocks": got,
            "free_after": sched.kv_cache_manager.block_pool.get_num_free_blocks(),
            "note": note,
        })
        return res

    # 入场（WAITING 侧）：各 30 token → 各 2 块（32 槽、余 2）
    r1 = make_request("r1", 30)
    r2 = make_request("r2", 30)
    a1 = sched.allocate_slots_for_waiting(r1, 30, 0, None)
    a2 = sched.allocate_slots_for_waiting(r2, 30, 0, None)
    for r in (r1, r2):
        r.status = RequestStatus.RUNNING
        r.num_computed_tokens = 30
    sched.running.extend([r1, r2])
    beats.append({
        "beat": "0 入场（两条 30-token prompt）",
        "request": "r1+r2",
        "total_tokens": 30,
        "cdiv_by_16": 2,
        "held_before": 0,
        "new_blocks": [a1.get_block_ids()[0], a2.get_block_ids()[0]],
        "free_after": sched.kv_cache_manager.block_pool.get_num_free_blocks(),
        "note": "各 2 块（32 槽，块内余量 2——decode 的前 2 步免账）",
    })

    # r1 长大：38（越界 33 后的第一拍）→ 44（块内）→ 52（再越界）
    res = beat("1", r1, 8, "token 33 越过第 2 块边界 → 本拍补 1 块")
    r1.num_computed_tokens = 38
    assert res is not None and res.get_block_ids()[0] == [5]

    res = beat("2", r1, 6, "仍在第 3 块内（44 ≤ 48）→ 0 新块，免账拍")
    r1.num_computed_tokens = 44
    assert res is not None and res is sched.kv_cache_manager.empty_kv_cache_blocks

    res = beat("3", r1, 8, "token 49 越过第 3 块边界 → 再补 1 块")
    r1.num_computed_tokens = 52
    assert res is not None and res.get_block_ids()[0] == [6]

    # r2 长大：46 → 补第 3 块（池只剩 1）
    res = beat("4", r2, 16, "r2 长到 46：越过边界 → 拿到池中最后 1 块")
    r2.num_computed_tokens = 46
    assert res is not None and res.get_block_ids()[0] == [7]

    # r1 再长：65 → cdiv 5 > 已持 4 → 需 1 块 > 空闲 0 → allocate_slots None →
    # while True 抢占环（环在 allocate_slots_for_running 内部：弹队尾 r2 →
    # _preempt_request → 重试成功）
    free_before_beat5 = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    res = beat(
        "5", r1, 13,
        f"token 65 越过第 4 块边界：预测 1 > 空闲 {free_before_beat5} → None → 环内 FCFS 弹队尾 r2、其块回池、重试成功",
    )
    r1.num_computed_tokens = 65
    preempt_record = {
        "trigger": "allocate_slots 返回 None（kv_cache_manager.py:L510-L527）→ allocate_slots_for_running 的 while True 环（scheduler.py:L576-L629）",
        "loop_action": "FCFS 弹 running 队尾 = 最新请求 r2（L614-L615）",
        "preempted": "r2",
        "r2_status_after": str(r2.status),
        "r2_num_computed_tokens_after": r2.num_computed_tokens,
        "r2_blocks_after": held("r2"),
        "r2_freed_eviction_order": [7, 4, 3],
        "retry_result": res.get_block_ids()[0] if res is not None else None,
        "note": "_preempt_request（L1274-L1308）：free 全部块 + num_computed_tokens=0 + PREEMPTED——外部行为 ch11 的块侧内景；r1 重试拿到 r2 刚归还的块 7（LRU 回收）",
    }

    # 节奏总表：token 数 → 应持块数
    rhythm = [
        {"tokens": t, "blocks_needed": -(-t // BLOCK_SIZE)}
        for t in [30, 31, 32, 33, 48, 49, 64, 65, 80]
    ]

    assert preempt_record["r2_status_after"] == "PREEMPTED"
    assert preempt_record["r2_num_computed_tokens_after"] == 0
    assert preempt_record["r2_blocks_after"] == 0
    assert preempt_record["retry_result"] == [7]
    assert beats[4]["free_after"] == 0  # 池干
    assert beats[5]["free_after"] == 2  # r2 归还 3 块、r1 重试取走 1 块

    out = {
        "driver": "run_m11_decode_growth.py",
        "mechanism": "m11 decode 稳态长块（scheduler.py:L576-L629 / single_type:L194-L200）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版",
        "config": {"num_gpu_blocks": NUM_BLOCKS, "usable": NUM_BLOCKS - 1, "block_size": BLOCK_SIZE},
        "beats": beats,
        "preemption_loop": preempt_record,
        "growth_rhythm_tokens_to_blocks": rhythm,
        "rhythm_note": "每 block_size=16 个 token 多要一块：31/32 免账、33 越界要第 3 块、49 要第 4 块……block_size 越大节奏越稀、尾部浪费上界越大（m13）",
        "emulation_note": "每拍的 num_computed_tokens 推进由 driver 手工模拟（真实由调度器乐观记账 + ⑤ 拍回填，ch10/ch9）",
    }

    dst = Path(__file__).resolve().parent / "m11_decode_growth.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(beats, ensure_ascii=False, indent=1))
    print(json.dumps(preempt_record, ensure_ascii=False))


if __name__ == "__main__":
    main()
