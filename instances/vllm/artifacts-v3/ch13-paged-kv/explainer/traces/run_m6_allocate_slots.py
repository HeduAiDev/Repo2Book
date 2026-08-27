"""Driver for m6 (allocate_slots 三段式) — host run against the ch13 companion.

五次调用把三段式与两个出口都走到（pool 10 块、可用 9）：
  1 WAITING r1 入场 100 token：预测 7 ≤ 空闲 9 → 分 [1..7]
  2 WAITING r2 入场 128 token：预测 8 > 空闲 2 → None（且零半截账）
  3 RUNNING r1 长到 116：差 1 ≤ 2 → 分 [8]
  4 WAITING r3 入场 16：预测 1 ≤ 1 → 分 [9]
  5 RUNNING r1 长到 132：差 1 > 空闲 0 → None（RUNNING 侧 None = ch11 抢占信号）
每步记录：第一段预测 vs 空闲 / 第二段挂命中（False 支 empty 判同短路）/ 第三段
分到的块 / 第四段写回（caching 关早退）。
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
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402

BLOCK_SIZE = 16
NUM_BLOCKS = 10
LAYER = "model.layers.0.self_attn.attn"


def make_manager() -> KVCacheManager:
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
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=512,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, n: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(n)))
    req.status = RequestStatus.WAITING
    return req


def main():
    mgr = make_manager()
    single = mgr.coordinator.single_type_managers[0]
    calls = []

    def call(label, side, req, num_new):
        free_before = mgr.block_pool.get_num_free_blocks()
        held_before = len(single.req_to_blocks.get(req.request_id, ()))
        # 第一段：预测器（与分配器同构——m5）
        predicted = single.get_num_blocks_to_allocate(
            request_id=req.request_id,
            num_tokens=req.num_computed_tokens + num_new,
            new_computed_blocks=[],
            total_computed_tokens=req.num_computed_tokens + num_new,
            num_local_computed_tokens=req.num_computed_tokens + num_new,
            num_tokens_main_model=req.num_computed_tokens + num_new,
        )
        result = mgr.allocate_slots(req, num_new)
        if result is None:
            got = None
            verdict = f"{predicted} > {free_before} → return None"
        else:
            got = result.get_block_ids()[0]
            verdict = f"{predicted} ≤ {free_before} → 过"
        rec = {
            "call": label,
            "side": side,
            "target_tokens": req.num_computed_tokens + num_new,
            "stage1_predicted": predicted,
            "stage1_free": free_before,
            "stage1_verdict": verdict,
            "stage2_attach_hits": "跳过（new_computed_block_list is empty_kv_cache_blocks.blocks 判同短路——前缀缓存关，命中恒空 → ch15）",
            "stage3_new_blocks": got,
            "stage4_cache_blocks": "早退（not enable_caching → L551-L552 直接返回，不进写回）",
            "free_after": mgr.block_pool.get_num_free_blocks(),
            "held_after": len(single.req_to_blocks.get(req.request_id, ())),
        }
        calls.append(rec)
        # 乐观推进由调度器做（ch10）；driver 手工模拟下一拍的 computed
        return result

    r1 = make_request("r1", 100)
    call("1", "WAITING r1 入场", r1, 100)
    r1.status = RequestStatus.RUNNING
    r1.num_computed_tokens = 100

    r2 = make_request("r2", 128)
    res2 = call("2", "WAITING r2 入场", r2, 128)

    call("3", "RUNNING r1 长大（+16）", r1, 16)
    r1.num_computed_tokens = 116

    r3 = make_request("r3", 16)
    call("4", "WAITING r3 入场", r3, 16)

    res5 = call("5", "RUNNING r1 长大（+16）", r1, 16)

    # None 无半截账的物证
    no_partial_state = {
        "after_call2": {
            "r2_in_req_to_blocks": "r2" in single.req_to_blocks,
            "free_blocks": calls[1]["free_after"],
            "note": "容量不够 → None，逻辑块表无 r2、空闲原封不动（2）——没有半截账",
        }
    }
    assert res2 is None and res5 is None
    assert calls[0]["stage3_new_blocks"] == [1, 2, 3, 4, 5, 6, 7]
    assert calls[2]["stage3_new_blocks"] == [8]
    assert calls[3]["stage3_new_blocks"] == [9]
    assert calls[1]["free_after"] == 2 and calls[4]["free_after"] == 0

    out = {
        "driver": "run_m6_allocate_slots.py",
        "mechanism": "m6 allocate_slots 三段式（kv_cache_manager.py:L344-L565；入口 scheduler.py:L973-L985 / L576-L629）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（enable_prefix_caching=False）",
        "config": {"num_gpu_blocks": NUM_BLOCKS, "usable": NUM_BLOCKS - 1, "block_size": BLOCK_SIZE},
        "calls": calls,
        "none_leaves_no_partial_state": no_partial_state,
        "note": "调用 5 的 None 发生在 RUNNING 侧——调度器把它喂进 while True 抢占环（scheduler.py:L576-L629，外部行为 ch11）；WAITING 侧 None 只 break（ch10）",
    }

    dst = Path(__file__).resolve().parent / "m6_allocate_slots.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(calls, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
