"""Driver for m1 (分页总布局：等大块池 + 每请求逻辑块表 + 块内槽位) — host run
against the ch13 subtract-only companion (pin vLLM v0.27.1, enable_prefix_caching=False
源码原生 NoPrefixCache 支).

Finger-count scenario, num_gpu_blocks=10 / block_size=16 (9 块可用——0 号被
null_block 占):
  r1 100-token prompt -> cdiv=7 块 [1..7]（112 槽，尾部浪费 12 < 16）
  r2  30-token prompt -> cdiv=2 块 [8,9]（32 槽，浪费 2）
  r1 完成 -> 逆序还块（驱逐序 [7,6,5,4,3,2,1] 挂队尾）
  r3  35-token prompt -> popleft 复用 [7,6,5]——逻辑连续、物理不相邻（分页的本体）
附带旧设计对照的算术（按 max_len=2048 连续预分配的内部碎片）与 r3 的槽位换算
预演（恒等式本体归 m9）。
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


def make_config(num_blocks: int) -> KVCacheConfig:
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[LAYER])
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[LAYER], kv_cache_spec=spec)],
    )


def make_manager(num_blocks: int) -> KVCacheManager:
    return KVCacheManager(
        kv_cache_config=make_config(num_blocks),
        max_model_len=256,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def make_request(req_id: str, prompt_tokens: int) -> Request:
    req = Request(request_id=req_id, prompt_token_ids=list(range(prompt_tokens)))
    req.status = RequestStatus.WAITING
    return req


def slot(block_id: int, pos: int) -> int:
    # 槽位恒等式预演（本体 = Triton kernel，m9 单独取证）
    return block_id * BLOCK_SIZE + pos % BLOCK_SIZE


def main():
    mgr = make_manager(NUM_BLOCKS)
    single = mgr.coordinator.single_type_managers[0]

    def ids(req_id):
        return [b.block_id for b in single.req_to_blocks[req_id]]

    out = {
        "driver": "run_m1_paged_layout.py",
        "mechanism": "m1 分页总布局：等大块池 + 每请求逻辑块表（block_pool.py:L175-L181 / single_type:L94-L97 / block_table.py:L105-L112）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch13 implementation/ 只做减法精简版（enable_prefix_caching=False 源码原生支）",
        "config": {
            "num_gpu_blocks": NUM_BLOCKS,
            "usable_blocks": NUM_BLOCKS - 1,
            "block_size": BLOCK_SIZE,
            "null_block_id": 0,
            "max_model_len": 256,
        },
        "events": [],
    }

    def ev(label, req, tokens, got, free_after, extra=None):
        capacity = BLOCK_SIZE * len(got) if tokens is not None else None
        rec = {
            "event": label,
            "request": req,
            "tokens": tokens,
            "cdiv_by_16": -(-tokens // BLOCK_SIZE) if tokens is not None else None,
            "logical_block_table": got,
            "slot_capacity": capacity,
            "tail_waste": (capacity - tokens) if tokens is not None else None,
            "pool_free_after": free_after,
        }
        if extra:
            rec.update(extra)
        out["events"].append(rec)
        return rec

    r1 = make_request("r1", 100)
    got1 = mgr.allocate_slots(r1, 100).get_block_ids()[0]
    ev("r1 入场（100-token prompt）", "r1", 100, got1, mgr.block_pool.get_num_free_blocks())

    r2 = make_request("r2", 30)
    got2 = mgr.allocate_slots(r2, 30).get_block_ids()[0]
    ev("r2 入场（30-token prompt）", "r2", 30, got2, mgr.block_pool.get_num_free_blocks())

    # r1 完成 -> 终局还块（门面 free -> 逆序 free_blocks）
    r1.status = RequestStatus.FINISHED_STOPPED
    mgr.free(r1)
    queue_after_free = [
        b.block_id for b in mgr.block_pool.free_block_queue.get_all_free_blocks()
    ]
    ev(
        "r1 完成：逆序还块", "r1", None, [], mgr.block_pool.get_num_free_blocks(),
        {
            "tokens": None,
            "cdiv_by_16": None,
            "slot_capacity": None,
            "tail_waste": None,
            "freed_eviction_order": list(reversed(got1)),
            "free_queue_head_to_tail": queue_after_free,
            "note": "7 块回池，驱逐序（队尾段）[7,6,5,4,3,2,1]——尾块最先处于被驱逐位",
        },
    )

    r3 = make_request("r3", 35)
    got3 = mgr.allocate_slots(r3, 35).get_block_ids()[0]
    r3_slots = {
        "token_0": {"block": got3[0], "slot": slot(got3[0], 0)},
        "token_15": {"block": got3[0], "slot": slot(got3[0], 15)},
        "token_16": {"block": got3[1], "slot": slot(got3[1], 16)},
        "token_32": {"block": got3[2], "slot": slot(got3[2], 32)},
        "token_34": {"block": got3[2], "slot": slot(got3[2], 34)},
    }
    ev(
        "r3 入场（35-token prompt，复用 r1 还回的块）", "r3", 35, got3,
        mgr.block_pool.get_num_free_blocks(),
        {
            "r3_slot_preview": r3_slots,
            "note": "逻辑块表 [7,6,5]：token 序列逻辑连续，物理块不相邻（块 7、6、5 是 r1 的尾三块）——分页的本体",
        },
    )

    # 汇总账
    paged_used_tokens = 100 + 30
    paged_capacity = (7 + 2) * BLOCK_SIZE
    out["pool_summary"] = {
        "usable_blocks": NUM_BLOCKS - 1,
        "held_after_r1_r2": 9,
        "tokens_held": paged_used_tokens,
        "slots_capacity": paged_capacity,
        "combined_tail_waste": paged_capacity - paged_used_tokens,
        "waste_upper_bound_per_request": BLOCK_SIZE - 1,
        "note": "两条请求合计浪费 14 个 token 位 < 2×16——每请求浪费 < 1 块",
    }

    # 旧设计对照（算术，非源码断言）：按请求最大长度一次性预分配连续显存
    reserved = 2048  # 论文举例的 max_len（arXiv:2309.06180 §2.2 "e.g., 2048 tokens"）
    out["old_design_contrast"] = {
        "reserved_slots_per_request": reserved,
        "r1_actual_tokens": 100,
        "r1_internal_waste_slots": reserved - 100,
        "r1_internal_frag_pct": round((reserved - 100) / reserved * 100, 2),
        "two_requests_reserved": 2 * reserved,
        "two_requests_used": paged_used_tokens,
        "two_requests_waste_pct": round(
            (2 * reserved - paged_used_tokens) / (2 * reserved) * 100, 2
        ),
        "paged_r1_waste": 12,
        "paper_utilization_pct_range": "20.4 - 38.2",
        "paper_note": "arXiv:2309.06180 §5.2：现有系统 KV 有效利用率 only 20.4% - 38.2%（是 38.2 不是 38.3）；§2.2 旧设计连续预分配；分页后同延迟吞吐 2-4x",
    }

    # 校验断言（写进 trace，供 lint/读者复核）
    assert got1 == [1, 2, 3, 4, 5, 6, 7]
    assert got2 == [8, 9]
    assert got3 == [7, 6, 5]
    assert queue_after_free[-7:] == [7, 6, 5, 4, 3, 2, 1]
    assert mgr.block_pool.get_num_free_blocks() == 4

    dst = Path(__file__).resolve().parent / "m1_paged_layout.json"
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out["pool_summary"], ensure_ascii=False))
    print(json.dumps(out["old_design_contrast"], ensure_ascii=False))


if __name__ == "__main__":
    main()
