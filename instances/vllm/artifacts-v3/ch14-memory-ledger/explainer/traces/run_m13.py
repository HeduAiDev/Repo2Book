# run_m13.py — m13 SWA 窗外回收与 null_block 原位占位 驱动脚本
# A. get_num_skipped_tokens 三型：SWA docstring 例（窗口 4/computed 7 → 4）、
#    full 恒 0、chunked 按 chunk 对齐（8:13→8 / 8→8 / 7→0）
# B. 窗外整块 free + null 原位换位：窗口 4、块 4、16-token 请求逐步推进，
#    块表形态 [b,b,b,b] → [NULL,b,b,b] → ...（每步记 skipped/回收/实持/free）
# C. 稳态：窗口 8、块 4、64-token → 60 算完时 13 块归池、实持 3
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from implementation.kv_cache_manager import KVCacheManager
from implementation.request import Request, RequestStatus

OUT = {}


def full_spec(bs=16):
    return FullAttentionSpec(
        block_size=bs, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def swa_spec(window, bs=16):
    return SlidingWindowSpec(
        block_size=bs, num_kv_heads=8, head_size=128, dtype=torch.float16,
        sliding_window=window,
    )


def chunked_spec(chunk, bs=16):
    return ChunkedLocalAttentionSpec(
        block_size=bs, num_kv_heads=8, head_size=128, dtype=torch.float16,
        attention_chunk_size=chunk,
    )


def make_manager(spec, num_blocks):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=[])],
        kv_cache_groups=[KVCacheGroupSpec(["s0"], spec)],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=4096,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        enable_caching=False,
    )


def block_shape(manager, req_id, pool):
    return ["NULL" if b.is_null else f"b{b.block_id}"
            for b in manager.req_to_blocks[req_id]]


# ------------------------------------------------ A. skipped_tokens 三型
mA = make_manager(swa_spec(4, bs=4), num_blocks=8)
swa = mA.coordinator.single_type_managers[0]
mFull = make_manager(full_spec(4), num_blocks=8)
full = mFull.coordinator.single_type_managers[0]
mC = make_manager(chunked_spec(8, bs=8), num_blocks=8)
cl = mC.coordinator.single_type_managers[0]
OUT["skipped_tokens_formula"] = {
    "swa": {
        "formula": "max(0, num_computed_tokens - sliding_window + 1)",
        "window": 4,
        "computed_7": swa.get_num_skipped_tokens(7),
        "computed_3": swa.get_num_skipped_tokens(3),
        "docstring": "tokens 0-3 在窗外被跳过（下一 token 的窗口是 4~7）",
    },
    "full": {
        "formula": "恒 0",
        "computed_10000": full.get_num_skipped_tokens(10000),
    },
    "chunked": {
        "formula": "num_computed_tokens // chunk x chunk（整 chunk 对齐）",
        "chunk": 8,
        "computed_13": cl.get_num_skipped_tokens(13),
        "computed_8": cl.get_num_skipped_tokens(8),
        "computed_7": cl.get_num_skipped_tokens(7),
    },
}

# ------------------------------------------------ B. null 原位换位逐步推进
WIN, BS, NTOK, POOL = 4, 4, 16, 8
m = make_manager(swa_spec(WIN, bs=BS), num_blocks=POOL)
mgr = m.coordinator.single_type_managers[0]
req = Request(request_id="r", prompt_token_ids=list(range(NTOK)))
req.status = RequestStatus.WAITING
m.allocate_slots(req, num_new_tokens=NTOK, full_sequence_must_fit=False)
steps = [{
    "processed": 0,
    "skipped_tokens": 0,
    "blocks_freed": 0,
    "shape": block_shape(mgr, "r", m.block_pool),
    "held": sum(1 for b in mgr.req_to_blocks["r"] if not b.is_null),
    "free_pool": m.block_pool.get_num_free_blocks(),
}]
for processed in (7, 11, 15):
    skipped = swa.get_num_skipped_tokens(processed)
    free_before = m.block_pool.get_num_free_blocks()
    m.remove_skipped_blocks("r", processed)
    steps.append({
        "processed": processed,
        "skipped_tokens": skipped,
        "blocks_freed": m.block_pool.get_num_free_blocks() - free_before,
        "shape": block_shape(mgr, "r", m.block_pool),
        "held": sum(1 for b in mgr.req_to_blocks["r"] if not b.is_null),
        "free_pool": m.block_pool.get_num_free_blocks(),
    })
OUT["null_in_place"] = {
    "window": WIN, "block_size": BS, "request_tokens": NTOK, "pool_blocks": POOL,
    "free_at_start": POOL - 1,
    "steps": steps,
    "invariant": "块表第 i 块 ↔ 第 i x block_size 个 token：null 占位保住位置 "
                 "对齐，注意力照常按表读、读到 null 的位置本来就在窗外不读",
}

# ------------------------------------------------ C. 稳态长序列
WIN2, BS2, NTOK2, POOL2 = 8, 4, 64, 64
m2 = make_manager(swa_spec(WIN2, bs=BS2), num_blocks=POOL2)
mgr2 = m2.coordinator.single_type_managers[0]
req2 = Request(request_id="long", prompt_token_ids=list(range(NTOK2)))
req2.status = RequestStatus.WAITING
m2.allocate_slots(req2, num_new_tokens=NTOK2, full_sequence_must_fit=False)
OUT["steady_state"] = {
    "window": WIN2, "block_size": BS2, "request_tokens": NTOK2, "pool_blocks": POOL2,
    "blocks_allocated_initial": len(mgr2.req_to_blocks["long"]),
    "processed": 60,
    "skipped_tokens": mgr2.get_num_skipped_tokens(60),
    "skipped_blocks_freed": mgr2.get_num_skipped_tokens(60) // BS2,
    "held_after": None,  # 填在下方
    "free_pool_after": None,
    "tokens_in_window": NTOK2 - (mgr2.get_num_skipped_tokens(60) // BS2) * BS2,
}
m2.remove_skipped_blocks("long", 60)
OUT["steady_state"]["held_after"] = sum(
    1 for b in mgr2.req_to_blocks["long"] if not b.is_null)
OUT["steady_state"]["free_pool_after"] = m2.block_pool.get_num_free_blocks()
OUT["steady_state"]["shape_head"] = block_shape(mgr2, "long", m2.block_pool)[:6]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m13.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
