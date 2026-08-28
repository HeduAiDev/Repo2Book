# run_m11.py — m11 回收感知准入上限 max_admission_blocks_per_request 驱动脚本
# A. cap 公式（SWA cdiv(window-1+in_flight, bs)+1 / chunked cdiv(chunk+in_flight, bs)）
# B. 多步推进：SWA 请求逐 chunk 分配，remove_skipped 先回收窗外块 → 实持块
#    停在 cap 之下（每步记窗外回收/新分/实持/池 free）
# C. 混合两组过 full-ISL 门：full 按整序列 + swa 夹到 cap
import json
import math
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


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def swa_spec(window, block_size=16):
    return SlidingWindowSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.float16,
        sliding_window=window,
    )


def chunked_spec(chunk, block_size=16):
    return ChunkedLocalAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.float16,
        attention_chunk_size=chunk,
    )


# ------------------------------------------------ A. cap 公式
OUT["cap_formula"] = {
    "swa_rule": "cap = cdiv(min(sliding_window - 1 + max_in_flight_tokens, "
                "max_model_len), block_size) + 1",
    "plus_one_reason": "窗口可能不从块首开始：block_size 4 存 6-token 窗 "
                       "[CDEF] 要两块 [XXCD][EF]（源码注释原例）",
    "cases": [
        {"spec": "swa window=8 bs=4 in_flight=8 max_len=64",
         "num_tokens": min(8 - 1 + 8, 64),
         "cap": swa_spec(8, 4).max_admission_blocks_per_request(8, 64)},
        {"spec": "swa window=7 bs=4 in_flight=0 max_len=100",
         "num_tokens": min(7 - 1 + 0, 100),
         "cap": swa_spec(7, 4).max_admission_blocks_per_request(0, 100)},
        {"spec": "swa window=512 bs=16 in_flight=0 max_len=4096",
         "num_tokens": min(512 - 1 + 0, 4096),
         "cap": swa_spec(512, 16).max_admission_blocks_per_request(0, 4096)},
        {"spec": "swa window=4096 bs=16 in_flight=0 max_len=100 (clamp)",
         "num_tokens": min(4096 - 1 + 0, 100),
         "cap": swa_spec(4096, 16).max_admission_blocks_per_request(0, 100)},
        {"spec": "chunked chunk=8 bs=4 in_flight=0 max_len=100",
         "num_tokens": min(8 + 0, 100),
         "cap": chunked_spec(8, 4).max_admission_blocks_per_request(0, 100),
         "note": "chunked 窗口从块首开始，无 +1"},
    ],
    "single_source": "spec.max_admission_blocks_per_request 同一方法喂启动期池"
                     "大小器（max_memory_usage_bytes = cap x page）与运行期准入门",
    "swa_max_memory_usage_is_cap_times_page": (
        swa_spec(512, 16).max_memory_usage_bytes(
            type("VC", (), {
                "max_in_flight_tokens": 0,
                "model_config": type("MC", (), {"max_model_len": 4096})(),
                "cache_config": None,
                "parallel_config": type("PC", (), {
                    "decode_context_parallel_size": 1})(),
            })()
        ) == swa_spec(512, 16).max_admission_blocks_per_request(0, 4096)
        * swa_spec(512, 16).page_size_bytes
    ),
}

# ------------------------------------------------ B. 多步推进（实持块停在上限下）
def make_manager(specs_by_group, num_blocks, max_model_len,
                 max_in_flight_tokens=None):
    groups = [KVCacheGroupSpec(names, spec) for names, spec in specs_by_group]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=[]) for _ in groups],
        kv_cache_groups=groups,
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=max_model_len,
        scheduler_block_size=max(g.kv_cache_spec.block_size for g in groups),
        hash_block_size=groups[0].kv_cache_spec.block_size,
        enable_caching=False,
        max_in_flight_tokens=max_in_flight_tokens,
    )


WIN, BS, IN_FLIGHT, MAXLEN, POOL = 8, 4, 8, 64, 16
m = make_manager([(["s0"], swa_spec(WIN, BS))], num_blocks=POOL,
                 max_model_len=MAXLEN, max_in_flight_tokens=IN_FLIGHT)
mgr = m.coordinator.single_type_managers[0]
cap = mgr._max_admission_blocks_per_request
req = Request(request_id="swa-req", prompt_token_ids=list(range(MAXLEN)))
req.status = RequestStatus.WAITING
steps = []
for step in range(1, 9):
    computed_before = req.num_computed_tokens
    free_before = m.block_pool.get_num_free_blocks()
    held_before = sum(1 for b in mgr.req_to_blocks["swa-req"] if not b.is_null)
    res = m.allocate_slots(req, num_new_tokens=8, full_sequence_must_fit=False)
    held_after = sum(1 for b in mgr.req_to_blocks["swa-req"] if not b.is_null)
    steps.append({
        "step": step,
        "computed_before": computed_before,
        "skipped_tokens_before_alloc": mgr.get_num_skipped_tokens(computed_before),
        "held_before": held_before,
        "held_after": held_after,
        "free_pool_after": m.block_pool.get_num_free_blocks(),
        "admitted": res is not None,
        "held_le_cap": held_after <= cap,
    })
    req.num_computed_tokens += 8
    req.status = RequestStatus.RUNNING
OUT["multi_step_plateau"] = {
    "window": WIN, "block_size": BS, "max_in_flight_tokens": IN_FLIGHT,
    "max_model_len": MAXLEN, "pool_blocks": POOL,
    "free_at_start": POOL - 1,  # null_block 恒占 1
    "cap": cap,
    "cap_formula_check": cap == math.ceil((WIN - 1 + IN_FLIGHT) / BS) + 1,
    "steps": steps,
    "plateau": "稳态实持 4 块 <= cap 5（+1 顶着窗口不在块首的最坏错位）",
}

# ------------------------------------------------ C. 混合两组过 full-ISL 门
m2 = make_manager([(["f0", "f1"], full_spec()), (["s0", "s1"], swa_spec(512))],
                  num_blocks=1000, max_model_len=4096, max_in_flight_tokens=0)
f_mgr, s_mgr = m2.coordinator.single_type_managers
full_required = f_mgr.get_num_blocks_to_allocate(
    "hy", 4096, (), 0, 0, 4096, apply_admission_cap=True)
swa_uncapped = s_mgr.get_num_blocks_to_allocate(
    "hy", 4096, (), 0, 0, 4096, apply_admission_cap=False)
swa_capped = s_mgr.get_num_blocks_to_allocate(
    "hy", 4096, (), 0, 0, 4096, apply_admission_cap=True)
req2 = Request(request_id="hy", prompt_token_ids=list(range(4096)))
req2.status = RequestStatus.WAITING
res2 = m2.allocate_slots(req2, num_new_tokens=4096, full_sequence_must_fit=True)
OUT["hybrid_gate"] = {
    "pool_blocks": 1000,
    "free_blocks": 999,
    "request_tokens": 4096,
    "full_group_required": full_required,
    "swa_group_uncapped": swa_uncapped,
    "swa_group_capped": swa_capped,
    "swa_cap": s_mgr._max_admission_blocks_per_request,
    "total_required": full_required + swa_capped,
    "admitted": res2 is not None,
    "verdict": "full 256 + swa 夹到 33 = 289 <= 999 放行——按整序列算 SWA 本要 "
               "256 块/组，准入上限换回并发",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m11.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
