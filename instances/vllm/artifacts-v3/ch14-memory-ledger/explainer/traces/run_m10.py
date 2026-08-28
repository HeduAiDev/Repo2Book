# run_m10.py — m10 full-ISL 准入门 驱动脚本
# #39734 超收对照：池 10 块 × block_size 16 = 160 token 容量；请求 200 token、
# 首个 chunked prefill chunk 只有 16 token。门开（默认）按整条序列算 → None
# 拒之门外；门关（旧行为=只查第一 chunk）→ 放进 → prefill 中途装不下死锁/OOM。
# 另收 reserved_blocks（异步 KV load 在途预约）一路。
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from implementation.kv_cache_manager import KVCacheManager
from implementation.request import Request, RequestStatus

OUT = {}


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def make_manager(num_blocks):
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=[])],
        kv_cache_groups=[KVCacheGroupSpec(["l0"], full_spec())],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=4096,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=False,
    )


def make_request(req_id, num_tokens, status=RequestStatus.WAITING):
    req = Request(request_id=req_id, prompt_token_ids=list(range(num_tokens)))
    req.status = status
    req.num_computed_tokens = 0
    return req


NUM_BLOCKS = 10
m = make_manager(NUM_BLOCKS)
free_at_start = m.block_pool.get_num_free_blocks()

OUT["setup"] = {
    "num_blocks": NUM_BLOCKS,
    "block_size": 16,
    "pool_capacity_tokens": NUM_BLOCKS * 16,
    "free_blocks_at_start": free_at_start,
    "free_note": "null_block 恒占 1 块（block_id 0），free = 10 - 1 = 9",
}

# ------------------------------------------------ 行 1：门开，整序列算 → None
m1 = make_manager(NUM_BLOCKS)
req = make_request("r1", num_tokens=200)
res_gate_on = m1.allocate_slots(req, num_new_tokens=16, full_sequence_must_fit=True)
OUT["gate_on"] = {
    "request_tokens": 200,
    "first_chunk_tokens": 16,
    "required_blocks_full_seq": -(-200 // 16),
    "required_blocks_first_chunk": 1,
    "free_blocks": free_at_start,
    "returned": "None" if res_gate_on is None else "blocks",
    "verdict": "13 > 9 → 拒之门外（#39734 的超收堵在这道门）",
}

# ------------------------------------------------ 行 2：门关，只查第一 chunk → 放进
m2 = make_manager(NUM_BLOCKS)
req2 = make_request("r2", num_tokens=200)
res_gate_off = m2.allocate_slots(req2, num_new_tokens=16, full_sequence_must_fit=False)
held = len(m2.coordinator.single_type_managers[0].req_to_blocks["r2"])
OUT["gate_off"] = {
    "request_tokens": 200,
    "first_chunk_tokens": 16,
    "required_blocks_first_chunk": 1,
    "returned": "None" if res_gate_off is None else "admitted",
    "blocks_held_after_chunk1": held,
    "free_blocks_after": m2.block_pool.get_num_free_blocks(),
    "verdict": "门只查第一 chunk：1 <= 9 放进——200 token 的请求池里只有 160 "
               "token 容量，prefill 到中途必然装不下（死锁/OOM 温床）",
}

# ------------------------------------------------ 行 3/4：reserved_blocks 预约
m3 = make_manager(NUM_BLOCKS)
req3 = make_request("r3", num_tokens=64)
res3 = m3.allocate_slots(req3, num_new_tokens=64, full_sequence_must_fit=False,
                         reserved_blocks=7)
req4 = make_request("r4", num_tokens=32)
res4 = m3.allocate_slots(req4, num_new_tokens=32, full_sequence_must_fit=False,
                         reserved_blocks=7)
OUT["reserved_blocks"] = {
    "free_blocks": free_at_start,
    "reserved_blocks": 7,
    "available_for_new": free_at_start - 7,
    "req3_tokens": 64,
    "req3_required": -(-64 // 16),
    "req3_returned": "None" if res3 is None else "admitted",
    "req4_tokens": 32,
    "req4_required": -(-32 // 16),
    "req4_returned": "None" if res4 is None else "admitted",
    "verdict": "required <= free - reserved：4 > 2 拒、2 <= 2 放（异步 KV load "
               "的在途预约，防挤死在途 prefill → ch16）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m10.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
