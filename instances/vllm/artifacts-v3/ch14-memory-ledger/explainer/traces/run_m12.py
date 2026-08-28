# run_m12.py — m12 精修版水位 watermark 驱动脚本
# 三条件对照（kv_cache_manager.py:L463-L470）：水位只在本步已有调度请求
# （has_scheduled_reqs）且请求状态 ∈ {WAITING, PREEMPTED} 时计入
# required_blocks —— RUNNING 涨块不受垫片约束、首拍空转不计入。
# 池 10 块、watermark=0.5 → watermark_blocks=5；free=9（null 占 1）。
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
from implementation.scheduler_config import SchedulerConfig

OUT = {}


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def make_manager(num_blocks, watermark):
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
        watermark=watermark,
    )


def make_request(req_id, num_tokens, status=RequestStatus.WAITING):
    req = Request(request_id=req_id, prompt_token_ids=list(range(num_tokens)))
    req.status = status
    req.num_computed_tokens = 0
    return req


NUM_BLOCKS, WM = 10, 0.5
m = make_manager(NUM_BLOCKS, WM)
OUT["setup"] = {
    "num_blocks": NUM_BLOCKS,
    "watermark": WM,
    "watermark_blocks": m.watermark_blocks,
    "formula": "watermark_blocks = int(watermark x num_blocks)",
    "free_blocks": m.block_pool.get_num_free_blocks(),
    "default_watermark": SchedulerConfig().watermark,
    "default_off": SchedulerConfig().watermark == 0.0,
}

# 行 1：WAITING + has_scheduled_reqs=True → 水位计入 → None
m1 = make_manager(NUM_BLOCKS, WM)
req1 = make_request("w1", num_tokens=80)
res1 = m1.allocate_slots(req1, num_new_tokens=80, full_sequence_must_fit=False,
                         has_scheduled_reqs=True)
OUT["waiting_with_scheduled"] = {
    "request_tokens": 80,
    "required_blocks": -(-80 // 16),
    "watermark_blocks": m1.watermark_blocks,
    "required_plus_watermark": -(-80 // 16) + m1.watermark_blocks,
    "free_blocks": m1.block_pool.get_num_free_blocks(),
    "returned": "None" if res1 is None else "admitted",
    "verdict": "5 + 5 = 10 > 9 → 拒（headroom 留给已 running 的请求长大）",
}

# 行 2：WAITING + has_scheduled_reqs=False（首拍空转）→ 水位不计入 → 放行
m2 = make_manager(NUM_BLOCKS, WM)
req2 = make_request("w2", num_tokens=80)
res2 = m2.allocate_slots(req2, num_new_tokens=80, full_sequence_must_fit=False,
                         has_scheduled_reqs=False)
OUT["waiting_first_step"] = {
    "request_tokens": 80,
    "required_blocks": -(-80 // 16),
    "watermark_blocks_applied": 0,
    "free_blocks": m2.block_pool.get_num_free_blocks(),
    "returned": "None" if res2 is None else "admitted",
    "verdict": "5 + 0 = 5 <= 9 → 放行（池空转时再保守就永远开不了工）",
}

# 行 3：RUNNING + has_scheduled_reqs=True → 精修版只对 WAITING/PREEMPTED → 放行
m3 = make_manager(NUM_BLOCKS, WM)
req3 = make_request("w3", num_tokens=80, status=RequestStatus.RUNNING)
res3 = m3.allocate_slots(req3, num_new_tokens=80, full_sequence_must_fit=False,
                         has_scheduled_reqs=True)
OUT["running_ignores_watermark"] = {
    "request_tokens": 80,
    "required_blocks": -(-80 // 16),
    "watermark_blocks_applied": 0,
    "free_blocks": m3.block_pool.get_num_free_blocks(),
    "returned": "None" if res3 is None else "admitted",
    "verdict": "RUNNING 涨块不扣水位——垫片只管『新客进门』，不管『在座长个』",
}

OUT["thrashing_motivation"] = {
    "source": "benchmarks/kv_cache_watermark.sh:L7-L16 官方论证",
    "config": "并发 200、input ~300、output ~4000（decode-heavy）、KV 池压到 "
              "均值需求 ~1.5x",
    "loop": "准入只预留输入长度（full-ISL 门管住）但输出长度未知不预留 → "
            "短时超收 → 集体增长 → 池尽 → 抢占刚准入者 → 重 prefill → 再抢占",
    "watermark_role": "水位留 headroom 让 running 请求有处可长，截断抖动循环",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m12.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
