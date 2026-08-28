# run_m15.py — m15 并发容量核算 驱动脚本
# uniform：Llama-2-7B 锚例（32 层、8GiB、page 256KiB → 1024 块；max_len 4096
# → 并发 4x、容量 16384 token）。混合：full 整序列 256 块 + swa cap 257 块
# （in_flight=8192 → cdiv(min(511+8192,4096),16)+1 = cdiv(4096,16)+1 = 257）
# → 每请求 513 块 → 并发 1024/513。
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.cache import CacheConfig
from implementation.config import ModelConfig, ParallelConfig, VllmConfig
from implementation.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from implementation.kv_cache_utils import (
    get_kv_cache_capacity,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
)

OUT = {}
GiB = 1 << 30


def make_cfg(max_model_len=4096):
    return VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len,
                                 original_max_model_len=max_model_len),
        cache_config=CacheConfig(enable_prefix_caching=False),
        parallel_config=ParallelConfig(),
    )


# ------------------------------------------------ uniform：Llama-2-7B 锚例
spec = FullAttentionSpec(
    block_size=16, num_kv_heads=32, head_size=128, dtype=torch.float16
)
specs = {f"model.layers.{i}.self_attn.attn": spec for i in range(32)}
cfg = make_cfg(4096)
configs = get_kv_cache_configs(cfg, [specs], [8 * GiB])
config = configs[0]
concurrency = get_max_concurrency_for_kv_cache_config(cfg, config)
num_tokens, max_concurrency = get_kv_cache_capacity(cfg, config)
OUT["uniform_llama7b"] = {
    "model": "Llama-2-7B",
    "available_kv_bytes": 8 * GiB,
    "page_size": spec.page_size_bytes,
    "num_blocks": config.num_blocks,
    "max_model_len": 4096,
    "blocks_per_request": math.ceil(4096 / 16),
    "max_concurrency": concurrency,
    "kv_cache_size_tokens": num_tokens,
    "identity": "tokens = concurrency x max_model_len = num_blocks x block_size",
    "check": num_tokens == 16384 and max_concurrency == 4.0,
}

# ------------------------------------------------ 混合：按组求和也正确
cfg2 = make_cfg(4096)
full_g = KVCacheGroupSpec(["f0", "f1"], FullAttentionSpec(
    block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16))
swa_g = KVCacheGroupSpec(["s0", "s1"], SlidingWindowSpec(
    block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16,
    sliding_window=512))
hybrid = KVCacheConfig(
    num_blocks=1024, kv_cache_tensors=[],
    kv_cache_groups=[full_g, swa_g],
)
max_in_flight = cfg2.max_in_flight_tokens
swa_cap = swa_g.kv_cache_spec.max_admission_blocks_per_request(max_in_flight, 4096)
full_blocks = math.ceil(4096 / 16)
conc_h = get_max_concurrency_for_kv_cache_config(cfg2, hybrid)
tokens_h, conc_h2 = get_kv_cache_capacity(cfg2, hybrid)
OUT["hybrid_sums_over_groups"] = {
    "max_in_flight_tokens": max_in_flight,
    "scheduler_max_num_batched_tokens": 8192,
    "num_blocks": 1024,
    "full_group_blocks_per_request": full_blocks,
    "swa_group_cap": swa_cap,
    "sliding_window_minus_one": 512 - 1,
    "swa_cap_derivation": "cdiv(min(512-1+8192, 4096), 16)+1 = cdiv(4096,16)+1",
    "blocks_per_request_sum": full_blocks + swa_cap,
    "max_concurrency": conc_h,
    "max_concurrency_rounded4": round(conc_h, 4),
    "kv_cache_size_tokens": tokens_h,
    "formula": "max_concurrency = num_blocks / sum_groups "
               "cdiv(每组 max_memory_usage_bytes, page)",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m15.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
