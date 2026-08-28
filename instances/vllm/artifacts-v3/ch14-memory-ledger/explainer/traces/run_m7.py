# run_m7.py — m7 张量共享布局驱动脚本（figure-only 机制的数字出处）
# 通用分支：group_size 个内存池、每池由每组各出一层共享 —— 3 组
# (full.0, full.1), (sw.0, sw.2), (sw.1) 的官方 ASCII 图例（源码注释）落成数字：
# 2 张量、每张量 page x num_blocks 字节、shared_by 各组出一层。
# 另收单组异宽（UniformTypeKVCacheSpecs）逐层张量一路。
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.cache import CacheConfig
from implementation.config import ModelConfig, ParallelConfig, VllmConfig
from implementation.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from implementation.kv_cache_utils import get_kv_cache_config_from_groups

OUT = {}
PAGE = 65536


def full_spec(heads=8):
    return FullAttentionSpec(
        block_size=16, num_kv_heads=heads, head_size=128, dtype=torch.float16
    )


def swa_spec(window=512):
    return SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16,
        sliding_window=window,
    )


def make_cfg():
    return VllmConfig(
        model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
        cache_config=CacheConfig(enable_prefix_caching=False),
        parallel_config=ParallelConfig(),
    )


# --------------------------------------------- 通用分支：group_size 池
full_g = KVCacheGroupSpec(["full.0", "full.1"], full_spec())
sw0 = KVCacheGroupSpec(["sw.0", "sw.2"], swa_spec(512))
sw1 = KVCacheGroupSpec(["sw.1"], swa_spec(512))
config = get_kv_cache_config_from_groups(
    make_cfg(), [full_g, sw0, sw1], available_memory=2 * PAGE * 10
)
OUT["general_group_size_pools"] = {
    "num_groups": 3,
    "group_layers": [["full.0", "full.1"], ["sw.0", "sw.2"], ["sw.1"]],
    "group_size": 2,
    "page_size": PAGE,
    "available_memory": 2 * PAGE * 10,
    "num_blocks": config.num_blocks,
    "num_tensors": len(config.kv_cache_tensors),
    "tensors": [
        {"tensor": i + 1, "size_bytes": t.size, "size_pages": t.size // PAGE,
         "shared_by": t.shared_by}
        for i, t in enumerate(config.kv_cache_tensors)
    ],
    "layout_note": "full.0, sw.0, sw.1 共一张张量；full.1, sw.2 共另一张 —— "
                   "一个 block_id 同一时刻只归一个组用，物理可共享",
}

# --------------------------------------------- 单组异宽：逐层一张张量
s0, s1 = full_spec(8), full_spec(4)
uni = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"l0": s0, "l1": s1})
group = KVCacheGroupSpec(["l0", "l1"], uni)
total_page = s0.page_size_bytes + s1.page_size_bytes
config2 = get_kv_cache_config_from_groups(make_cfg(), [group],
                                          available_memory=total_page * 5)
OUT["single_group_per_layer_tensors"] = {
    "l0_num_kv_heads": s0.num_kv_heads,
    "l1_num_kv_heads": s1.num_kv_heads,
    "l0_page": s0.page_size_bytes,
    "l1_page": s1.page_size_bytes,
    "aggregated_page": total_page,
    "num_blocks": config2.num_blocks,
    "tensors": [
        {"tensor": i + 1, "size_bytes": t.size, "shared_by": t.shared_by}
        for i, t in enumerate(config2.kv_cache_tensors)
    ],
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m7.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
