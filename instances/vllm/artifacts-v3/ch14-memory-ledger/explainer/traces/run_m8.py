# run_m8.py — m8 resolve_kv_cache_block_sizes 驱动脚本（scheduler=LCM / hash=GCD）
# 六路：单组两粒度同值；多组 16+32 → (32,16)；prefix_match_unit=8 覆盖 → (32,8)；
# unit=5 不整除 → ValueError；无缓存无 connector → hash 回退 scheduler 粒度；
# mamba 非 align（块 64 ≠ cache 16）→ 回退 (64,64)。
import json
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
    MambaSpec,
)
from implementation.kv_cache_utils import resolve_kv_cache_block_sizes

OUT = {}


def full_spec(block_size=16):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def mamba_spec(block_size=64):
    return MambaSpec(
        block_size=block_size, shapes=((4096,),), dtypes=(torch.uint8,),
        mamba_cache_mode="none",
    )


def make_cfg(cache):
    return VllmConfig(
        model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
        cache_config=cache,
        parallel_config=ParallelConfig(),
    )


def two_group_config(bs_a=16, bs_b=32, spec_b=None):
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], full_spec(bs_a)),
            KVCacheGroupSpec(["b"], spec_b or full_spec(bs_b)),
        ],
    )


OUT["case1_single_group"] = {
    "block_size": 16,
    "resolved": list(resolve_kv_cache_block_sizes(
        KVCacheConfig(num_blocks=10, kv_cache_tensors=[],
                      kv_cache_groups=[KVCacheGroupSpec(["a"], full_spec(16))]),
        make_cfg(CacheConfig(enable_prefix_caching=True)),
    )),
    "rule": "单组：scheduler = hash = block_size x dcp",
}

OUT["case2_multi_group_lcm_gcd"] = {
    "group_block_sizes": [16, 32],
    "resolved": list(resolve_kv_cache_block_sizes(
        two_group_config(16, 32), make_cfg(CacheConfig(enable_prefix_caching=True)),
    )),
    "lcm": 32, "gcd": 16,
    "rule": "scheduler=lcm(16,32)=32（num_computed_tokens 对齐不变量）hash=gcd=16",
}

OUT["case3_prefix_match_unit_override"] = {
    "group_block_sizes": [16, 32],
    "prefix_match_unit": 8,
    "resolved": list(resolve_kv_cache_block_sizes(
        two_group_config(16, 32),
        make_cfg(CacheConfig(enable_prefix_caching=True, prefix_match_unit=8)),
    )),
    "rule": "prefix_match_unit 覆盖 GCD → hash=8",
}

try:
    resolve_kv_cache_block_sizes(
        two_group_config(16, 32),
        make_cfg(CacheConfig(enable_prefix_caching=True, prefix_match_unit=5)),
    )
    OUT["case4_non_divisible_unit"] = {"prefix_match_unit": 5, "raised": False}
except ValueError:
    OUT["case4_non_divisible_unit"] = {
        "prefix_match_unit": 5,
        "16 % 5": 16 % 5, "32 % 5": 32 % 5,
        "raised": True,
        "rule": "每组块大小都必须整除 hash_block_size，否则 ValueError",
    }

OUT["case5_no_caching_falls_back"] = {
    "group_block_sizes": [16, 32],
    "enable_prefix_caching": False,
    "resolved": list(resolve_kv_cache_block_sizes(
        two_group_config(16, 32), make_cfg(CacheConfig(enable_prefix_caching=False)),
    )),
    "rule": "无前缀缓存也无 connector → hash 退回 scheduler 粒度（哈希没人消费）",
}

OUT["case6_mamba_non_align_backs_off"] = {
    "group_block_sizes": [16, 64],
    "mamba_block_size": 64,
    "cache_block_size": 16,
    "resolved": list(resolve_kv_cache_block_sizes(
        two_group_config(16, 64, spec_b=mamba_spec(64)),
        make_cfg(CacheConfig(enable_prefix_caching=True)),
    )),
    "rule": "mamba_cache_mode != align（块 64 != cache 16）破坏整除性 → hash 回退 scheduler",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m8.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
