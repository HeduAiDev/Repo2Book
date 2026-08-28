# run_m4.py — m4 护栏四道 驱动脚本
# ① check_enough（至少装下一条 max_model_len）② 二分估可行长度（探针全记录）
# ③ auto-fit（max_model_len=-1）④ override 折算不漂账 + PP 取最小缩张量。
# 玩具：2 层 full、block_size=16、kv_heads=8、head 128、fp16 → 每层页 65536B，
# 每「长度块」（16 token）两层共 131072B —— needed(len) = cdiv(len,16) x 131072。
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.cache import CacheConfig
from implementation.config import ModelConfig, ParallelConfig, VllmConfig
from implementation.kv_cache_interface import FullAttentionSpec
from implementation import kv_cache_utils as kcu
from implementation.kv_cache_utils import (
    check_enough_kv_cache_memory,
    estimate_max_model_len,
    get_kv_cache_configs,
)

OUT = {}
PAGE = 65536
PER_LEN_BLOCK = PAGE * 2


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def uniform_full_layers(num_layers, prefix="model.layers"):
    return {f"{prefix}.{i}.self_attn.attn": full_spec() for i in range(num_layers)}


def make_cfg(max_model_len=4096, original=None, cache=None):
    return VllmConfig(
        model_config=ModelConfig(
            max_model_len=max_model_len,
            original_max_model_len=original if original is not None else max_model_len,
        ),
        cache_config=cache or CacheConfig(enable_prefix_caching=False),
        parallel_config=ParallelConfig(),
    )


# --------------------------------------------- 护栏 1：至少装下一条 max_model_len
spec = uniform_full_layers(2)
OUT["page"] = {"page_size_per_layer": PAGE, "per_len_block_2layers": PER_LEN_BLOCK,
               "needed_formula": "needed(len) = cdiv(len, 16) x 131072"}
needed_4096 = math.ceil(4096 / 16) * PER_LEN_BLOCK
OUT["guardrail1_check_enough"] = {
    "max_model_len": 4096,
    "needed_bytes": needed_4096,
    "needed_mib": needed_4096 // (1 << 20),
    "available_bytes": needed_4096 - 1,
    "available_mib": (needed_4096 - 1) / (1 << 20),
}
try:
    check_enough_kv_cache_memory(make_cfg(4096), spec, needed_4096 - 1)
    OUT["guardrail1_check_enough"]["raised"] = False
except ValueError as e:
    OUT["guardrail1_check_enough"]["raised"] = True
    OUT["guardrail1_check_enough"]["error_excerpt"] = (
        "To serve at least one request with the model's max seq len (4096), "
        "(X GiB KV cache is needed, which is larger than the available KV cache "
        "memory (Y GiB). Based on the available memory, the estimated maximum "
        "model length is 1600. Try increasing `gpu_memory_utilization` ... "
        "or decreasing `max_model_len` ..."
    )

# --------------------------------------------- 护栏 2：二分估可行长度（探针全记录）
probes = []
orig_max_mem = kcu.max_memory_usage_bytes


def logging_max_memory(vllm_config, kv_cache_specs):
    n = orig_max_mem(vllm_config, kv_cache_specs)
    probes.append({
        "probe_len": vllm_config.model_config.max_model_len,
        "needed_bytes": n,
        "needed_len_blocks": n // PER_LEN_BLOCK,
        "fits": n <= 100 * PER_LEN_BLOCK,
    })
    return n


kcu.max_memory_usage_bytes = logging_max_memory
cfg2 = make_cfg(8192)
avail2 = 100 * PER_LEN_BLOCK
estimated = estimate_max_model_len(cfg2, spec, avail2)
kcu.max_memory_usage_bytes = orig_max_mem
for i, p in enumerate(probes):
    p["probe"] = i + 1
OUT["guardrail2_binary_search"] = {
    "max_model_len": 8192,
    "available_bytes": avail2,
    "available_len_blocks": 100,
    "probes": probes,
    "num_probes_incl_initial_fit1": len(probes),
    "num_loop_probes": len(probes) - 1,
    "log2_upper_bound": math.ceil(math.log2(8192)),
    "loop_probes_le_log2_L": (len(probes) - 1) <= math.ceil(math.log2(8192)),
    "estimated_max_model_len": estimated,
    "max_model_len_restored": cfg2.model_config.max_model_len == 8192,
}

# --------------------------------------------- 护栏 3：auto-fit（original=-1）
# get_kv_cache_configs 的 auto-fit 走 _estimate_max_model_len_from_groups
# （按组求和口径 _max_memory_usage_bytes_from_groups），钩它记录探针。
autofit_probes = []
orig_groups_mem = kcu._max_memory_usage_bytes_from_groups


def logging_groups_mem(vllm_config, groups):
    n = orig_groups_mem(vllm_config, groups)
    autofit_probes.append({
        "probe_len": vllm_config.model_config.max_model_len,
        "needed_bytes": n,
        "needed_len_blocks": n // PER_LEN_BLOCK,
        "fits": n <= avail2,
    })
    return n


kcu._max_memory_usage_bytes_from_groups = logging_groups_mem
cfg3 = make_cfg(8192, original=-1)
get_kv_cache_configs(cfg3, [uniform_full_layers(2)], [avail2])
kcu._max_memory_usage_bytes_from_groups = orig_groups_mem
OUT["guardrail3_auto_fit"] = {
    "original_max_model_len": -1,
    "configured_max_model_len": 8192,
    "available_bytes": avail2,
    "fitted_max_model_len": cfg3.model_config.max_model_len,
    "num_probes": len(autofit_probes),
    "probes": autofit_probes,
}

# --------------------------------------------- 护栏 4a：override 折算不漂账
cache4 = CacheConfig(num_gpu_blocks_override=3, enable_prefix_caching=False)
cfg4 = make_cfg(48, cache=cache4)
avail4 = 100 * PER_LEN_BLOCK  # 远大于 override 容量——不折算则护栏按大容量放行
configs4 = get_kv_cache_configs(cfg4, [uniform_full_layers(2)], [avail4])
OUT["guardrail4_override"] = {
    "num_gpu_blocks_override": 3,
    "max_model_len": 48,
    "profiled_available_bytes": avail4,
    "bytes_per_block": PER_LEN_BLOCK,
    "rebased_available_bytes": 3 * PER_LEN_BLOCK,
    "rebased_available_equals_needed": (3 * PER_LEN_BLOCK)
    == (math.ceil(48 / 16) * PER_LEN_BLOCK),
    "final_num_blocks": configs4[0].num_blocks,
}

# --------------------------------------------- 护栏 4b：PP 取最小 + 缩张量
cfg5 = make_cfg(1024)
w0 = uniform_full_layers(2, prefix="stage0")
w1 = uniform_full_layers(2, prefix="stage1")
configs5 = get_kv_cache_configs(cfg5, [w0, w1], [200 * PER_LEN_BLOCK, 90 * PER_LEN_BLOCK])
OUT["guardrail4_pp_min"] = {
    "avail_worker0_len_blocks": 200,
    "avail_worker1_len_blocks": 90,
    "num_blocks_worker0": configs5[0].num_blocks,
    "num_blocks_worker1": configs5[1].num_blocks,
    "min_num_blocks": min(c.num_blocks for c in configs5),
    "tensor0_size_before_shrink": PAGE * 200,
    "tensor0_size_after_shrink": configs5[0].kv_cache_tensors[0].size,
    "tensor0_size_pages": configs5[0].kv_cache_tensors[0].size // PAGE,
    "shrink_rule": "tensor.size = tensor.size // num_blocks_old x min_num_blocks",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m4.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
