# run_m5.py — m5 混合组化 get_kv_cache_groups 驱动脚本
# 三场景：① Gemma3 式 10 SWA + 2 full（5:1 模式 ×2）→ 6 组 × 2 层、
# layers[i::n] 交错；② 12 SW + 13 full → 1.5 启发式取 13 补成 13/13
# （padding warning「may waste at most 8.33%」）；③ --disable-hybrid-kv-cache-
# manager 回退：SWA 当 full 分配（warning 原话）。场景 2 捕 caplog 记录 warning。
import io
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from implementation.cache import CacheConfig
from implementation.config import ModelConfig, ParallelConfig, VllmConfig
from implementation.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
)
from implementation.kv_cache_utils import get_kv_cache_groups
from implementation.scheduler_config import SchedulerConfig

OUT = {}


def full_spec():
    return FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16
    )


def swa_spec(window=512):
    return SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.float16,
        sliding_window=window,
    )


def make_cfg(scheduler=None):
    return VllmConfig(
        model_config=ModelConfig(max_model_len=4096, original_max_model_len=4096),
        cache_config=CacheConfig(enable_prefix_caching=False),
        scheduler_config=scheduler or SchedulerConfig(),
        parallel_config=ParallelConfig(),
    )


def gemma3_like(num_swa, num_full, window=512):
    spec = {}
    for i in range(num_swa):
        spec[f"model.layers.{i}.self_attn.attn"] = swa_spec(window)
    for j in range(num_full):
        spec[f"model.layers.{num_swa + j}.self_attn.attn"] = full_spec()
    return spec


def brief(groups):
    return [
        {
            "group": i + 1,
            "layers": [n.split(".")[2] for n in g.layer_names],
            "spec_type": type(g.kv_cache_spec).__name__,
        }
        for i, g in enumerate(groups)
    ]


# --------------------------------------------- 场景 1：uniform 单组（多数模型）
cfg = make_cfg()
uniform = {f"model.layers.{i}.self_attn.attn": full_spec() for i in range(32)}
groups_u = get_kv_cache_groups(cfg, dict(uniform))
OUT["uniform_model"] = {
    "num_layers": 32,
    "num_groups": len(groups_u),
    "layers_per_group": len(groups_u[0].layer_names),
}

# --------------------------------------------- 场景 2：Gemma3 式 10 SWA + 2 full
cfg = make_cfg()
spec2 = gemma3_like(num_swa=10, num_full=2)
groups2 = get_kv_cache_groups(cfg, dict(spec2))
OUT["gemma3_10swa_2full"] = {
    "swa_layers": 10,
    "full_layers": 2,
    "group_size": 2,
    "num_groups": len(groups2),
    "groups": brief(groups2),
    "interleave_rule": "layers[i::num_groups] 进第 i 组（PP 时避免某 stage 出空组）",
}

# --------------------------------------------- 场景 3：12 SW + 13 full（1.5 启发式）
root = logging.getLogger()
root.setLevel(logging.WARNING)
buf = io.StringIO()
handler = logging.StreamHandler(buf)
root.addHandler(handler)
cfg = make_cfg()
spec3 = gemma3_like(num_swa=12, num_full=13)
groups3 = get_kv_cache_groups(cfg, dict(spec3))
root.removeHandler(handler)
OUT["gptoss_12sw_13full"] = {
    "swa_layers": 12,
    "full_layers": 13,
    "min_layers": 12,
    "max_layers": 13,
    "one_point_five_x_min": 12 * 1.5,
    "heuristic": "max(13) < 1.5 x min(12)=18 → group_size 取 13（而非 12）",
    "num_groups": len(groups3),
    "groups": brief(groups3),
    "padding_layers_added": 13 - 12,
    "waste_bound_pct": round((13 - 12) / 12 * 100, 2),
    "warning": next((l for l in buf.getvalue().splitlines() if "padding" in l), ""),
}

# --------------------------------------------- 场景 4：disable 回退（SWA 当 full）
buf2 = io.StringIO()
handler2 = logging.StreamHandler(buf2)
root.addHandler(handler2)
cfg = make_cfg(scheduler=SchedulerConfig(disable_hybrid_kv_cache_manager=True))
spec4 = gemma3_like(num_swa=5, num_full=1)
groups4 = get_kv_cache_groups(cfg, spec4)
root.removeHandler(handler2)
OUT["disable_fallback"] = {
    "swa_layers": 5,
    "full_layers": 1,
    "num_groups": len(groups4),
    "merged_spec_type": type(groups4[0].kv_cache_spec).__name__,
    "merged_sliding_window_recorded": groups4[0].kv_cache_spec.sliding_window,
    "warning_excerpt": "Hybrid KV cache manager is disabled ... we do not enable "
                       "any optimizations for saving KV cache memory (e.g., "
                       "dropping the KV cache outside the sliding window).",
    "warning_full": buf2.getvalue().strip().splitlines()[0] if buf2.getvalue() else "",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m5.json"), "w",
          encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print("ok")
