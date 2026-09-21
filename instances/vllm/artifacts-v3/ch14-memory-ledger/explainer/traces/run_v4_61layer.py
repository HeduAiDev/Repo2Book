# ch14 重写二补：社区实读 61 层配置（30 c4 + 31 c128，fp8_ds_mla）对 pin 复算。
# 驱动复用 run_v4_cache_groups.py 的桩替身（仅配置/指标类被替身；分组、近似
# GCD、packed 布局全部走 pin 算法代码零改动）。核四件事：
#   1. group_and_unify 四组 num_layer_tuples 是否 [31, 61, 30, 31]
#   2. _approximate_gcd 是否选 d=31、总 pad=2
#   3. 终五组 dense width 是否 1,435,968/1,160,640/1,123,200/1,244,160/1,017,792
#   4. packed block_stride = max(dense) 与 num_blocks 整除关系
# 层分布口径：社区文章 csdn-vllm-analysis-11-dsv4-layout 实读实发权重
# （compress_ratios 61 项 = 30 个 4 + 31 个 128；pin 内无 61 层算例，仅有
# kv_cache_utils.py:L1693-L1697 docstring 玩具 11+10 与 tests 5 层玩具）。
import json
import os
import sys
import types
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PIN_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "source"))
sys.path.insert(0, PIN_ROOT)
os.environ["PYTHONHASHSEED"] = "0"


class _DummyMeta(type):
    def __getattr__(cls, name):
        return _Dummy


class _Dummy(metaclass=_DummyMeta):
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return _Dummy()


class _CfgModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


cfg = _CfgModule("vllm.config")
cfg.VllmConfig = _Dummy
cfg.CacheConfig = _Dummy
cfg.get_current_vllm_config = lambda *a, **k: None
sys.modules["vllm.config"] = cfg
cfg.__path__ = []
kv_events_stub = types.ModuleType("vllm.config.kv_events")
kv_events_stub.KVEventsConfig = _Dummy
sys.modules["vllm.config.kv_events"] = kv_events_stub
metrics_stub = types.ModuleType("vllm.v1.metrics")
metrics_stub.__path__ = []
sys.modules["vllm.v1.metrics"] = metrics_stub
stats_stub = types.ModuleType("vllm.v1.metrics.stats")
stats_stub.KVCacheEvictionEvent = _Dummy
sys.modules["vllm.v1.metrics.stats"] = stats_stub
req_stub = types.ModuleType("vllm.v1.request")
req_stub.Request = _Dummy
sys.modules["vllm.v1.request"] = req_stub
v1u = types.ModuleType("vllm.v1.utils")
v1u.tensor_data = lambda *a, **k: None
sys.modules["vllm.v1.utils"] = v1u

import torch  # noqa: E402
from vllm.v1.core import kv_cache_utils as kcu  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    UniformTypeKVCacheSpecs,
)

FP8_DS_MLA = "fp8_ds_mla"
SLIDING_WINDOW = 8192
BLOCK_SIZE = 256


def swa_spec():
    return __import__("vllm.v1.kv_cache_interface", fromlist=["x"]).SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=SLIDING_WINDOW, cache_dtype_str=FP8_DS_MLA,
        alignment=576, model_version="deepseek_v4")


def main_kv_spec(r):
    return __import__("vllm.v1.kv_cache_interface", fromlist=["x"]).MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        compress_ratio=r, cache_dtype_str=FP8_DS_MLA, alignment=576,
        model_version="deepseek_v4")


def indexer_spec():
    return __import__("vllm.v1.kv_cache_interface", fromlist=["x"]).MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=132, dtype=torch.uint8,
        compress_ratio=4, alignment=576)


def attn_compressor_state_spec(r):
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    coff = 1 + (r == 4)
    return SlidingWindowMLASpec(
        block_size=4 if r == 4 else 8, num_kv_heads=1,
        head_size=2 * coff * 512, dtype=torch.float32,
        sliding_window=coff * r, alignment=576)


def indexer_compressor_state_spec():
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
    return SlidingWindowMLASpec(
        block_size=4, num_kv_heads=1, head_size=2 * 2 * 128,
        dtype=torch.float32, sliding_window=8, alignment=576)


def layer_specs(r):
    out = {"swa": swa_spec()}
    if r <= 1:
        return out
    out["main"] = main_kv_spec(r)
    if r == 4:
        out["indexer"] = indexer_spec()
        out["indexer_state"] = indexer_compressor_state_spec()
    out["attn_state"] = attn_compressor_state_spec(r)
    return out


def build_model(compress_ratios):
    specs = {}
    for li, r in enumerate(compress_ratios):
        eff = max(1, r)
        for k, s in layer_specs(eff).items():
            specs[f"model.layers.{li}.{k}"] = s
    return specs


out = {"params": {
    "pin": "vLLM v0.27.1（instances/vllm/source，行号基线）",
    "layer_config_source": "社区实读实发权重（csdn-vllm-analysis-11-dsv4-layout）："
                           "compress_ratios 61 项 = 30 c4 + 31 c128；"
                           "pin 内无 61 层算例（docstring 玩具 11+10 在 "
                           "kv_cache_utils.py:L1693-L1697；tests 玩具 [0,0,4,128,0] 在 "
                           "test_indexer_deepseek_v4_slot_mapping.py:L23）",
    "layout": "fp8_ds_mla 默认（alignment=576、uint8）",
    "stub_note": "同 run_v4_cache_groups.py：仅配置/指标类被替身；分组、近似 GCD、"
                 "packed 布局全部走 pin 算法代码零改动",
}}

ratios = [4] * 30 + [128] * 31
specs = build_model(ratios)
out["spec_census"] = {
    "total_specs": len(specs),
    "by_kind": dict(Counter(n.rsplit(".", 1)[-1] for n in specs)),
    "distinct_page_sizes": sorted({s.page_size_bytes for s in specs.values()}),
}

grouped = kcu.group_and_unify_kv_cache_specs(specs)
assert grouped is not None and len(grouped) == 4
g_info = []
for g in grouped:
    kinds = Counter(
        f"{type(s).__name__}({s.block_size},{s.sliding_window})"
        if isinstance(s, __import__("vllm.v1.kv_cache_interface", fromlist=["x"]).SlidingWindowMLASpec)
        else f"{type(s).__name__}({s.block_size})"
        for s in g.kv_cache_specs.values())
    g_info.append({
        "num_layers": len(g.kv_cache_specs),
        "member_kinds": dict(kinds),
        "page_sizes": dict(Counter(
            s.page_size_bytes for s in g.kv_cache_specs.values())),
        "dense_width": g.page_size_bytes,
        "num_layer_tuples": g.get_num_layer_tuples(),
        "block_size": g.block_size,
    })

tuples = [g.get_num_layer_tuples() for g in grouped]
chosen = kcu._approximate_gcd(tuples, lower_bound=tuples[0])
pad_total = sum((chosen - (x % chosen)) % chosen for x in tuples)
rounded = [((x + chosen - 1) // chosen) * chosen for x in tuples]

groups = kcu._get_kv_cache_groups_uniform_groups(grouped)
final = []
for i, g in enumerate(groups):
    spec = g.kv_cache_spec
    dense = spec.page_size_bytes if isinstance(spec, UniformTypeKVCacheSpecs) else None
    final.append({
        "gid": i,
        "num_layers": len(g.layer_names),
        "spec": (f"Uniform[{dict(Counter(type(s).__name__ for s in spec.kv_cache_specs.values()))}]"
                 if isinstance(spec, UniformTypeKVCacheSpecs) else type(spec).__name__),
        "block_size": spec.block_size,
        "member_pages": dict(Counter(
            s.page_size_bytes for s in spec.kv_cache_specs.values()))
        if isinstance(spec, UniformTypeKVCacheSpecs) else None,
        "dense_width": dense,
    })

block_stride, layers_by_offset = kcu._get_packed_kv_cache_layout(groups)
per_group_dense = [g["dense_width"] for g in final]

out["grouping"] = {
    "group_and_unify": g_info,
    "num_layer_tuples_list": tuples,
    "approx_gcd": {"chosen": chosen, "pad_total": pad_total,
                   "rounded_up": rounded},
    "final_groups": final,
    "per_group_dense_width": per_group_dense,
    "packed_layout": {
        "block_stride": block_stride,
        "block_stride_is_max_dense": block_stride == max(per_group_dense),
        "num_distinct_offsets": len(layers_by_offset),
        "offset0_num_layers": len(layers_by_offset[0]),
    },
    "community_check": {
        "expected_dense_widths": [1435968, 1160640, 1123200, 1244160, 1017792],
        "match": per_group_dense == [1435968, 1160640, 1123200, 1244160, 1017792],
        "expected_tuples": [31, 61, 30, 31],
        "tuples_match": tuples == [31, 61, 30, 31],
        "expected_gcd": {"d": 31, "pad": 2},
        "gcd_match": chosen == 31 and pad_total == 2,
    },
    "num_blocks_divisor": {
        "note": "num_blocks = available_memory // block_stride "
                "(kv_cache_utils.py:L1342)",
        "block_stride": block_stride,
    },
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_61layer.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", path)
print("tuples:", tuples, "| gcd d =", chosen, "pad =", pad_total)
print("dense widths:", per_group_dense)
print("block_stride:", block_stride,
      "| community match:",
      out["grouping"]["community_check"]["match"],
      out["grouping"]["community_check"]["gcd_match"])
