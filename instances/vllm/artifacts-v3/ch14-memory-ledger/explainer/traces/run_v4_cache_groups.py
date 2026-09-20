# ch15 v4-cache-spec-census / v4-cache-uniform-groups 驱动：DeepSeek V4 缓存面的
# spec census、装包分组、近似 GCD、packed 布局与两套块尺寸——全部跑 pin 源码
# （vLLM v0.27.1，vllm/v1/core/kv_cache_utils.py 与 vllm/v1/kv_cache_interface.py
# 零改动；仅 vllm.config / vllm.v1.request / vllm.v1.utils / vllm.v1.metrics 四处
# 配置/指标桩替身绕开 transformers v5 与 GPU 依赖——算法路径不经过它们）。
# 场景：
#   A. 玩具 census：compress_ratios=[0,0,4,128,0]（官方回归测试同款玩具表，
#      tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py:L23）→
#      逐层复刻五类 spec 生产者，数每层报几条账、全模型四种页宽。
#   B. 官方 docstring 算例形状：11 个 c4 层 + 10 个 c128 层（sliding_window=8192、
#      fp8_ds_mla 默认布局）→ group_and_unify_kv_cache_specs → 4 个
#      UniformTypeKVCacheSpecs → _approximate_gcd → 终 5 组 → packed 布局
#      block_stride → resolve_kv_cache_block_sizes 得 (scheduler, hash)=(256, 4)。
#   C. 对照：等页统一支路 unify_kv_cache_spec_page_size 对同一 85 条 spec 必然
#      NotImplementedError（三个较小页宽都除不尽最大页 37440）——真跑真抛。
#   D. generate_scheduler_kv_cache_config：UniformType → 任取内层 spec 的解包。
import hashlib
import json
import math
import os
import sys
import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# traces → explainer → ch15-prefix-caching → artifacts-v3 → vllm(实例) → source
PIN_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "source"))
sys.path.insert(0, PIN_ROOT)
os.environ["PYTHONHASHSEED"] = "0"


# ---------- 桩：绕开 host 没有的重配置依赖（算法代码零改动） ----------
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
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

FP8_DS_MLA = "fp8_ds_mla"
SLIDING_WINDOW = 8192  # DeepseekV4Attention 的 window_size（全层同一个）
BLOCK_SIZE = 256       # 管理面 block_size（DeepseekSparseSWABackend 偏好 256）


# ---------- 五类 spec 生产者：逐字复刻 pin 的构造实参 ----------
def swa_spec():
    """DeepseekV4SWACache.get_kv_cache_spec（sparse_swa.py:L87-L102，fp8_ds_mla）。"""
    return SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=SLIDING_WINDOW, cache_dtype_str=FP8_DS_MLA,
        alignment=576, model_version="deepseek_v4")


def main_kv_spec(r):
    """DeepseekV4Attention.get_kv_cache_spec 主 KV（attention.py:L664-L674）。"""
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        compress_ratio=r, cache_dtype_str=FP8_DS_MLA, alignment=576,
        model_version="deepseek_v4")


def indexer_spec():
    """DeepseekV4IndexerCache.get_kv_cache_spec（attention.py:L698-L710；
    fp8 布局 head_dim=128+128//32*4=132，attention.py:L787-L791）。"""
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=132, dtype=torch.uint8,
        compress_ratio=4, alignment=576)


def attn_compressor_state_spec(r):
    """注意力压缩器的 CompressorStateCache（compressor.py:L173-L203）：
    state_dim=2*coff*512，coff=1+(r==4)；c4→(块4,窗8)、c128→(块8,窗128)。"""
    coff = 1 + (r == 4)
    return SlidingWindowMLASpec(
        block_size=4 if r == 4 else 8, num_kv_heads=1,
        head_size=2 * coff * 512, dtype=torch.float32,
        sliding_window=coff * r, alignment=576)


def indexer_compressor_state_spec():
    """indexer 内嵌压缩器：head=128 → state_dim=2*2*128=512（compressor.py:L295-L300
    的 state_dim 公式，head_dim=128 分支）。"""
    return SlidingWindowMLASpec(
        block_size=4, num_kv_heads=1, head_size=2 * 2 * 128,
        dtype=torch.float32, sliding_window=8, alignment=576)


def layer_specs(r):
    """一层 DeepseekV4Attention 名下的全部缓存账（census 的逐层口径）。"""
    out = {"swa": swa_spec()}  # 每层都挂：c1 层唯一的缓存面
    if r <= 1:
        return out  # attention.py:L655-L659：主 KV 返回 None
    out["main"] = main_kv_spec(r)
    if r == 4:
        out["indexer"] = indexer_spec()
        out["indexer_state"] = indexer_compressor_state_spec()
    out["attn_state"] = attn_compressor_state_spec(r)
    return out


def spec_row(name, kind, spec):
    return {"layer": name, "kind": kind, "class": type(spec).__name__,
            "block_size": spec.block_size,
            "storage_block_size": spec.storage_block_size,
            "real_page_size_bytes": spec.real_page_size_bytes,
            "page_size_bytes": spec.page_size_bytes}


def build_model(compress_ratios):
    specs, census = {}, []
    for li, r in enumerate(compress_ratios):
        eff = max(1, r)  # attention.py:L210-L213：max(1,·) 把 0 折成 1
        per_layer = layer_specs(eff)
        census.append({"layer": li, "compress_ratio": r, "effective": eff,
                       "num_specs": len(per_layer),
                       "specs": [spec_row(f"L{li}", k, s)
                                 for k, s in per_layer.items()]})
        for k, s in per_layer.items():
            specs[f"model.layers.{li}.{k}"] = s
    return specs, census


out = {"params": {
    "pin": "vLLM v0.27.1（instances/vllm/source，行号基线）",
    "layout": "fp8_ds_mla 默认（alignment=576、uint8）",
    "sliding_window": SLIDING_WINDOW,
    "block_size_mgmt": BLOCK_SIZE,
    "stubbed_modules": ["vllm.config", "vllm.config.kv_events",
                        "vllm.v1.metrics", "vllm.v1.metrics.stats",
                        "vllm.v1.request", "vllm.v1.utils"],
    "stub_note": "仅配置/指标类被替身；分组、近似 GCD、packed 布局、块尺寸解析"
                 "全部走 pin 算法代码零改动",
}}

# ---------- A. 玩具 census：[0,0,4,128,0] ----------
ratios = [0, 0, 4, 128, 0]
specs_toy, census_toy = build_model(ratios)
pages_toy = sorted({s.page_size_bytes for s in specs_toy.values()})
out["census_toy"] = {
    "compress_ratios": ratios,
    "source_of_ratios": "tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py:L23",
    "per_layer": [{"layer": c["layer"], "ratio": c["compress_ratio"],
                   "num_specs": c["num_specs"],
                   "kinds": [r["kind"] for r in c["specs"]]} for c in census_toy],
    "total_specs": sum(c["num_specs"] for c in census_toy),
    "distinct_page_sizes": pages_toy,
    "page_census_detail": [
        {"page": p, "pages_of": sorted({
            f'{r["kind"]}(r={c["effective"]})' if r["kind"] in ("main", "attn_state", "indexer_state")
            else r["kind"]
            for c in census_toy for r in c["specs"] if r["page_size_bytes"] == p})}
        for p in pages_toy],
}

# ---------- B. 官方算例形状：11 c4 + 10 c128 ----------
ratios_big = [4] * 11 + [128] * 10
specs_big, census_big = build_model(ratios_big)
pages_big = sorted({s.page_size_bytes for s in specs_big.values()})
from collections import Counter  # noqa: E402
page_counts = Counter(s.page_size_bytes for s in specs_big.values())

grouped = kcu.group_and_unify_kv_cache_specs(specs_big)
assert grouped is not None and len(grouped) == 4
g_info = []
for g in grouped:
    kinds = Counter(
        f"{type(s).__name__}({s.block_size},{s.sliding_window})"
        if isinstance(s, SlidingWindowMLASpec) else f"{type(s).__name__}({s.block_size})"
        for s in g.kv_cache_specs.values())
    g_info.append({
        "num_layers": len(g.kv_cache_specs),
        "member_kinds": dict(kinds),
        "page_sizes": dict(Counter(
            s.page_size_bytes for s in g.kv_cache_specs.values())),
        "group_page_sum": g.page_size_bytes,
        "num_layer_tuples": g.get_num_layer_tuples(),
        "block_size": g.block_size,
    })

tuples = [g.get_num_layer_tuples() for g in grouped]
approx_scan = []
for d in range(tuples[0], max(tuples) + 1):
    pad = sum((d - (x % d)) % d for x in tuples)
    approx_scan.append({"d": d, "pad": pad})
chosen = kcu._approximate_gcd(tuples, lower_bound=tuples[0])
rounded = [((x + chosen - 1) // chosen) * chosen for x in tuples]

groups = kcu._get_kv_cache_groups_uniform_groups(grouped)
final = [{
    "gid": i,
    "num_layers": len(g.layer_names),
    "spec": (f"Uniform[{dict(Counter(type(s).__name__ for s in g.kv_cache_spec.kv_cache_specs.values()))}]"
             if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
             else type(g.kv_cache_spec).__name__),
    "block_size": g.kv_cache_spec.block_size,
    "member_pages": dict(Counter(
        s.page_size_bytes for s in g.kv_cache_spec.kv_cache_specs.values()))
    if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else None,
} for i, g in enumerate(groups)]

block_stride, layers_by_offset = kcu._get_packed_kv_cache_layout(groups)
offset_sharing = [{"offset": o, "num_layers": len(v),
                   "groups_sharing": len({n.rsplit(".", 2)[-2] if ".state_cache" in n or ".k_cache" in n else n
                                          for n in v})}
                  for o, v in sorted(layers_by_offset.items())]
# offset 0 上到底压了几个组的层：按『每组的第一个层名』判
first_layers = {g.layer_names[0] for g in groups}
offset0_layers = layers_by_offset[0]
mla_width = 11 * 37440 + 11 * 8640 + 10 * 1728

# resolve_kv_cache_block_sizes：调度器/哈希两套块尺寸
ccfg = types.SimpleNamespace(block_size=256, prefix_match_unit=None,
                             enable_prefix_caching=True)
pcfg = types.SimpleNamespace(decode_context_parallel_size=1)
vcfg = types.SimpleNamespace(cache_config=ccfg, parallel_config=pcfg,
                             kv_transfer_config=None)
kv_cfg = KVCacheConfig(num_blocks=4, kv_cache_tensors=[],
                       kv_cache_groups=groups)
scheduler_bs, hash_bs = kcu.resolve_kv_cache_block_sizes(kv_cfg, vcfg)
group_bs = [g.kv_cache_spec.block_size for g in groups]

# ---------- C. 等页统一支路必败（真跑真抛） ----------
unify_err = None
try:
    kcu.unify_kv_cache_spec_page_size(specs_big)
except NotImplementedError as e:
    unify_err = str(e).splitlines()[0]
divis = {p: 37440 % p for p in pages_big if p != 37440}

# ---------- D. 调度器侧解包 ----------
sched_cfg = kcu.generate_scheduler_kv_cache_config([kv_cfg])
unwrapped = [{"gid": i, "inner_class": type(g.kv_cache_spec).__name__,
              "block_size": g.kv_cache_spec.block_size}
             for i, g in enumerate(sched_cfg.kv_cache_groups)]

out["groups_11_10"] = {
    "compress_ratios_shape": "11 c4 + 10 c128（kv_cache_utils.py:L1693-L1697 官方算例）",
    "total_specs": len(specs_big),
    "distinct_page_sizes": pages_big,
    "page_counts": dict(page_counts),
    "group_and_unify": g_info,
    "approx_gcd": {"values": tuples, "lower_bound": tuples[0],
                   "scan": approx_scan, "chosen": chosen,
                   "rounded_up": rounded},
    "final_groups": final,
    "packed_layout": {
        "block_stride": block_stride,
        "num_distinct_offsets": len(layers_by_offset),
        "offset0_num_layers": len(offset0_layers),
        "offset0_groups_sharing": sum(1 for fl in first_layers if fl in offset0_layers),
        "mla_group_dense_width": mla_width,
        "mla_width_formula": "11*37440 + 11*8640 + 10*1728",
    },
    "block_sizes": {
        "group_block_sizes": group_bs,
        "scheduler_block_size": scheduler_bs,
        "hash_block_size": hash_bs,
        "lcm": math.lcm(*group_bs),
        "gcd": math.gcd(*group_bs),
        "prefix_match_unit": None,
    },
    "page_unify_fallback": {
        "error": unify_err,
        "remainders_of_max": divis,
    },
    "scheduler_unwrap": unwrapped,
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v4_cache_groups.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", path)
print("census_toy: total", out["census_toy"]["total_specs"], "specs, pages",
      out["census_toy"]["distinct_page_sizes"])
print("groups:", [g["num_layers"] for g in final], "stride", block_stride,
      "sched/hash", scheduler_bs, hash_bs)
print("unify raises:", unify_err is not None, "| remainders:", divis)
