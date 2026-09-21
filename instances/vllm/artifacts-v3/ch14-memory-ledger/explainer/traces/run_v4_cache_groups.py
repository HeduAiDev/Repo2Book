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

# ===========================================================================
# 场景 E（ch14 重写二补素材驱动）：真实 61 层配置 packed 全账 + 两组简化示教例。
# 输出独立文件 v4_cache_groups_61layer.json。同样只桩配置/指标类——布局函数
# _get_packed_kv_cache_layout / _get_kv_cache_config_packed / group_and_unify /
# _approximate_gcd / _get_kv_cache_groups_uniform_groups 全部走 pin 零改动。
# 切片四步链为 attn_utils.py:L225-L233 packing 分支的逐字复刻（host 无法
# import gpu worker 模块链；四步为纯 torch 张量操作，数值逐位可对照）。
# ===========================================================================
from vllm.v1.kv_cache_interface import KVCacheGroupSpec  # noqa: E402

out61 = {"params": {
    "pin": "vLLM v0.27.1（instances/vllm/source，行号基线）",
    "layer_config_source": "社区实读实发权重（research/hybrid-v4-background.json "
                           "entry csdn-vllm-analysis-11-dsv4-layout）：compress_ratios "
                           "61 项 = 30 c4 + 31 c128；pin 内无 61 层算例（docstring 玩具 "
                           "11+10 在 kv_cache_utils.py:L1693-L1697；tests 玩具 "
                           "[0,0,4,128,0] 在 test_indexer_deepseek_v4_slot_mapping.py:L23）",
    "layout": "fp8_ds_mla 默认（alignment=576、uint8）",
    "sliding_window": SLIDING_WINDOW,
    "state_window_formula": "状态桶窗口 = coff*compress_ratio：c4→8、c128→128"
                            "（compressor.py:L174-L176）",
    "stubbed_modules": ["vllm.config", "vllm.config.kv_events",
                        "vllm.v1.metrics", "vllm.v1.metrics.stats",
                        "vllm.v1.request", "vllm.v1.utils"],
    "stub_note": "仅配置/指标类被替身；分组、近似 GCD、packed 布局、块尺寸解析"
                 "全部走 pin 算法代码零改动；与 analyst 独立驱动 run_v4_61layer.py"
                 "互证（两份独立 trace 同源同果）",
    "slicing_note": "视图切片四步链逐字复刻自 vllm/v1/worker/gpu/attn_utils.py"
                    ":L225-L233 packing 分支（kv_raw_tensor.view(-1, block_stride)"
                    "[:, offset:offset+page_bytes].view(dtype).view(shape)）——"
                    "host 无法 import gpu worker 模块链，复刻为纯 torch 张量操作",
}}

# ---------- E1. 两组简化示教例：A 组 2 层同页 37440、B 组 8640+32832 ----------
# 页宽直接复用 pin spec 类的真实页宽（swa=37440、indexer=8640、c128 attn_state=32832），
# 层数刻意选 2/2 让读者心算跟上——布局语义（offset 累进/跨组重叠/block_stride=max）
# 与 61 层真实配置完全同构。
gA = KVCacheGroupSpec(layer_names=["a0", "a1"],
                      kv_cache_spec=UniformTypeKVCacheSpecs(
                          block_size=64,
                          kv_cache_specs={"a0": swa_spec(), "a1": swa_spec()}))
gB = KVCacheGroupSpec(layer_names=["b0", "b1"],
                      kv_cache_spec=UniformTypeKVCacheSpecs(
                          block_size=8,
                          kv_cache_specs={"b0": indexer_spec(),
                                          "b1": attn_compressor_state_spec(128)}))
toy_groups = [gA, gB]
toy_stride, toy_offsets = kcu._get_packed_kv_cache_layout(toy_groups)

AVAIL_T = 1000000  # 示教显存（字节）：刻意不整除，演示 floor 除法的余数闲置
toy_vcfg = types.SimpleNamespace(
    cache_config=types.SimpleNamespace(num_gpu_blocks_override=None))
toy_num_blocks, toy_tensors = kcu._get_kv_cache_config_packed(
    toy_vcfg, toy_groups, AVAIL_T)
toy_total = toy_stride * toy_num_blocks

# 切片四步链（attn_utils.py:L225-L233 packing 分支逐字复刻，走真实 torch 张量）
slab = torch.zeros(toy_total, dtype=torch.int8)
def slice_view(layer, offset, page_bytes, dtype):
    return slab.view(-1, toy_stride)[:, offset:offset + page_bytes].view(dtype)
toy_slice = []
for g, gid in ((gA, "A"), (gB, "B")):
    off = 0
    for ln in g.layer_names:
        spec = g.kv_cache_spec.kv_cache_specs[ln]
        page = spec.page_size_bytes
        v = slice_view(ln, off, page, spec.dtype)
        toy_slice.append({
            "layer": ln, "group": gid,
            "offset": off, "page_bytes": page,
            "view_expr": f"view(-1,{toy_stride})[:,{off}:{off + page}]",
            "out_shape": list(v.shape),
            "dtype": str(v.dtype).replace("torch.", ""),
        })
        off += page

out61["toy_two_group_packed"] = {
    "groups": [
        {"gid": "A", "layer_names": ["a0", "a1"],
         "member_pages": [37440, 37440],
         "dense_width": gA.kv_cache_spec.page_size_bytes,
         "formula": "2*37440"},
        {"gid": "B", "layer_names": ["b0", "b1"],
         "member_pages": [8640, 32832],
         "dense_width": gB.kv_cache_spec.page_size_bytes,
         "formula": "8640+32832"},
    ],
    "block_stride": toy_stride,
    "block_stride_is_max_dense": toy_stride == max(
        gA.kv_cache_spec.page_size_bytes, gB.kv_cache_spec.page_size_bytes),
    "layers_by_offset": {str(o): v for o, v in sorted(toy_offsets.items())},
    "num_distinct_offsets": len(toy_offsets),
    "offset0_shared_by": toy_offsets[0],
    "offset0_cross_group_overlap": True,
    "available_memory_bytes": AVAIL_T,
    "num_blocks": toy_num_blocks,
    "floor_remainder_bytes": AVAIL_T - toy_total,
    "total_size": toy_total,
    "tensors": [{"offset": t.offset, "block_stride": t.block_stride,
                 "size": t.size, "shared_by": t.shared_by}
                for t in toy_tensors],
    "slicing_views": toy_slice,
}

# ---------- E2. 真实 61 层配置：30 c4 + 31 c128 ----------
ratios61 = [4] * 30 + [128] * 31
specs61 = build_model(ratios61)[0]
by_kind61 = Counter(n.rsplit(".", 1)[-1] for n in specs61)
grouped61 = kcu.group_and_unify_kv_cache_specs(specs61)
assert grouped61 is not None and len(grouped61) == 4
g_info61 = []
for g in grouped61:
    pages = Counter(s.page_size_bytes for s in g.kv_cache_specs.values())
    formula = " + ".join(f"{c}*{p}" for p, c in sorted(pages.items()))
    g_info61.append({
        "num_layers": len(g.kv_cache_specs),
        "page_sizes": dict(pages),
        "dense_width": g.page_size_bytes,
        "dense_width_formula": formula,
        "num_layer_tuples": g.get_num_layer_tuples(),
        "block_size": g.block_size,
    })

tuples61 = [g.get_num_layer_tuples() for g in grouped61]
scan61 = [{"d": d, "pad": sum((d - (x % d)) % d for x in tuples61)}
          for d in range(tuples61[0], max(tuples61) + 1)]
chosen61 = kcu._approximate_gcd(tuples61, lower_bound=tuples61[0])
rounded61 = [((x + chosen61 - 1) // chosen61) * chosen61 for x in tuples61]
pad_detail = [{"bucket_tuples": x, "rounded_up": rr, "pad": rr - x}
              for x, rr in zip(tuples61, rounded61)]

groups61 = kcu._get_kv_cache_groups_uniform_groups(grouped61)
final61 = []
for i, g in enumerate(groups61):
    spec = g.kv_cache_spec
    pages = (Counter(s.page_size_bytes for s in spec.kv_cache_specs.values())
             if isinstance(spec, UniformTypeKVCacheSpecs) else Counter())
    formula = " + ".join(f"{c}*{p}" for p, c in sorted(pages.items()))
    final61.append({
        "gid": i, "num_layers": len(g.layer_names),
        "block_size": spec.block_size,
        "member_pages": dict(pages),
        "dense_width": spec.page_size_bytes if isinstance(
            spec, UniformTypeKVCacheSpecs) else None,
        "dense_width_formula": formula or None,
    })

stride61, offsets61 = kcu._get_packed_kv_cache_layout(groups61)
dense_list61 = [g["dense_width"] for g in final61]
AVAIL_10G = 10737418240  # 10 GiB 示教显存（社区文章 csdn-vllm-analysis-11 同口径）
cfg61 = types.SimpleNamespace(
    cache_config=types.SimpleNamespace(num_gpu_blocks_override=None))
nb_10g, _ = kcu._get_kv_cache_config_packed(cfg61, groups61, AVAIL_10G)

out61["layer61"] = {
    "compress_ratios_shape": "30 c4 + 31 c128（社区实读实发权重，61 层）",
    "spec_census": {
        "total_specs": len(specs61),
        "by_kind": dict(by_kind61),
        "total_formula": "30*5 + 31*3 = 150 + 93",
        "distinct_page_sizes": sorted({s.page_size_bytes for s in specs61.values()}),
    },
    "group_and_unify": g_info61,
    "num_layer_tuples_list": tuples61,
    "approx_gcd": {"scan": scan61, "chosen": chosen61,
                   "pad_total": sum(rr - x for x, rr in zip(tuples61, rounded61)),
                   "rounded_up": rounded61, "pad_detail": pad_detail},
    "final_groups": final61,
    "per_group_dense_width": dense_list61,
    "packed_layout": {
        "block_stride": stride61,
        "block_stride_is_max_dense": stride61 == max(dense_list61),
        "num_distinct_offsets": len(offsets61),
        "offset0_num_layers": len(offsets61[0]),
    },
    "num_blocks_10gib": {
        "available_bytes": AVAIL_10G, "available_gib": 10,
        "num_blocks": nb_10g,
        "note": "num_blocks = available_memory // block_stride "
                "(kv_cache_utils.py:L1342)",
    },
    "community_check": {
        "expected_dense_widths": [1435968, 1160640, 1123200, 1244160, 1017792],
        "match": dense_list61 == [1435968, 1160640, 1123200, 1244160, 1017792],
        "expected_num_blocks_10gib": 7477,
        "num_blocks_match": nb_10g == 7477,
        "source": "research/hybrid-v4-background.json csdn-vllm-analysis-11-dsv4-layout",
    },
}

path61 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "v4_cache_groups_61layer.json")
with open(path61, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out61, f, ensure_ascii=False, indent=1)
print("wrote", path61)
print("toy: stride", toy_stride, "offsets", {o: v for o, v in sorted(toy_offsets.items())},
      "num_blocks", toy_num_blocks, "total", toy_total)
print("61L: tuples", tuples61, "d =", chosen61, "pad =",
      out61["layer61"]["approx_gcd"]["pad_total"])
print("61L: dense", dense_list61, "stride", stride61,
      "10GiB blocks", nb_10g, "community match",
      out61["layer61"]["community_check"]["match"],
      out61["layer61"]["community_check"]["num_blocks_match"])
