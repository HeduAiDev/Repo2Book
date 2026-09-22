# ch14 三补素材驱动：tensor-declaration-account——三类模型「KVCacheTensor 声明数 vs
# 物理分配数」对账（Uniform 单型 / Grouped 混合同页 / V4 packed）。全部跑 pin 真码
# （vLLM v0.27.1，vllm/v1/core/kv_cache_utils.py 零改动；仅 vllm.config /
# vllm.v1.metrics / vllm.v1.request / vllm.v1.utils 四处配置/指标桩替身——同
# run_v4_cache_groups.py，算法路径不经过它们）。场景：
#   U. Uniform 单型：32 层同 FullAttentionSpec（block16×8头×(128+128)×bf16 → 页 65536）
#      → get_kv_cache_groups → get_kv_cache_config_from_groups @1GiB → 32 条声明（每条
#      block_stride=0、shared_by=单层）→ 物理分配 32 张独立 zeros。
#   G. Grouped 混合同页：6 full + 4 swa（同页 65536）→ _get_kv_cache_groups_uniform_
#      page_size 均衡拆分 [3,3,4] → 一般支路 group_size=4 条池声明（3 池×shared_by 3 层
#      + 1 池×1 层）→ 物理 4；full 桶垫 2 层（warning 落 trace）。
#   P. V4 packed：61 层（30 c4 + 31 c128，社区实读口径同 v4_cache_groups_61layer.json）
#      → 终五组 → _get_packed_kv_cache_layout：202 个 offset 桶逐组构成（新开 offset
#      91/27/0/57/27）+ shared_by 直方图 + 跨组相撞例 → _get_kv_cache_config_packed
#      @10GiB → 202 条声明全部 block_stride>0 → 物理 1 块 slab。
# 逐组 offset 阶梯为 pin 布局循环的逐字重放（同样的 layer_names 顺序、同样的累加），
# 并与 pin 的 layers_by_offset 全量断言一致——重放数字 = pin 数字。
import json
import logging
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


# ---------- 桩：绕开 host 没有的重配置依赖（算法代码零改动，同 run_v4_61layer.py） ----------
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
    FullAttentionSpec,
    SlidingWindowSpec,
    SlidingWindowMLASpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

GIB = 1073741824  # 1 GiB（U/G 示教显存）
AVAIL_10G = 10737418240  # 10 GiB（P 场景，社区文章同口径）


def make_vcfg():
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
        cache_config=types.SimpleNamespace(num_gpu_blocks_override=None),
        kv_transfer_config=None,
    )


# ---------- 捕获 pin 的 padding warning（让「垫 2 层 / 33.33%」落 trace） ----------
_warn_records = []


class _Cap(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            _warn_records.append(record.getMessage())


logging.getLogger().addHandler(_Cap())
for ln in ("vllm", "vllm.v1.core.kv_cache_utils"):
    logging.getLogger(ln).addHandler(_Cap())
    logging.getLogger(ln).setLevel(logging.WARNING)

out = {"params": {
    "pin": "vLLM v0.27.1（instances/vllm/source，行号基线）",
    "layout": "U/G 用通用 FullAttention/SlidingWindow spec（bf16）；P 用 fp8_ds_mla "
              "默认（alignment=576、uint8）",
    "sliding_window": 8192,
    "stubbed_modules": ["vllm.config", "vllm.config.kv_events",
                        "vllm.v1.metrics", "vllm.v1.metrics.stats",
                        "vllm.v1.request", "vllm.v1.utils"],
    "stub_note": "仅配置/指标类被替身；分组、均衡拆分、一般支路池声明、packed 布局、"
                 "packed 配置全部走 pin 算法代码零改动（get_kv_cache_groups / "
                 "get_kv_cache_config_from_groups / _get_packed_kv_cache_layout / "
                 "_get_kv_cache_config_packed）",
    "physical_count_rule": "gpu_model_runner.py:L7312-L7353：block_stride>0 的第一条"
                           "分配一次 packed_backing、其后全部别名；block_stride=0 的"
                           "声明各自独立 torch.zeros——物理数=Σ(stride==0)+有 packed 则 1",
    "available_memory": {"uniform_grouped_gib": 1, "packed_gib": 10},
}}

# ===========================================================================
# U. Uniform 单型：32 层同 FullAttentionSpec → 32 声明 / 32 物理
# ===========================================================================
specs_u = {
    f"model.layers.{i}.k_cache": FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    for i in range(32)}
page_u = next(iter(specs_u.values())).page_size_bytes
groups_u = kcu.get_kv_cache_groups(make_vcfg(), specs_u)
cfg_u = kcu.get_kv_cache_config_from_groups(make_vcfg(), groups_u, GIB)
tensors_u = cfg_u.kv_cache_tensors
sizes_u = sorted({t.size for t in tensors_u})
out["uniform_32"] = {
    "num_layers": 32,
    "spec": "FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, bf16)",
    "page_size_bytes": page_u,
    "page_formula": "2*16*8*128*2",
    "num_groups": len(groups_u),
    "group_layer_counts": [len(g.layer_names) for g in groups_u],
    "available_bytes": GIB,
    "available_gib": 1,
    "num_blocks": cfg_u.num_blocks,
    "num_blocks_formula": "1073741824 // 65536 // 32",
    "num_declarations": len(tensors_u),
    "decl_shared_by_sizes": dict(Counter(len(t.shared_by) for t in tensors_u)),
    "all_block_stride_zero": all(t.block_stride == 0 for t in tensors_u),
    "physical_allocations": sum(1 for t in tensors_u if t.block_stride == 0),
    "per_tensor_bytes": sizes_u[0],
    "per_tensor_mib": sizes_u[0] / (1024 * 1024),
    "total_bytes": sum(t.size for t in tensors_u),
    "total_equals_available": sum(t.size for t in tensors_u) == GIB,
}

# ===========================================================================
# G. Grouped 混合同页：6 full + 4 swa → 3 组 [3,3,4] → 4 池声明 / 4 物理
# ===========================================================================
specs_g = {}
for i in range(6):
    specs_g[f"model.layers.{i}.full"] = FullAttentionSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
for i in range(6, 10):
    specs_g[f"model.layers.{i}.sw"] = SlidingWindowSpec(
        block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16,
        sliding_window=1024)
pages_g = {s.page_size_bytes for s in specs_g.values()}
assert len(pages_g) == 1  # 同页（SlidingWindowSpec real_page=16*8*(128+128)*2 同为 65536）
groups_g = kcu.get_kv_cache_groups(make_vcfg(), specs_g)
cfg_g = kcu.get_kv_cache_config_from_groups(make_vcfg(), groups_g, GIB)
tensors_g = cfg_g.kv_cache_tensors
out["grouped_6f4sw"] = {
    "num_layers": len(specs_g),
    "composition": "6 FullAttentionSpec + 4 SlidingWindowSpec(窗 1024)，同页",
    "page_size_bytes": pages_g.pop(),
    "num_groups": len(groups_g),
    "group_layer_counts": [len(g.layer_names) for g in groups_g],
    "group_specs": [type(g.kv_cache_spec).__name__ for g in groups_g],
    "group_members": {f"G{i}": list(g.layer_names) for i, g in enumerate(groups_g)},
    "padding_warnings": [w for w in _warn_records if "padding" in w],
    "available_bytes": GIB,
    "available_gib": 1,
    "num_blocks": cfg_g.num_blocks,
    "num_blocks_formula": "1073741824 // 65536 // 4",
    "num_declarations": len(tensors_g),
    "decl_shared_by_sizes": dict(Counter(len(t.shared_by) for t in tensors_g)),
    "pool_shared_by": [list(t.shared_by) for t in tensors_g],
    "all_block_stride_zero": all(t.block_stride == 0 for t in tensors_g),
    "physical_allocations": sum(1 for t in tensors_g if t.block_stride == 0),
    "per_pool_bytes": tensors_g[0].size,
    "total_bytes": sum(t.size for t in tensors_g),
    "total_equals_available": sum(t.size for t in tensors_g) == GIB,
}

# ===========================================================================
# P. V4 packed：61 层 → 5 组 → 202 offset 桶 → 1 slab
# ===========================================================================
FP8_DS_MLA = "fp8_ds_mla"
SLIDING_WINDOW = 8192
BLOCK_SIZE = 256


def swa_spec():
    return SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=SLIDING_WINDOW, cache_dtype_str=FP8_DS_MLA,
        alignment=576, model_version="deepseek_v4")


def main_kv_spec(r):
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        compress_ratio=r, cache_dtype_str=FP8_DS_MLA, alignment=576,
        model_version="deepseek_v4")


def indexer_spec():
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=132, dtype=torch.uint8,
        compress_ratio=4, alignment=576)


def attn_compressor_state_spec(r):
    coff = 1 + (r == 4)
    return SlidingWindowMLASpec(
        block_size=4 if r == 4 else 8, num_kv_heads=1,
        head_size=2 * coff * 512, dtype=torch.float32,
        sliding_window=coff * r, alignment=576)


def indexer_compressor_state_spec():
    return SlidingWindowMLASpec(
        block_size=4, num_kv_heads=1, head_size=2 * 2 * 128,
        dtype=torch.float32, sliding_window=8, alignment=576)


def layer_specs(r):
    o = {"swa": swa_spec()}
    if r <= 1:
        return o
    o["main"] = main_kv_spec(r)
    if r == 4:
        o["indexer"] = indexer_spec()
        o["indexer_state"] = indexer_compressor_state_spec()
    o["attn_state"] = attn_compressor_state_spec(r)
    return o


ratios61 = [4] * 30 + [128] * 31
specs61 = {}
for li, r in enumerate(ratios61):
    for k, s in layer_specs(max(1, r)).items():
        specs61[f"model.layers.{li}.{k}"] = s

grouped61 = kcu.group_and_unify_kv_cache_specs(specs61)
groups61 = kcu._get_kv_cache_groups_uniform_groups(grouped61)
stride61, layers_by_offset = kcu._get_packed_kv_cache_layout(groups61)

# ---- 逐组 offset 阶梯重放（pin 布局循环逐字同序）并与 pin 全量断言一致 ----
layer_to_gid = {ln: gi for gi, g in enumerate(groups61) for ln in g.layer_names}
seen = set()
per_group_replay = []
members_replay = {}
for gi, g in enumerate(groups61):
    spec = g.kv_cache_spec
    off = 0
    own = []
    for ln in g.layer_names:
        page = (spec.kv_cache_specs[ln].page_size_bytes
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec.page_size_bytes)
        own.append(off)
        members_replay.setdefault(off, []).append(ln)
        off += page
    new = [o for o in own if o not in seen]
    per_group_replay.append({
        "gid": gi,
        "kind": f"G{gi}({len(g.layer_names)}层,块{spec.block_size})",
        "layers": len(g.layer_names),
        "own_offsets": len(own),
        "new_offsets": len(new),
        "dense_width": off,
    })
    seen.update(own)
assert members_replay == {o: list(v) for o, v in layers_by_offset.items()}, \
    "阶梯重放与 pin layers_by_offset 不一致"

hist = dict(Counter(len(v) for v in layers_by_offset.values()))
total_layer_refs = sum(k * c for k, c in hist.items())

# ---- offset 0 桶与代表性跨组相撞 ----
offset0 = list(layers_by_offset[0])
cross_buckets = [
    {"offset": o, "members": list(v),
     "member_gids": sorted({layer_to_gid[n] for n in v})}
    for o, v in sorted(layers_by_offset.items())
    if len({layer_to_gid[n] for n in v}) > 1]
named = ["model.layers.32.swa", "model.layers.13.main", "model.layers.9.main",
         "model.layers.10.indexer_state", "model.layers.54.attn_state",
         "model.layers.19.indexer_state"]
named_offsets = {}
for o, v in layers_by_offset.items():
    for n in named:
        if n in v:
            named_offsets[n] = o

vcfg61 = types.SimpleNamespace(
    cache_config=types.SimpleNamespace(num_gpu_blocks_override=None))
nb61, tensors61 = kcu._get_kv_cache_config_packed(vcfg61, groups61, AVAIL_10G)
total61 = stride61 * nb61

out["packed_61"] = {
    "compress_ratios_shape": "30 c4 + 31 c128（社区实读实发权重，61 层；同 "
                             "traces/v4_cache_groups_61layer.json 口径）",
    "num_layers_specs": len(specs61),
    "spec_census_formula": "30*5 + 31*3 = 150 + 93",
    "num_groups": len(groups61),
    "group_layer_counts": [len(g.layer_names) for g in groups61],
    "per_group_dense_width": [r["dense_width"] for r in per_group_replay],
    "block_stride": stride61,
    "block_stride_is_max_dense": stride61 == max(
        r["dense_width"] for r in per_group_replay),
    "num_distinct_offsets": len(layers_by_offset),
    "per_group_offset_account": per_group_replay,
    "new_offsets_sum": sum(r["new_offsets"] for r in per_group_replay),
    "new_offsets_sum_formula": "91+27+0+57+27",
    "shared_by_histogram": hist,
    "shared_by_layer_refs_total": total_layer_refs,
    "offset0_bucket": offset0,
    "offset0_num_layers": len(offset0),
    "cross_group_bucket_count": len(cross_buckets),
    "cross_group_examples": cross_buckets[:8],
    "named_layer_offsets": named_offsets,
    "available_bytes": AVAIL_10G,
    "available_gib": 10,
    "num_blocks": nb61,
    "num_declarations": len(tensors61),
    "all_block_stride_positive": all(t.block_stride > 0 for t in tensors61),
    "physical_allocations": 1,  # 全部 packed ⇒ Allocate once（L7328-L7336）
    "slab_bytes": total61,
    "slab_formula": "1435968*7477",
    "floor_remainder_bytes": AVAIL_10G - total61,
    "community_check": {
        "expected_offsets": 202,
        "offsets_match": len(layers_by_offset) == 202,
        "expected_histogram": {"1": 167, "2": 31, "3": 3, "5": 1},
        "histogram_match": hist == {1: 167, 2: 31, 3: 3, 5: 1},
        "expected_new_offsets": [91, 27, 0, 57, 27],
        "new_offsets_match": [r["new_offsets"] for r in per_group_replay]
        == [91, 27, 0, 57, 27],
        "expected_num_blocks_10gib": 7477,
        "num_blocks_match": nb61 == 7477,
        "source": "dossier/supplement-tensor-account.json sup-tensor-2（host 桩跑互证）",
    },
}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "tensor_decl_account.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", path)
print("U: page", page_u, "blocks", cfg_u.num_blocks,
      "decls", len(tensors_u), "phys",
      out["uniform_32"]["physical_allocations"])
print("G: groups", out["grouped_6f4sw"]["group_layer_counts"],
      "blocks", cfg_g.num_blocks, "decls", len(tensors_g), "phys",
      out["grouped_6f4sw"]["physical_allocations"])
print("G: warnings", out["grouped_6f4sw"]["padding_warnings"])
print("P: offsets", len(layers_by_offset), "hist", hist,
      "new", [r["new_offsets"] for r in per_group_replay])
print("P: stride", stride61, "blocks", nb61, "slab", total61,
      "match", out["packed_61"]["community_check"]["offsets_match"],
      out["packed_61"]["community_check"]["histogram_match"],
      out["packed_61"]["community_check"]["new_offsets_match"],
      out["packed_61"]["community_check"]["num_blocks_match"])
