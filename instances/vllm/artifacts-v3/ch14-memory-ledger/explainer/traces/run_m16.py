# run_m16.py — m16 DeepSeek V4 账本户口：六类缓存自报 spec 的页字节 census
# SECTION A（pin 直跑）：从 pin 源码（instances/vllm/source，v0.27.1）import 真实
#   MLAAttentionSpec/SlidingWindowMLASpec 类，按 V4 四处自报点的参数（fp8_ds_mla
#   布局）构造六类缓存 spec，读 real_page_size_bytes / page_size_bytes（对齐 576
#   由 _apply_alignment_padding 在 __post_init__ 里真跑）。
#   参数出处（每个都在 V4 模型侧源码里）：
#   - SWA 缓存（每层一挂）：sparse_swa.py:L87-L102，block_size=64（L82）、
#     compress_ratio 缺省 1、alignment 576；与 C4A 主 KV 同页的设计意图见 L77-L82 注释。
#   - 主 KV：attention.py:L655-L674，block_size=cache_config.block_size（V4 后端族
#     preferred 256，sparse_swa.py:L119-L121；协商链归 ch22）、compress_ratio=4/128
#     （逐层表 attention.py:L210-L213）、fp8_ds_mla 下 584B/槽
#     （kv_cache_interface.py:L408-L413：448 NoPE + 128 RoPE + 8 fp8 scale）。
#   - 索引器缓存：attention.py:L698-L710 + L792-L798，uint8、head 132
#     （=128 fp8 + 4 fp32 scale，attention.py:L788）、compress_ratio 传入 → 无
#     model_version 字段 → 走元素公式（kv_cache_interface.py:L417-L426）。
#   - 压缩器状态缓存：compressor.py:L155-L203 + L295-L300，fp32、
#     state_dim=2*coff*512（coff=1+(cr==4)）、block_size=4/8、窗=coff*cr（L175-L176）。
#   窗口 2048 为示教代入（config.sliding_window 实值随权重发布，源码只读：
#   attention.py:L206）；census 布局 = 源码注释自带示例（kv_cache_utils.py:L1693-L1697：
#   11 个 C4 层 + 10 个 C128 层 + 21 个滑窗缓存）。
# SECTION B（转写走查）：分组按 kv_cache_utils.py:L1592-L1632（group_and_unify：
#   MLA 桶 + SlidingWindowMLASpec 按 (block_size, sliding_window) 分桶）与
#   L1670-L1754（uniform_groups：元组数取 _approximate_gcd，L1635-L1667）逐行转写。
#   ⚠️ 转写非直跑：host 缺 transformers v5，无法 import vllm.config 链
#   （kv_cache_utils.py:L16 顶部 from vllm.config import VllmConfig）；转写逻辑 1:1、
#   行号锚见各函数 docstring。页字节（SECTION A）是 pin 类直跑产出。
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../source")
import torch  # noqa: E402

from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec  # noqa: E402

OUT = {"trace_source": {"section_A": "pin 类直跑（vllm/v1/kv_cache_interface.py，v0.27.1）",
                        "section_B": "分组走查转写（kv_cache_utils.py:L1592-L1754，host 缺 transformers v5 无法直跑）"}}

# ---------------- SECTION A：census（pin spec 类直跑） ----------------
def swa_cache():
    # sparse_swa.py:L91-L102 的报数（fp8_ds_mla → uint8/alignment 576）
    return SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=512,
                                dtype=torch.uint8, sliding_window=2048,
                                cache_dtype_str="fp8_ds_mla", alignment=576,
                                model_version="deepseek_v4")

def main_kv(cr):
    # attention.py:L664-L674 的报数（cache block_size 256）
    return MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=512,
                            dtype=torch.uint8, compress_ratio=cr,
                            cache_dtype_str="fp8_ds_mla", alignment=576,
                            model_version="deepseek_v4")

def indexer_kv():
    # attention.py:L702-L710 的报数（无 model_version → 元素公式；head 132 = L788）
    return MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=132,
                            dtype=torch.uint8, compress_ratio=4, alignment=576)

def comp_state(cr):
    # compressor.py:L196-L203 的报数（fp32；state_dim=2*coff*512，coff=1+(cr==4)）
    coff = 1 + (cr == 4)
    return SlidingWindowMLASpec(block_size=4 if cr == 4 else 8, num_kv_heads=1,
                                head_size=2 * coff * 512, dtype=torch.float32,
                                sliding_window=coff * cr, alignment=576)

def census_row(name, site, spec, window):
    slots = spec.block_size // spec.compress_ratio
    return {
        "spec_class": type(spec).__name__, "self_report_site": site,
        "block_size": spec.block_size, "compress_ratio": spec.compress_ratio,
        "head_size": spec.head_size, "storage_slots_per_page": slots,
        "bytes_per_slot": spec.real_page_size_bytes // slots,
        "page_raw": spec.real_page_size_bytes, "page_padded": spec.page_size_bytes,
        "manager": "SlidingWindowManager(窗口 cap)" if window else "FullAttentionManager(全历史)",
        "window": window or None,
    }

census = {
    "swa_cache(每层一挂)": census_row("swa", "sparse_swa.py:L87-L102", swa_cache(), 2048),
    "c4a_main_kv": census_row("c4a", "attention.py:L655-L674", main_kv(4), None),
    "c4i_indexer_kv": census_row("c4i", "attention.py:L698-L710", indexer_kv(), None),
    "c128a_main_kv": census_row("c128a", "attention.py:L655-L674", main_kv(128), None),
    "c4_compressor_state": census_row("comp4", "compressor.py:L191-L203", comp_state(4), 8),
    "c128_compressor_state": census_row("comp128", "compressor.py:L191-L203", comp_state(128), 256),
}
OUT["census"] = census

pages = [r["page_padded"] for r in census.values()]
OUT["page_set"] = sorted(set(pages))
OUT["num_page_sizes"] = len(set(pages))
OUT["same_page_pair"] = {
    "swa_cache": census["swa_cache(每层一挂)"]["page_padded"],
    "c4a_main_kv": census["c4a_main_kv"]["page_padded"],
    "equal": census["swa_cache(每层一挂)"]["page_padded"] == census["c4a_main_kv"]["page_padded"],
    "design_note": "sparse_swa.py:L77-L82：两类块共享同一物理张量，故必须同页",
}

# ---------------- SECTION B：分组走查（转写 kv_cache_utils.py:L1592-L1754） ----------------
def build_layer_specs(n_c4a=11, n_c128a=10):
    """源码注释示例布局（kv_cache_utils.py:L1693-L1697）：11 C4 + 10 C128，
    每层一挂 swa_cache → 21 个滑窗缓存 spec。层名用真实模块路径形态。"""
    spec = {}
    for i in range(n_c4a):
        spec[f"model.layers.{i}.self_attn"] = main_kv(4)
        spec[f"model.layers.{i}.self_attn.swa_cache"] = swa_cache()
        spec[f"model.layers.{i}.self_attn.indexer.k_cache"] = indexer_kv()
        spec[f"model.layers.{i}.self_attn.compressor.state_cache"] = comp_state(4)
    for i in range(n_c4a, n_c4a + n_c128a):
        spec[f"model.layers.{i}.self_attn"] = main_kv(128)
        spec[f"model.layers.{i}.self_attn.swa_cache"] = swa_cache()
        spec[f"model.layers.{i}.self_attn.compressor.state_cache"] = comp_state(128)
    return spec

def group_and_unify_transcribed(spec):
    """转写 kv_cache_utils.py:L1592-L1632（group_and_unify_kv_cache_specs）。"""
    if not any(isinstance(s, SlidingWindowMLASpec) for s in spec.values()):  # L1599-L1602
        return None
    if len({s.page_size_bytes for s in spec.values()}) <= 1:               # L1604-L1607
        return None
    mla, swa_buckets = {}, defaultdict(dict)                                # L1609-L1618
    for name, s in spec.items():
        if isinstance(s, SlidingWindowMLASpec):
            swa_buckets[(s.block_size, s.sliding_window)][name] = s
        elif isinstance(s, MLAAttentionSpec):
            mla[name] = s
    return [mla, *swa_buckets.values()]                                     # L1632

def approximate_gcd_transcribed(values, lower_bound=None):
    """转写 kv_cache_utils.py:L1635-L1667（_approximate_gcd）：
    取 d 最小化总向上 padding，平手取大 d。"""
    min_d = max(1, lower_bound if lower_bound is not None else 1)
    best_d, best_pad = min_d, None
    for d in range(min_d, max(values) + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad, best_d = pad, d
    return best_d

def uniform_groups_transcribed(grouped):
    """转写 kv_cache_utils.py:L1670-L1754（_get_kv_cache_groups_uniform_groups）。"""
    tuples_per_group = [Counter(s.page_size_bytes for s in g.values()).most_common(1)[0][1]
                        for g in grouped]                                    # L1698-L1700
    num_layer_tuples = approximate_gcd_transcribed(
        tuples_per_group, lower_bound=tuples_per_group[0])                   # L1702-L1704
    groups = [{"kind": "mla_tuple_group", "layers": len(grouped[0]),
               "layer_names": sorted(grouped[0]),
               "group_page_sum": sum(s.page_size_bytes for s in grouped[0].values())}]
    for bucket in grouped[1:]:                                               # L1721-L1752
        layers_per_size = defaultdict(list)
        for n, s in bucket.items():
            layers_per_size[s.page_size_bytes].append(n)
        per_size = len(next(iter(layers_per_size.values())))
        num_tuple_groups = -(-per_size // num_layer_tuples)                  # cdiv, L1734
        layer_tuples = list(zip(*layers_per_size.values()))                  # L1735
        for i in range(num_tuple_groups):
            names = [n for t in layer_tuples[i::num_tuple_groups] for n in t]
            groups.append({"kind": "swa_mla_bucket_group", "layers": len(names),
                           "layer_names": sorted(names),
                           "group_page_sum": sum(bucket[n].page_size_bytes for n in names)})
    return num_layer_tuples, groups

spec = build_layer_specs()
grouped = group_and_unify_transcribed(spec)
num_layer_tuples, groups = uniform_groups_transcribed(grouped)

# 桶清单（与 group_and_unify_transcribed 的分桶键一致，重新数一遍便于 JSON 直读）
swa_buckets = defaultdict(dict)
for n, s in spec.items():
    if isinstance(s, SlidingWindowMLASpec):
        swa_buckets[(s.block_size, s.sliding_window)][n] = s
bucket_list = [{"bucket": "mla(全部 MLAAttentionSpec)", "specs": len(grouped[0])}] + [
    {"bucket": f"swa_mla(block_size,window)={k}", "specs": len(v)} for k, v in swa_buckets.items()
]
OUT["grouping_walk"] = {
    "census_example": "kv_cache_utils.py:L1693-L1697 注释自带示例：11 C4 + 10 C128（共 21 层，各挂 swa_cache）",
    "total_spec_count": len(spec),
    "buckets": bucket_list,
    "num_layer_tuples_approx_gcd": num_layer_tuples,
    "groups": groups,
    "num_groups": len(groups),
    "groups_summary": (f"共 {len(groups)} 组：MLA 元组组 1 个（{groups[0]['layers']} 层 = 11 个"
                       f"[C4I,C4A,C128] 元组，C128A 10 层补齐到 11）+ SWA 桶切 2 组（11+10）"
                       f"+ C4 状态桶 1 组（11 层）+ C128 状态桶 1 组（10 层）"),
    "eagle_note": "开 EAGLE/MTP 时最后一层的 swa_cache 所在组被标 is_eagle_group（L1757-L1778）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m16.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
