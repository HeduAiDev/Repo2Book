#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ch36 部署账本复算 driver（DSV4-Flash × 96GB 档位卡）。

本章无精简版（dossier subtraction_plan 留空、kind=code 的部署章）：本 driver 不跑 vLLM
引擎，而是把「启动账本」的每一步算术落到可执行的代码上——

  1) 页大小/对齐补齐全部由 **pin 源码的真实 KVCacheSpec 类** 算出（直接 import
     instances/vllm/source 的 vllm.v1.kv_cache_interface——host 可干净导入，spec 类
     只需 torch dtype 常量、无 GPU 行为）；
  2) 分组/定账四个函数从 pin 的 vllm/v1/core/kv_cache_utils.py **逐字复制**
     （# SOURCE: 行锚标注；该模块在 host 因 transformers v4/v5 门槛不可整模块导入，
     复制体与 pin 逐字符一致、仅去掉与本账无关的调用方）；
  3) spec 清单按三处 call site 逐字镜像构造：
       - vllm/models/deepseek_v4/attention.py:L664-L674（压缩主账 MLAAttentionSpec）
       - vllm/models/deepseek_v4/attention.py:L702-L710 + L782-L791（indexer 账）
       - vllm/v1/attention/backends/mla/sparse_swa.py:L91-L102（窗口账）
       - vllm/models/deepseek_v4/compressor.py:L196-L203（状态账；注意 attention 压缩器
         head=512 与 indexer 压缩器 head=128 各建一份 CompressorStateCache——
         compressor.py:L295-L300 无条件创建，C4A 层因此有 twin 状态账）
     模型形状数字取自 deepseek-ai/DeepSeek-V4-Flash 的 config.json（HF 官方，
     2026-09-22 取回：hidden 4096 / moe_mid 2048 / 256 专家 / 43 层 / head_dim 512 /
     qk_rope 64 / sliding_window 128 / index_head_dim 128 / expert_dtype fp4 /
     num_nextn_predict_layers 1 / vocab 129280 / compress_ratios 44 项）。

取证环境：host Windows / Miniconda Python 3.11.11 / torch 2.11.0+cu128（spec 类仅用
dtype 常量，无 GPU 行为差异字段）。docker daemon 未运行、且单张 96GB 卡装不下
155-160GiB checkpoint——无真实启动日志可取，故 explainer trace_source=manual、
本 driver 的输出（ch36_ledger_arithmetic.json）是数字的运行取证。

用法：/d/Env/Miniconda/python run_ch36_ledger.py   （在 explainer/traces/ 下）
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PIN = Path(r"E:\Laboratory\Repo2Book\instances\vllm\source")
sys.path.insert(0, str(PIN))

# host 装的是 transformers 4.57，pin v0.27.1 的 vllm.config 链在
# transformers_utils/config.py:L52-L56 有 v5 硬门槛——KVCacheSpecRegistry 的惰性
# _ensure_registered 会拉整条引擎 import 链。改走 register() 公开 API 预登记本章
# 用到的两个 spec 类型（映射逐字镜像 single_type_kv_cache_manager.py:L1894-L1922
# 的 register_all_kvcache_specs：SlidingWindowMLASpec 基类=自己、MLAAttentionSpec
# 基类=FullAttentionSpec——manager 类与分组数学无关，占位 object）。注册表非空后
# _ensure_registered 早退，import 级联不触发；uniform-type 判定与真路逐位一致。
import torch  # noqa: E402

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry  # noqa: E402

KVCacheSpecRegistry.register(
    SlidingWindowMLASpec, manager_class=object,
    uniform_type_base_spec=SlidingWindowMLASpec,
)
KVCacheSpecRegistry.register(
    MLAAttentionSpec, manager_class=object,
    uniform_type_base_spec=FullAttentionSpec,
)

try:
    from vllm.utils.math_utils import cdiv, round_up
except Exception:  # pragma: no cover
    def cdiv(a, b):
        return -(-a // b)

    def round_up(value, divisor):
        return (value + divisor - 1) // divisor * divisor

GIB = 1 << 30

# --------------------------------------------------------------------------
# pin 逐字复制区（vllm/v1/core/kv_cache_utils.py，v0.27.1 @ 6e448d0ea）
# --------------------------------------------------------------------------
# SOURCE: vllm/v1/core/kv_cache_utils.py:L1635-L1667
def _approximate_gcd(values, *, lower_bound=None):
    """Pick a chunk size that minimizes total upward padding."""
    if not values:
        raise ValueError("values must be non-empty")
    if any(x <= 0 for x in values):
        raise ValueError(f"values must be positive, got: {list(values)!r}")
    min_d = max(1, lower_bound if lower_bound is not None else 1)
    max_d = max(values)
    if min_d > max_d:
        return min_d
    best_d = min_d
    best_pad = None
    for d in range(min_d, max_d + 1):
        pad = sum((d - (x % d)) % d for x in values)
        if best_pad is None or pad < best_pad or (pad == best_pad and d > best_d):
            best_pad = pad
            best_d = d
    return best_d


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1592-L1632
def group_and_unify_kv_cache_specs(kv_cache_spec):
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return None
    mla_specs = {}
    grouped_swa_mla_specs = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec
    assert len(mla_specs) > 0
    mla_uniform_spec = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform_spec is not None
    swa_uniform_specs = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)
    return [mla_uniform_spec, *swa_uniform_specs]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1670-L1754
def _get_kv_cache_groups_uniform_groups(grouped_specs):
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    num_layer_tuples_per_group = [
        g_spec.get_num_layer_tuples() for g_spec in grouped_specs
    ]
    num_layer_tuples = _approximate_gcd(
        num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0]
    )
    num_layer_tuples_per_group = [
        round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group
    ]
    swa_mla_specs = grouped_specs[1:]
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in swa_mla_specs
        for spec in group.kv_cache_specs.values()
    )
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        layers_per_size = defaultdict(list)
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            layers_per_size[layer_spec.page_size_bytes].append(layer_name)
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            group_layer_names = [
                name for layer_tuple in group_layer_tuples for name in layer_tuple
            ]
            group_layer_specs = {
                name: sm_spec.kv_cache_specs[name] for name in group_layer_names
            }
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )
    return [full_mla_group, *swa_mla_groups]


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1283-L1305
def _get_packed_kv_cache_layout(kv_cache_groups):
    """Lay out each cache group densely in one shared block slab."""
    layers_by_offset = defaultdict(list)
    block_stride = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        byte_offset = 0
        for layer_name in group.layer_names:
            if isinstance(spec, UniformTypeKVCacheSpecs):
                page_size = spec.kv_cache_specs[layer_name].page_size_bytes
            else:
                page_size = spec.page_size_bytes
            layers_by_offset[byte_offset].append(layer_name)
            byte_offset += page_size
        block_stride = max(block_stride, byte_offset)
    assert block_stride > 0
    return block_stride, layers_by_offset


# SOURCE: vllm/v1/core/kv_cache_utils.py:L951-L958（get_max_concurrency_for_kv_cache_config 主体）
def max_concurrency(kv_cache_groups, num_blocks, vllm_config):
    num_blocks_per_request = sum(
        cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in kv_cache_groups
    )
    return num_blocks / num_blocks_per_request, num_blocks_per_request


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1890-L1937（_max_memory_usage_bytes_from_groups 的
# UniformTypeKVCacheSpecs 全组特化分支——"only DeepseekV4 for now"）
def dsv4_max_memory_usage_bytes(kv_cache_groups, vllm_config):
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())
    num_layer_tuples = max(
        group.kv_cache_spec.get_num_layer_tuples() for group in kv_cache_groups
    )
    total = 0
    detail = []
    for group in kv_cache_groups:
        pages = group.kv_cache_spec.max_memory_usage_pages(vllm_config)
        b = num_layer_tuples * pages * layer_tuple_bytes
        total += b
        detail.append({"layers": len(group.layer_names), "pages": pages, "bytes": b})
    return total, layer_tuple_bytes, num_layer_tuples, detail


# SOURCE: vllm/v1/core/kv_cache_utils.py:L1953-L1985（_estimate_max_model_len_from_groups）
def estimate_max_model_len(kv_cache_groups, vllm_config, available_memory):
    original_max = vllm_config.model_config.max_model_len

    def fits(model_len):
        vllm_config.model_config.max_model_len = model_len
        return dsv4_max_memory_usage_bytes(kv_cache_groups, vllm_config)[0] <= available_memory

    left, right = 1, original_max
    if not fits(left):
        return 0
    result = 1
    while left <= right:
        mid = (left + right) // 2
        if fits(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    vllm_config.model_config.max_model_len = original_max
    return result


# --------------------------------------------------------------------------
# spec 构造区：逐字镜像三处 call site + HF config.json 官方值
# HEAD_DIM=512, ROPE=64, WINDOW=128, BLOCK=256(配方), INDEX_HEAD=128
# --------------------------------------------------------------------------
HEAD_DIM = 512
ROPE_DIM = 64
WINDOW = 128
BLOCK = 256          # 官方配方 --block-size 256（sparse MLA kernel 硬约束）
INDEX_HEAD = 128
MXFP4_BLOCK = 32     # vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:L11
QUANT_BLOCK = 128    # attention.py:L769 self.quant_block_size = 128
QUANT = get_kv_quant_mode("fp8_ds_mla")  # startswith("fp8") -> FP8_PER_TENSOR


# mirror sparse_swa.py:L91-L102（每层一份，含 MTP；fp8_ds_mla -> dtype uint8）
def swa_spec():
    return SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=HEAD_DIM, dtype=torch.uint8,
        sliding_window=WINDOW, cache_dtype_str="fp8_ds_mla", alignment=576,
        model_version="deepseek_v4", kv_quant_mode=QUANT,
    )


# mirror attention.py:L664-L674（压缩主账）
def main_spec(ratio):
    return MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=HEAD_DIM, dtype=torch.uint8,
        compress_ratio=ratio, cache_dtype_str="fp8_ds_mla", alignment=576,
        model_version="deepseek_v4", kv_quant_mode=QUANT,
    )


# mirror attention.py:L702-L710 + L782-L791（indexer 账；无 cache_dtype_str/model_version）
def indexer_spec(use_fp4: bool):
    if use_fp4:
        head = INDEX_HEAD // 2 + INDEX_HEAD // MXFP4_BLOCK      # 64 + 4 = 68
    else:
        head = INDEX_HEAD + INDEX_HEAD // QUANT_BLOCK * 4       # 128 + 4 = 132
    return MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=head, dtype=torch.uint8,
        compress_ratio=4, alignment=576,
    )


# mirror compressor.py:L155-L203（状态账；fp32 恒定；coff = 1+(ratio==4)）
def state_spec(ratio, head_dim):
    coff = 1 + (ratio == 4)
    state_dim = 2 * coff * head_dim      # compressor.py:L296 kv_state+score_state 两半
    block = 4 if ratio == 4 else 8
    win = coff * ratio                   # C4: 8 / C128: 128
    return SlidingWindowMLASpec(
        block_size=block, num_kv_heads=1, head_size=state_dim,
        dtype=torch.float32, sliding_window=win, alignment=576,
    )


def indexer_state_spec():
    return state_spec(4, INDEX_HEAD)     # indexer 压缩器 head=128 -> state_dim 512


def build_specs(use_fp4_indexer: bool = True):
    """返回 (spec dict, census 说明)。MTP 的 SWA spec 最后插入（镜像真实注册序，
    _annotate_eagle 取 next(reversed(...)) 找最后一层）。"""
    ratios = [0, 0] + [4, 128] * 20 + [4, 0]   # config.json compress_ratios 44 项
    assert len(ratios) == 44
    specs = {}
    census = Counter()
    for i, r in enumerate(ratios[:43]):         # 43 个 dense 层（末位 0 是 MTP 占位不读）
        ratio = max(1, r)                       # attention.py:L210-L211
        p = f"L{i:02d}"
        specs[f"{p}.swa_cache"] = swa_spec(); census["swa(64,128)"] += 1
        if ratio == 4:
            specs[f"{p}.main"] = main_spec(4); census["mla.C4A"] += 1
            specs[f"{p}.indexer.k_cache"] = indexer_spec(use_fp4_indexer)
            census["mla.C4I"] += 1
            specs[f"{p}.compressor.state"] = state_spec(4, HEAD_DIM)
            census["state(4,8).attn"] += 1
            specs[f"{p}.indexer.compressor.state"] = indexer_state_spec()
            census["state(4,8).indexer"] += 1
        elif ratio == 128:
            specs[f"{p}.main"] = main_spec(128); census["mla.C128"] += 1
            specs[f"{p}.compressor.state"] = state_spec(128, HEAD_DIM)
            census["state(8,128)"] += 1
        else:
            census["pure_window"] += 1
    specs["L43.mtp.swa_cache"] = swa_spec(); census["swa(64,128)"] += 1
    census["mtp"] += 1
    return specs, dict(census)


def group_ledger(use_fp4_indexer=True):
    specs, census = build_specs(use_fp4_indexer)
    grouped = group_and_unify_kv_cache_specs(specs)
    groups = _get_kv_cache_groups_uniform_groups(grouped)
    stride, layers_by_offset = _get_packed_kv_cache_layout(groups)
    tuple_counts = [g.get_num_layer_tuples() for g in grouped]
    gcd_walk = {
        "num_layer_tuples_per_bucket": tuple_counts,
        "chosen_d": _approximate_gcd(tuple_counts, lower_bound=tuple_counts[0]),
        "pad_by_d": {
            str(d): sum((d - (x % d)) % d for x in tuple_counts)
            for d in range(tuple_counts[0], max(tuple_counts) + 1)
        },
    }
    per_bucket = []
    for g in grouped:
        pages = Counter(s.page_size_bytes for s in g.kv_cache_specs.values())
        per_bucket.append({
            "layers": len(g.kv_cache_specs),
            "num_layer_tuples(most_common)": g.get_num_layer_tuples(),
            "page_counts": {f"{k}B": v for k, v in sorted(pages.items())},
            "group_page_bytes": g.page_size_bytes,
        })
    final_groups = []
    for g in groups:
        pages = Counter(
            g.kv_cache_spec.kv_cache_specs[n].page_size_bytes for n in g.layer_names
        )
        final_groups.append({
            "layers": len(g.layer_names),
            "page_counts": {f"{k}B": v for k, v in sorted(pages.items())},
            "stack_bytes(sum of layer pages)": g.kv_cache_spec.page_size_bytes,
        })
    # eagle 组：含最后一层（MTP SWA）的组
    eagle = next(i for i, g in enumerate(groups) if "L43.mtp.swa_cache" in g.layer_names)
    return {
        "census": census,
        "total_specs": len(specs),
        "approximate_gcd_walk": gcd_walk,
        "buckets(UniformTypeKVCacheSpecs before split)": per_bucket,
        "final_groups": final_groups,
        "num_groups": len(groups),
        "eagle_group_index": eagle,
        "block_stride": stride,
        "offsets": {str(k): len(v) for k, v in sorted(layers_by_offset.items())},
    }


def make_vc(max_model_len, in_flight):
    # 真实 VllmConfig 的属性面（spec 方法只读这几个字段；max_in_flight_tokens 在
    # 真身上是 property = max_concurrent_batches × max_num_batched_tokens，
    # vllm/config/vllm.py:L553-L561——async 服务默认 2×8192=16384）
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=in_flight,
    )


def scenario(stride, groups, pool_bytes, max_model_len, in_flight):
    vc = make_vc(max_model_len, in_flight)
    num_blocks = pool_bytes // stride
    conc, per_req_total = max_concurrency(groups, num_blocks, vc)
    needed, tuple_bytes, n_tuples, detail = dsv4_max_memory_usage_bytes(groups, vc)
    out = {
        "pool_GiB": round(pool_bytes / GIB, 3),
        "max_model_len": max_model_len,
        "in_flight_tokens": in_flight,
        "num_blocks": num_blocks,
        "blocks_per_request": per_req_total,
        "check_enough_needed_bytes": needed,
        "check_enough_needed_GiB": round(needed / GIB, 3),
        "passes_check_enough": needed <= pool_bytes,
        "max_concurrency": round(conc, 4),
        "kv_cache_size_tokens": int(conc * max_model_len),
        "per_group_pages": [d["pages"] for d in detail],
    }
    if needed > pool_bytes:
        out["estimated_max_model_len"] = estimate_max_model_len(
            groups, make_vc(max_model_len, in_flight), pool_bytes
        )
    return out


def main():
    res = {"pin": "vLLM v0.27.1 (6e448d0ea); spec 类 = pin 真码; 分组/定账 = pin 逐字复制"}

    # ---- 权重账（m3）：纯整数算术，shape 全部 nvidia/model.py:L202-L246 ----
    H, I, E, LAYERS = 4096, 2048, 256, 43
    w13_val = (2 * I) * (H // 2)             # uint8 每字节 2 个 fp4
    w13_sf = (2 * I) * (H // 32)             # ue8m0 每 32 值 1 字节
    w2_val = H * (I // 2)
    w2_sf = H * (I // 32)
    per_expert = w13_val + w13_sf + w2_val + w2_sf
    params_per_expert = 3 * I * H
    per_layer = per_expert * E
    all_experts = per_layer * LAYERS
    embed_head = 2 * 129280 * 4096 * 2       # bf16, tie=false
    res["weights"] = {
        "params_per_expert": params_per_expert,
        "w13_packed_B": w13_val, "w13_scale_B": w13_sf,
        "w2_packed_B": w2_val, "w2_scale_B": w2_sf,
        "values_B": w13_val + w2_val, "scales_B": w13_sf + w2_sf,
        "scale_overhead_is_exactly_1_of_17": (w13_sf + w2_sf) * 17 == per_expert,
        "bytes_per_expert": per_expert,
        "bytes_per_param": per_expert / params_per_expert,
        "bytes_per_layer_moe": per_layer,
        "GiB_per_layer_moe": round(per_layer / GIB, 4),
        "expert_params_total": params_per_expert * E * LAYERS,
        "expert_bytes_total": all_experts,
        "expert_GiB_total": round(all_experts / GIB, 2),
        "expert_GB_total_decimal": round(all_experts / 1e9, 2),
        "embed_lmhead_B_bf16": embed_head,
        "embed_lmhead_GiB": round(embed_head / GIB, 4),
    }

    # ---- 584B 槽位与 576 对齐（m6）：真 spec 类算页 ----
    ledgers = {
        "SWA 窗口账 block64": swa_spec(),
        "C4A 压缩主账 ratio4": main_spec(4),
        "C128 压缩主账 ratio128": main_spec(128),
        "C4I indexer fp4": indexer_spec(True),
        "C4I indexer fp8": indexer_spec(False),
        "C4 状态账(attn head512)": state_spec(4, HEAD_DIM),
        "C4 状态账(indexer head128)": indexer_state_spec(),
        "C128 状态账": state_spec(128, HEAD_DIM),
    }
    res["pages"] = {}
    for name, s in ledgers.items():
        raw = s.real_page_size_bytes
        res["pages"][name] = {
            "block_size": s.block_size,
            "storage_block_size": s.storage_block_size,
            "page_size_bytes(after 576 alignment)": s.page_size_bytes,
            "raw_page_size_bytes": raw,
            "pad_B": s.page_size_bytes - raw,
            "pad_pct": round((s.page_size_bytes - raw) / raw * 100, 2),
            "per_token_amortized_B(per 256-token main block where applicable)": (
                s.page_size_bytes / (s.block_size * s.compress_ratio)
            ),
        }

    # ---- indexer 两档对比（m8）----
    fp4_page = indexer_spec(True).page_size_bytes
    fp8_page = indexer_spec(False).page_size_bytes
    res["indexer_compare"] = {
        "head_fp4": 68, "head_fp8": 132,
        "page_fp4": fp4_page, "page_fp8": fp8_page,
        "slot_68B_formula": "128//2 + 128//32",
        "slot_132B_formula": "128 + 128//128*4",
    }

    # ---- 分组与 packed 定账（m10/m11）：fp4 与 fp8 indexer 两档 ----
    res["grouping_fp4_indexer"] = group_ledger(True)
    res["grouping_fp8_indexer"] = group_ledger(False)

    # ---- 每 token 摊销账（m17，稳态窗口、主块 256 token 摊销）----
    m6_fp4 = res["grouping_fp4_indexer"]
    c4a_tok = res["pages"]["C4A 压缩主账 ratio4"]["page_size_bytes(after 576 alignment)"] / 256
    c128_tok = res["pages"]["C128 压缩主账 ratio128"]["page_size_bytes(after 576 alignment)"] / 256
    idx_fp4_tok = fp4_page / 256
    idx_fp8_tok = fp8_page / 256
    steady_fp4 = 21 * c4a_tok + 20 * c128_tok + 21 * idx_fp4_tok
    steady_fp8 = 21 * c4a_tok + 20 * c128_tok + 21 * idx_fp8_tok
    llama70_gqa_tok = 2 * 8 * 128 * 2 * 80          # 2(K,V)×8 kv 头×128 维×2B×80 层
    dsv4_nocompress_tok = 584 * 44                   # 全部 44 本账存全量 token
    res["amortized_per_token"] = {
        "C4A_main_B": c4a_tok, "C128_main_B": c128_tok,
        "indexer_fp4_pool_B": idx_fp4_tok, "indexer_fp4_payload_B": 68 / 4,
        "indexer_fp8_pool_B": idx_fp8_tok, "indexer_fp8_payload_B": 132 / 4,
        "steady_total_fp4_B": steady_fp4,
        "steady_total_fp8_B": steady_fp8,
        "window_and_state_bounded": "窗口账/状态账按请求有界（窗口+chunk），不随全长线性涨",
        "llama70b_gqa_B": llama70_gqa_tok,
        "ratio_vs_gqa": round(llama70_gqa_tok / steady_fp4, 1),
        "dsv4_if_no_compression_B": dsv4_nocompress_tok,
    }

    # ---- 测量三件套（m12）：方程与假设填充 ----
    total_96gib = 96 * GIB
    util = 0.92
    requested = -(-int(total_96gib * util) // 1)     # ceil（utils.py:L414-L416 用 math.ceil）
    res["measurement"] = {
        "total_GiB_assumed": 96,
        "gpu_memory_utilization_default": util,       # config/cache.py:L68
        "requested_B": requested,
        "requested_GiB": round(requested / GIB, 2),
        "est_weights_TP2_GiB": 78.7,
        "est_activation_peak_GiB": 3.6,
        "est_cudagraph_GiB": 1.0,
        "available_GiB_rounded": 5.0,
        "note": "权重/激活/图池为量级估计（est）；num_blocks 及其后全部由假设 available 精确算出",
    }

    # ---- 护栏与并发场景（m13/m18）----
    groups = _get_kv_cache_groups_uniform_groups(
        group_and_unify_kv_cache_specs(build_specs(True)[0])
    )
    stride = res["grouping_fp4_indexer"]["block_stride"]
    scenarios = []
    for pool_gib, mml, ifl in [
        (5, 131072, 16384),   # A: 5GiB 池 128K 上下文、服务默认 chunk 8192
        (5, 131072, 4096),    # B: 同池砍 chunk 到 2048
        (8, 1048576, 16384),  # C: 8GiB 池 1M 上下文、chunk 8192
        (8, 1048576, 4096),   # D: 同池 chunk 2048
        (8, 131072, 16384),   # E: 8GiB 池 128K、chunk 8192
        (5, 13572, 16384),    # F: 场景 A 护栏二分估出的可行长度
        (44, 131072, 16384),  # G: TP4 权重减半后的量级池
    ]:
        scenarios.append(scenario(stride, groups, int(pool_gib * GIB), mml, ifl))
    res["scenarios"] = scenarios

    # ---- indexer 档对块数的杠杆（m8/m18）----
    stride_fp8 = res["grouping_fp8_indexer"]["block_stride"]
    res["indexer_lever"] = {
        "stride_fp4": stride, "stride_fp8": stride_fp8,
        "blocks_gain_fp4_vs_fp8_pct": round((stride_fp8 / stride - 1) * 100, 2),
        "blocks_8GiB_fp4": (8 * GIB) // stride,
        "blocks_8GiB_fp8": (8 * GIB) // stride_fp8,
    }

    # 误读口径对照（m11）：『一块=Σ全模型逐层页』的直觉值 vs 实际 max
    stacks = [g["stack_bytes(sum of layer pages)"] for g in m6_fp4["final_groups"]]
    res["misread_sum_all_groups"] = {
        "sum_of_all_group_stacks": sum(stacks),
        "actual_block_stride_max": stride,
        "overestimate_factor": round(sum(stacks) / stride, 2),
    }

    out = HERE / "ch36_ledger_arithmetic.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"written: {out}")
    print(json.dumps({
        "stride_fp4": stride,
        "stride_fp8": stride_fp8,
        "census": res["grouping_fp4_indexer"]["census"],
        "num_groups": res["grouping_fp4_indexer"]["num_groups"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
