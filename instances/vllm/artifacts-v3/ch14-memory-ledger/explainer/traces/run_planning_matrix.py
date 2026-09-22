# ch14 四补素材驱动：planning-width-bucket-matrix——组×宽度桶矩阵（Pro 61 层 +
# Flash 43 层双矩阵）。教会读者读社区 gpu_kv_planning 图（组×宽度桶矩阵 /
# valid-wasted 逐格）的 pin 侧账面：行 = 4 种页宽桶（三对同页把 7 种缓存形态收进
# 4 行）、列 = 5 个分配组、格 = valid（放什么 spec/几层/共几 Byte）或浪费归因。
# 全部跑 pin 真码（vLLM v0.27.1，kv_cache_utils.py 分组/近似 GCD/packed 布局零
# 改动；仅 vllm.config / vllm.config.kv_events / vllm.v1.metrics / v1.metrics.stats /
# vllm.v1.request / vllm.v1.utils 六处配置/指标桩替身——同 run_v4_cache_groups.py
# 桩面）。矩阵本体 = 对 pin 布局输出的只读后处理：
#   - 逐组 offset 阶梯重放（与 pin 的 layers_by_offset 全量断言一致）；
#   - valid 格 = 组内该页宽层数 × 页宽；
#   - 浪费格 = 该桶【整条】条带完全落在本组密排尾 [dense, block_stride) 内的
#     并集字节（骑跨 dense 边界的条带不计；跨组同 offset 折叠后去重，「加权」
#     为不去重口径 = 条带数 × 页宽）——与 dossier/supplement-planning-matrix.json
#     sup-plan-3 的 40 个格值逐一断言相等（supplement_check 落 trace）；
#   - 行闲置 = block_stride − dense（每列唯一可加总的浪费量）。
# 互证：Pro 的 dense/stride/tuples/7477 块对 traces/v4_cache_groups_61layer.json
# 逐项断言；Flash 的 stride 1002240 与 ch36 登记的 fp8 档 block_stride 一致。
import json
import os
import sys
import types
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# traces → explainer → ch14-memory-ledger → artifacts-v3 → vllm(实例) → source
PIN_ROOT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "source"))
sys.path.insert(0, PIN_ROOT)
os.environ["PYTHONHASHSEED"] = "0"


# ---------- 桩：绕开 host 没有的重配置依赖（算法代码零改动，同 run_v4_cache_groups.py） ----------
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
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

FP8_DS_MLA = "fp8_ds_mla"
BLOCK_SIZE = 256  # 管理面 block_size（DeepseekSparseSWABackend 偏好 256）
AVAIL_10G = 10737418240  # 10 GiB 示教显存（社区文章同口径）


# ---------- 五类 spec 生产者：逐字复刻 pin 的构造实参（窗值参数化） ----------
def swa_spec(window):
    """DeepseekV4SWACache.get_kv_cache_spec（sparse_swa.py:L87-L102，fp8_ds_mla）。"""
    return SlidingWindowMLASpec(
        block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=window, cache_dtype_str=FP8_DS_MLA,
        alignment=576, model_version="deepseek_v4")


def main_kv_spec(r):
    """DeepseekV4Attention.get_kv_cache_spec 主 KV（attention.py:L664-L674）。"""
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        compress_ratio=r, cache_dtype_str=FP8_DS_MLA, alignment=576,
        model_version="deepseek_v4")


def indexer_spec():
    """DeepseekV4IndexerCache.get_kv_cache_spec（attention.py:L698-L710；head 132）。"""
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=132, dtype=torch.uint8,
        compress_ratio=4, alignment=576)


def attn_compressor_state_spec(r):
    """注意力压缩器状态（compressor.py:L173-L203）：coff=1+(r==4)。"""
    coff = 1 + (r == 4)
    return SlidingWindowMLASpec(
        block_size=4 if r == 4 else 8, num_kv_heads=1,
        head_size=2 * coff * 512, dtype=torch.float32,
        sliding_window=coff * r, alignment=576)


def indexer_compressor_state_spec():
    """indexer 内嵌压缩器：head=128 → state_dim=512（compressor.py:L295-L300）。"""
    return SlidingWindowMLASpec(
        block_size=4, num_kv_heads=1, head_size=2 * 2 * 128,
        dtype=torch.float32, sliding_window=8, alignment=576)


def layer_specs(r, window):
    """一层名下的全部缓存账（插入顺序 = pin 收集顺序：swa→main→indexer→
    indexer_state→attn_state——G0 元组阶梯由这个顺序决定）。"""
    out = {"swa": swa_spec(window)}
    if r <= 1:
        return out  # attention.py:L655-L659：主 KV 返回 None
    out["main"] = main_kv_spec(r)
    if r == 4:
        out["indexer"] = indexer_spec()
        out["indexer_state"] = indexer_compressor_state_spec()
    out["attn_state"] = attn_compressor_state_spec(r)
    return out


def build_model(ratios, window):
    specs = {}
    for li, r in enumerate(ratios):
        eff = max(1, r)  # attention.py:L210-L213：max(1,·) 把 0 折成 1
        for k, s in layer_specs(eff, window).items():
            specs[f"model.layers.{li}.{k}"] = s
    return specs


def union_len(intervals):
    """区间并集总长（跨组同 offset 的条带去重）。"""
    if not intervals:
        return 0
    ivs = sorted(intervals)
    tot, cs, ce = 0, ivs[0][0], ivs[0][1]
    for s, e in ivs[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def kind_label(name, ratios):
    """main/attn_state 按源层压缩比细分（main_c4 / attn_state_c128 …）。"""
    kind = name.rsplit(".", 1)[-1]
    li = int(name.split(".")[2])
    r = max(1, ratios[li])
    if kind == "main":
        return f"main_c{r}"
    if kind == "attn_state":
        return f"attn_state_c{r}"
    return kind


def run_matrix(ratios, window):
    specs = build_model(ratios, window)
    grouped = kcu.group_and_unify_kv_cache_specs(specs)
    assert grouped is not None and len(grouped) == 4
    tuples = [g.get_num_layer_tuples() for g in grouped]
    chosen = kcu._approximate_gcd(tuples, lower_bound=tuples[0])
    groups = kcu._get_kv_cache_groups_uniform_groups(grouped)
    stride, layers_by_offset = kcu._get_packed_kv_cache_layout(groups)
    vcfg = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(num_gpu_blocks_override=None))
    num_blocks, _ = kcu._get_kv_cache_config_packed(vcfg, groups, AVAIL_10G)

    # ---- 逐组 offset 阶梯重放（pin 布局循环逐字同序）并与 pin 全量断言一致 ----
    replay_map = {}
    ladders, all_strips = [], []  # all_strips: (offset, page, kind_label, gid)
    dense_list = []
    for gi, g in enumerate(groups):
        spec = g.kv_cache_spec
        off, lad = 0, []
        for ln in g.layer_names:
            page = (spec.kv_cache_specs[ln].page_size_bytes
                    if isinstance(spec, UniformTypeKVCacheSpecs)
                    else spec.page_size_bytes)
            lab = kind_label(ln, ratios)
            lad.append({"layer": ln, "kind": lab, "page": page, "offset": off})
            all_strips.append((off, page, lab, gi))
            replay_map.setdefault(off, []).append(ln)
            off += page
        ladders.append(lad)
        dense_list.append(off)
    assert replay_map == {o: list(v) for o, v in layers_by_offset.items()}, \
        "阶梯重放与 pin layers_by_offset 不一致"

    # ---- 组×桶矩阵 ----
    buckets = sorted({s.page_size_bytes for s in specs.values()}, reverse=True)
    group_labels = []
    for gi, g in enumerate(groups):
        spec = g.kv_cache_spec
        group_labels.append(f"G{gi}({len(g.layer_names)}层,块{spec.block_size})")
    rows = []
    for b in buckets:
        cells = []
        for gi in range(len(groups)):
            members = [e for e in ladders[gi] if e["page"] == b]
            if members:
                kinds = Counter(e["kind"] for e in members)
                cells.append({
                    "gid": gi, "type": "valid",
                    "content": " + ".join(f"{k}×{c}" for k, c in sorted(kinds.items())),
                    "layers": len(members), "bytes": len(members) * b})
            else:
                dense = dense_list[gi]
                in_tail = [(o, o + p) for (o, p, _lab, _g) in all_strips
                           if p == b and o >= dense]
                cells.append({
                    "gid": gi, "type": "wasted",
                    "bytes_union_in_tail": union_len(in_tail),
                    "bytes_weighted": len(in_tail) * b,
                    "n_strips": len(in_tail)})
        rows.append({"bucket": b, "cells": cells})

    row_idle = [{"gid": gi,
                 "idle_bytes": stride - dense_list[gi],
                 "idle_pct": round((stride - dense_list[gi]) / stride * 100, 3)}
                for gi in range(len(groups))]

    # ---- 别名不可加总实证：逐组「三/四浪费格并集之和 vs 行闲置」 ----
    aliasing = []
    for gi in range(len(groups)):
        cells_sum = sum(c["bytes_union_in_tail"] for r in rows for c in r["cells"]
                        if c["gid"] == gi and c["type"] == "wasted")
        aliasing.append({"gid": gi, "wasted_cells_sum": cells_sum,
                         "row_idle": stride - dense_list[gi],
                         "cells_sum_exceeds_idle": cells_sum > stride - dense_list[gi]})

    return {
        "tuples_per_bucket": tuples, "gcd_chosen": chosen,
        "pad_total": sum((-x) % chosen for x in tuples),
        "block_stride": stride,
        "num_blocks_10gib": num_blocks,
        "floor_remainder_bytes": AVAIL_10G - stride * num_blocks,
        "groups": [{"gid": gi, "label": group_labels[gi],
                    "block_size": groups[gi].kv_cache_spec.block_size,
                    "layers": len(groups[gi].layer_names),
                    "dense": dense_list[gi]} for gi in range(len(groups))],
        "matrix": {"buckets": buckets, "rows": rows},
        "row_idle": row_idle,
        "aliasing_proof": aliasing,
    }


out = {"params": {
    "pin": "vLLM v0.27.1（instances/vllm/source，行号基线）",
    "method": "host 桩跑（同 run_v4_cache_groups.py 桩面：六处配置/指标替身，分组/"
              "近似 GCD/packed 布局/块数换算全部 pin 算法零改动；矩阵=对布局输出的"
              "只读后处理——逐组 offset 阶梯重放与 pin layers_by_offset 全量断言一致）",
    "matrix_convention": "行=4 页宽桶（降序）、列=5 组；valid 格=组内该页宽层数×页宽；"
                         "浪费格=该桶整条条带完全落在本组密排尾 [dense, block_stride) "
                         "内的并集字节（骑跨 dense 边界的条带不计；跨组同 offset 折叠"
                         "后去重，bytes_weighted 为不去重口径=条带数×页宽）；行闲置="
                         "block_stride−dense 是每列唯一可加总的浪费量（浪费格间跨桶"
                         "别名、不可加总，见 aliasing_proof）",
    "pro_ratios": "[4]*30 + [128]*31，sliding_window=8192（示教窗，dense/tuples/"
                  "stride 不随窗值变）——社区实读实发权重口径（61 层）",
    "flash_ratios": "[0,0] + [4,128]*20 + [4]，sliding_window=128（Flash config.json "
                    "官方窗）——43 层，本轮 pin 真跑",
    "cross_refs": "Pro dense/stride/tuples/7477 块对 traces/v4_cache_groups_61layer"
                  ".json 逐项断言（pro61.trace61_crosscheck）；矩阵 40 格与 dossier/"
                  "supplement-planning-matrix.json sup-plan-3 逐一断言相等"
                  "（supplement_check）；Flash stride 1002240 与 ch36 登记的 fp8 档"
                  " block_stride 一致",
}}


# ===========================================================================
# Pro 61 层：30 c4 + 31 c128
# ===========================================================================
ratios_pro = [4] * 30 + [128] * 31
pro = run_matrix(ratios_pro, 8192)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "v4_cache_groups_61layer.json"), encoding="utf-8") as f:
    trace61 = json.load(f)["layer61"]
pro["trace61_crosscheck"] = {
    "dense_match": [g["dense"] for g in pro["groups"]]
    == trace61["per_group_dense_width"],
    "stride_match": pro["block_stride"] == trace61["packed_layout"]["block_stride"],
    "tuples_match": pro["tuples_per_bucket"] == trace61["num_layer_tuples_list"],
    "num_blocks_match": pro["num_blocks_10gib"] == trace61["num_blocks_10gib"]["num_blocks"],
    "expected_num_blocks_10gib": 7477,
}
assert pro["trace61_crosscheck"]["dense_match"]
assert pro["trace61_crosscheck"]["stride_match"]
assert pro["trace61_crosscheck"]["tuples_match"]
assert pro["trace61_crosscheck"]["num_blocks_match"]
out["pro61"] = pro

# ===========================================================================
# Flash 43 层：[0,0] + [4,128]*20 + [4]，窗 128
# ===========================================================================
ratios_flash = [0, 0] + [4, 128] * 20 + [4]
flash = run_matrix(ratios_flash, 128)
assert flash["block_stride"] == 1002240  # ch36 登记的 fp8（8640 档）block_stride
out["flash43"] = flash

# ===========================================================================
# 博客三桶对照（推导口径：博客原文桶页宽未转录，按 pin 页宽代入 T1/T2/T3——
# supplement sup-plan-3 blog3bucket_matrix 同款）
# ===========================================================================
out["blog3bucket_derived"] = {
    "provenance": "推导口径：结构性结论（valid/wasted 格分布）= research 双转录"
                  "（community-narrative-figures.json A2-mqc-dsv4 + hybrid-v4-"
                  "background.json dsv4-kvcache-management-blog）；Byte 数字 = 按 pin "
                  "页宽代入 T1/T2/T3",
    "t1_main_c128_page": 1728, "t2_indexer_family_page": 8640,
    "t3_main_c4_swa_page": 37440,
    "per_block_footprint_derived": 1728 + 8640 + 37440,
    "footprint_note": "47808 = pin 主包一个层元组 [C4I, C4A, C128] 的宽度 = Flash G0 "
                      "阶梯的循环步长（pin 的块 = G0 整栈 block_stride：Pro 1435968 / "
                      "Flash 1002240，max 不是 sum）",
    "pin_fourth_width": 32832,
    "fourth_width_note": "pin 给 attn_state 对独立第四桶；博客三桶制里状态组『进 "
                         "T3/T2』（垫页算术未转录），pin 不垫",
}

# ===========================================================================
# 运行时行桥接（sup-plan-4 口径）：len=258 请求逐组 cdiv 需块数——矩阵列（组）
# 的运行时读法；管理面块长 = 五组 block_size（本轮 pin 实跑值）。cdiv 用 pin
# 自己的函数（kv_cache_utils 从 vllm.utils.math_utils 导入的 cdiv，零改动）。
# ===========================================================================
LEN_REQ = 258
pro_bs = [g["block_size"] for g in pro["groups"]]
pin_blocks_258 = [kcu.cdiv(LEN_REQ, bs) for bs in pro_bs]
BLOG_BLOCKS_258 = [2, 2, 2, 33, 65]  # 研究包双转录（博客图下 258 行）
assert pin_blocks_258 == [2, 5, 5, 65, 33], pin_blocks_258
assert BLOG_BLOCKS_258[0] == kcu.cdiv(LEN_REQ, 256)
assert BLOG_BLOCKS_258[3:] == [kcu.cdiv(LEN_REQ, 8), kcu.cdiv(LEN_REQ, 4)]
out["cdiv_258"] = {
    "request_len_tokens": LEN_REQ,
    "group_block_sizes": pro_bs,
    "pin_blocks": pin_blocks_258,
    "pin_formula_per_group": [f"cdiv(258,{bs})" for bs in pro_bs],
    "blog_blocks": BLOG_BLOCKS_258,
    "blog_provenance": "研究包双转录（community-narrative-figures.json A2-mqc-dsv4 "
                       "/ hybrid-v4-background.json dsv4-kvcache-management-blog）："
                       "博客 2/2/2/33/65——swa 组博客 2 vs pin 5，差在博客版 swa 走 "
                       "256 管理面、pin 是 64（被 C4A 主 KV 块形反定，sparse_swa.py:"
                       "L77-L82）；33=cdiv(258,8)、65=cdiv(258,4) 两版本同数",
    "source": "dossier/supplement-planning-matrix.json sup-plan-4 worked_example_"
              "material.cdiv_258_table（pin 侧真跑复算、博客侧转录）",
}

# ===========================================================================
# supplement_check：与 dossier/supplement-planning-matrix.json sup-plan-3 的
# worked_example_material 40 格 + 行闲置 + 头条数字逐一断言相等
# ===========================================================================
SUP_PRO_CELLS = {  # (bucket, gid) -> bytes_union_in_tail（wasted）或 bytes（valid）
    (37440, 0): 1123200, (37440, 1): 1160640, (37440, 2): 1123200,
    (37440, 3): 112320, (37440, 4): 290880,
    (32832, 0): 0, (32832, 1): 65664, (32832, 2): 98496,
    (32832, 3): 984960, (32832, 4): 1017792,
    (8640, 0): 259200, (8640, 1): 60480, (8640, 2): 69120,
    (8640, 3): 259200, (8640, 4): 112320,
    (1728, 0): 53568, (1728, 1): 53568, (1728, 2): 53568,
    (1728, 3): 53568, (1728, 4): 53568,
}
SUP_PRO_IDLE = [(0, 0.0), (275328, 19.174), (312768, 21.781),
                (191808, 13.357), (418176, 29.122)]
SUP_PRO_G4_37440_WEIGHTED = 449280
SUP_FLASH_CELLS = {
    (37440, 0): 786240, (37440, 1): 823680, (37440, 2): 786240,
    (37440, 3): 74880, (37440, 4): 293184,
    (32832, 0): 0, (32832, 1): 32832, (32832, 2): 65664,
    (32832, 3): 689472, (32832, 4): 656640,
    (8640, 0): 181440, (8640, 1): 43200, (8640, 2): 60480,
    (8640, 3): 181440, (8640, 4): 100800,
    (1728, 0): 34560, (1728, 1): 5184, (1728, 2): 6912,
    (1728, 3): 3456, (1728, 4): 12096,
}
SUP_FLASH_IDLE = [(0, 0.0), (178560, 17.816), (216000, 21.552),
                  (131328, 13.103), (345600, 34.483)]


def check_cells(mat, sup):
    bad = []
    for r in mat["rows"]:
        for c in r["cells"]:
            got = (c.get("bytes_union_in_tail") if c["type"] == "wasted"
                   else c.get("bytes"))
            if got != sup[(r["bucket"], c["gid"])]:
                bad.append({"bucket": r["bucket"], "gid": c["gid"],
                            "got": got, "supplement": sup[(r["bucket"], c["gid"])]})
    return bad


def check_idle(row_idle, sup):
    return [{"gid": r["gid"], "got": [r["idle_bytes"], r["idle_pct"]],
             "supplement": list(s)} for r, s in zip(row_idle, sup)
            if [r["idle_bytes"], r["idle_pct"]] != list(s)]


sc = {
    "pro61_cells_mismatch": check_cells(pro["matrix"], SUP_PRO_CELLS),
    "pro61_idle_mismatch": check_idle(pro["row_idle"], SUP_PRO_IDLE),
    "pro61_stride_match": pro["block_stride"] == 1435968,
    "pro61_dense_match": [g["dense"] for g in pro["groups"]]
    == [1435968, 1160640, 1123200, 1244160, 1017792],
    "pro61_gcd_match": pro["gcd_chosen"] == 31,
    "pro61_g4_37440_weighted_match": next(
        c["bytes_weighted"] for r in pro["matrix"]["rows"] for c in r["cells"]
        if r["bucket"] == 37440 and c["gid"] == 4) == SUP_PRO_G4_37440_WEIGHTED,
    "pro61_g4_cells_sum": next(a["wasted_cells_sum"] for a in pro["aliasing_proof"]
                               if a["gid"] == 4),
    "flash43_cells_mismatch": check_cells(flash["matrix"], SUP_FLASH_CELLS),
    "flash43_idle_mismatch": check_idle(flash["row_idle"], SUP_FLASH_IDLE),
    "flash43_stride_match": flash["block_stride"] == 1002240,
    "flash43_dense_match": [g["dense"] for g in flash["groups"]]
    == [1002240, 823680, 786240, 870912, 656640],
    "flash43_gcd_match": flash["gcd_chosen"] == 22,
    "flash43_pad_match": flash["pad_total"] == 5,
    "flash43_num_blocks_match": flash["num_blocks_10gib"] == 10713,
    "flash43_floor_remainder_match": flash["floor_remainder_bytes"] == 421120,
    "source": "dossier/supplement-planning-matrix.json sup-plan-3 worked_example_"
              "material（本轮 pin 真跑与 supplement 逐格断言）",
}
ok = (not sc["pro61_cells_mismatch"] and not sc["pro61_idle_mismatch"]
      and sc["pro61_stride_match"] and sc["pro61_dense_match"]
      and sc["pro61_gcd_match"] and sc["pro61_g4_37440_weighted_match"]
      and not sc["flash43_cells_mismatch"] and not sc["flash43_idle_mismatch"]
      and sc["flash43_stride_match"] and sc["flash43_dense_match"]
      and sc["flash43_gcd_match"] and sc["flash43_pad_match"]
      and sc["flash43_num_blocks_match"] and sc["flash43_floor_remainder_match"])
sc["all_match"] = ok
assert ok, f"supplement 对账失败: {sc}"
out["supplement_check"] = sc

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "planning_matrix.json")
with open(path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", path)
print("pro61: stride", pro["block_stride"], "tuples", pro["tuples_per_bucket"],
      "d", pro["gcd_chosen"], "blocks", pro["num_blocks_10gib"])
print("pro61 idle:", [(r["idle_bytes"], r["idle_pct"]) for r in pro["row_idle"]])
print("flash43: stride", flash["block_stride"], "tuples",
      flash["tuples_per_bucket"], "d", flash["gcd_chosen"], "pad",
      flash["pad_total"], "blocks", flash["num_blocks_10gib"],
      "rem", flash["floor_remainder_bytes"])
print("flash43 idle:", [(r["idle_bytes"], r["idle_pct"]) for r in flash["row_idle"]])
print("supplement_check all_match:", sc["all_match"],
      "| pro G4 cells_sum", sc["pro61_g4_cells_sum"])
