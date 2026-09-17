"""ch28-m11 驱动脚本 —— expert_dtype 分发: 同一个 '.scale' 后缀翻成两套参数名 + expert_mapping 落位。

跑法(host, 纯 CPU, 无 torch 依赖): python run_ch29_m11_scale_suffix_mapper.py
输出: ch29_m11_scale_suffix_mapper.json(与本脚本同目录)

素材对应 dossier 机制 ch28-m11(needs_worked_example)。

真源(逐字拷贝):
- _make_deepseek_v4_weights_mapper 的三张映射表(regex/prefix/suffix/substr)逐字拷贝自
  vllm/models/deepseek_v4/nvidia/model.py:L1383-L1417:
  fp4 路径两条 regex((\.experts\.\d+\.w[123])\.scale$ → \1.weight_scale 先,
  \.scale$ → .weight_scale_inv 后), fp8 路径一条(全部 .scale → .weight_scale_inv)。
- WeightsMapper._map_name_with_shard 的应用顺序逐字复刻自
  vllm/model_executor/models/utils.py:L98-L138: regex(按 dict 序逐条 search+sub,
  后条可作用于前条产物) → substr → prefix → suffix。
- make_deepseek_v4_expert_params_mapping 逐字拷贝自 model.py:L149-L165;
  树内测试 tests/models/test_deepseek_v4_mega_moe.py:L20-L33 断言其输出
  (2 专家 6 元组), 本脚本复跑同参对拍。
- 惰性解析(expert_dtype property)是 vllm_config 时序问题, 不在 host 复现范围,
  引 quant_config.py:L50-L70(docstring NOTE: 构造发生在 set_current_vllm_config
  之前, 急切读恒见默认 'fp4' 且 silently misroute Flash-Base)。

三件:
① 8 个代表性 checkpoint 键 × {fp4, fp8} 两条路径的 mapper 产物对照——
  专家 scale 与非专家 scale 在 fp4 路径分家、在 fp8 路径合流。
② expert_mapping 落位演示: mapper 产物 '...experts.1.w1.weight_scale' 再经
  expert_mapping 翻成参数名 'experts.w13_weight_scale'(含 w13 打包)。
③ 与树内测试对拍(make_deepseek_v4_expert_params_mapping(2))。
"""
import json
import re
from pathlib import Path

# ---- make_deepseek_v4_expert_params_mapping 逐字 (model.py:L149-L165) ----


def make_deepseek_v4_expert_params_mapping(num_experts: int) -> list:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]


# ---- _make_deepseek_v4_weights_mapper 的映射表逐字 (model.py:L1383-L1417) ----
FP4_SCALE_REGEX = {
    re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
    re.compile(r"\.scale$"): ".weight_scale_inv",
}
FP8_SCALE_REGEX = {
    re.compile(r"\.scale$"): ".weight_scale_inv",
}
PREFIX = {
    "layers.": "model.layers.",
    "embed.": "model.embed.",
    "norm.": "model.norm.",
    "hc_head": "model.hc_head",
    "mtp.": "model.mtp.",
}
SUFFIX = {
    "head.weight": "lm_head.weight",
    "embed.weight": "embed_tokens.weight",
    ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
}
SUBSTR = {
    ".shared_experts.w2": ".shared_experts.down_proj",
}


def map_name(key: str, scale_regex: dict) -> str:
    """WeightsMapper._map_name_with_shard 的顺序复刻 (utils.py:L98-L138):
    regex(dict 序) → substr → prefix → suffix; DSV4 mapper 无 stacked/renaming。"""
    for pattern, new_key in scale_regex.items():
        if pattern.search(key):
            key = pattern.sub(new_key, key)
    for substr, new_key in SUBSTR.items():
        if substr in key:
            key = key.replace(substr, new_key, 1)
    for prefix, new_key in PREFIX.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_key, 1)
    for suffix, new_key in SUFFIX.items():
        if key.endswith(suffix):
            key = new_key.join(key.rsplit(suffix, 1))
    return key


KEYS = [
    "layers.3.ffn.experts.17.w1.scale",
    "layers.3.ffn.experts.17.w2.scale",
    "layers.3.ffn.experts.17.w3.weight",
    "layers.3.ffn.gate.scale",
    "layers.3.self_attn.wq_a.scale",
    "layers.3.ffn.shared_experts.w2.scale",
    "head.weight",
    "embed.weight",
]

cases = {}
for k in KEYS:
    cases[k] = {
        "fp4(expert_dtype='fp4')": map_name(k, FP4_SCALE_REGEX),
        "fp8(expert_dtype='fp8')": map_name(k, FP8_SCALE_REGEX),
    }

# expert_mapping 落位: mapper 之后, load_weights 主循环用 expert_mapping 再翻一次
mapping = make_deepseek_v4_expert_params_mapping(2)
placement = {}
for mapped in ["model.layers.3.ffn.experts.1.w1.weight_scale",
               "model.layers.3.ffn.experts.1.w2.weight",
               "model.layers.3.ffn.experts.1.w3.weight"]:
    for param_name, weight_name, expert_id, shard_id in mapping:
        if weight_name in mapped:
            placement[mapped] = mapped.replace(weight_name, param_name)
            break

tree_expected = [
    ("experts.w13_", "experts.0.w1.", 0, "w1"),
    ("experts.w2_", "experts.0.w2.", 0, "w2"),
    ("experts.w13_", "experts.0.w3.", 0, "w3"),
    ("experts.w13_", "experts.1.w1.", 1, "w1"),
    ("experts.w2_", "experts.1.w2.", 1, "w2"),
    ("experts.w13_", "experts.1.w3.", 1, "w3"),
]
cross_check_ok = mapping == tree_expected

out = {
    "env": "host Miniconda python 3.11.11 (纯 CPU, 无 torch), pin=vLLM v0.27.1 (6e448d0ea)",
    "refs_copied_verbatim": [
        "vllm/models/deepseek_v4/nvidia/model.py:L1383-L1417 (_make_deepseek_v4_weights_mapper 映射表)",
        "vllm/model_executor/models/utils.py:L98-L138 (WeightsMapper 应用顺序 regex→substr→prefix→suffix)",
        "vllm/models/deepseek_v4/nvidia/model.py:L149-L165 (make_deepseek_v4_expert_params_mapping)",
    ],
    "lazy_resolution_note": "expert_dtype property 惰性解析: 构造期 set_current_vllm_config 未生效时读恒返回默认 'fp4', 首次在生效环境读到才定型并 info_once (quant_config.py:L50-L70 docstring NOTE 原话: eagerly 读会 'silently misroute Flash-Base checkpoints'); is_scale_e8m0=(expert_dtype=='fp4')",
    "cases": cases,
    "expert_mapping_placement": placement,
    "expert_mapping_tree_test_cross_check": {
        "expected": [list(t) for t in tree_expected],
        "got": [list(t) for t in mapping],
        "ok": bool(cross_check_ok),
        "provenance": "tests/models/test_deepseek_v4_mega_moe.py:L20-L33",
    },
}

dst = Path(__file__).parent / "ch29_m11_scale_suffix_mapper.json"
with open(dst, "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", dst)
print(json.dumps(cases, ensure_ascii=True, indent=1))
print("cross_check_ok:", cross_check_ok)
print(json.dumps(placement, ensure_ascii=True, indent=1))
