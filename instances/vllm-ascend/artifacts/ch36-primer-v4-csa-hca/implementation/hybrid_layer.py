"""ch36 §1 / §2.3 (paper.md, arXiv:2606.19348) -- CSA/HCA 层间交错的『开关表』与单层装配。

论文原句(§1):"we design a hybrid attention mechanism combining Compressed Sparse
Attention (CSA) and Heavily Compressed Attention (HCA)"——具体交错成什么样的比例/
顺序,论文没有给出逐层数组(那是发行版配置文件里的数字),但"每层读一个 compress_ratio,
4=CSA、128=HCA、其余=稠密"这件"怎么读、怎么决定挂什么模块"的逻辑,论文与落地代码
完全一致,这正是本文件要落地的部分。

落地:vllm_ascend/utils.py:L105-110(get_dsv4_compress_ratio,逐层开关表读取入口)、
vllm_ascend/models/deepseek_v4.py:L790-834(DeepseekV4Attention.__init__ 依
compress_ratio 挂载 Compressor/Indexer)、L1044-1048(DeepseekV4Model.make_layers
按 compress_ratios 交错建层)、vllm_ascend/models/layer/attention/layer.py:
L174-192(DSAAttention.get_kv_cache_spec,compress_ratio<=1 走独立 SWA 缓存)。
"""
from dataclasses import dataclass


# PAPER: §1 引言"a hybrid attention mechanism combining CSA and HCA";落地对照
# vllm_ascend/utils.py:L105-110 —— 逐层读 compress_ratios,越界或未配置视为稠密(0)
def get_dsv4_compress_ratio(compress_ratios: list[int] | None, layer_idx: int) -> int:
    """compress_ratios 为 None 或 layer_idx 越界(如 MTP 附加层未配置)时,该层按稠密处理。"""
    if compress_ratios is None or layer_idx >= len(compress_ratios):
        return 0
    return compress_ratios[layer_idx]


# PAPER: §2.3 首段(CSA/HCA 两种层的区分标准);落地对照 deepseek_v4.py:L790-834 ——
# 一层注意力的装配描述:compress_ratio 决定 kind(CSA/HCA/dense)、是否挂 Compressor
# (compress_ratio>1 必挂)、是否挂 Indexer(仅 compress_ratio==4 才挂,HCA 已把序列压到
# 1/128,块数极少,稠密注意力已足够便宜,不需要 top-k 稀疏选块)
# PAPER: §2.3 首段(见上)
@dataclass
class LayerSpec:
    layer_idx: int
    compress_ratio: int
    kind: str          # "CSA" | "HCA" | "dense"
    has_compressor: bool
    has_indexer: bool


# PAPER: §2.3 首段"CSA integrates both compression and sparse attention...HCA aims
# for extreme compression...consolidating...into a single entry"——由 compress_ratio
# 决定这一层是哪一种;落地对照 deepseek_v4.py:L790-834
def build_layer_spec(layer_idx: int, compress_ratio: int) -> LayerSpec:
    if compress_ratio == 4:
        return LayerSpec(layer_idx, compress_ratio, "CSA", has_compressor=True, has_indexer=True)
    if compress_ratio == 128:
        return LayerSpec(layer_idx, compress_ratio, "HCA", has_compressor=True, has_indexer=False)
    if compress_ratio <= 1:
        return LayerSpec(layer_idx, compress_ratio, "dense", has_compressor=False, has_indexer=False)
    raise ValueError(f"Only support compress_ratio in [4, 128] (or <=1 for dense). Got: {compress_ratio}")


# PAPER: §2.3 首段 + §1 —— 按整份 compress_ratios 交错建出整个模型的层序列;
# 落地对照 vllm_ascend/models/deepseek_v4.py:L1044-1048(DeepseekV4Model.make_layers)
def build_model_layers(compress_ratios: list[int]) -> list[LayerSpec]:
    return [build_layer_spec(i, get_dsv4_compress_ratio(compress_ratios, i)) for i in range(len(compress_ratios))]


# PAPER: §2.3.4 "the attention module...achieves remarkable efficiency" 的落地前提——
# 不同 kind 的层用不同形状的 KV 缓存;落地对照
# vllm_ascend/models/layer/attention/layer.py:L174-192(DSAAttention.get_kv_cache_spec)
def kv_cache_spec_for_layer(spec: LayerSpec) -> str:
    """compress_ratio<=1 的稠密层走独立的滑窗(SWA)缓存;CSA/HCA 层走带 compress_ratio
    标记的 MLA 式缓存(缓存的是压缩后的 KV 条目,不是原始 token 的 KV)。"""
    if spec.kind == "dense":
        return "SWA"
    return f"MLA(compress_ratio={spec.compress_ratio})"
