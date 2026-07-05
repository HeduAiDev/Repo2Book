# vllm_ascend/utils.py —— subtract-only 精简版（ch16 KV 显存几何用到的辅助函数）
#
# 只摘出本章 _allocate / _reshape / bind 三步会调用的纯函数；utils.py 其余上千行
# （平台探测 / 通信 / 量化辅助 等）与本章正交，整体折叠。
# SUBTRACTED: vllm_ascend/utils.py 中除下列三个函数外的全部内容（数百个符号），
#   与「KV 张量分配/重排/绑定」无关，按 subtraction_plan 折叠。
from typing import Any


# SOURCE: vllm_ascend/utils.py:L82
def extract_dsv4_layer_index(config: Any, layer_name: str) -> int:
    """Extract DSV4 index for config per-layer arrays.

    Runtime module names keep their original MTP namespace, e.g. ``mtp.0``.
    When indexing config-level arrays such as ``compress_ratios``, MTP layers
    are addressed after the main model layers.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    layer_idx = extract_layer_index(layer_name)
    # TODO(zzzzwwjj): the layer idx of mtp should be aligned with vLLM
    if ".mtp." in f".{layer_name}." and layer_idx < config.num_hidden_layers:
        return config.num_hidden_layers + layer_idx
    return layer_idx


# SOURCE: vllm_ascend/utils.py:L1441
def calc_split_factor(num_list: list[int]):
    total = sum(num_list)
    return [total / num for num in num_list]


# SOURCE: vllm_ascend/utils.py:L1546
def kv_cache_spec_uses_sparse_c8(kv_cache_spec) -> bool:
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    return isinstance(kv_cache_spec, MLAAttentionSpec) and bool(getattr(kv_cache_spec, "cache_sparse_c8", False))
