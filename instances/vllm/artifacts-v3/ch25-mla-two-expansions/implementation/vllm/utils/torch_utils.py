# SOURCE: vllm/utils/torch_utils.py
# ch25 消费面：is_quantized_kv_cache / kv_cache_dtype_str_to_dtype /
# get_dtype_size / LayerName opaque 族（_encode/_resolve）/
# direct_register_custom_op（统一算子真实注册——infer_schema 同真实）。
from __future__ import annotations

from typing import Any

import torch

# LayerNameType: torch>=2.11 用 str（# SOURCE: vllm/utils/torch_utils.py
#   LayerNameType 的版本分岔——本包按 str 承载）
LayerNameType = str


# SOURCE: vllm/utils/torch_utils.py:L75-L80 is_quantized_kv_cache —— 逐字
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
    )


# SOURCE: vllm/utils/torch_utils.py kv_cache_dtype_str_to_dtype —— 减法子集
#   （auto→model dtype 逐字；fp8 变体与打包布局按 host 需要承载）
def kv_cache_dtype_str_to_dtype(kv_cache_dtype: str, model_config=None) -> torch.dtype:
    # SOURCE: vllm/utils/torch_utils.py kv_cache_dtype_str_to_dtype
    if kv_cache_dtype == "auto":
        if model_config is not None:
            return model_config.dtype
        return torch.get_default_dtype()
    dtype = getattr(torch, kv_cache_dtype, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.uint8  # fp8_ds_mla 等打包布局（dtype 由 spec 侧特算）


# SOURCE: vllm/utils/torch_utils.py get_dtype_size —— 逐字
def get_dtype_size(dtype: torch.dtype) -> int:
    # SOURCE: vllm/utils/torch_utils.py get_dtype_size
    return torch.tensor([], dtype=dtype).element_size()


# ── LayerName opaque 族（# SOURCE: vllm/utils/torch_utils.py:L832-L889
#    —— 图内不烘焙字符串的编码层；host 直调路径只过 _resolve 的恒等回退）──


# SOURCE: vllm/utils/torch_utils.py _encode_layer_name —— 减法子集（str
#   恒等承载——无 torch.compile 图）
def _encode_layer_name(layer_name: str) -> str:
    # SOURCE: vllm/utils/torch_utils.py _encode_layer_name —— HOST SEAM 恒等
    return layer_name


# SOURCE: vllm/utils/torch_utils.py _resolve_layer_name —— 减法子集
def _resolve_layer_name(layer_name: Any) -> str:
    # SOURCE: vllm/utils/torch_utils.py _resolve_layer_name —— HOST SEAM 恒等
    return layer_name


# SOURCE: vllm/utils/torch_utils.py:L895 vllm_lib —— 逐字（算子库载体）
try:
    vllm_lib = torch.library.Library("vllm", "FRAGMENT")  # noqa
except RuntimeError:  # 已注册（重复 import）
    pass


# SOURCE: vllm/utils/torch_utils.py:L901-L939 direct_register_custom_op ——
#   逐字主干（infer_schema 推签名 + define + impl + fake——统一算子面照此
#   挂进 torch.ops.vllm）
def direct_register_custom_op(
    op_name: str,
    op_func: Any,
    mutates_args: list[str] | None = None,
    fake_impl: Any = None,
    target_lib=None,
    dispatch_key: str | None = None,
    tags: tuple = (),
) -> None:
    # SOURCE: vllm/utils/torch_utils.py:L901-L939 direct_register_custom_op
    """Define a custom op that runs `op_func` natively."""
    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key

    if hasattr(torch.ops.vllm, op_name):
        return  # HOST SEAM：重复注册守护（同 op 二次 import 的幂等位）

    from torch.library import infer_schema  # noqa: TID251  (真实 torch 内部 API 位)

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
