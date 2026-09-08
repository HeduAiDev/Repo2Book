# SOURCE: vllm/utils/torch_utils.py
# ch23 消费面三件：set_default_torch_dtype（装配期 dtype 上下文，base_loader
# 消费）、LayerName/_resolve_layer_name/_encode_layer_name（Attention 算子的
# layer_name 通道）、direct_register_custom_op 位（SUBTRACTED——torch.library
# 注册面归 ch19，本章直调 Python 算子函数）。
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, TypeAlias, Union

import torch


# SOURCE: vllm/utils/torch_utils.py:L145-L150 set_default_torch_dtype（逐字）
@contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    # SOURCE: vllm/utils/torch_utils.py:L145-L150 set_default_torch_dtype（逐字）
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


# SOURCE: vllm/utils/torch_utils.py:L836-L845 LayerName —— HOST SEAM：str 直通
#   （真实为 torch>=2.11 的 opaque 类型防算子特化；host 编译面未启——
#   _USE_LAYERNAME=False 同款退化）
LayerNameType: TypeAlias = str


# SOURCE: vllm/utils/torch_utils.py:L882-L884 _resolve_layer_name
def _resolve_layer_name(layer_name: Union[str, LayerNameType]) -> str:
    """Unwrap a LayerName to str, or return str unchanged."""
    return layer_name


# SOURCE: vllm/utils/torch_utils.py:L887-L889 _encode_layer_name —— HOST SEAM：
#   str 直返（_USE_LAYERNAME=False 分支的逐字退化）
def _encode_layer_name(layer_name: str) -> Union[str, LayerNameType]:
    """Wrap a str layer name as LayerName when enabled."""
    # SOURCE: vllm/utils/torch_utils.py:L887-L889 _encode_layer_name —— HOST SEAM
    return layer_name


# SOURCE: vllm/utils/torch_utils.py:L901-L934 direct_register_custom_op
#   SUBTRACTED（torch.library 注册面归 ch19；本章算子以 Python 函数直调，
#   同控制流）
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    **kwargs,
):
    # SOURCE: vllm/utils/torch_utils.py:L901-L934 direct_register_custom_op
    raise NotImplementedError("SUBTRACTED: torch.library 注册面 → ch19")


# SOURCE: vllm/utils/torch_utils.py:L395-L401 kv_cache_dtype_str_to_dtype
#   （逐字——"auto" 跟随模型 dtype 的解析）
def kv_cache_dtype_str_to_dtype(
    kv_cache_dtype: str, model_config
) -> torch.dtype:
    # SOURCE: vllm/utils/torch_utils.py:L395-L401 kv_cache_dtype_str_to_dtype
    if kv_cache_dtype == "auto":
        # Model config may not be specified for unit tests, default to float16
        return model_config.dtype if model_config else torch.half
    STR_DTYPE_TO_TORCH_DTYPE = {
        "auto": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    # SUBTRACTED: fp8/nvfp4 等量化 KV dtype 项——ch27 域
    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
