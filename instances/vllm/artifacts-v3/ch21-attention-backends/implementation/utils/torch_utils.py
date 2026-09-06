# Subtract-only companion for v3 ch21 — vllm/utils/torch_utils.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus 章范围
# 外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# Kept surface: is_quantized_kv_cache / canonicalize_singleton_dim_strides /
# kv_cache_dtype_str_to_dtype（flash_attn 消费面）、LayerName opaque 族
# （forward op 的 layer_name 参数编码，ch19 已立）、direct_register_custom_op
# （统一算子注册面——unified_kv_cache_update/unified_attention_with_output
# 的 torch.ops.vllm 路径真实可走）。
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from packaging import version
from torch.library import Library, infer_schema

from .._host_seams import envs

# SUBTRACTED: vllm/utils/torch_utils.py L1-L74 的工具族（np/dtype 映射表、
#   PIN_MEMORY、async_tensor_h2d 等）——ch13/ch17/ch22 域，本章零调用。


# SOURCE: vllm/utils/torch_utils.py:L32-L42 STR_DTYPE_TO_TORCH_DTYPE ——本章
#   消费子集（kv_cache_dtype_str_to_dtype 的查表面；fp4/int8 等量化档随
#   ch27 域删）
STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
}


# SOURCE: vllm/utils/torch_utils.py:L75-L80 is_quantized_kv_cache ——（逐字）
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
    )


# SUBTRACTED: kv_cache_uses_per_token_head_scales / is_strictly_contiguous
#   （L82-L115）——量化标定域（ch27）/本章零调用。


# SOURCE: vllm/utils/torch_utils.py:L118-L145 canonicalize_singleton_dim_strides
#   ——（逐字）修 size=1 维的退化步长（FA3/4 TMA 的 ≥16 字节对齐要求）
def canonicalize_singleton_dim_strides(t: torch.Tensor) -> torch.Tensor:  # SOURCE: vllm/utils/torch_utils.py
    """Fix degenerate strides on size=1 dimensions for CUDA TMA compatibility.

    PyTorch allows any stride on a size=1 dim (is_contiguous() is always True
    there), so a size=1 dim may have stride=1 (2 bytes for bf16) instead of
    the canonical product(shape[i+1:]).  CUDA TMA on H100+ requires all
    non-outermost strides to be ≥16-byte aligned; stride=1 triggers
    cudaErrorIllegalInstruction.  Zero-copy: patches stride metadata only via
    as_strided; returns t unchanged if all size=1 strides are already canonical.
    """
    if 1 not in t.shape:
        return t
    strides = list(t.stride())
    shape = t.shape
    prev_stride = 1
    changed = False
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == 1 and strides[i] != prev_stride:
            strides[i] = prev_stride
            changed = True
        prev_stride = strides[i] * shape[i]
    if not changed:
        return t
    return t.as_strided(t.shape, strides)


# SUBTRACTED: set_default_torch_dtype / set_default_torch_num_threads /
#   current_stream / weak_ref_tensor(s)（L147-L763）——ch19 捕获域与 ch17
#   工具域，本章零调用。

# SOURCE: vllm/utils/torch_utils.py:L785-L786 _is_torch_equal_or_newer（逐字）
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


# SOURCE: vllm/utils/torch_utils.py:L789 is_torch_equal_or_newer（逐字）
def is_torch_equal_or_newer(target: str) -> bool:
    return _is_torch_equal_or_newer(torch.__version__, target)


# SOURCE: vllm/utils/torch_utils.py:L832 HAS_OPAQUE_TYPE（逐字）
HAS_OPAQUE_TYPE = is_torch_equal_or_newer("2.11.0.dev")

# Allow toggling LayerName usage via environment variable.
# Defaults to True on torch >= 2.11, False otherwise.
# Set VLLM_USE_LAYERNAME=0 to disable even on torch >= 2.11.
# SOURCE: vllm/utils/torch_utils.py:L834-L837 _USE_LAYERNAME（逐字）
_USE_LAYERNAME = HAS_OPAQUE_TYPE and envs.VLLM_USE_LAYERNAME

if HAS_OPAQUE_TYPE:
    from torch._opaque_base import OpaqueBase
else:
    OpaqueBase = object  # type: ignore[misc, assignment]


# SOURCE: vllm/utils/torch_utils.py:L845-L864 LayerName —— opaque 层名类型
#   （torch.compile 把它 lift 成图输入而非烤成常量）
class LayerName(OpaqueBase):  # type: ignore[misc]
    """Wraps a module name string for use as a torch opaque type.

    When torch >= 2.11, this is registered as a hoisted value-type opaque
    object so that torch.compile lifts it as a graph input instead of baking
    it as a constant.  This avoids per-layer recompilation for custom ops
    that accept layer name strings (attention, MOE, KV cache, etc.).
    """

    # SOURCE: vllm/utils/torch_utils.py:L854-L855 LayerName.__init__
    def __init__(self, value: str):
        self.value = value

    # SOURCE: vllm/utils/torch_utils.py:L857-L858 LayerName.__eq__
    def __eq__(self, other):
        return isinstance(other, LayerName) and self.value == other.value

    # SOURCE: vllm/utils/torch_utils.py:L860-L861 LayerName.__hash__
    def __hash__(self):
        return hash(self.value)

    # SOURCE: vllm/utils/torch_utils.py:L863-L864 LayerName.__fx_repr__
    def __fx_repr__(self):
        return (f"LayerName({self.value!r})", {"LayerName": LayerName})


if HAS_OPAQUE_TYPE:
    # SOURCE: vllm/utils/torch_utils.py:L867-L870 register_opaque_type(hoist)
    from torch._library.opaque_object import register_opaque_type

    register_opaque_type(LayerName, typ="value", hoist=True)

# On torch >= 2.11 (with VLLM_USE_LAYERNAME enabled), custom op
# layer_name parameters use LayerName; otherwise they remain plain str.
if TYPE_CHECKING:
    from typing import TypeAlias

    LayerNameType: TypeAlias = str | LayerName
else:
    # SOURCE: vllm/utils/torch_utils.py:L879 LayerNameType 运行期取值（逐字）
    LayerNameType = LayerName if _USE_LAYERNAME else str


# SOURCE: vllm/utils/torch_utils.py:L882-L884 _resolve_layer_name（逐字）
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    """Unwrap a LayerName to str, or return str unchanged."""
    return layer_name.value if isinstance(layer_name, LayerName) else layer_name


# SOURCE: vllm/utils/torch_utils.py:L887-L889 _encode_layer_name（逐字）
def _encode_layer_name(layer_name: str) -> str | LayerName:
    """Wrap a str layer name as LayerName when enabled."""
    return LayerName(layer_name) if _USE_LAYERNAME else layer_name


# SUBTRACTED: supports_xpu_graph（L892-L894——XPU 平台谓词）。

# create a library to hold the custom op
# SOURCE: vllm/utils/torch_utils.py:L897-L898 vllm_lib（逐字）
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


# SOURCE: vllm/utils/torch_utils.py:L901-L939 direct_register_custom_op ——（逐字）
#   统一算子的低开销注册面（unified_kv_cache_update 等）
def direct_register_custom_op(  # SOURCE: vllm/utils/torch_utils.py
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to bind the operator to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: The lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is being used.
    """
    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from .._host_seams import current_platform

        dispatch_key = current_platform.dispatch_key

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


# SOURCE: vllm/utils/torch_utils.py:L395-L401 kv_cache_dtype_str_to_dtype ——
#   （逐字；ModelConfig 类型面以 Any 注记承载——ch03 配置域）
def kv_cache_dtype_str_to_dtype(  # SOURCE: vllm/utils/torch_utils.py
    kv_cache_dtype: str, model_config: Any
) -> torch.dtype:
    if kv_cache_dtype == "auto":
        # Model config may not be specified for unit tests, default to float16
        return model_config.dtype if model_config else torch.half
    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]


# SOURCE: vllm/utils/torch_utils.py:L212-L214 get_dtype_size ——（逐字）
def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# SOURCE: vllm/utils/torch_utils.py:L414-L416 nvfp4_kv_cache_full_dim ——（逐字）
def nvfp4_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for NVFP4 KV cache: fp4 data + fp8 block scales."""
    return head_size // 2 + head_size // 16
