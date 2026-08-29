# Subtract-only companion for v3 ch19 — vllm/utils/torch_utils.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 章范围外域段以 SUBTRACTED+归属注记收窄（见 impl-notes §范围裁剪）。
#
# Kept surface: the LayerName opaque-type cluster (m06), current_stream /
# weak_ref_tensors (m14 capture/replay inputs), is_torch_equal_or_newer,
# HAS_OPAQUE_TYPE, and direct_register_custom_op (the registration face the
# attention op trio goes through).
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from packaging import version
from torch.library import Library, infer_schema

from .._host_seams import envs, init_logger

logger = init_logger(__name__)

# SUBTRACTED: vllm/utils/torch_utils.py 其余工具族（async_tensor_h2d /
#   get_accelerator_view_from_cpu_tensor / supports_xpu_graph / CUDA device
#   string helpers 等，L40-L660——ch17 立过的通用工具域，本章零调用）。


# SOURCE: vllm/utils/torch_utils.py:L641-L652 torch.cuda.set_stream patch ——
#   让 set_stream 顺手记 TLS，current_stream 不必每次重建 stream 对象
_current_stream_tls = threading.local()

_prev_set_stream = torch.cuda.set_stream


# SOURCE: vllm/utils/torch_utils.py:L648-L652 _patched_set_stream
def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    _current_stream_tls.value = stream
    _prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


# SOURCE: vllm/utils/torch_utils.py:L657-L659 _StreamPlaceholder
class _StreamPlaceholder:
    def __init__(self):  # SOURCE: vllm/utils/torch_utils.py:L657-L659
        self.synchronize = lambda: None


# SOURCE: vllm/utils/torch_utils.py:L662-L699 current_stream —— 捕获/回放
#   共用同一专用 stream 的取流入口（CUDA graph 捕获不能用默认流）
def current_stream() -> torch.cuda.Stream:  # SOURCE: vllm/utils/torch_utils.py:L662-L699
    """
    replace `torch.cuda.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    from .._host_seams import current_platform

    if not hasattr(_current_stream_tls, "value") or _current_stream_tls.value is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        # On ROCm using the default 0 stream in combination with RCCL
        # is hurting performance.
        # On CUDA, we capture and replay cudagraph on the same stream,
        # so we need to avoid using the default stream as well. The default
        # stream cannot be used for cudagraph capture, see
        # https://github.com/pytorch/pytorch/blob/42ad9edfb754743fdae3276ade43de000beb4f60/aten/src/ATen/CUDAGraph.cpp#L77
        # for more details. Therefore, we create a dedicated stream per process.
        if current_platform.is_rocm() or current_platform.is_cuda():
            # torch.cuda.set_stream here is the alias of _pathed_set_stream
            torch.cuda.set_stream(torch.cuda.Stream())
        elif current_platform.is_cpu():
            _current_stream_tls.value = _StreamPlaceholder()
        else:
            current_stream = current_platform.current_stream
            if current_stream is not None:
                _current_stream_tls.value = current_stream()
            else:
                raise ValueError(
                    "Fail to set current stream, current platform "
                    "may not support current_stream with torch API"
                )
    return _current_stream_tls.value


# SUBTRACTED: aux_stream（L702-L722——MoE shared-expert overlap 的辅助流）与
#   get_accelerator_view_from_cpu_tensor（L766-L781——UVA 视图，ch14 域）。


# SOURCE: vllm/utils/torch_utils.py:L725-L735 weak_ref_tensor
def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    This ignores 0-size tensors as those don't allocate any memory.
    """
    if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        return torch.ops._C.weak_ref_tensor(tensor)
    else:
        return tensor


# SOURCE: vllm/utils/torch_utils.py:L738-L763 weak_ref_tensors —— 捕获输出
#   弱引用（省显存；末片才安全）；IntermediateTensors 臂随 PP 域删
def weak_ref_tensors(  # SOURCE: vllm/utils/torch_utils.py:L738-L763
    tensors: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor | list[Any] | tuple[Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    # SUBTRACTED: IntermediateTensors 分支（L755-L762——PP 域）。
    raise ValueError("Invalid type for tensors")


# SOURCE: vllm/utils/torch_utils.py:L785-L786 _is_torch_equal_or_newer
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


# SOURCE: vllm/utils/torch_utils.py:L789 is_torch_equal_or_newer
def is_torch_equal_or_newer(target: str) -> bool:
    return _is_torch_equal_or_newer(torch.__version__, target)


# SOURCE: vllm/utils/torch_utils.py:L832 HAS_OPAQUE_TYPE
HAS_OPAQUE_TYPE = is_torch_equal_or_newer("2.11.0.dev")

# Allow toggling LayerName usage via environment variable.
# Defaults to True on torch >= 2.11, False otherwise.
# Set VLLM_USE_LAYERNAME=0 to disable even on torch >= 2.11.
# SOURCE: vllm/utils/torch_utils.py:L834-L837 _USE_LAYERNAME
_USE_LAYERNAME = HAS_OPAQUE_TYPE and envs.VLLM_USE_LAYERNAME

if HAS_OPAQUE_TYPE:
    from torch._opaque_base import OpaqueBase
else:
    OpaqueBase = object  # type: ignore[misc, assignment]


# SOURCE: vllm/utils/torch_utils.py:L845-L864 LayerName —— opaque 层名类型：
#   torch.compile 把它 lift 成图输入而非烤成常量，避免逐层重编译
class LayerName(OpaqueBase):  # type: ignore[misc]
    """Wraps a module name string for use as a torch opaque type.

    When torch >= 2.11, this is registered as a hoisted value-type opaque
    object so that torch.compile lifts it as a graph input instead of baking
    it as a constant.  This avoids per-layer recompilation for custom ops
    that accept layer name strings (attention, MOE, KV cache, etc.).
    """

    def __init__(self, value: str):  # SOURCE: vllm/utils/torch_utils.py:L854-L855
        self.value = value

    def __eq__(self, other):  # SOURCE: vllm/utils/torch_utils.py:L857-L858
        return isinstance(other, LayerName) and self.value == other.value

    def __hash__(self):  # SOURCE: vllm/utils/torch_utils.py:L860-L861
        return hash(self.value)

    def __fx_repr__(self):  # SOURCE: vllm/utils/torch_utils.py:L863-L864
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
    # SOURCE: vllm/utils/torch_utils.py:L879 LayerNameType 运行期取值
    LayerNameType = LayerName if _USE_LAYERNAME else str


# SOURCE: vllm/utils/torch_utils.py:L882-L884 _resolve_layer_name
def _resolve_layer_name(layer_name: str | LayerName) -> str:
    """Unwrap a LayerName to str, or return str unchanged."""
    return layer_name.value if isinstance(layer_name, LayerName) else layer_name


# SOURCE: vllm/utils/torch_utils.py:L887-L889 _encode_layer_name
def _encode_layer_name(layer_name: str) -> str | LayerName:
    """Wrap a str layer name as LayerName when enabled."""
    return LayerName(layer_name) if _USE_LAYERNAME else layer_name


# SUBTRACTED: supports_xpu_graph（L892-L894——XPU 平台谓词）。

# create a library to hold the custom op
# SOURCE: vllm/utils/torch_utils.py:L897-L898 vllm_lib
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


# SOURCE: vllm/utils/torch_utils.py:L901-L939 direct_register_custom_op —
#   统一算子（unified_attention_with_output 等）的低开销注册面
def direct_register_custom_op(  # SOURCE: vllm/utils/torch_utils.py:L901-L939
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
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of
    the library object. If you want to bind the operator to a different
    library, make sure the library object is alive when the operator
    is used.
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
