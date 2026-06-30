# 精简版（只做减法）— 对照真实源码 vllm_ascend/meta_registration.py
# C++ 侧把 _C_ascend::<op> 真实现注册到 PrivateUse1 后，本文件在 Python 层给「没有
# C++ meta」的算子补 Meta 派发键实现：只 empty 推形状、不真算，让算子能进 torch.compile/aclgraph。
import torch
from torch.library import Library

# SUBTRACTED: 顶部 42 行 how-to docstring（讲 Python vs C++ 两种 meta 写法、如何新增），
#             开发指引非运行逻辑；保留下面一句点题。  原 vllm_ascend/meta_registration.py:L6-L42
# Both approaches enable tracing/shape inference in PyTorch — essential for `torch.compile` and aclgraph.

from vllm_ascend.utils import is_310p

lib = Library("_C_ascend", "IMPL")


# SOURCE: vllm_ascend/meta_registration.py:L47
def register_meta_if_necessary(ns: str, op_name: str, fn, overload: str = ""):
    if overload != "":
        op_name = op_name + "." + overload
    schema_to_find = ns + "::" + op_name
    meta_impl_list = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
    if schema_to_find in meta_impl_list:
        return
    lib.impl(op_name, fn, "Meta")


# SOURCE: vllm_ascend/meta_registration.py:L57
def get_masked_input_and_mask_meta(
    input: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
):
    masked_input = torch.empty_like(input)
    mask = torch.empty_like(input).to(torch.bool)

    return masked_input, mask


# SOURCE: vllm_ascend/meta_registration.py:L71
def bgmv_expand_meta(
    x: torch.Tensor, weight: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, slice_offset: int, slice_size: int
):
    y_out = torch.empty_like(y)
    return y_out


# SOURCE: vllm_ascend/meta_registration.py:L78
def sgmv_expand_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_indices: torch.Tensor,
    seq_len: torch.Tensor,
    y: torch.Tensor,
    slice_offset: int,
    slice_size: int,
):
    y_out = torch.empty_like(y)
    return y_out


if not is_310p():
    register_meta_if_necessary("_C_ascend", "get_masked_input_and_mask", get_masked_input_and_mask_meta)
    register_meta_if_necessary("_C_ascend", "bgmv_expand", bgmv_expand_meta)
    register_meta_if_necessary("_C_ascend", "sgmv_expand", sgmv_expand_meta)
