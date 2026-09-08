# SOURCE: vllm/ir/ops/layernorm.py
# RMSNorm 的数学本体（m1：融合 add-norm 的二元组语义证据）。算子注册/
# 输入生成器/容差覆盖删除（见包 __init__ 注记）；maybe_inplace 的 inplace
# 变体对返回值与 native 调用等价（op.py:L530-L534 未启 torch wrap 时同走
# native impl），HOST SEAM 以直调承载。
from __future__ import annotations

import torch
from torch import Tensor


# SOURCE: vllm/ir/ops/layernorm.py:L10-L22 rms_norm（逐字——去 @register_op）
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


# SUBTRACTED: rms_norm 的 input_generator / tolerance override
#   （vllm/ir/ops/layernorm.py:L25-L41）——算子测试基建，本章不进。


# SOURCE: vllm/ir/ops/layernorm.py:L44-L61 fused_add_rms_norm（native 数学逐字）
def _fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused add and weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)

    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


# SOURCE: vllm/ir/op.py:L481-L498 IrOpInplace（allow_inplace 装载了 maybe_inplace
#   重载）——HOST SEAM：直调 native（op.py:L530-L534 未启 torch wrap 的同款路径；
#   inplace 变体只额外允许原地写第一激活参数，返回值等价）
class _IrOpInplaceSeam:
    # SOURCE: vllm/ir/op.py:L515 —— maybe_inplace 名字位
    name = "fused_add_rms_norm.maybe_inplace"

    # SOURCE: vllm/ir/op.py:L529-L533 IrOpInplaceOverload.__call__ —— HOST SEAM
    def __call__(self, *args, **kwargs):
        return _fused_add_rms_norm(*args, **kwargs)


# SOURCE: vllm/ir/ops/layernorm.py:L44 fused_add_rms_norm 算子对象
#   （IrOpInplace 实例；maybe_inplace 属性位 op.py:L497）
class _FusedAddRMSNormOp:
    maybe_inplace = _IrOpInplaceSeam()

    # SOURCE: vllm/ir/op.py:L529-L533 —— 默认重载同样直调 native
    def __call__(self, *args, **kwargs):
        return _fused_add_rms_norm(*args, **kwargs)


fused_add_rms_norm = _FusedAddRMSNormOp()
