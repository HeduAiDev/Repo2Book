# SOURCE: vllm/model_executor/layers/layernorm.py
# ch23 消费面：RMSNorm（m1 双 RMSNorm 积木 + 融合 add-norm 的二元组语义）。
# SUBTRACTED：poly_norm/GemmaRMSNorm/RMSNormGated/LayerNorm 等其余 norm 族
#   （layernorm.py 其余）——模型特例域。
from __future__ import annotations

import torch
import torch.nn as nn

import vllm.envs as envs
import vllm.ir as ir
from vllm.model_executor.custom_op import CustomOp

# SUBTRACTED: import vllm.kernels / batch_invariant 面（layernorm.py:L8-L15）
#   ——kernel 注册与 batch invariant 数学域（ch20）；host 走 forward_native


# SOURCE: vllm/model_executor/layers/layernorm.py:L36-L37 RMSNorm 注册装饰
@CustomOp.register("rms_norm")
# SOURCE: vllm/model_executor/layers/layernorm.py:L37 RMSNorm
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # SOURCE: vllm/model_executor/layers/layernorm.py:L45-L73 __init__（逐字）
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

        # When has_weight=False, pass weight=None so implementations that
        # support a weightless path can skip the per-channel multiply.
        # Implementations that require weight (e.g. oink) fall back via IR
        # op priority when weight=None is unsupported.
        self.pass_weight = self.has_weight
        self.pass_weight_add = self.has_weight

    # SOURCE: vllm/model_executor/layers/layernorm.py:L74-L95 forward_native
    #   （逐字——融合 add-norm 的二元组语义本体：residual 非 None 时
    #   ir.ops.fused_add_rms_norm.maybe_inplace 返回 (x, residual) 二元组）
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        # SOURCE: vllm/model_executor/layers/layernorm.py:L74-L95 forward_native
        if residual is None:
            return ir.ops.rms_norm(
                x,
                self.weight.data if self.pass_weight else None,
                self.variance_epsilon,
                self.variance_size_override,
            )
        else:
            return ir.ops.fused_add_rms_norm.maybe_inplace(
                x,
                residual,
                self.weight.data if self.pass_weight_add else None,
                self.variance_epsilon,
                self.variance_size_override,
            )

    # SOURCE: vllm/model_executor/layers/layernorm.py:L96-L114 forward_cuda
    #   —— 减法子集（VLLM_BATCH_INVARIANT 分支删除——batch invariant 数学域；
    #   主支委托 forward_native 逐字）
    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SUBTRACTED: envs.VLLM_BATCH_INVARIANT 的
        #   rms_norm_batch_invariant 分支（layernorm.py:L98-L111）——ch20 域
        # SOURCE: vllm/model_executor/layers/layernorm.py:L96-L114 forward_cuda
        return self.forward_native(x, residual)

    # SOURCE: vllm/model_executor/layers/layernorm.py:L117-L120 forward_xpu（逐字）
    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    # SOURCE: vllm/model_executor/layers/layernorm.py:L122-L125 extra_repr（逐字）
    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s
