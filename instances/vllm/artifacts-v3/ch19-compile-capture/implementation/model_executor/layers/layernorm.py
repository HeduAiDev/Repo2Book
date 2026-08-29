# Subtract-only companion for v3 ch19 — vllm/model_executor/layers/layernorm.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: RMSNorm as the worked CustomOp
# instance (register + forward_native/forward_cuda/forward_xpu, m01).
# SUBTRACTED: poly_norm（L19-L32——poly-norm CUDA kernel 面）、其余 norm 层
#   （GemmaRMSNorm 等，L124 起——模型域）、vllm.kernels 副作用导入（CUDA
#   kernel 注册，容器域）。
from __future__ import annotations

import torch
import torch.nn as nn

from ..._host_seams import envs, init_logger, ir, rms_norm_batch_invariant
from ..custom_op import CustomOp

logger = init_logger(__name__)

# SUBTRACTED: poly_norm（L19-L32——poly-norm CUDA kernel 注册面，容器域）。


# --8<-- [start:rms_norm]
# SOURCE: vllm/model_executor/layers/layernorm.py:L36-L42 RMSNorm 注册头 ——
#   @CustomOp.register("rms_norm") 把类挂进 op_registry
@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    # --8<-- [end:rms_norm]

    # SOURCE: vllm/model_executor/layers/layernorm.py:L46-L72 __init__
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

    # SOURCE: vllm/model_executor/layers/layernorm.py:L74-L94 forward_native ——
    #   PyTorch 原生实现（编译器可融合；经 vllm.ir 的 IrOp 引用面）
    def forward_native(  # SOURCE: vllm/model_executor/layers/layernorm.py:L74-L94
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
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

    # SOURCE: vllm/model_executor/layers/layernorm.py:L96-L115 forward_cuda ——
    #   手工 kernel 槽位（VLLM_BATCH_INVARIANT 实验分支保留原文，默认回落
    #   forward_native）
    def forward_cuda(  # SOURCE: vllm/model_executor/layers/layernorm.py:L96-L115
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if envs.VLLM_BATCH_INVARIANT:
            assert self.variance_size_override is None, (
                "Batch invariance is not supported for variance_size_override"
            )
            pass_weight = (
                self.pass_weight_add if residual is not None else self.pass_weight
            )
            return rms_norm_batch_invariant(
                x,
                self.weight.data if pass_weight else None,
                self.variance_epsilon,
                residual=residual,
            )

        return self.forward_native(x, residual)

    # SOURCE: vllm/model_executor/layers/layernorm.py:L117-L122 forward_xpu
    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)

    # SOURCE: vllm/model_executor/layers/layernorm.py:L124-L127 extra_repr
    def extra_repr(self) -> str:
        s = f"hidden_size={self.weight.data.size(0)}"
        s += f", eps={self.variance_epsilon}"
        return s


# SUBTRACTED: GemmaRMSNorm/... 其余 norm 层与函数（L130-L325——模型域；
#   RMSNorm 三实现之外的本章零调用）。
