# 只做减法的精简版 —— 对照基座 vLLM v0.21.0
# 真实文件：vllm/model_executor/layers/layernorm.py
#
# 基座 RMSNorm 是「身」：AscendRMSNorm 继承它、复用 weight/variance_epsilon，只覆写 forward_oot。
import torch
import torch.nn as nn

# SUBTRACTED: import vllm.kernels（副作用导入内核，L10）/ poly_norm 辅助函数（L20-33）—— 与 RMSNorm 顶替无关。
from vllm import envs, ir
from vllm.config import get_current_vllm_config
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant


@CustomOp.register("rms_norm")
class RMSNorm(CustomOp):
    # SOURCE: vllm/model_executor/layers/layernorm.py:L38
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/layernorm.py:L47
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

        # Cheat and predict whether native impl will be dispatched to (affects weight passing).
        priority = get_current_vllm_config().kernel_config.ir_op_priority
        var_override = self.variance_size_override is not None
        native_rms_norm = priority.rms_norm[0] == "native" or var_override
        native_add_rms_norm = priority.fused_add_rms_norm[0] == "native" or var_override
        self.pass_weight = self.has_weight or not native_rms_norm
        self.pass_weight_add = self.has_weight or not native_add_rms_norm

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/model_executor/layers/layernorm.py:L82
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

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/model_executor/layers/layernorm.py:L104
        if (
            envs.VLLM_BATCH_INVARIANT
            and residual is None
            and self.variance_size_override is None
        ):
            return rms_norm_batch_invariant(x, self.weight.data, self.variance_epsilon)

        return self.forward_native(x, residual)

    # SUBTRACTED: forward_xpu(L118-123) / extra_repr(L125-128) —— 非本章关心。

# SUBTRACTED: GemmaRMSNorm / RMSNormGated 等其余归一化算子（layernorm.py 余下）—— 与 RMSNorm 同构，不在主线。
