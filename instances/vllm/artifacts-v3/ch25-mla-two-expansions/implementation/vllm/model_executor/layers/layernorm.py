# SOURCE: vllm/model_executor/layers/layernorm.py
# ch25 消费面：RMSNorm（q_a_layernorm / kv_a_layernorm 两块积木）——
# __init__ + forward_native 数学逐字（融合 add 位删除——残差穿针归 ch23）。
from __future__ import annotations

import torch
from torch import nn


# SOURCE: vllm/model_executor/layers/layernorm.py RMSNorm —— 减法子集
class RMSNorm(nn.Module):
    """Root mean square normalization.

    Computes
        x -> w * x / sqrt(mean(x^2) + eps)

    where the reduction is over the last dimension.
    """

    # SOURCE: vllm/model_executor/layers/layernorm.py RMSNorm.__init__
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/layernorm.py __init__ —— 逐字
        #（requires_grad=False：推理承载——与加载后权重一致）
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = eps

    # SOURCE: vllm/model_executor/layers/layernorm.py forward_native —— 逐字
    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/model_executor/layers/layernorm.py forward_native —— 逐字
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x, residual

    def forward(self, x, residual=None):
        # SOURCE: vllm/model_executor/layers/layernorm.py 平台派发位 ——
        #   HOST SEAM：host 直调 forward_native（真实 forward_cuda 走融合
        #   kernel——ch19/ch23 域）
        out, res = self.forward_native(x, residual)
        if residual is not None:
            return out, res
        return out
