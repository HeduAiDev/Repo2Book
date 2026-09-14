# SOURCE: vllm/model_executor/layers/layernorm.py
# ch26 消费面（ch25 同款切面）：LayerNorm（indexer k_norm=LayerNorm(eps=1e-6)——
# forward_native 数学逐字）+ RMSNorm（q_a/kv_a norm 与 V4 压缩机 norm 的消费面）。
from __future__ import annotations

import torch
from torch import nn


# SOURCE: vllm/model_executor/layers/layernorm.py RMSNorm —— HOST SEAM：
#   forward_native 数学逐字（fp32 归一 + weight 乘）
class RMSNorm(nn.Module):
    """RMSNorm: https://arxiv.org/abs/1910.07467."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # SOURCE: vllm/model_executor/layers/layernorm.py RMSNorm.__init__
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # SOURCE: vllm/model_executor/layers/layernorm.py forward_native ——
        #   数学逐字
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        return x


# SOURCE: vllm/model_executor/layers/layernorm.py LayerNorm —— HOST SEAM：
#   nn.LayerNorm 同语义承载（indexer 的 k_norm——真实为 cutlass 融合面，
#   数学=标准 LayerNorm）
class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # SOURCE: vllm/model_executor/layers/layernorm.py LayerNorm.__init__
        super().__init__()
        self.inner = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.eps = eps

    def forward(self, x):
        # SOURCE: vllm/model_executor/layers/layernorm.py LayerNorm —— HOST
        #   SEAM（无仿射——indexer k_norm 无 weight）
        return self.inner(x)
