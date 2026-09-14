# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py
# ch26 消费面（ch25 同款切面）：get_rope 工厂 + GPT-J(interleave)/NeoX 两形态
# 的旋转数学。indexer 专属 RoPE 与主 RoPE 同厂不同参：
#   主 RoPE    deepseek_v2.py:L1075-L1080 is_neox_style=False（GPT-J interleave）
#   indexer RoPE deepseek_v2.py:L1122-L1127 is_neox_style=not
#     indexer_rope_interleave（DSV3.2 置 True → NeoX 取反 = GPT-J interleave，
#     与主 RoPE 相反的开关语义）
# cos_sin_cache 布局：[max_pos, rot_dim]——前半 per-pair cos、后半 per-pair sin
#（fused_indexer_q_rope_quant 核与 V4 fused 核的同一布局契约）。
# SUBTRACTED：deepseek_yarn/llama_scaling 的 mscale 数学（→ ch24 primer 域）、
#   CUDA kernel 派发位（host 走 forward_native 数学）。
from __future__ import annotations

import torch
from torch import nn


# SOURCE: vllm/model_executor/layers/rotary_embedding/rotary_embedding.py
#   DeepseekScalingRotaryEmbedding 家族 —— HOST SEAM：default 型数学承载
class RotaryEmbedding(nn.Module):
    # SOURCE: vllm/model_executor/layers/rotary_embedding/rotary_embedding.py
    #   RotaryEmbedding.__init__ —— 减法子集（base=10000 default 型）
    def __init__(self, head_size: int, max_position: int, base: float = 10000.0,
                 is_neox_style: bool = True):
        super().__init__()
        self.head_size = head_size
        self.max_position = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        inv_freq = 1.0 / (base ** (torch.arange(
            0, head_size, 2, dtype=torch.float32) / head_size))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # SOURCE: cos_sin_cache 布局（前半 cos/后半 sin——逐对）
        self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)

    def _rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x [..., rot_dim] → 旋转后的同形张量（fp32 数学）。"""
        # SOURCE: vllm/model_executor/layers/rotary_embedding/rotary_embedding.py
        #   forward_native 数学 —— HOST SEAM 逐式
        rot_dim = x.shape[-1]
        cos = self.cos_sin_cache[positions, : rot_dim // 2].float()  # [T, half]
        sin = self.cos_sin_cache[positions, rot_dim // 2:].float()
        while cos.dim() < x.dim():  # 头维广播（q 带头维、k 带 [1] 头维）
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        if self.is_neox_style:
            # NeoX：前后半各为一半对
            half = rot_dim // 2
            x0, x1 = x[..., :half].float(), x[..., half:].float()
            r0 = x0 * cos - x1 * sin
            r1 = x1 * cos + x0 * sin
            return torch.cat((r0, r1), dim=-1).to(x.dtype)
        else:
            # interleaved (GPT-J)：偶奇对 (x[2i], x[2i+1])
            x0, x1 = x[..., 0::2].float(), x[..., 1::2].float()
            r0 = x0 * cos - x1 * sin
            r1 = x1 * cos + x0 * sin
            out = torch.empty_like(x, dtype=torch.float32)
            out[..., 0::2] = r0
            out[..., 1::2] = r1
            return out.to(x.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """shape-preserving 旋转（q/k 任意前导维，最后一维=rot_dim）。"""
        # SOURCE: vllm/model_executor/layers/rotary_embedding/rotary_embedding.py
        #   forward —— HOST SEAM（kernel 派发位删，数学同式）
        return self._rotate(q, positions), self._rotate(k, positions)

    def enabled(self) -> bool:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/rotary_embedding.py
        #   enabled —— HOST SEAM：False（inplace 融合面是 ROCm/自定义核域）
        return False

    __call__ = forward


# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py get_rope ——
#   减法子集（default 型直建；yarn 家族 → ch24 primer 域）
def get_rope(
    head_size: int,
    rotary_dim: int | None = None,
    max_position: int = 8192,
    base: float = 10000.0,
    rope_scaling: dict | None = None,
    is_neox_style: bool = True,
    dtype=None,
    rope_parameters: dict | None = None,
):
    # SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py get_rope
    params = rope_parameters or rope_scaling or {}
    rope_type = params.get("rope_type", "default")
    if rope_type != "default":
        # SUBTRACTED: deepseek_yarn/llama_scaling 的 mscale 数学——ch24 primer
        #   域（本章测试与正文用 default 型）
        raise NotImplementedError(
            f"rope_type={rope_type} scaling math is ch24's domain."
        )
    return RotaryEmbedding(head_size, max_position, base,
                           is_neox_style=is_neox_style)
