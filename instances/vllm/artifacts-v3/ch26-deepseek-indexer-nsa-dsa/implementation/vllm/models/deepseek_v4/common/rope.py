# SOURCE: vllm/models/deepseek_v4/common/rope.py
# ch26 消费面（ch25 同源切面）：build_deepseek_v4_rope——V4 旋转位（GPT-J
# interleave；compress_ratio>1 时位置坐标在压缩位上取 cos/sin——indexer 打分
# 活在 //ratio 坐标系的 RoPE 面）。HOST SEAM：真实为 V4 专属 RotaryEmbedding
# 家族；host 以同签名 GPT-J 数学承载。
from __future__ import annotations

import torch
from torch import nn


# SOURCE: vllm/models/deepseek_v4/common/rope.py DeepseekV4ScalingRotary-
#   Embedding 家族 —— HOST SEAM：GPT-J interleave 数学 + 压缩位坐标
# SOURCE: vllm/models/deepseek_v4/common/rope.py —— HOST SEAM（锚点双置）
class _V4Rotary(nn.Module):
    # SOURCE: vllm/models/deepseek_v4/common/rope.py —— HOST SEAM（锚点双置）
    def __init__(self, head_dim: int, rope_head_dim: int, max_position: int,
                 compress_ratio: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.compress_ratio = compress_ratio
        self.is_neox_style = False  # GPT-J interleaved
        half = rope_head_dim // 2
        inv = 1.0 / (base ** (torch.arange(0, half).float() / half))
        # cos_sin_cache 布局：[max_pos, rope_head_dim] 前半 cos 后半 sin
        #（与 fused compress/indexer_q 核同一布局契约）
        pos = torch.arange(max_position).float()
        ang = torch.outer(pos, inv)
        self.cos_sin_cache = torch.cat([ang.cos(), ang.sin()], dim=-1)

    # SOURCE: vllm/models/deepseek_v4/common/rope.py —— HOST SEAM（锚点双置）
    def forward(self, positions, q, k):
        """GPT-J 旋转（任意前导维；位置先 //ratio 折到压缩位）。"""
        half = self.rope_head_dim // 2
        p = positions
        if self.compress_ratio > 1:
            p = (p // self.compress_ratio) * self.compress_ratio
        cos = self.cos_sin_cache[p, :half].float()
        sin = self.cos_sin_cache[p, half:].float()

        # SOURCE: vllm/models/deepseek_v4/common/rope.py —— HOST SEAM（锚点双置）
        def rot(x):
            x0, x1 = x[..., 0::2].float(), x[..., 1::2].float()
            r0 = x0 * cos - x1 * sin
            r1 = x1 * cos + x0 * sin
            out = torch.empty_like(x, dtype=torch.float32)
            out[..., 0::2] = r0
            out[..., 1::2] = r1
            return out.to(x.dtype)

        return rot(q), rot(k)


# SOURCE: vllm/models/deepseek_v4/common/rope.py build_deepseek_v4_rope
#   —— HOST SEAM 同签名
def build_deepseek_v4_rope(config, head_dim: int, rope_head_dim: int,
                           max_position_embeddings: int, compress_ratio: int):
    # SOURCE: vllm/models/deepseek_v4/common/rope.py build_deepseek_v4_rope
    return _V4Rotary(head_dim, rope_head_dim, max_position_embeddings,
                     compress_ratio)
