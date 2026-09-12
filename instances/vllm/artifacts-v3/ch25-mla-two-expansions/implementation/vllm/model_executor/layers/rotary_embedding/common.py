# SOURCE: vllm/model_executor/layers/rotary_embedding/common.py
# ch25 消费面：ApplyRotaryEmb 的静态数学（neox/interleave 两式逐字）——
# deepseek 系 is_neox_style=False → GPT-J interleaved 分支。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/layers/rotary_embedding/common.py ApplyRotaryEmb
class ApplyRotaryEmb:
    def __init__(
        self,
        is_neox_style: bool = True,
    ):
        # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py —— 逐字位
        self.is_neox_style = is_neox_style

    @staticmethod
    def forward_static(
        q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py
        #   forward_static —— 逐字（neox 半分 / GPT-J 偶奇交错两式）
        if is_neox_style:
            cos = cos.repeat(1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 2).unsqueeze(-2)
        else:
            # GPT-J style: interleaved pairs
            cos = cos.repeat_interleave(2, dim=1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=1).unsqueeze(-2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)

        if is_neox_style:
            # Rotate halves
            q_rot = torch.cat((-q[..., 1::2], q[..., ::2]), dim=-1)
        else:
            # GPT-J style: rotate each pair (even, odd) -> (-odd, even)
            q_perm = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
            q_rot = q_perm.reshape(q_perm.shape[:-2] + (-1,))
        return q * cos + q_rot * sin
