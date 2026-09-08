# SOURCE: vllm/model_executor/layers/rotary_embedding/common.py
# ch23 消费面：ApplyRotaryEmb.forward_static——neox/GPT-J 两式旋转数学本体。
# SUBTRACTED：CustomOp 基类与 flash_attn triton 面（common.py 其余）——
#   编译/kernel 域。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/layers/rotary_embedding/common.py:L125 ApplyRotaryEmb
class ApplyRotaryEmb:
    # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py:L128-L136
    #   __init__（减法子集：属性位逐字；flash_attn triton 探测删除——平台域）
    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ):
        # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py:L128-L136
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        # SUBTRACTED: apply_rotary_emb_flash_attn triton 模块探测
        #   （common.py:L138-L143）——平台 kernel 域

    # SUBTRACTED: CustomOp 基类继承（common.py:L125）——编译派发域

    # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py:L145-L185
    #   forward_static（逐字——旋转数学：neox 式对半分块、GPT-J 式隔行交错）
    @staticmethod
    def forward_static(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size (optional), seq_len, num_heads, head_size]
            cos: [seq_len, head_size // 2]
            sin: [seq_len, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style.
            enable_fp32_compute: Temporarily convert x, cos, sin to FP32 dtype
                                 for higher accuracy.
        """
        # SOURCE: vllm/model_executor/layers/rotary_embedding/common.py:L145-L185
        origin_dtype = x.dtype
        if enable_fp32_compute:
            x = x.float()

        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)

        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        if enable_fp32_compute:
            output = output.to(origin_dtype)
        return output
