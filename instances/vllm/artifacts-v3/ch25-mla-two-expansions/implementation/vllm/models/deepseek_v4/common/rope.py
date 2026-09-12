# SOURCE: vllm/models/deepseek_v4/common/rope.py
# HOST SEAM：build_deepseek_v4_rope 的 host 载体。真实按 compress_ratio
# 构建「压缩感知」RoPE（位置 pos → pos // compress_ratio 的重标定——C4/C128
# 层的潜格位置语义，机制归 ch26）；本载体返回 GPT-J 式基础 rotary（装配面
# 同签名，forward 数学未被本章测试消费）。
from __future__ import annotations

from typing import Any

from vllm.model_executor.layers.rotary_embedding import get_rope


# SOURCE: vllm/models/deepseek_v4/common/rope.py:L9 build_deepseek_v4_rope
#   —— HOST SEAM：签名承载 + default rope
def build_deepseek_v4_rope(
    config: Any,
    head_dim: int,
    rope_head_dim: int,
    max_position_embeddings: int,
    compress_ratio: int,
):
    # SOURCE: vllm/models/deepseek_v4/common/rope.py:L9 —— HOST SEAM
    #（compress≠1 的位置重标定归 ch26）
    return get_rope(
        rope_head_dim,
        max(max_position_embeddings // max(1, compress_ratio), 16),
        is_neox_style=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
