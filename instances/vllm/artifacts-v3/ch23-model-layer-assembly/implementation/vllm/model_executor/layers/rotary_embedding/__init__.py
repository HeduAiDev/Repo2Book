# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py
# ch23 消费面：get_rope 工厂（must_keep——LlamaAttention._init_rotary_emb 的
# 位置编码工厂调用；正文当黑盒工厂，实现给 default scaling 主路径的真实子集）。
# SUBTRACTED：RoPE 变体族分发（mrope/llama3/yarn/fope/dual_chunk/phi3 等
#   20+ 变体与 dual_chunk_attention_config 面）——非本章主题（章节 elide 明示
#   「RoPE 数学非本章主题」）；_ROPE_DICT 缓存保留。
from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py _ROPE_DICT
#   实例缓存（逐字位）
_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}


# SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py:L33 get_rope
#   —— 减法子集（签名与 default-scaling 主路径逐字；变体族 dispatch 删除）
def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    # SOURCE: vllm/model_executor/layers/rotary_embedding/__init__.py:L33 get_rope
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_parameters is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    # SUBTRACTED: dual_chunk_attention_config 的 key 装配
    #   （rotary_embedding/__init__.py:L53-L61）——DCP 域归 ch34

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    if rotary_dim := rope_parameters.get("rope_dim", None):
        pass
    else:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
        rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    # SUBTRACTED: 变体族 dispatch（dual_chunk/mrope/fope/linear/yarn/llama3/
    #   dynamic_ntk/…rotary_embedding/__init__.py:L86-L~230）——各模型特例
    assert scaling_type == "default"
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
    )
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
