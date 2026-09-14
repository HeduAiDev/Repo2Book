# SOURCE: vllm/model_executor/layers/attention_layer_base.py
# ch26 消费面（ch25 同源切面）：AttentionLayerBase——模型层向后端暴露
# get_kv_cache_spec/get_attn_backend 的注册协议（IndexCache/SWA cache/
# 压缩机状态缓存/MLAAttention 全经此挂进 static_forward_context）。
# 真实为非 Module 的协议基类（消费类以 (nn.Module, AttentionLayerBase)
# 双继承形态组合）。
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheSpec


# SOURCE: vllm/model_executor/layers/attention_layer_base.py
#   AttentionLayerBase —— 减法子集（协议面）
class AttentionLayerBase(ABC):
    """The base class for attention layers.

    Defines the interfaces required for attention layers, including
    KV cache spec reporting and attention backend resolution.
    """

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config) -> "KVCacheSpec":
        """Get the KV cache spec of this layer."""
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py
        raise NotImplementedError

    @abstractmethod
    def get_attn_backend(self) -> type["AttentionBackend"]:
        """Get the attention backend class of this layer."""
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py
        raise NotImplementedError
