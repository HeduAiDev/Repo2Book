# SOURCE: vllm/model_executor/layers/attention_layer_base.py —— 全文逐字
#   （get_attn_backend/get_kv_cache_spec 抽象 + bind_kv_cache 默认实现；
#   impl 注解按本章类型面）
"""Base class for attention-like layers."""

from abc import ABC, abstractmethod

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec


# SOURCE: vllm/model_executor/layers/attention_layer_base.py:L12 AttentionLayerBase
class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "object"
    # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L22
    #   supports_dcp —— 字段位（delete[0] 后无 DCP 消费点，保形）
    supports_dcp: bool = True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L24-L30
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L32-L35
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L37-L42
        pass
