# SOURCE: vllm/model_executor/layers/attention_layer_base.py
# ch23 消费面：AttentionLayerBase——attention-like 层的公共基类（Attention 的
# 基类之一；bind_kv_cache 的占位 kv_cache 替换位）。
# SUBTRACTED：get_kv_cache_spec 抽象位（KV cache 规格体系归 ch22 已立）——
#   本章插座面不构造规格。
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from vllm.v1.attention.backend import AttentionBackend


# SOURCE: vllm/model_executor/layers/attention_layer_base.py:L14 AttentionLayerBase
class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L23-L24 impl 位
    #   与 supports_dcp（逐字）
    impl: "object"
    supports_dcp: bool = True

    # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L26-L33
    #   bind_kv_cache（逐字——占位 kv_cache 张量由此替换）
    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L26-L33
        self.kv_cache = kv_cache

    # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L34-L39
    #   get_attn_backend 抽象位（逐字）
    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:
        """Get the attention backend class for this layer."""
        # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L34-L39
        pass

    # SUBTRACTED: get_kv_cache_spec 抽象位（attention_layer_base.py:L41-L46）
    #   ——KV cache 规格体系（FullAttentionSpec/SlidingWindowSpec…）归 ch22
