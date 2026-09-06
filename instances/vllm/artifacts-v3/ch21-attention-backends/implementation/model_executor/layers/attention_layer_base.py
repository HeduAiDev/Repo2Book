# Subtract-only companion for v3 ch21 — vllm/model_executor/layers/
# attention_layer_base.py (pin v0.27.1 / 6e448d0ea). 全文逐字：Attention/
# Mamba 等「attention 类层」的公共基类——get_attn_backend/get_kv_cache_spec
# 抽象面 + bind_kv_cache 默认实现（initialize_attn_backend 的归组循环与
# bind_kv_cache 的落层都吃这个接口）。
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from ...v1.attention.backend import AttentionBackend, AttentionImpl
from ...v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    from ..._host_seams import VllmConfigSeam as VllmConfig


# SOURCE: vllm/model_executor/layers/attention_layer_base.py:L12-L44
#   AttentionLayerBase ——（逐字）
class AttentionLayerBase(ABC):
    """
    Base class for attention-like layers (Attention, Mamba, etc.)
    that support the v1 engine.

    This provides a common interface for getting attention backends
    from different layer types.
    """

    impl: "AttentionImpl"
    supports_dcp: bool = True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:  # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L25-L32
        """Bind the allocated KV cache tensor to this layer.

        The default stores the cache view as-is; subclasses (e.g. Mamba)
        override this to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]:  # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L34-L37
        """Get the attention backend class for this layer."""
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: "VllmConfig") -> KVCacheSpec | None:  # SOURCE: vllm/model_executor/layers/attention_layer_base.py:L39-L44
        """
        Get the KV cache spec for this layer.
        May be None if the layer does not need KV cache.
        """
        pass
