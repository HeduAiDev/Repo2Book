# SOURCE: vllm/v1/attention/backend.py
# ch23 消费面：AttentionType 枚举（DECODER 主值——llama.py/attention.py 的
# attn_type 语言）+ AttentionBackend 抽象骨架（get_impl_cls/get_name 两语义位）
# + AttentionMetadata 基类（ForwardContext.attn_metadata 的类型面）。
# SUBTRACTED：CommonAttentionMetadata 全 dataclass 与 quant_utils 常量、
#   AttentionImpl 的完整接口（ch21/ch22 域——本章只保插座消费的类型面）。
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import torch


# SOURCE: vllm/v1/attention/backend.py:L33 AttentionType（逐字）
class AttentionType(str, Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


# SOURCE: vllm/v1/attention/backend.py:L404 AttentionMetadata（逐字——占位基类）
class AttentionMetadata:
    pass


# SOURCE: vllm/v1/attention/backend.py:L56 AttentionBackend —— 减法子集
#   （类头 + 两语义位逐字；supported dtpes/静态工厂族删除——ch21 域）
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    # SUBTRACTED: supported_dtypes/supported_kv_cache_dtypes 类变量
    #   （backend.py:L58-L64）——后端能力面归 ch21

    # Does attention's forward() include kv cache update?
    # SOURCE: vllm/v1/attention/backend.py:L65-L66 forward_includes_kv_
    #   cache_update —— 两语义位之一（Attention.forward 的算子选择键）
    forward_includes_kv_cache_update: bool = True

    # SOURCE: vllm/v1/attention/backend.py:L70-L72 get_supported_kernel_block_sizes
    #   —— SUBTRACTED（kernel 块尺寸面归 ch21/ch22）

    # SOURCE: vllm/v1/attention/backend.py:L73-L78 get_name 抽象位（逐字）
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backend.py:L73-L78 get_name 抽象位（逐字）
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py get_impl_cls 抽象位——后端交出
    #   impl 类的工厂位（Attention.__init__ L420 消费）
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py get_impl_cls 抽象位——后端交出
        raise NotImplementedError
