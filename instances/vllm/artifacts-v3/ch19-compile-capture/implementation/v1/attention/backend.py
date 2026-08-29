# Subtract-only companion for v3 ch19 — vllm/v1/attention/backend.py
# (pin v0.27.1 / 6e448d0ea). Kept surface = the contract faces ch19 consumes:
#   * AttentionType — Attention 构造参数（attention.py 消费）;
#   * AttentionBackend 类头 + forward_includes_kv_cache_update —
#     Attention.forward 的 KV 写分叉开关（why_chains[8] cost 面）;
#   * AttentionMetadata — ForwardContext.attn_metadata 的类型标记（真源
#     L404-L405 即两行 marker：metadata 本体由各后端 builder 产出，ch20/ch21 域）;
#   * AttentionCGSupport + AttentionMetadataBuilder.get_cudagraph_support ——
#     站 5/m16 的 min_cg_support 最弱链接口（后端能力向上传染；选择器本体归 ch21）。
# SUBTRACTED：其余全部（CommonAttentionMetadata 字段族、AttentionImpl 协议、
# AttentionLayer Protocol——后端/实现域，ch20/ch21 章）。
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import torch

if TYPE_CHECKING:
    from ...config import VllmConfig


# SOURCE: vllm/v1/attention/backend.py:L33-L47 AttentionType —— 层类型枚举
#   （str Enum 以兼容 torch.compile）
class AttentionType(str, Enum):  # SOURCE: vllm/v1/attention/backend.py:L33-L47
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


# SOURCE: vllm/v1/attention/backend.py:L56-L67 AttentionBackend 类头 ——
#   抽象后端基类（supported_dtypes/forward_includes_kv_cache_update 两个
#   ClassVar 是 ch19 消费面：后者决定 KV 写是否拆独立算子）
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    # SUBTRACTED: get_supported_kernel_block_sizes（L69-L71）与
    #   get_kv_cache_shape/get_kv_cache_block_dim 等静态接口（L73-L401）、
    #   is_ssm 等（L399-L401）——kernel 形状协商域，ch20/ch21。

    @staticmethod
    @abstractmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backend.py:L73-L76
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type:  # SOURCE: vllm/v1/attention/backend.py:L78-L81
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # SOURCE: vllm/v1/attention/backend.py:L83-L86
        # -> Type["AttentionMetadataBuilder"]
        raise NotImplementedError


# SOURCE: vllm/v1/attention/backend.py:L404-L405 AttentionMetadata —— marker
#   类：metadata 本体由各后端的 AttentionMetadataBuilder 产出（ch20/ch21 域），
#   ch19 只把它作为 ForwardContext/算子签名的类型标记
class AttentionMetadata:  # SOURCE: vllm/v1/attention/backend.py:L404-L405
    pass


# SUBTRACTED: CommonAttentionMetadata 数据类（L411-L601——逐拍共享张量集，
#   ch20/ch22 域：query_start_loc/seq_lens/block_table 的消费侧）。

M = TypeVar("M")


# SOURCE: vllm/v1/attention/backend.py:L606-L620 AttentionCGSupport ——
#   后端 CUDA graph 支持档位（最弱链降级的值域；ch21 的优先级表产它、
#   ch19 的 _check_and_update_cudagraph_mode 消费它）
class AttentionCGSupport(Enum):  # SOURCE: vllm/v1/attention/backend.py:L606-L620
    """Constants for the cudagraph support of the attention backend
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""
    NEVER = 0
    """NO cudagraph support"""


# SOURCE: vllm/v1/attention/backend.py:L623-L657 AttentionMetadataBuilder ——
#   类头 + _cudagraph_support ClassVar + get_cudagraph_support（最弱链查询面；
#   builder 的其余接口是 ch20/ch21 的 metadata 装配域）
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int | None = None

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: Any,
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):  # SOURCE: vllm/v1/attention/backend.py:L637-L648
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(  # SOURCE: vllm/v1/attention/backend.py:L650-L657
        cls, vllm_config: "VllmConfig", kv_cache_spec: Any
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class."""
        return cls._cudagraph_support

    # SUBTRACTED: _init_reorder_batch_threshold 与 builder 运行期接口
    #   （L659-L775——metadata 装配域，ch20/ch21）。

# SUBTRACTED: AttentionLayer Protocol / AttentionImplBase / AttentionImpl /
#   MLAAttentionImpl（L775-L1100——后端实现域，ch20/ch21；ch19 的算子转调面
#   经 attn.impl 注入，见 attention.py 范围注记）。
