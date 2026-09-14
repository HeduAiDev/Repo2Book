# SOURCE: vllm/v1/attention/backend.py
# ch26 切面（ch25 同源切面 + 本章消费增量）：协议层本章消费面——
#   AttentionType + MultipleOf + AttentionBackend ABC（get_name/get_impl_cls/
#   get_builder_cls 三抽象 + get_supported_kernel_block_sizes 默认位 +
#   get_kv_cache_shape + get_kv_cache_stride_order + get_supported_head_sizes +
#   is_mla/is_sparse 身份证 + supports_pcp）+ AttentionCGSupport +
#   get_cudagraph_support + CommonAttentionMetadata（split_decodes_and_prefills
#   与 builder.build 的消费字段 + token_to_req_indices）+ AttentionMetadataBuilder
#   （__init__/build 中心入口/_init_reorder_batch_threshold 非 spec 分支）+
#   AttentionLayer 协议 + AttentionImplBase（is_sparse/supports_dense_mha_
#   prefill 旗标）+ MLAAttentionImpl（forward_mha/forward_mqa 双抽象）。
# SUBTRACTED：FlashAttention/优先级选择面长尾（→ ch21）、DCP/PCP 钩子族
#   （delete[0]）、AttentionImpl 通用插座（→ ch21/ch23）。
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Generic, Optional, TypeVar

import numpy as np
import torch

from vllm.v1.kv_cache_interface import KVCacheSpec  # noqa: F401  (类型位)


# SOURCE: vllm/v1/attention/backend.py AttentionType —— 逐字常量族
class AttentionType:
    """The type of attention, used for different mask shapes."""
    DECODER = "decoder"
    ENCODER_ONLY = "encoder-only"
    ENCODER_DECODER = "encoder-decoder"


# SOURCE: vllm/v1/attention/backend.py:L49-L53 MultipleOf —— 逐字
class MultipleOf:
    base: int

    def __init__(self, base: int):
        # SOURCE: vllm/v1/attention/backend.py:L49-L53（锚点双置）
        self.base = base


# SOURCE: vllm/v1/attention/backend.py AttentionBackend —— 减法子集
#   （四身份证 + kernel block sizes + shape/head_size + is_mla/is_sparse +
#   supports_pcp；validate_configuration 聚合器与优先级选择面归 ch21）
class AttentionBackend(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backend.py（锚点双置）
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py（锚点双置）
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py（锚点双置）
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/v1/attention/backend.py:L69-L71 —— 逐字默认位
        return [MultipleOf(1)]

    @classmethod
    def supports_block_size(cls, block_size: int) -> bool:
        # SOURCE: vllm/v1/attention/backend.py:L183-L186 supports_block_size
        supported_sizes = cls.get_supported_kernel_block_sizes()
        return block_size in supported_sizes or any(
            isinstance(s, MultipleOf) and block_size % s.base == 0
            for s in supported_sizes
        )

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # SOURCE: vllm/v1/attention/backend.py:L195-L201 —— 逐字
        supported_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_sizes:
            return default_block_size

        if cls.supports_block_size(default_block_size):
            return default_block_size

        return min(s.base if isinstance(s, MultipleOf) else s for s in supported_sizes)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/v1/attention/backend.py get_supported_head_sizes 默认位
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # SOURCE: vllm/v1/attention/backend.py:L159-L161 supports_head_size
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    # SOURCE: vllm/v1/attention/backend.py:L120 get_kv_cache_stride_order
    #   —— 默认位
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py:L120 —— 默认位
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py:L285-L289 supports_pcp —— 逐字
    @classmethod
    # SOURCE: vllm/v1/attention/backend.py:L285-L289（锚点双置）
    def supports_pcp(cls) -> bool:
        try:
            return cls.get_impl_cls().supports_pcp
        except NotImplementedError:
            return False

    @classmethod
    def is_mla(cls) -> bool:
        # SOURCE: vllm/v1/attention/backend.py AttentionBackend.is_mla 默认位
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        # SOURCE: vllm/v1/attention/backend.py AttentionBackend.is_sparse 默认位
        return False

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py（锚点双置）
        raise NotImplementedError

    # SUBTRACTED: get_kv_cache_stride_order 的层维探针/validate_configuration/
    #   supports_compute_capability/优先级族——ch21 域


# SOURCE: vllm/v1/attention/backend.py:L404 AttentionMetadata —— 标记基类
class AttentionMetadata:
    # SOURCE: vllm/v1/attention/backend.py:L404（锚点双置）
    pass


# SOURCE: vllm/v1/attention/backend.py:L606-L621 AttentionCGSupport —— 逐字
class AttentionCGSupport(Enum):
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


T = TypeVar("T", bound=AttentionMetadata)
M = TypeVar("M", bound=AttentionMetadata)


# SOURCE: vllm/v1/attention/backend.py:L412 CommonAttentionMetadata —— 减法
#   子集（本章消费字段：query_start_loc(_cpu)/seq_lens/num_reqs/num_actual_
#   tokens/max_query_len/max_seq_len/block_table_tensor/slot_mapping/causal/
#   dcp_local_seq_lens/is_prefilling/seq_lens_cpu_upper_bound +
#   token_to_req_indices；mm/rswa 长尾 → ch15/ch16 域）
@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    causal: bool | torch.Tensor = True

    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """Sequence lengths of the local rank in decode context parallelism world"""

    is_prefilling: torch.Tensor | None = None
    """(batch_size,) bool tensor: True if request is still in prefill phase
    (num_computed_tokens < prompt_tokens). Used by some backends to
    distinguish actual decodes from short extends."""

    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    """(batch_size,) CPU upper bound on seq_lens. Precise for prefill rows
    and for all rows outside async spec decode; optimistic for async-spec
    decode rows (assumes every draft was accepted). Not safe for kernels
    that need exact per-row context lengths on decode rows."""

    # WARNING: Deprecated fields. Will be removed in a future release.
    _num_computed_tokens_cpu: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None

    def batch_size(self) -> int:
        # SOURCE: vllm/v1/attention/backend.py:L535-L536 batch_size
        return self.seq_lens.shape[0]

    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        """Build or reuse the per-token request index mapping."""
        # SOURCE: vllm/v1/attention/backend.py:L573-L591 token_to_req_indices
        #   —— 逐字
        num_tokens = self.num_actual_tokens
        if self._token_to_req_indices_cache is not None:
            assert self._token_to_req_indices_cache.device == buffer.device
            assert self._token_to_req_indices_cache.dtype == torch.int32
            assert self._token_to_req_indices_cache.shape[0] >= num_tokens
            return self._token_to_req_indices_cache[:num_tokens]

        starts = np.asarray(self.query_start_loc_cpu, dtype=np.int32)
        query_lens = np.diff(starts)
        token_to_req_indices = np.repeat(
            np.arange(query_lens.shape[0], dtype=np.int32), query_lens
        )
        num_mapped_tokens = token_to_req_indices.shape[0]
        assert buffer.shape[0] >= max(num_mapped_tokens, num_tokens)
        buffer[:num_mapped_tokens] = torch.from_numpy(token_to_req_indices)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]
        return self._token_to_req_indices_cache[:num_tokens]


# SOURCE: vllm/v1/attention/backend.py:L623 AttentionMetadataBuilder —— 减法
#   子集（__init__/build 抽象入口/get_cudagraph_support/_init_reorder_batch_
#   threshold 非 spec 分支）
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int | None = None

    def __init__(
        self,
        kv_cache_spec: "KVCacheSpec",
        layer_names: list[str],
        vllm_config: Any,
        device: torch.device,
    ):
        # SOURCE: vllm/v1/attention/backend.py:L636-L649 builder __init__
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],
        vllm_config: Any,
        kv_cache_spec: "KVCacheSpec",
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class."""
        # SOURCE: vllm/v1/attention/backend.py:L651-L657 —— 逐字
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        # SOURCE: vllm/v1/attention/backend.py:L659-L693 —— 减法子集
        self.reorder_batch_threshold = reorder_batch_threshold
        # SUBTRACTED: supports_spec_as_decode 的 num_speculative_tokens 抬
        #   阈值段（backend.py:L666-L683）——spec-decode 归 ch33

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        # SOURCE: vllm/v1/attention/backend.py build 抽象入口（锚点双置）
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.
        """
        raise NotImplementedError

    # SUBTRACTED: update_block_table/build_for_cudagraph_capture——ch19/ch21 域


# SOURCE: vllm/v1/attention/backend.py:L775-L795 AttentionLayer —— 协议位
class AttentionLayer:
    # SOURCE: vllm/v1/attention/backend.py:L775 AttentionLayer（锚点双置）
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float


# SOURCE: vllm/v1/attention/backend.py:L796 AttentionImplBase —— 减法子集
#   （sparse 路由旗标 + dense-MHA prefill 旗标 + lse 底数契约注释）
class AttentionImplBase(ABC, Generic[T]):
    # SOURCE: vllm/v1/attention/backend.py:L796（锚点双置）
    """Base class for attention implementations.

    Contains common attributes and initialization logic shared by both
    standard AttentionImpl and MLAAttentionImpl. Does not define a forward
    method - subclasses define their own forward interfaces.
    """

    # Whether this attention impl uses a sparse (top-k) attention path.
    # Used by MLA to route between the dense-MHA prefill and sparse-MQA paths.
    is_sparse: ClassVar[bool] = False

    # Whether this impl can run a dense MHA prefill for the sparse layers
    # (instead of routing everything through the top-k MQA path).
    # SOURCE: vllm/v1/attention/backend.py:L810 supports_dense_mha_prefill
    supports_dense_mha_prefill: ClassVar[bool] = True

    # Whether the attention impl can return the softmax lse for decode.
    can_return_lse_for_decode: bool = False

    # Base of the logarithm used by this backend when returning softmax lse.
    # True  => natural log (lse = ln(sum(exp(qk))))
    # False => base 2      (lse = log2(sum(exp(qk))))
    lse_base_on_e: bool = True

    # SUBTRACTED: supports_pcp 旗标位（delete[0]——DCP/PCP 归 ch34）


# SOURCE: vllm/v1/attention/backend.py:L1009-L1101 MLAAttentionImpl —— 减法
#   子集（forward_mha/forward_mqa 双抽象）
class MLAAttentionImpl(AttentionImplBase[T], Generic[T]):
    """MLA attention implementation with forward_mqa and forward_mha methods."""

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: Any,
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        # SOURCE: vllm/v1/attention/backend.py:L1009-L1101（锚点双置）
        raise NotImplementedError

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        """MHA-style prefill forward pass."""
        # SOURCE: vllm/v1/attention/backend.py:L1041-L1050 forward_mha 默认位
        raise NotImplementedError

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """MQA-style decode forward pass."""
        # SOURCE: vllm/v1/attention/backend.py:L1052-L1062 forward_mqa 抽象
        raise NotImplementedError
