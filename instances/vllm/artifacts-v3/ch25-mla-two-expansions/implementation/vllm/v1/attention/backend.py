# SOURCE: vllm/v1/attention/backend.py
# ch25 切面（站 3/5/m10）：协议层本章消费面——
#   AttentionType + AttentionBackend ABC（四身份证 + get_kv_cache_shape/
#   get_supported_head_sizes/supports_head_size + is_mla 家族身份证）+
#   CommonAttentionMetadata（split_decodes_and_prefills 与 builder.build 的
#   消费字段）+ AttentionMetadataBuilder（build 中心入口 + reorder_batch_
#   threshold 阈值声明位 + _init_reorder_batch_threshold 的非 spec 分支）+
#   AttentionLayer 协议 + AttentionImplBase/MLAAttentionImpl（do_kv_cache_
#   update 写腿 + forward_mha/forward_mqa 双抽象）。
# SUBTRACTED：AttentionCGSupport/FlashAttention 长尾字段族（ch21 域）、
#   AttentionImpl（通用插座 impl——ch21/ch23 域）、DCP/PCP 钩子族（delete[0]）。
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
        # SOURCE: vllm/v1/attention/backend.py:L49-L53（锚点双置：声明上方注释同文）
        self.base = base


# SOURCE: vllm/v1/attention/backend.py:L56 AttentionBackend —— 减法子集
#   （四身份证 + shape/head_size 面 + is_mla；validate_configuration 聚合器
#   与优先级选择面归 ch21）
class AttentionBackend(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backend.py:L56（锚点双置：声明上方注释同文）
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py:L56（锚点双置：声明上方注释同文）
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type:
        # SOURCE: vllm/v1/attention/backend.py:L56（锚点双置：声明上方注释同文）
        raise NotImplementedError

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # SOURCE: vllm/v1/attention/backend.py:L159-L161 supports_head_size
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/v1/attention/backend.py get_supported_head_sizes 默认位
        return []

    # SOURCE: vllm/v1/attention/backend.py:L120 get_kv_cache_stride_order
    #   —— 默认位（子类覆写声明物理序；MLACommonBackend 覆写 (0,1,2)）
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py:L120 —— 默认恒等序位
        raise NotImplementedError

    # SOURCE: vllm/v1/attention/backend.py:L206 indexes_kv_by_block_stride
    #   —— 语义逐字（runner.get_kv_cache_spec 回填 spec 的探针）
    @classmethod
    def indexes_kv_by_block_stride(cls) -> bool:
        # SOURCE: vllm/v1/attention/backend.py:L206（锚点双置：声明上方注释同文）
        """Whether the backend reads KV pages by the runtime block stride.

        True when ``num_blocks`` is the outermost physical dimension of the KV
        cache, so the backend tolerates a non-contiguous block dim. This gates
        page size padding and cross-layer uniform KV layout.

        Returns:
            True if the backend's physical KV layout is num-blocks-first. False
            otherwise, including when the backend does not define a layered
            stride order.
        """
        try:
            kv_cache_stride_order = cls.get_kv_cache_stride_order(
                include_num_layers_dimension=False
            )
            layered_kv_cache_stride_order = cls.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
        except (AttributeError, NotImplementedError):
            return False

        # Check that attention backend includes a layers dimension.
        if len(layered_kv_cache_stride_order) != len(kv_cache_stride_order) + 1:
            return False

        # stride_order[0] == 0 means num_layers stays first in physical
        # layout (identity permutation), so indexing by block stride is
        # not supported.
        return layered_kv_cache_stride_order[0] != 0

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backend.py:L206（锚点双置：声明上方注释同文）
        raise NotImplementedError

    @classmethod
    def is_mla(cls) -> bool:
        # SOURCE: vllm/v1/attention/backend.py AttentionBackend.is_mla 默认位
        #   （MLACommonBackend 覆写为 True——家族身份证）
        return False

    # SUBTRACTED: get_supported_kernel_block_sizes/get_kv_cache_stride_order/
    #   supports_combination/validate_configuration/全能力探针族——ch21 域


class AttentionMetadata:
    # SOURCE: vllm/v1/attention/backend.py:L404 AttentionMetadata —— 标记基类
    pass


# SOURCE: vllm/v1/attention/backend.py:L606-L621 AttentionCGSupport —— 逐字
#   （builder 的 CG 支持档声明——本章只有 FlashMLAMetadataBuilder 引用其
#   UNIFORM_BATCH 档）
class AttentionCGSupport(Enum):
    # SOURCE: vllm/v1/attention/backend.py:L606-L621（锚点双置：声明上方注释同文）
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
#   dcp_local_seq_lens/is_prefilling/seq_lens_cpu_upper_bound；mm/rswa/
#   replayssm 长尾 → ch15/ch16 域）
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
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
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
    (num_computed_tokens < num_prompt_tokens). Used by some backends to
    distinguish actual decodes from short extends."""

    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    """(batch_size,) CPU upper bound on seq_lens. Precise for prefill rows
    and for all rows outside async spec decode; optimistic for async-spec
    decode rows (assumes every draft was accepted). Not safe for kernels
    that need exact per-row context lengths on decode rows."""

    # WARNING: Deprecated fields. Will be removed in a future release (v0.15.0)
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    _num_computed_tokens_cache: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None

    def batch_size(self) -> int:
        # SOURCE: vllm/v1/attention/backend.py:L535-L536 batch_size
        return self.seq_lens.shape[0]

    def token_to_req_indices(self, buffer: torch.Tensor) -> torch.Tensor:
        """Build or reuse the per-token request index mapping."""
        # SOURCE: vllm/v1/attention/backend.py:L573-L591 token_to_req_indices
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
        # copy from CPU to GPU
        buffer[:num_mapped_tokens] = torch.from_numpy(token_to_req_indices)
        if num_mapped_tokens < num_tokens:
            buffer[num_mapped_tokens:num_tokens].zero_()
        self._token_to_req_indices_cache = buffer[: max(num_mapped_tokens, num_tokens)]
        return self._token_to_req_indices_cache[:num_tokens]


# SOURCE: vllm/v1/attention/backend.py:L623 AttentionMetadataBuilder —— 减法
#   子集（reorder_batch_threshold 阈值声明位 + _init_reorder_batch_threshold
#   非 spec 分支逐字 + build 抽象入口；spec-decode 抬阈值联动按 delete[4] 删）
class AttentionMetadataBuilder(ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).
    # Do not access directly. Call get_cudagraph_support() instead.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?
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

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        # SOURCE: vllm/v1/attention/backend.py:L659-L693 _init_reorder_batch_
        #   threshold —— 减法子集
        self.reorder_batch_threshold = reorder_batch_threshold
        # SUBTRACTED: supports_spec_as_decode 的 num_speculative_tokens 抬
        #   阈值段（backend.py:L666-L683）——delete[4]（spec-decode 归 ch33）

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
        # SOURCE: vllm/v1/attention/backend.py:L659-L693（锚点双置：声明上方注释同文）
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.
        """
        raise NotImplementedError

    # SUBTRACTED: get_cudagraph_support/update_block_table/build_for_cudagraph_
    #   capture（backend.py:L646-L657、L695+）——ch19/ch21 域 + delete[4]


# SOURCE: vllm/v1/attention/backend.py:L775-L795 AttentionLayer —— 逐字协议
#   （真实为 typing.Protocol——本章以普通类承载同一属性契约面）
class AttentionLayer:
    # SOURCE: vllm/v1/attention/backend.py:L775 AttentionLayer —— 协议位
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _k_scale_cpu: torch.Tensor
    _v_scale: torch.Tensor
    _v_scale_cpu: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor


# SOURCE: vllm/v1/attention/backend.py:L796 AttentionImplBase —— 减法子集
#   （MLA 路由旗标 + can_return_lse_for_decode + lse_base_on_e 底数契约注释）
class AttentionImplBase(ABC, Generic[T]):
    # SOURCE: vllm/v1/attention/backend.py:L796（锚点双置：声明上方注释同文）
    """Base class for attention implementations.

    Contains common attributes and initialization logic shared by both
    standard AttentionImpl and MLAAttentionImpl. Does not define a forward
    method - subclasses define their own forward interfaces.
    """

    # Whether this impl uses a sparse (top-k) attention path. Used by MLA to
    # route between the dense-MHA prefill and sparse-MQA paths.
    is_sparse: ClassVar[bool] = False

    # Whether the attention impl can return the softmax lse for decode.
    # Some features like decode context parallelism require the softmax lse.
    can_return_lse_for_decode: bool = False

    # Base of the logarithm used by this backend when returning softmax lse.
    # True  => natural log (lse = ln(sum(exp(qk))))
    #          -- e.g. Triton MLA, FlashAttention, FlashMLA, Cutlass MLA
    # False => base 2      (lse = log2(sum(exp(qk))))
    #          -- e.g. FlashInfer trtllm-gen MLA
    # The DCP combine kernel (cp_lse_ag_out_rs / dcp_a2a_lse_reduce in
    # vllm/v1/attention/ops/common.py) branches on this via its IS_BASE_E
    # constexpr; getting it wrong silently corrupts the cross-shard
    # softmax denominator.
    lse_base_on_e: bool = True

    # SUBTRACTED: supports_dense_mha_prefill 旗标位（sparse 路由面，
    #   delete[3]）、supports_pcp（delete[0]）


# SOURCE: vllm/v1/attention/backend.py:L1009-L1101 MLAAttentionImpl —— 减法
#   子集（forward_mha/forward_mqa 双抽象 + do_kv_cache_update 写腿；
#   fused_output_quant_supported 按 delete[2] 删）
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
        # SOURCE: vllm/v1/attention/backend.py:L1009-L1101（锚点双置：声明上方注释同文）
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

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/attention/backend.py:L1078-L1101 do_kv_cache_update
        #   —— 逐字（MLA 写腿：cat 潜向量 + 分页散写）
        if kv_cache.numel() == 0:
            return
        from vllm import _custom_ops as ops

        ops.concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype=kv_cache_dtype,
            scale=k_scale,
        )
