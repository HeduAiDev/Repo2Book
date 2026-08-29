# SOURCE: vllm/v1/attention/backend.py
# ch22 切面：CommonAttentionMetadata（站10 收束载体——block_table_tensor 读腿
# + slot_mapping 写腿一起过桥；两个 deprecated 属性的措辞就是 D2H 禁忌的成文
# 纪律 WC2）+ AttentionBackend 的两个语义位（forward_includes_kv_cache_update
# 默认 True——m7 双口径裁决的由来；get_supported_kernel_block_sizes——m9 的
# 约束源）。后端选择/validate 全景归 ch21。
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, TypeVar

import numpy as np
import torch

from ._host_seams import deprecated


# SOURCE: vllm/v1/attention/backend.py:L33 AttentionType（字符串枚举、
#   torch.compile 兼容）
class AttentionType(str, Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # SOURCE: vllm/v1/attention/backend.py:L38-L45
    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


# SOURCE: vllm/v1/attention/backend.py:L49 MultipleOf —— kernel 块尺寸约束记法
class MultipleOf:
    base: int

    # SOURCE: vllm/v1/attention/backend.py:L52 MultipleOf.__init__
    def __init__(self, base: int):
        self.base = base


# SOURCE: vllm/v1/attention/backend.py:L56 AttentionBackend —— 本章只保留
#   两个语义位，validate/选择全景 → ch21
class AttentionBackend(ABC):
    """Abstract class for attention backends."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    # SOURCE: vllm/v1/attention/backend.py:L66-L67 —— KV 写是否含在 forward()
    #   里（FlashAttentionBackend 改 False → 双口径裁决的由来，m7）
    # Does attention's forward() include kv cache update?
    forward_includes_kv_cache_update: bool = True

    # SOURCE: vllm/v1/attention/backend.py:L69-L71 —— kernel 块尺寸约束声明
    @staticmethod
    # SOURCE: vllm/v1/attention/backend.py:L70 AttentionBackend.get_supported_kernel_block_sizes
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    # SUBTRACTED: get_name/get_impl_cls/get_builder_cls/get_kv_cache_shape/
    #   validate_attention_backend 等抽象面（L73-L401）——后端选择域 → ch21。


# SOURCE: vllm/v1/attention/backend.py:L404 AttentionMetadata —— 后端私有
#   metadata 的公共标记基类
# SOURCE: vllm/v1/attention/backend.py:L404 AttentionMetadata
class AttentionMetadata:
    pass


# SOURCE: vllm/v1/attention/backend.py:L408 T
T = TypeVar("T", bound=AttentionMetadata)


# SOURCE: vllm/v1/attention/backend.py:L411 CommonAttentionMetadata —— 跨后端
#   共享元数据（本章主角之一：块表张量与 slot_mapping 在此收束）
@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both GPU and CPU versions.
    """

    # SOURCE: vllm/v1/attention/backend.py:L420-L435
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

    # SOURCE: vllm/v1/attention/backend.py:L437-L438 读腿表 + 写腿索引
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    # SOURCE: vllm/v1/attention/backend.py:L440
    causal: bool | torch.Tensor = True

    # SOURCE: vllm/v1/attention/backend.py:L442-L486 可选字段面（注解与
    #   docstring 逐字；本章配置下恒 None/默认）
    # Needed by FastPrefillAttentionBuilder
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None

    # Needed by CrossAttentionBuilder
    encoder_seq_lens: torch.Tensor | None = None
    encoder_seq_lens_cpu: np.ndarray | None = None

    dcp_local_seq_lens: torch.Tensor | None = None
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """Sequence lengths of the local rank in decode context parallelism world"""

    positions: torch.Tensor | None = None
    """(num_actual_tokens,) token positions.  Optional; set when the caller
    has positions available so that builders can pre-compute position-dependent
    sparse metadata for DeepSeek V4 C128A layers."""

    is_prefilling: torch.Tensor | None = None
    """(batch_size,) bool tensor: True if request is still in prefill phase
    (num_computed_tokens < num_prompt_tokens). Used by some backends to
    distinguish actual decodes from short extends."""

    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    """(batch_size,) CPU upper bound on seq_lens. Precise for prefill rows
    and for all rows outside async spec decode; optimistic for async-spec
    decode rows (assumes every draft was accepted). Not safe for kernels
    that need exact per-row context lengths on decode rows."""

    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None = None
    """PrefixLM bidirectional ranges for multimodal tokens. Maps
    request index to list of (start, end) token position ranges
    where bidirectional attention should apply. None for text-only
    batches or non-PrefixLM models."""

    rswa_prefix_lens: torch.Tensor | None = None
    """(batch_size,) per-request prefix length (prompt/image token count) for
    Reference Sliding Window Attention (R-SWA). Tokens with logical index below
    this stay globally visible; later generated tokens additionally see a
    fixed sliding window. None disables R-SWA. The attention backend copies this
    into its own persistent buffer and reads ``rswa_window`` from model config."""

    replayssm_decode_base_cpu: torch.Tensor | None = None
    """(batch_size,) CPU ring origin for Mamba2 ReplaySSM decode: num_computed
    at the current decode run's last full-state write. write_pos counts from
    here, so a preemption-resumed request re-anchors past the prompt boundary."""

    # WARNING: Deprecated fields. Will be removed in a future release (v0.15.0)
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    # SUBTRACTED: _num_computed_tokens_cache / _token_to_req_indices_cache
    #   两缓存字段（L492-L493）——随其缓存方法删（→ ch21 消费域）。

    # SOURCE: vllm/v1/attention/backend.py:L495 batch_size
    def batch_size(self) -> int:
        return self.seq_lens.shape[0]

    # SOURCE: vllm/v1/attention/backend.py:L498 naive_query_lens
    def naive_query_lens(self) -> torch.Tensor:
        """Naive because it assumes that query ends where the next query starts."""
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    # SOURCE: vllm/v1/attention/backend.py:L502 replace
    def replace(self, **kwargs) -> "CommonAttentionMetadata":
        return replace(self, **kwargs)

    # SOURCE: vllm/v1/attention/backend.py:L505-L516 seq_lens_cpu —— D2H 之忌
    #   的成文纪律（deprecated 措辞原文：'Prefer using device seq_lens directly
    #   to avoid implicit H<>D sync.'）
    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    # SOURCE: vllm/v1/attention/backend.py:L513 seq_lens_cpu（deprecated 属性本体）
    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu

    # SOURCE: vllm/v1/attention/backend.py:L518-L533 num_computed_tokens_cpu
    #   （措辞原文：'... which breaks full async scheduling.'）
    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
    async scheduling. If a CPU copy is needed, it can be derived from
    query_start_loc_cpu and seq_lens.
    Will be removed in a future release, please migrate as soon as possible.
    """
    )
    # SOURCE: vllm/v1/attention/backend.py:L527 num_computed_tokens_cpu（deprecated 属性本体）
    def num_computed_tokens_cpu(self) -> torch.Tensor:
        if self._num_computed_tokens_cpu is None:
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu

    # SUBTRACTED: compute_num_computed_tokens / token_to_req_indices / unpadded
    #   三方法（L535-L620）——后端消费域 → ch21。
