# SOURCE: vllm/v1/kv_cache_interface.py
# spec 家族（前缀缓存切面）：KVCacheSpec（block_size 本体）→ AttentionSpec →
# FullAttentionSpec（supports_fine_grained_hash_lookup 的宿主 spec）/
# SlidingWindowSpec（reachable_block_mask 稀疏驻留的宿主）/ MambaSpec
# （mamba_cache_mode="align" 是 enable_partial_hash_hits 的装配前提）→
# KVCacheTensor/KVCacheGroupSpec/KVCacheConfig（has_mamba_layers 与
# needs_kv_cache_zeroing 两个属性被 coordinator/调度面读）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 7 条 扩展注意力族：RSWASpec/SinkFullAttentionSpec/CrossAttentionSpec/
#     EncoderOnly/HiddenStateCacheSpec/UniformTypeKVCacheSpecs/
#     MLA/SlidingWindowMLA/TQ 族（L353-L515、L630-L938 各段）；
#   页物理公式族（page_size_bytes/real_page_size_bytes/max_memory_usage_
#     bytes 等 → ch13/ch14 已建；本章只消费 block_size 与类型标识）；
#   get_kv_cache_spec_kind/get_kv_cache_spec_sliding_window（事件元数据面，
#     第 1 条 observations 一并删）；KVQuantMode（量化 → ch27，spec 里
#     kv_quant_mode 字段保留默认 NONE 作账位）。
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from .math_utils import cdiv

if TYPE_CHECKING:
    from .config import VllmConfig  # noqa: F401 — 类型账位（构造期校验归 ch03）


# SOURCE: vllm/v1/kv_cache_interface.py:L33 KVQuantMode
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L40-L46
    NONE = 0

    # SUBTRACTED: FP8_PER_TENSOR/INT8_PER_TOKEN_HEAD/FP8_PER_TOKEN_HEAD/
    #   INT4_PER_TOKEN_HEAD/NVFP4/TURBOQUANT 枚举位（量化布局 → ch27；
    #   本章 spec 恒 NONE，字段仅保留默认账位）。


# SOURCE: vllm/v1/kv_cache_interface.py:L107 KVCacheSpec
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    # SOURCE: vllm/v1/kv_cache_interface.py:L113-L114
    block_size: int

    # SUBTRACTED: page_size_bytes/max_memory_usage_bytes/merge 等（→ ch13/14）。


# SOURCE: vllm/v1/kv_cache_interface.py:L183 AttentionSpec
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L185-L190
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None

    # SUBTRACTED: indexes_kv_by_block_stride（页 pad 特路 → ch14）与页物理
    #   公式族（real_page_size_bytes 等 → ch13/14）。


# SOURCE: vllm/v1/kv_cache_interface.py:L234 FullAttentionSpec
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in model runner.
    In this case, we use FullAttentionSpec and record the sliding window size.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L245-L252
    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    # SUBTRACTED: non_causal 字段（L258-L267——调度策略面 → ch10）。

    # SOURCE: vllm/v1/kv_cache_interface.py:L262 __post_init__
    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L263-L264
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # SUBTRACTED: max_memory_usage_bytes/merge_window_sizes/merge/
    #   real_page_size_bytes（→ ch13/14 定账与页公式面）。


# SOURCE: vllm/v1/kv_cache_interface.py:L518 ChunkedLocalAttentionSpec
# SUBTRACTED: is_uniform_with_collection/max_memory_usage_bytes（L548-L555
#   ——定账面 → ch14）；类保留最小壳（第 7 条：manager 内部删、注册表条目
#   完整性保留）。
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L520 attention_chunk_size
    attention_chunk_size: int

    # SOURCE: vllm/v1/kv_cache_interface.py:L522 max_admission_blocks_per_request
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate, so requests
        admitted by startup can also be admitted at runtime.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for at most one chunk window plus
        # the in-flight tokens, since frees happen on the processed-token basis.
        # SOURCE: vllm/v1/kv_cache_interface.py:L534-L539
        num_tokens = min(
            self.attention_chunk_size + max_in_flight_tokens, max_model_len
        )
        return cdiv(num_tokens, self.block_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L478 RSWASpec
# SUBTRACTED: 类本体（dossier.delete 第 7 条——R-SWA 特化注意力；最小壳仅
#   保留 rswa_window 字段供 manager 壳构造）。
@dataclass(frozen=True, kw_only=True)
class RSWASpec(FullAttentionSpec):
    """KV cache spec for Reference Sliding Window Attention (R-SWA).

    Prefill (image + text prompt) tokens are always globally visible.
    Only the last ``rswa_window`` generated tokens are kept in the KV cache;
    gap blocks (between the prefill tail and the current decode window) are
    evicted during each decode step to bound memory at
    O(prefix_blocks + window_blocks).
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L489 rswa_window
    rswa_window: int


# SOURCE: vllm/v1/kv_cache_interface.py:L558 SlidingWindowSpec
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L560-L561
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    # SOURCE: vllm/v1/kv_cache_interface.py:L563 __post_init__
    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L564-L565
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # SOURCE: vllm/v1/kv_cache_interface.py:L587 max_admission_blocks_per_request
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.

        `max_in_flight_tokens` is the max tokens scheduled but not yet settled
        (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`.
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the in-flight tokens (frees happen on the
        # processed-token basis); never more than `max_model_len`.
        # SOURCE: vllm/v1/kv_cache_interface.py:L601-L604
        num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        # SOURCE: vllm/v1/kv_cache_interface.py:L605-L608
        return cdiv(num_tokens, self.block_size) + 1


# SUBTRACTED: SlidingWindowMLASpec（L630-L706——DSV4 特路，第 7 条）。


# SOURCE: vllm/v1/kv_cache_interface.py:L770 CrossAttentionSpec
# SUBTRACTED: max_memory_usage_bytes（L774-L779——encoder 账面 → ch13/14；
#   最小壳保留注册表条目完整性，第 7 条）。
@dataclass(frozen=True, kw_only=True)
class CrossAttentionSpec(AttentionSpec):
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """  # SOURCE: vllm/v1/kv_cache_interface.py:L770（类本体仅此 docstring——
    #   max_memory_usage_bytes 随第 7 条删）


# SOURCE: vllm/v1/kv_cache_interface.py:L783 SinkFullAttentionSpec
# SUBTRACTED: merge 方法族（L787-L830——组化面 → ch14）；最小壳保留
#   sink_len 字段，第 7 条。
@dataclass(frozen=True, kw_only=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L784 sink_len
    sink_len: int | None = None


# SOURCE: vllm/v1/kv_cache_interface.py:L709 MambaSpec
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L711-L716
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    # SUBTRACTED: mamba_type 字段（mamba 后端装配面）与页公式/对齐推导方法族
    #   （→ ch14；本章消费 mamba_cache_mode 与 block_size 两语义位）。


# SUBTRACTED: UniformTypeKVCacheSpecs 与 packed 布局族（L837-L938——单组异宽
#   聚合 spec → ch14；本章哈希面不触）。


# SOURCE: vllm/v1/kv_cache_interface.py:L945 KVCacheTensor
@dataclass
class KVCacheTensor:
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L951-L954
    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)


# SOURCE: vllm/v1/kv_cache_interface.py:L957 KVCacheGroupSpec
@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    # SOURCE: vllm/v1/kv_cache_interface.py:L961-L966
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # SUBTRACTED: is_eagle_group 字段（L967-L969——dossier.delete 第 3 条
    #   eagle/投机解码草稿层标注，→ ch33）


# SOURCE: vllm/v1/kv_cache_interface.py:L972 KVCacheConfig
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L978-L989
    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L991 has_mamba_layers
    @property
    def has_mamba_layers(self) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L993
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    # SUBTRACTED: has_mixed_precision_kv_cache（L995-L1011——零清通道 → ch13；
    #   needs_kv_cache_zeroing 相应只保留 mamba 半边语义账位）

    # SOURCE: vllm/v1/kv_cache_interface.py:L1013 needs_kv_cache_zeroing
    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L1022（混合精度半边随
        #   has_mixed_precision_kv_cache 删 → ch13）
        return self.has_mamba_layers
