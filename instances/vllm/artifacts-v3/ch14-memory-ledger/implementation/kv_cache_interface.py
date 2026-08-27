# SOURCE: vllm/v1/kv_cache_interface.py
# spec 家族 + 账本产物（m5/m6/m7/m11 的原料）：KVCacheSpec 基类 →
# AttentionSpec（页物理公式 real_page_size_bytes = 2×bs×kv_heads×head_dim
# ×dtype）→ FullAttentionSpec（全历史；head_size+head_size_v 版页公式）/
# ChunkedLocalAttentionSpec / SlidingWindowSpec（两个回收型——带
# max_admission_blocks_per_request 单源准入上限）/ MambaSpec（状态型页，
# pad 统一的特例）→ UniformTypeKVCacheSpecs（同型异宽单组）→
# KVCacheTensor/KVCacheGroupSpec/KVCacheConfig（账本的产物与传递形态）。
# SUBTRACTED（dossier.delete 批准项的落点）：
#   第 4 条 DSV4/SlidingWindowMLA 特路：MLAAttentionSpec/SlidingWindowMLASpec/
#     TQFullAttentionSpec（L353-L475、L630-L706）及 FullAttentionSpec.merge
#     里的 MLA 排除桩；
#   第 5 条 扩展注意力族：RSWASpec（L478-L515）、SinkFullAttention/
#     CrossAttention/EncoderOnly/HiddenStateCacheSpec（散布）；
#   第 6 条 eagle：is_eagle_group 字段（KVCacheGroupSpec）；
#   get_kv_cache_spec_kind/get_kv_cache_spec_sliding_window（事件元数据面，
#     kv_cache_manager L172-L178 一并删——观测/级联旁路）；
#   KVQuantMode 的映射函数族（get_kv_quant_mode/is_quantized_kv_cache/
#     kv_cache_uses_per_token_head_scales——ch13/27 消费面）与
#     is_turboquant（TQ 族随第 4 条删）；
#   nvfp4/int4 的 head_dim 特判（量化布局 → ch27）。
from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from .kv_cache_spec_registry import KVCacheSpecRegistry
from .math_utils import cdiv
from .torch_utils import get_dtype_size

if TYPE_CHECKING:
    from .config import VllmConfig  # noqa: F401 — 类型账位（构造期校验归 ch03）


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/kv_cache_interface.py:L33 KVQuantMode
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L40-L46
    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp
    NVFP4 = 5  # packed fp4 data + fp8 block scales

    # SUBTRACTED: TURBOQUANT（第 4 条 TQ 族）。

    # SOURCE: vllm/v1/kv_cache_interface.py:L48-L55 is_per_token_head
    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        # SOURCE: vllm/v1/kv_cache_interface.py:L51-L55
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    # SOURCE: vllm/v1/kv_cache_interface.py:L57-L60 is_nvfp4
    @property
    def is_nvfp4(self) -> bool:
        """True for NVFP4 packed quantization mode."""
        # SOURCE: vllm/v1/kv_cache_interface.py:L60
        return self == KVQuantMode.NVFP4


# SUBTRACTED: get_kv_quant_mode / is_quantized_kv_cache /
#   kv_cache_uses_per_token_head_scales（L68-L91——ch13/27 消费面）；
#   KVCacheSpecKind（L94-L104——事件元数据面）。


# SOURCE: vllm/v1/kv_cache_interface.py:L107 KVCacheSpec
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    # SOURCE: vllm/v1/kv_cache_interface.py:L113-L114
    block_size: int

    # SOURCE: vllm/v1/kv_cache_interface.py:L116 page_size_bytes property
    @property
    def page_size_bytes(self) -> int:
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L123
        raise NotImplementedError

    # SOURCE: vllm/v1/kv_cache_interface.py:L126 storage_block_size property
    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L128
        return self.block_size

    # SOURCE: vllm/v1/kv_cache_interface.py:L130 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L137
        raise NotImplementedError

    # SOURCE: vllm/v1/kv_cache_interface.py:L139 max_num_blocks_per_req
    def max_num_blocks_per_req(self, vllm_config: "VllmConfig", max_len: int) -> int:
        """
        The number of block table entries needed per request, i.e., the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L149
        return cdiv(max_len, self.block_size)

    # SOURCE: vllm/v1/kv_cache_interface.py:L151 copy_with_new_block_size
    def copy_with_new_block_size(self, block_size: int) -> Self:
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L155
        return replace(self, block_size=block_size)

    # SOURCE: vllm/v1/kv_cache_interface.py:L157 merge
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L162-L165
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    # SOURCE: vllm/v1/kv_cache_interface.py:L167 is_uniform_with_collection
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L173-L180
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L183 AttentionSpec
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L185-L190
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    # SOURCE: vllm/v1/kv_cache_interface.py:L192 unpadded_page_size_bytes
    @property
    def unpadded_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L194
        unpadded = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        # SOURCE: vllm/v1/kv_cache_interface.py:L198-L201
        if self.kv_quant_mode.is_per_token_head:
            unpadded += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        return unpadded

    # SOURCE: vllm/v1/kv_cache_interface.py:L204 page_size_bytes
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L206-L209（对外口径：叠 padding）
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L211 real_page_size_bytes
    @property
    def real_page_size_bytes(self) -> int:
        # 页的物理形状公式本体：2(K,V) × block_size × num_kv_heads ×
        # head_dim × dtype 字节。一块页（每层）装 block_size 个 token 的 K 和 V。
        # SUBTRACTED: nvfp4/int4 的 head_dim 特判（L213-L219——量化布局 →
        #   ch27；无量化时 head_dim 原样）
        # SOURCE: vllm/v1/kv_cache_interface.py:L220-L226
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    # SOURCE: vllm/v1/kv_cache_interface.py:L228 max_num_blocks_per_req
    def max_num_blocks_per_req(self, vllm_config: "VllmConfig", max_len: int) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L229-L231
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


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

    # SOURCE: vllm/v1/kv_cache_interface.py:L245-L253
    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    """
    Default to None for not using sliding window attention.
    """
    attention_chunk_size: int | None = None

    non_causal: bool = False
    """
    Whether the layer attends non-causally (e.g. Prefix LM). Carried on the
    spec so the engine core, which collects specs from all workers before the
    scheduler is built, can adjust scheduling policy (chunked prefill / prefix
    caching) regardless of tensor-parallel layout. It does not affect the KV
    cache layout itself.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L262 __post_init__
    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L263-L264
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    # SOURCE: vllm/v1/kv_cache_interface.py:L266 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L267-L271
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L273 merge_window_sizes
    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        # SOURCE: vllm/v1/kv_cache_interface.py:L274-L283
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    # SOURCE: vllm/v1/kv_cache_interface.py:L285 merge
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L291-L293
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        # SOURCE: vllm/v1/kv_cache_interface.py:L295-L302
        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        # SUBTRACTED: MLAAttentionSpec 排除桩（L303-L305——第 4 条）
        # SOURCE: vllm/v1/kv_cache_interface.py:L306-L320
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            # If any layer in the group is non-causal, treat the group as
            # non-causal so the engine core disables incompatible scheduling.
            non_causal=any(spec.non_causal for spec in specs),
        )
        # SOURCE: vllm/v1/kv_cache_interface.py:L321-L332
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec

    # SOURCE: vllm/v1/kv_cache_interface.py:L335 real_page_size_bytes
    @property
    def real_page_size_bytes(self) -> int:
        # Full 版页公式：K 的 head_size + V 的 head_size_v 打进内容维
        # SUBTRACTED: nvfp4/int4 的 last_dim 特判（L337-L345——量化布局 →
        #   ch27；无量化时 last_dim = head_size + head_size_v）
        # SOURCE: vllm/v1/kv_cache_interface.py:L346-L350
        last_dim = self.head_size + self.head_size_v
        return (
            self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype)
        )


# SUBTRACTED: TQFullAttentionSpec / MLAAttentionSpec / HiddenStateCacheSpec /
#   RSWASpec（L353-L375、L388-L475、L478-L515——dossier.delete 第 4/5 条：
#   TQ/MLA/DSV4 特路、HiddenState、R-SWA）。


# SOURCE: vllm/v1/kv_cache_interface.py:L518 ChunkedLocalAttentionSpec
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

    # SOURCE: vllm/v1/kv_cache_interface.py:L541 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L542-L546（cap × page——与
        #   运行期准入门同源的启动期池大小口径）
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L548 is_uniform_with_collection
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L551-L555（同 chunk 尺寸才同型）
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


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

    # SOURCE: vllm/v1/kv_cache_interface.py:L567 real_page_size_bytes
    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L580-L585（K/V 异宽版公式；
        #   nvfp4 分支 L570-L579 → ch27）
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

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

    # SOURCE: vllm/v1/kv_cache_interface.py:L610 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L611-L618（cap × page——
        #   启动期池大小与运行期准入门的单源铁律落点）
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L620 is_uniform_with_collection
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L623-L627（同窗口才同型）
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# SUBTRACTED: SlidingWindowMLASpec（L630-L706——dossier.delete 第 4 条
#   DSV4/SlidingWindowMLA 特路）。


# SOURCE: vllm/v1/kv_cache_interface.py:L709 MambaSpec
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L711-L716
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_type: object = None  # SUBTRACTED: MambaAttentionBackendEnum 装配（mamba 后端域）
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    # SOURCE: vllm/v1/kv_cache_interface.py:L718 page_size_bytes
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L719-L727（页 = 状态形状
        #   逐元素求和；pad 后取 padded——unify 的 pad 目标）
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    # SOURCE: vllm/v1/kv_cache_interface.py:L729 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L730-L738（三种 mamba_cache_
        #   mode 的口径：all=全长状态 / align=2 块 / none=1 块）
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    # SUBTRACTED: max_num_blocks_per_req 的 align 行宽推导（L740-L757——
    #   worker 侧块表行宽（状态槽位寻址）→ 邻章；本章 mamba 只进定账算术。


# SUBTRACTED: MambaSpec 其余方法（L758-L835——对齐推导/等宽合并/mamba
#   后端族，mamba 状态管理深讲不在本书 spine）。


# SOURCE: vllm/v1/kv_cache_interface.py:L837 UniformTypeKVCacheSpecs
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L845
    kv_cache_specs: dict[str, KVCacheSpec]

    # SOURCE: vllm/v1/kv_cache_interface.py:L847 page_size_bytes
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L849（聚合页 = 各层页之和——
        #   「单组异宽」布局的分账基准）
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    # SOURCE: vllm/v1/kv_cache_interface.py:L851 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L852-L856
        max_num_pages = max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )
        return max_num_pages * self.page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L858 is_uniform_type
    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L866-L871
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    # SOURCE: vllm/v1/kv_cache_interface.py:L873 from_specs
    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None:
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type
        of KV cache spec. Return None if not.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L879-L883
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # SUBTRACTED: DSV4 专用三件（get_page_sizes/get_num_layer_tuples/
    #   max_memory_usage_pages，L885-L898——dossier.delete 第 4 条）。


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
    # SUBTRACTED: is_eagle_group 默认值（L967-L969——dossier.delete 第 6 条
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

    # SUBTRACTED: has_mamba_layers（L991-L993——mamba 调度位 → 邻章）。

    # SOURCE: vllm/v1/kv_cache_interface.py:L995-L1011 has_mixed_precision_kv_cache
    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
        # SUBTRACTED: UniformTypeKVCacheSpecs 逐层展开分支（L998-L1005 的
        #   if 面删；聚合 spec 的逐层内景 → dossier.delete 第 12 条）
        # SOURCE: vllm/v1/kv_cache_interface.py:L998-L1011
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            group_spec = group.kv_cache_spec
            group_specs = [group_spec]
            kv_cache_precisions.update(
                (spec.dtype, spec.kv_quant_mode)
                for spec in group_specs
                if isinstance(spec, AttentionSpec)
            )
        return len(kv_cache_precisions) > 1

    # SOURCE: vllm/v1/kv_cache_interface.py:L1013 needs_kv_cache_zeroing
    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """Whether newly allocated KV cache blocks must be zeroed before use.

        Required for Mamba layers, whose state is read before it is fully written
        (#35219), and for mixed-precision caches, where a block reused across
        groups can be reinterpreted under a different precision and decode stale
        bytes to NaN/Inf. Uniform-precision caches skip zeroing.
        """
        # SOURCE: vllm/v1/kv_cache_interface.py:L1022（mamba 半边随
        #   has_mamba_layers 删——零清通道归 ch13；混合精度半边保留）
        return self.has_mixed_precision_kv_cache
