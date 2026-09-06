# Subtract-only companion for v3 ch21 — vllm/v1/kv_cache_interface.py
# (pin v0.27.1 / 6e448d0ea). 本章消费面：KVCacheSpecKind 十种组 kind
# （backend_per_kind 的键空间）、KVCacheSpec/AttentionSpec 族（归组的等价类
# 原料——copy_with_new_block_size / max_num_blocks_per_req / page_size_bytes）、
# UniformTypeKVCacheSpecs（同名层混合规格的载体）、KVCacheGroupSpec/
# KVCacheTensor/KVCacheConfig（分组与裸显存账本）。
from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from math import prod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from ..utils.math_utils import cdiv
from ..utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    from .._host_seams import VllmConfigSeam as VllmConfig

# SUBTRACTED: init_logger / KVCacheSpecRegistry / MambaAttentionBackendEnum
#   import（L16-L20）——注册表与 mamba 枚举域（delete[4] 连带：is_uniform_
#   with_collection 的注册表判别归 ch13）。


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/kv_cache_interface.py:L33-L65 KVQuantMode ——（逐字）
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    FP8_PER_TENSOR = 1  # per-tensor scales (current fp8 path)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head dynamic scales for int8
    FP8_PER_TOKEN_HEAD = 3  # per-token-head dynamic scales for fp8
    INT4_PER_TOKEN_HEAD = 4  # packed 2×int4/byte, RHT + asymmetric zp
    NVFP4 = 5  # packed fp4 data + fp8 block scales
    TURBOQUANT = 6  # Hadamard-rotated Lloyd-Max quant, packed K+V per slot

    @property
    def is_per_token_head(self) -> bool:  # SOURCE: vllm/v1/kv_cache_interface.py:L48-L55
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:  # SOURCE: vllm/v1/kv_cache_interface.py:L57-L60
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4

    @property
    def is_turboquant(self) -> bool:  # SOURCE: vllm/v1/kv_cache_interface.py:L62-L65
        """True for turboquant quantization mode."""
        return self == KVQuantMode.TURBOQUANT


# SOURCE: vllm/v1/kv_cache_interface.py:L68-L82 get_kv_quant_mode ——（逐字）
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("turboquant_"):
        return KVQuantMode.TURBOQUANT
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


# SOURCE: vllm/v1/kv_cache_interface.py:L85-L86 is_quantized_kv_cache ——（逐字）
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


# SUBTRACTED: kv_cache_uses_per_token_head_scales（L89-L91）——量化标定域
#   （ch27），本章零调用。


# SOURCE: vllm/v1/kv_cache_interface.py:L94-L104 KVCacheSpecKind ——（逐字）
#   十种组 kind——backend_per_kind 覆写表的键空间
class KVCacheSpecKind(str, Enum):  # SOURCE: vllm/v1/kv_cache_interface.py
    FULL_ATTENTION = "full_attention"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW = "sliding_window"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    MAMBA = "mamba"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    SINK_FULL_ATTENTION = "sink_full_attention"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


# SOURCE: vllm/v1/kv_cache_interface.py:L107-L165 KVCacheSpec —— 层自报 KV
#   规格的基类（分组的等价类原料）
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L116-L124
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L126-L128
        return self.block_size

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L130-L137
        """
        The maximum possible memory usage of this KV cache in bytes.

        Returns:
            The KV cache size in bytes
        """
        raise NotImplementedError

    def max_num_blocks_per_req(self, vllm_config: "VllmConfig", max_len: int) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L139-L149
        """
        The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        return cdiv(max_len, self.block_size)

    def copy_with_new_block_size(self, block_size: int) -> Self:  # SOURCE: vllm/v1/kv_cache_interface.py:L151-L155
        """
        Create a new KVCacheSpec from self but replacing the block size.
        """
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:  # SOURCE: vllm/v1/kv_cache_interface.py:L157-L165
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    # SUBTRACTED: is_uniform_with_collection（L167-L180）——uniform 分组判别
    #   走 KVCacheSpecRegistry（ch13 分组器域），本章零调用。


# SOURCE: vllm/v1/kv_cache_interface.py:L183-L231 AttentionSpec ——（逐字）
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def unpadded_page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L192-L202
        unpadded = self.real_page_size_bytes
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        if self.kv_quant_mode.is_per_token_head:
            unpadded += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        return unpadded

    @property
    def page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L204-L209
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L211-L226
        if self.kv_quant_mode.is_nvfp4:
            # Packed layout: fp4 data + fp8 block scales per head.
            from ..utils.torch_utils import nvfp4_kv_cache_full_dim

            head_dim = nvfp4_kv_cache_full_dim(self.head_size)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    def max_num_blocks_per_req(self, vllm_config: "VllmConfig", max_len: int) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L228-L231
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


# SOURCE: vllm/v1/kv_cache_interface.py:L234-L333 FullAttentionSpec ——（逐字
#   除 MLA 断言；merge_window_sizes/merge 是归组等价类的判别面）
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

    def __post_init__(self):  # SOURCE: vllm/v1/kv_cache_interface.py:L262-L264
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L266-L271
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:  # SOURCE: vllm/v1/kv_cache_interface.py:L273-L283
        if len(window_sizes) == 0:
            return None
        elif len(window_sizes) == 1:
            return window_sizes.pop()
        else:
            raise ValueError(
                "All attention layers in the same KV cache group must have the "
                "same window size."
            )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:  # SOURCE: vllm/v1/kv_cache_interface.py:L285-L333
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        # SUBTRACTED: MLAAttentionSpec 断言（L303-L305）——MLA spec 族
        #   → ch24/25 域（delete[0] 同域）。
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


# SUBTRACTED: FullAttentionSpec.real_page_size_bytes（L335-L361）与
#   TQFullAttentionSpec（L362-L387）——NVFP4/turboquant 量化 spec 覆写
#   → ch27 域（基类 AttentionSpec.real_page_size_bytes 已在上方逐字保留）。


# SOURCE: vllm/v1/kv_cache_interface.py:L558-L561 SlidingWindowSpec —— 字段面
#   + real_page_size_bytes（逐字；max_admission_blocks_per_request → ch15
#   滑窗管理域）
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    def __post_init__(self):  # SOURCE: vllm/v1/kv_cache_interface.py:L563-L565
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def real_page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L567-L585
        # Mirror ``FullAttentionSpec.real_page_size_bytes`` for NVFP4 KV cache.
        if self.kv_quant_mode.is_nvfp4:
            from ..utils.torch_utils import nvfp4_kv_cache_full_dim

            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
            return (
                self.block_size
                * self.num_kv_heads
                * last_dim
                * get_dtype_size(self.dtype)
            )
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )


# SUBTRACTED: MLAAttentionSpec / HiddenStateCacheSpec / RSWASpec /
#   ChunkedLocalAttentionSpec（L388-L557）、SlidingWindowMLASpec（L630-L708）、
#   CrossAttentionSpec（L769-L779）、SinkFullAttentionSpec（L782-L835）——
#   MLA/R-SWA/chunked-local/enc-dec/sink 的 spec 族 → ch24/25 与 enc-dec 域
#   （delete[0] 同域）。


# SOURCE: vllm/v1/kv_cache_interface.py:L709-L716 MambaSpec —— 字段面（mamba_
#   type 字段随 MambaAttentionBackendEnum delete[4] 连带删）
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    num_speculative_blocks: int = 0

    @property
    def page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L718-L727
        page_size = sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    # SUBTRACTED: mamba_type 字段（L714——MambaAttentionBackendEnum，
    #   delete[4] 连带删）；max_num_blocks_per_req/is_uniform_with_collection
    #   （L740-L759）——mamba 显存账与 uniform 判别 → ch13/ch22 域（本章只做
    #   isinstance 判别；max_memory_usage_bytes 下方逐字保留）。

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L729-L738（逐字）
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)


# SOURCE: vllm/v1/kv_cache_interface.py:L762-L766 EncoderOnlyAttentionSpec ——（逐字）
@dataclass(frozen=True, kw_only=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L764-L766
        # Encoder-only layers do not need KV cache
        return 0


# SOURCE: vllm/v1/kv_cache_interface.py:L836-L849 UniformTypeKVCacheSpecs ——
#   字段面 + page_size_bytes（分组器产出的混合规格载体；本章消费其
#   kv_cache_specs 子表与 block_size）
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:  # SOURCE: vllm/v1/kv_cache_interface.py:L847-L849
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    # SUBTRACTED: max_memory_usage_bytes/is_uniform_type/from_specs/
    #   get_page_sizes/get_num_layer_tuples/max_memory_usage_pages
    #   （L851-L899）——uniform 分组器域（ch13），本章零调用。


# SOURCE: vllm/v1/kv_cache_interface.py:L901-L930 get_kv_cache_spec_kind ——
#   spec → KVCacheSpecKind（MLA/sink/chunked-local/cross 的 spec 族分支随
#   其类删，→ ch24/25 与 enc-dec 域）
def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:  # SOURCE: vllm/v1/kv_cache_interface.py
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    return KVCacheSpecKind.UNKNOWN


# SUBTRACTED: get_kv_cache_spec_sliding_window / get_kv_cache_spec_non_causal
#   等辅助（L933 起）——调度面消费（ch10/ch15），本章零调用。


# SOURCE: vllm/v1/kv_cache_interface.py:L945-L954 KVCacheTensor ——（逐字）
#   裸显存账本条目：size/shared_by/offset/block_stride
@dataclass
class KVCacheTensor:  # SOURCE: vllm/v1/kv_cache_interface.py
    """
    A class for specifying how the workers should initialize the KV cache.
    """

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)


# SOURCE: vllm/v1/kv_cache_interface.py:L957-L969 KVCacheGroupSpec ——（逐字）
@dataclass
class KVCacheGroupSpec:  # SOURCE: vllm/v1/kv_cache_interface.py
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py:L972-L989 KVCacheConfig ——（逐字头部）
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

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

    @property
    def has_mamba_layers(self) -> bool:  # SOURCE: vllm/v1/kv_cache_interface.py:L991-L993
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    # SUBTRACTED: has_mixed_precision_kv_cache（L995-L1011）——混精度显存账
    #   → ch14/ch27 域。
