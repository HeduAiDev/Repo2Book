# SOURCE: vllm/v1/kv_cache_interface.py
# ch25 切面（站 5-6 / m09/m11/m12/m13）：KVCacheSpec 谱系的本章消费面——
#   KVQuantMode/get_kv_quant_mode/is_quantized_kv_cache + KVCacheSpec/
#   AttentionSpec/FullAttentionSpec（merge 全链）+ _apply_alignment_padding +
#   MLAAttentionSpec（584B/656B 特账 + merge 四字段断言）+ HiddenStateCacheSpec
#   + SlidingWindowSpec/SlidingWindowMLASpec + UniformTypeKVCacheSpecs +
#   KVCacheGroupSpec。
# SUBTRACTED：TQ/RSWA/ChunkedLocal/Sink/Mamba/EncoderOnly/Cross spec 族与
#   KVCacheTensor/KVCacheConfig 布局面——ch13/ch14/ch27 域（章界收窄 →
#   各章）；nvfp4/turboquant 分支归 ch27。
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# KV cache quantization mode
# ---------------------------------------------------------------------------


# SOURCE: vllm/v1/kv_cache_interface.py:L33-L65 KVQuantMode —— 减法子集
#   （NONE/FP8 族保真；is_per_token_head/is_nvfp4/is_turboquant 属性位保留，
#   TURBOQUANT 档归 ch27）
class KVQuantMode(IntEnum):
    # SOURCE: vllm/v1/kv_cache_interface.py:L33-L65（锚点双置：声明上方注释同文）
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

    @property
    def is_per_token_head(self) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L33-L65（锚点双置：声明上方注释同文）
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )

    @property
    def is_nvfp4(self) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L33-L65（锚点双置：声明上方注释同文）
        """True for NVFP4 packed quantization mode."""
        return self == KVQuantMode.NVFP4


# SOURCE: vllm/v1/kv_cache_interface.py:L68-L82 get_kv_quant_mode —— 逐字
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
    # SUBTRACTED: turboquant_ 前缀档（L78-L79）——ch27 域
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        return KVQuantMode.FP8_PER_TENSOR
    return KVQuantMode.NONE


# SOURCE: vllm/v1/kv_cache_interface.py:L85-L86 is_quantized_kv_cache —— 逐字
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


# SOURCE: vllm/v1/kv_cache_interface.py:L107-L180 KVCacheSpec —— 减法子集
#   （page_size_bytes/storage_block_size/merge/is_uniform_with_collection
#   承重；max_memory_usage/max_num_blocks_per_req 的分配面归 ch13/ch14）
@dataclass(frozen=True)
class KVCacheSpec:
    # SOURCE: vllm/v1/kv_cache_interface.py:L107-L180（锚点双置：声明上方注释同文）
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L107-L180（锚点双置：声明上方注释同文）
        """
        The size of a page with `block_size` tokens in bytes.

        Returns:
            The page size
        """
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L107-L180（锚点双置：声明上方注释同文）
        return self.block_size

    @classmethod
    def merge(cls, specs: list) -> "KVCacheSpec":
        # SOURCE: vllm/v1/kv_cache_interface.py:L107-L180（锚点双置：声明上方注释同文）
        """
        Merge a list of KVCacheSpec objects into a single KVCacheSpec object.
        """
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, "KVCacheSpec"]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L107-L180（锚点双置：声明上方注释同文）
        """
        Whether this KVCacheSpec is uniform with all specs of all layers.
        """
        uniform_type_base_spec = _get_uniform_type_base_spec(self)
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec) for spec in kv_cache_specs.values()
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L183-L231 AttentionSpec —— 减法子集
#   （real_page_size_bytes 的 nvfp4 支删除——ch27；unpadded/page 两属性逐字）
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L183-L231（锚点双置：声明上方注释同文）
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def unpadded_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L183-L231（锚点双置：声明上方注释同文）
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
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L183-L231（锚点双置：声明上方注释同文）
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L183-L231（锚点双置：声明上方注释同文）
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
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


# SOURCE: vllm/v1/kv_cache_interface.py:L234-L333 FullAttentionSpec —— 减法
#   子集（head_size_v/sliding_window/attention_chunk_size/non_causal 字段 +
#   __post_init__/merge 全链逐字；max_memory_usage 分配面归 ch13）
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L234-L333（锚点双置：声明上方注释同文）
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

    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L234-L333（锚点双置：声明上方注释同文）
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @classmethod
    def merge_window_sizes(cls, window_sizes: set) -> int | None:
        # SOURCE: vllm/v1/kv_cache_interface.py:L234-L333（锚点双置：声明上方注释同文）
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
    def merge(cls, specs: list) -> "FullAttentionSpec":
        # SOURCE: vllm/v1/kv_cache_interface.py:L234-L333（锚点双置：声明上方注释同文）
        """
        Merge a list of FullAttentionSpec objects into a single
        FullAttentionSpec object.
        """
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )

        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
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


# SOURCE: vllm/v1/kv_cache_interface.py:L353-L359 _apply_alignment_padding —— 逐字
def _apply_alignment_padding(spec):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L388-L468 MLAAttentionSpec —— 逐字
#   （must_keep：MLAAttention.get_kv_cache_spec 的产物；DSV4 专属字段 +
#   storage_block_size + 584B/656B 特账 + merge 四字段断言）
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L388-L468（锚点双置：声明上方注释同文）
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1  # Default to 1 for no compression.
    model_version: str | None = None
    # Marks draft groups that flatten a non-causal query block into decode rows.
    non_causal_multi_token_decode: bool = False

    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L388-L468（锚点双置：声明上方注释同文）
        super().__post_init__()
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L388-L468（锚点双置：声明上方注释同文）
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L388-L468（锚点双置：声明上方注释同文）
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            self.storage_block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list) -> "MLAAttentionSpec":
        # SOURCE: vllm/v1/kv_cache_interface.py:L388-L468（锚点双置：声明上方注释同文）
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, and KV block "
            "stride indexing."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        return merged_spec


# SOURCE: vllm/v1/kv_cache_interface.py:L471-L475 HiddenStateCacheSpec —— 逐字
@dataclass(frozen=True, kw_only=True)
class HiddenStateCacheSpec(MLAAttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L471-L475（锚点双置：声明上方注释同文）
    """Marker for hidden-state cache layers used by extract_hidden_states."""

    pass


# SOURCE: vllm/v1/kv_cache_interface.py:L518-L560 ChunkedLocalAttentionSpec
#   —— 减法子集（字段 + is_uniform_with_collection；admission/max_memory
#   面归 ch13——_promote_local_kv_cache_specs 消费其类型面）
@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L549-L556 —— 逐字
        return all(
            isinstance(spec, ChunkedLocalAttentionSpec)
            and spec.attention_chunk_size == self.attention_chunk_size
            for spec in kv_cache_specs.values()
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L562-L628 SlidingWindowSpec —— 减法
#   子集（real_page_size 的 nvfp4 支删除——ch27；admission/max_memory 面
#   归 ch13）
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L562-L628（锚点双置：声明上方注释同文）
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L562-L628（锚点双置：声明上方注释同文）
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L562-L628（锚点双置：声明上方注释同文）
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * get_dtype_size(self.dtype)
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L562-L628（锚点双置：声明上方注释同文）
        return all(
            isinstance(spec, SlidingWindowSpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L631-L714 SlidingWindowMLASpec —— 减法
#   子集（584B 特账 + storage÷compress + merge 逐字）
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        assert self.model_version in (None, "deepseek_v4"), (
            f"Unsupported model version: {self.model_version}"
        )
        return (
            self.storage_block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list) -> "SlidingWindowMLASpec":
        # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
        assert all(isinstance(spec, SlidingWindowMLASpec) for spec in specs), (
            "All attention layers in the same KV cache group must be "
            "SlidingWindowMLASpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        sliding_window_set = set(spec.sliding_window for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(sliding_window_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, sliding "
            "window size, and KV block stride indexing."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            sliding_window=sliding_window_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
        )

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L631-L714（锚点双置：声明上方注释同文）
        return all(
            isinstance(spec, SlidingWindowMLASpec)
            and spec.sliding_window == self.sliding_window
            for spec in kv_cache_specs.values()
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L837-L935 UniformTypeKVCacheSpecs
#   —— 逐字（get_num_layer_tuples 等 DSV4 注记族保留——grouped 分组消费）
@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
    """
    A KV cache spec for multiple layers with the same type of attention. Here,
    same types means always need the same number of token slots. For example,
    sliding window attentions with different window sizes are not the same type
    and should not be merged into one UniformTypeKVCacheSpecs.
    """

    kv_cache_specs: dict[str, KVCacheSpec]

    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
        """
        Whether all layers have the same type of KV cache spec.

        Uses the registry to determine grouping base classes, so custom specs
        that inherit from FullAttentionSpec are treated as full attention.
        """
        block_sizes = set(spec.block_size for spec in kv_cache_specs.values())
        if len(block_sizes) > 1:
            # Different block sizes, not uniform.
            return False
        first_spec = next(iter(kv_cache_specs.values()))
        return first_spec.is_uniform_with_collection(kv_cache_specs)

    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]):
        # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
        """
        Return a SameTypeKVCacheSpecs object if all layers have the same type of
        KV cache spec. Return None if not.
        """
        if cls.is_uniform_type(kv_cache_specs):
            block_size = next(iter(kv_cache_specs.values())).block_size
            return cls(block_size=block_size, kv_cache_specs=kv_cache_specs)
        else:
            return None

    # NOTE: below util functions are only used by DeepseekV4 for now.
    def get_page_sizes(self) -> list[int]:
        # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
        return list(set(spec.page_size_bytes for spec in self.kv_cache_specs.values()))

    def get_num_layer_tuples(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L837-L935（锚点双置：声明上方注释同文）
        return Counter(
            spec.page_size_bytes for spec in self.kv_cache_specs.values()
        ).most_common(1)[0][1]

    # SUBTRACTED: max_memory_usage_bytes/max_memory_usage_pages（L849-L855、
    #   L926-L932）——分配面归 ch13/ch14


# SOURCE: vllm/v1/kv_cache_interface.py:L958-L968 KVCacheGroupSpec —— 逐字
@dataclass
class KVCacheGroupSpec:
    # SOURCE: vllm/v1/kv_cache_interface.py:L958-L968（锚点双置：声明上方注释同文）
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


# ── spec 注册表（分组等价类的基类解析）─────────────────────────────────
# SOURCE: vllm/v1/kv_cache_spec_registry.py KVCacheSpecRegistry —— HOST SEAM
#   子集：get_uniform_type_base_spec 的 MRO 走查语义逐字；register 的
#   manager_class 面（single_type_kv_cache_manager 的注册族）归 ch14——
#   此处只登记 uniform_type_base_spec 映射（对照
#   vllm/v1/core/single_type_kv_cache_manager.py:L1881-L1938 的注册表）。
_REGISTRY_KVCACHESPEC_LIST: dict = {}


# SOURCE: vllm/v1/kv_cache_spec_registry.py:L80-L122 KVCacheSpecRegistry.register
#   —— HOST SEAM 子集（manager 位省略）
def _register_spec_base(kvcache_spec_cls, uniform_type_base_spec) -> None:
    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L88-L91 —— HOST SEAM 子集
    assert issubclass(kvcache_spec_cls, uniform_type_base_spec), (
        f"{kvcache_spec_cls.__name__} must inherit from its declared "
        f"uniform_type_base_spec {uniform_type_base_spec.__name__}."
    )
    _REGISTRY_KVCACHESPEC_LIST[kvcache_spec_cls] = uniform_type_base_spec


# 对照 SOURCE: vllm/v1/core/single_type_kv_cache_manager.py:L1881-L1938
#   register_all_kvcache_specs 的 base-spec 注册面（manager 族省略——ch14）
_register_spec_base(FullAttentionSpec, FullAttentionSpec)
_register_spec_base(SlidingWindowSpec, SlidingWindowSpec)
_register_spec_base(SlidingWindowMLASpec, SlidingWindowMLASpec)
_register_spec_base(MLAAttentionSpec, FullAttentionSpec)
_register_spec_base(HiddenStateCacheSpec, FullAttentionSpec)


# SOURCE: vllm/v1/kv_cache_spec_registry.py:L128-L152 get_uniform_type_base_spec
#   —— HOST SEAM：MRO 走查语义逐字
def _get_uniform_type_base_spec(kv_cache_spec: KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_spec_registry.py:L128-L152（锚点双置：声明上方注释同文）
    kvcache_spec_cls = type(kv_cache_spec)

    # Walk up the MRO to find a registered base class
    for base in kvcache_spec_cls.__mro__:
        if base in _REGISTRY_KVCACHESPEC_LIST:
            return _REGISTRY_KVCACHESPEC_LIST[base]

    return None
