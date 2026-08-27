# SOURCE: vllm/v1/kv_cache_interface.py
# 页的物理形状载体（m10/m13）：AttentionSpec.real_page_size_bytes =
# 2 × block_size × num_kv_heads × head_dim × dtype 字节——「一块多大」的唯一
# 公式；FullAttentionSpec = 本章单组全注意力主路径的 spec；KVCacheConfig =
# 启动期账本（num_blocks 与 needs_kv_cache_zeroing 的单一事实源）。
# SUBTRACTED: 混合/多组 spec 家族（dossier.delete 第 4 条）：SlidingWindowSpec/
#   MambaSpec/RSWASpec/CrossAttentionSpec/SinkFullAttentionSpec/ChunkedLocal
#   AttentionSpec/MLAAttentionSpec/TQFullAttentionSpec/SlidingWindowMLASpec/
#   EncoderOnlyAttentionSpec/HiddenStateCacheSpec/UniformTypeKVCacheSpecs 及
#   各自的 admission cap（L522-L608——回收型准入门 → ch14）；KVCacheSpecRegistry
#   注册面；get_kv_cache_spec_kind/get_kv_cache_spec_sliding_window（第 11 条）。
import copy
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from .math_utils import cdiv
from .torch_utils import get_dtype_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig  # noqa: F401 — 类型账位（构造期校验归 ch03）


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
    TURBOQUANT = 6  # Hadamard-rotated Lloyd-Max quant, packed K+V per slot

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
        """True for NVFP4 packed quantization."""
        # SOURCE: vllm/v1/kv_cache_interface.py:L60
        return self == KVQuantMode.NVFP4

    # SOURCE: vllm/v1/kv_cache_interface.py:L62-L65 is_turboquant
    @property
    def is_turboquant(self) -> bool:
        """True for turboquant quantization."""
        # SOURCE: vllm/v1/kv_cache_interface.py:L65
        return self == KVQuantMode.TURBOQUANT


# SOURCE: vllm/v1/kv_cache_interface.py:L68 get_kv_quant_mode
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    # SOURCE: vllm/v1/kv_cache_interface.py:L70-L82
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


# SOURCE: vllm/v1/kv_cache_interface.py:L85 is_quantized_kv_cache
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    # SOURCE: vllm/v1/kv_cache_interface.py:L86
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


# SOURCE: vllm/v1/kv_cache_interface.py:L89 kv_cache_uses_per_token_head_scales
def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    # SOURCE: vllm/v1/kv_cache_interface.py:L91
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


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

    # SUBTRACTED: is_uniform_with_collection（L167-L180——registry 注册族，
    #   dossier.delete 第 4 条）。


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
    Whether the layer attends non-causally (e.g., Prefix LM). Carried on the
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
    # SOURCE: vllm/v1/kv_cache_interface.py:L961-L969
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    # SUBTRACTED: is_eagle_group 默认值与 eagle 装配（dossier.delete 第 5 条
    #   ——投机解码草稿头专用，→ ch33）


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

    # SUBTRACTED: has_mamba_layers（L991-L993——Mamba spec 族，第 4 条）；
    #   has_mixed_precision_kv_cache 里 UniformTypeKVCacheSpecs 分支
    #   （L998-L1005——第 4 条）。

    # SOURCE: vllm/v1/kv_cache_interface.py:L995-L1011 has_mixed_precision_kv_cache
    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
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
        # SOURCE: vllm/v1/kv_cache_interface.py:L1022（mamba 半边删——第 4 条；
        #   混合精度半边保留）
        return self.has_mixed_precision_kv_cache
