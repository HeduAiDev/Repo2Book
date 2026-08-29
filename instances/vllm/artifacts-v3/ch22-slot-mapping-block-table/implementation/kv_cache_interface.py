# SOURCE: vllm/v1/kv_cache_interface.py
# ch22 切面：KV cache spec 的消费面——may_reinitialize_input_batch 按 spec
# kind 选 SlotMappingMode（m10）、prepare_kernel_block_sizes 按	spec 类型定
# kernel 块尺寸（m9）、_get_slot_mappings/_get_block_table 读组结构与
# layer_names（m12）。分页账本/混合组化的调度侧故事归 ch13/ch14——本文件只
# 留本章消费到的类型面与判别函数。
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TYPE_CHECKING

import torch

from .math_utils import cdiv

if TYPE_CHECKING:
    pass

# HOST SEAM：vllm/utils/torch_utils.py:L212 get_dtype_size 的等价装配
#   （dtype.itemsize）。
_get_dtype_size = lambda dtype: dtype.itemsize  # noqa: E731


# SOURCE: vllm/v1/kv_cache_interface.py:L94 KVCacheSpecKind
class KVCacheSpecKind(str, Enum):
    # SOURCE: vllm/v1/kv_cache_interface.py:L95-L103（全枚举逐字）
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


# SOURCE: vllm/v1/kv_cache_interface.py:L107 KVCacheSpec —— 单层 KV 格式基类
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int

    # SUBTRACTED: page_size_bytes / storage_block_size / max_memory_usage_bytes
    #   抽象属性与 copy_with_new_block_size / merge / is_uniform_with_collection
    #   （L113-L222）——ch13/ch14 账本域，本章消费面不触及。

    # SOURCE: vllm/v1/kv_cache_interface.py:L139 max_num_blocks_per_req ——
    #   「每请求的块表行宽」：worker 侧块表行长即由它定
    # SOURCE: vllm/v1/kv_cache_interface.py:L139 max_num_blocks_per_req
    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        """
        The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.
        """
        return cdiv(max_len, self.block_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L~225 AttentionSpec —— 注意力类 spec
#   中间基类（真实类带 dtype/head_size/head_size_v 等字段；本章消费面只判
#   类型，字段面 SUBTRACTED → ch13/ch14）。
@dataclass(frozen=True, kw_only=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L~224 AttentionSpec（中间基类——字段面 → ch13/ch14）
class AttentionSpec(KVCacheSpec):
    pass


# SOURCE: vllm/v1/kv_cache_interface.py:L235 FullAttentionSpec
@dataclass(frozen=True, kw_only=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L235 FullAttentionSpec
class FullAttentionSpec(AttentionSpec):
    """When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding window
    attention are regarded as full attention in KV cache manager."""

    # SUBTRACTED: head_size_v/dtype/sliding_window 等字段与 real_page_size_
    #   bytes/max_memory_usage_bytes/max_num_blocks_per_req 的 DCP 分片版
    #   （L226-L231）——本章消费面只判 kind + 基类行宽公式。


# SOURCE: vllm/v1/kv_cache_interface.py:L710 MambaSpec —— 状态缓存组
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    # SUBTRACTED: mamba_type 枚举字段（L714）——后端选择面 → ch21。
    page_size_padded: int | None = None
    mamba_cache_mode: str = "none"
    num_speculative_blocks: int = 0

    # SOURCE: vllm/v1/kv_cache_interface.py:L716 page_size_bytes
    @property
    # SOURCE: vllm/v1/kv_cache_interface.py:L716 page_size_bytes
    def page_size_bytes(self) -> int:
        page_size = sum(
            prod(shape) * _get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size

    # SOURCE: vllm/v1/kv_cache_interface.py:L728 max_memory_usage_bytes
    def max_memory_usage_bytes(self, vllm_config) -> int:
        if vllm_config.cache_config.mamba_cache_mode == "all":
            max_model_len = vllm_config.model_config.max_model_len
            return (
                cdiv(max_model_len, self.block_size) + self.num_speculative_blocks
            ) * self.page_size_bytes
        elif vllm_config.cache_config.mamba_cache_mode == "align":
            return self.page_size_bytes * (2 + self.num_speculative_blocks)
        else:
            return self.page_size_bytes * (1 + self.num_speculative_blocks)

    # SOURCE: vllm/v1/kv_cache_interface.py:L740 max_num_blocks_per_req
    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        # Mamba state is replicated across DCP/PCP ranks, never sharded, so
        # no CP scaling applies.
        if vllm_config.cache_config.mamba_cache_mode == "align":
            # Block table rows are position-indexed over the full sequence
            # even though only 2 + num_speculative_blocks state blocks are
            # resident at a time (earlier states are nulled out by
            # remove_skipped_blocks), so the row length must cover max_len
            # rather than max_memory_usage_bytes.
            return cdiv(max_len, self.block_size) + self.num_speculative_blocks
        return cdiv(self.max_memory_usage_bytes(vllm_config), self.page_size_bytes)


# SOURCE: vllm/v1/kv_cache_interface.py:L763 EncoderOnlyAttentionSpec
@dataclass(frozen=True, kw_only=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L765-L770（encoder-only 层无 KV）
    def max_memory_usage_bytes(self, vllm_config) -> int:
        # Encoder-only layers do not need KV cache
        return 0


# SOURCE: vllm/v1/kv_cache_interface.py:L837 UniformTypeKVCacheSpecs ——
#   多层同型合组的 spec 容器（may_reinitialize/prepare_kernel 对它拆内层）
@dataclass(frozen=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L837 UniformTypeKVCacheSpecs
class UniformTypeKVCacheSpecs(KVCacheSpec):
    kv_cache_specs: dict[str, KVCacheSpec]

    # SUBTRACTED: page_size_bytes/max_memory_usage_bytes/is_uniform_type/
    #   from_specs/get_page_sizes 等（L840-L912）——ch14 混合组化域。


# SOURCE: vllm/v1/kv_cache_interface.py:L901 get_kv_cache_spec_kind
def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    # SOURCE: vllm/v1/kv_cache_interface.py:L902-L908 Uniform 容器拆内层
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_kinds = {
            get_kv_cache_spec_kind(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(inner_kinds) == 1:
            return next(iter(inner_kinds))
        return KVCacheSpecKind.UNKNOWN
    # SUBTRACTED: SlidingWindowMLA/MLA/SinkFull/ChunkedLocal/SlidingWindow/
    #   Cross 六类判别（L909-L921、L933-L935）——后端选择/滑窗域 → ch21，
    #   本章模型只含 full/mamba/encoder-only 三类。
    # Keep subclass checks before base classes so specialized specs keep their
    # more precise kind.
    # SOURCE: vllm/v1/kv_cache_interface.py:L925-L932
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    return KVCacheSpecKind.UNKNOWN


# SOURCE: vllm/v1/kv_cache_interface.py:L958 KVCacheGroupSpec —— 一组共享
#   同一张块表的层
@dataclass
# SOURCE: vllm/v1/kv_cache_interface.py:L958 KVCacheGroupSpec
class KVCacheGroupSpec:
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


# SOURCE: vllm/v1/kv_cache_interface.py:L971 KVCacheConfig
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list
    """How should model runner initialize the KV cache tensors for each layer
    (SUBTRACTED: KVCacheTensor 装配面 → ch13；本章恒空表)"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """
    The kv cache groups of the model.
    For models with only one type of attention, there is only one group that
    contains all layers.
    For models with multiple types of attention, there will be multiple groups,
    see `_get_kv_cache_config_uniform_page_size` for more details.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L989 has_mamba_layers
    @property
    # SOURCE: vllm/v1/kv_cache_interface.py:L989 has_mamba_layers
    def has_mamba_layers(self) -> bool:
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    # SUBTRACTED: has_mixed_precision_kv_cache（L992 起）——量化域 → ch14。
