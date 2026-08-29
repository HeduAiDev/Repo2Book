# SOURCE: vllm/v1/kv_cache_interface.py
# 本章消费面（ch13/14 已立、ch16 只消费的池侧类型账）：spec 家族最小面
# （FullAttention 主角 + Mamba/Cross/EncoderOnly 的 isinstance 位）+
# KVCacheConfig 的 needs_kv_cache_zeroing 谓词——站 6/站 10 清零护栏的开关。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   定账/组化/布局族（get_kv_cache_configs/张量共享布局——ch14 全量切面）；
#   SWA/ChunkedLocal/RSWA/SinkFull/UniformType spec（混合命中调和 → ch15；
#     本章单 full 组 + NoPrefixCache 多组两形态）；
#   KVQuantMode 只留 NONE（量化 KV → 后章）；
#   page_size/max_memory_usage_bytes 族（显存算术 → ch14）。
from dataclasses import dataclass
from enum import IntEnum

import torch


# SOURCE: vllm/v1/kv_cache_interface.py:L33 KVQuantMode（NONE 单支——量化族删）
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0


# SOURCE: vllm/v1/kv_cache_interface.py:L108 KVCacheSpec
@dataclass(frozen=True)
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    # SOURCE: vllm/v1/kv_cache_interface.py:L115-L116
    block_size: int


# SOURCE: vllm/v1/kv_cache_interface.py:L184 AttentionSpec
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L185-L190
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE


# （full attention：全历史驻留、从不窗外回收——本章主配置）
@dataclass(frozen=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L235
class FullAttentionSpec(AttentionSpec):
    """
    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding
    window attention are regarded as full attention in KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    in attention backends.
    """


# （get_block_ids_for_computed_tokens 的 isinstance 位——裁剪时原样保留）
# SOURCE: vllm/v1/kv_cache_interface.py:L763 EncoderOnlyAttentionSpec
@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L763（空壳——isinstance 位）
    pass


# SOURCE: vllm/v1/kv_cache_interface.py class CrossAttentionSpec(AttentionSpec)
@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py（cross-attention——encoder-
    #   decoder 专用；本章构造期被拒，isinstance 位保留）
    """
    KV cache spec for cross-attention layers in encoder-decoder models.
    """


# SOURCE: vllm/v1/kv_cache_interface.py:L710 MambaSpec（混合组化位——
#   has_mamba_layers 谓词与 Hybrid coordinator 的 align 组判定用；
#   mamba 状态管理 → 邻章）
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L711-L717（字段面逐字——
    #   mamba_type 枚举/num_speculative_blocks 删，账位缩）
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = None
    mamba_cache_mode: str = "none"


# SOURCE: vllm/v1/kv_cache_interface.py:L946 KVCacheTensor
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


# SOURCE: vllm/v1/kv_cache_interface.py:L958 KVCacheGroupSpec
@dataclass
class KVCacheGroupSpec:
    """
    Represents a group of model layers that share the same KV cache block table.
    These layers are regarded as one layer in the KV cache manager.
    """

    # The names of model layers in this group
    # SOURCE: vllm/v1/kv_cache_interface.py:L963-L968
    layer_names: list[str]
    # The KV cache spec of this manager layer
    kv_cache_spec: KVCacheSpec
    # Whether this group contains EAGLE/MTP draft attention layers.
    is_eagle_group: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py:L973 KVCacheConfig
@dataclass
class KVCacheConfig:
    """
    The KV cache configuration of a model.
    """

    # SOURCE: vllm/v1/kv_cache_interface.py:L978-L986
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
        # SOURCE: vllm/v1/kv_cache_interface.py:L992
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)

    # SOURCE: vllm/v1/kv_cache_interface.py:L995 has_mixed_precision_kv_cache
    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """Whether attention groups store their KV cache at more than one precision."""
        # SOURCE: vllm/v1/kv_cache_interface.py:L998-L1010（UniformTypeKVCacheSpecs
        #   分支删——布局族归 ch14；每组 (dtype, quant) 去重后 >1 即混合精度）
        kv_cache_precisions: set[tuple[torch.dtype, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, AttentionSpec):
                kv_cache_precisions.add((group_spec.dtype, group_spec.kv_quant_mode))
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
        # SOURCE: vllm/v1/kv_cache_interface.py:L1022
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache
