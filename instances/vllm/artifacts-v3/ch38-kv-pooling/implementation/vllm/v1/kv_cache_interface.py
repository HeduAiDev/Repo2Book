# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存规格与配置：池化边界把它当**输入几何**消费。

# SOURCE: vllm/v1/kv_cache_interface.py:L94-L1010
# SUBTRACTED: 量化模式的 nvfp4/int4 头维折算、RSWA/ChunkedLocal/Sink/Encoder
#   等特化规格、max_memory_usage 预算推导（ch14 的面）——本章只保留
#   「一种注意力规格 + 页字节数 = 2*block_size*num_kv_heads*head_size*dtype_size」
#   这一条主路 + 池化面消费的类型位/谓词/分组容器。
"""

from dataclasses import dataclass, field
from enum import Enum
from math import prod as _prod
from typing import Any

import torch

from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import get_dtype_size


# SOURCE: vllm/v1/kv_cache_interface.py:L94-L105
class KVCacheSpecKind(str, Enum):
    """KV cache spec kind, used for KV events."""

    UNKNOWN = "unknown"
    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    MLA_ATTENTION = "mla_attention"
    SLIDING_WINDOW_MLA = "sliding_window_mla"
    SINK_FULL_ATTENTION = "sink_full_attention"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    MAMBA = "mamba"
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"
    CROSS_ATTENTION = "cross_attention"


# SOURCE: vllm/v1/kv_cache_interface.py:L107-L180
@dataclass(frozen=True)
class KVCacheSpec:
    """Base class for specifying the KV cache format of a layer."""

    block_size: int
    """Number of tokens per block."""

    # SOURCE: vllm/v1/kv_cache_interface.py:L117-L130（基类抽象）
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L117-L130
        raise NotImplementedError

    # SOURCE: vllm/v1/kv_cache_interface.py:L139-L150
    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L139-L150
        return cdiv(max_len, self.block_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L183-L231
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L183-L231
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    # SOURCE: vllm/v1/kv_cache_interface.py:L192-L202
    @property
    def unpadded_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L192-L202
        return self.real_page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L204-L209
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L204-L209
        return self.unpadded_page_size_bytes

    # SOURCE: vllm/v1/kv_cache_interface.py:L211-L226
    # SUBTRACTED: 量化头维折算（nvfp4 / int4 per-token-head）——ch27 的面。
    # SOURCE: vllm/v1/kv_cache_interface.py:L211-L226
    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L211-L226
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )


# SOURCE: vllm/v1/kv_cache_interface.py:L234-L280
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L234-L280
    """全注意力组规格：本章正典（池化搬运的 KV 都是它）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L389-L400
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L389-L400
    """MLA 组规格（类型位；MLA 复制页推导已按减法计划删除）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L559-L600
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L559-L600
    """滑动窗口组规格：SWA 尾扫查找的窗口来源。"""

    sliding_window: int


# SOURCE: vllm/v1/kv_cache_interface.py:L710-L780
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L710-L780
    """SSM 组规格——本章保留页几何（worker 规范化分支消费）与类型位。

    # SUBTRACTED: conv 分解/投机 scratch 槽、mamba_cache_mode 深分支
    #   （resolve_mamba_align_size 已按减法计划删除）。
    """

    shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple = ()
    page_size_padded: int | None = None
    mamba_cache_mode: str = "none"

    # SOURCE: vllm/v1/kv_cache_interface.py:L719-L726
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L719-L726
        page_size = sum(
            _prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
        if self.page_size_padded is not None:
            assert self.page_size_padded >= page_size
            return self.page_size_padded
        return page_size


# SOURCE: vllm/v1/kv_cache_interface.py:L770-L780
@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L770-L780
    """交叉注意力（MooncakeStore 部署门的类型位）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L837-L870
@dataclass(frozen=True, kw_only=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L837-L870
    """把一个组里逐层的同类型规格合起来（worker 规范化的类型位）。"""

    kv_cache_specs: dict[str, KVCacheSpec] = field(default_factory=dict)


# SOURCE: vllm/v1/kv_cache_interface.py:L901-L931
def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    # SOURCE: vllm/v1/kv_cache_interface.py:L901-L931
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
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return KVCacheSpecKind.MLA_ATTENTION
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return KVCacheSpecKind.SLIDING_WINDOW
    if isinstance(kv_cache_spec, MambaSpec):
        return KVCacheSpecKind.MAMBA
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


# SOURCE: vllm/v1/kv_cache_interface.py:L933-L943
def get_kv_cache_spec_sliding_window(kv_cache_spec: KVCacheSpec) -> int | None:
    # SOURCE: vllm/v1/kv_cache_interface.py:L933-L943
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        inner_windows = {
            get_kv_cache_spec_sliding_window(spec)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        return next(iter(inner_windows)) if len(inner_windows) == 1 else None
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        return kv_cache_spec.sliding_window
    return None


# SOURCE: vllm/v1/kv_cache_interface.py:L945-L955
@dataclass
class KVCacheTensor:
    # SOURCE: vllm/v1/kv_cache_interface.py:L945-L955
    """How the workers should initialize the KV cache tensor of a layer."""

    size: int  # size of the KV cache tensor in bytes
    shared_by: list[str]  # layer names that share the same KV cache tensor
    offset: int = 0  # byte offset of this layer within a contiguous block
    block_stride: int = 0  # total bytes per block in a packed layout (0 = not packed)


# SOURCE: vllm/v1/kv_cache_interface.py:L957-L970
@dataclass
class KVCacheGroupSpec:
    # SOURCE: vllm/v1/kv_cache_interface.py:L957-L970
    """Represents a group of model layers that share the same KV cache block table."""

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    is_eagle_group: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py:L972-L1010
@dataclass
class KVCacheConfig:
    # SOURCE: vllm/v1/kv_cache_interface.py:L972-L1010
    """The KV cache configuration of a model."""

    num_blocks: int
    """The number of KV cache blocks"""
    kv_cache_tensors: list[KVCacheTensor]
    """How should model runner initialize the KV cache tensors for each layer"""
    kv_cache_groups: list[KVCacheGroupSpec]
    """The kv cache groups of the model."""

    # SOURCE: vllm/v1/kv_cache_interface.py:L991-L993
    @property
    def has_mamba_layers(self) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L991-L993
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)


__all__ = [
    "KVCacheSpecKind",
    "KVCacheSpec",
    "AttentionSpec",
    "FullAttentionSpec",
    "MLAAttentionSpec",
    "SlidingWindowSpec",
    "MambaSpec",
    "CrossAttentionSpec",
    "UniformTypeKVCacheSpecs",
    "get_kv_cache_spec_kind",
    "get_kv_cache_spec_sliding_window",
    "KVCacheTensor",
    "KVCacheGroupSpec",
    "KVCacheConfig",
]
