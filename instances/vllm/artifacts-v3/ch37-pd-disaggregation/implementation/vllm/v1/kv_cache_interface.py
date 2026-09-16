# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 缓存规格与配置：connector 推算 NIXL 区域/描述符几何的输入。

# SOURCE: vllm/v1/kv_cache_interface.py:L120-L1010
# SUBTRACTED: 混合分配器（HMA）的分组推导、页大小 padding 协商、量化模式的
#   nvfp4/int4 头维折算、Eagle/扩散等特化规格——本章只保留「一种注意力规格 +
#   页字节数 = 2*block_size*num_kv_heads*head_size*dtype_size」这一条主路。
#   MambaSpec 保留为**类型**（多处 isinstance 判定要它），其深分支已按减法计划删除。
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.utils.torch_utils import get_dtype_size


# SOURCE: vllm/v1/kv_cache_interface.py:L108-L180
@dataclass(frozen=True)
class KVCacheSpec:
    """Base class for specifying the KV cache format of a layer."""

    block_size: int
    """Number of tokens per block."""

    # SOURCE: vllm/v1/kv_cache_interface.py:L117-L130（基类抽象）
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L117-L130（基类抽象）
        raise NotImplementedError

    # SOURCE: vllm/v1/kv_cache_interface.py:L139-L150（基类抽象）
    def max_num_blocks_per_req(self, vllm_config: Any, max_len: int) -> int:
        from vllm.utils.math_utils import cdiv

        return cdiv(max_len, self.block_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L184-L231
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
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
    """全注意力组的规格：本章正典（P/D 交接的 KV 都是它）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L559-L600
@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L559-L600
    """滑动窗口组：get_exchange_clipped_blocks 的尾剪裁就为它而设
    （块为整段序列长度分配，超窗的头块不该过线）。"""

    sliding_window: int = 0


# SOURCE: vllm/v1/kv_cache_interface.py:L710-L780
@dataclass(frozen=True, kw_only=True)
class MambaSpec(KVCacheSpec):
    """SSM 组的规格——本章只保留**类型**用于 isinstance 判定与字段位。

    # SUBTRACTED: 状态槽几何/conv 分解/投机 scratch 槽（`num_speculative_blocks`
    保字段、深分支全删）——混合模型传输剪裁归 ch14/ch16 已立的 HMA 语义。
    """

    shapes: tuple = ()
    dtypes: tuple = ()
    num_speculative_blocks: int = 0

    # SOURCE: vllm/v1/kv_cache_interface.py:L719-L726
    @property
    def page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L719-L726
        return 0


# SOURCE: vllm/v1/kv_cache_interface.py:L389-L400
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(AttentionSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L389-L400
    """MLA 组的规格（类型位；MLA 深分支已按减法计划删除）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L631-L650
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L631-L650
    """MLA + 滑窗（类型位）。"""


# SOURCE: vllm/v1/kv_cache_interface.py:L837-L870
@dataclass(frozen=True, kw_only=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    # SOURCE: vllm/v1/kv_cache_interface.py:L837-L870
    """把一个组里逐层的同类型规格合起来（DSA indexer 场景）。"""

    kv_cache_specs: dict[str, KVCacheSpec] = field(default_factory=dict)


# SOURCE: vllm/v1/kv_cache_interface.py:L946-L955
@dataclass
class KVCacheTensor:
    # SOURCE: vllm/v1/kv_cache_interface.py:L946-L955
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
    """The KV cache configuration of a model."""

    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]

    # SOURCE: vllm/v1/kv_cache_interface.py:L991-L993
    @property
    def has_mamba_layers(self) -> bool:
        # SOURCE: vllm/v1/kv_cache_interface.py:L991-L993
        return any(isinstance(g.kv_cache_spec, MambaSpec) for g in self.kv_cache_groups)


__all__ = [
    "KVCacheSpec",
    "AttentionSpec",
    "FullAttentionSpec",
    "SlidingWindowSpec",
    "MLAAttentionSpec",
    "SlidingWindowMLASpec",
    "MambaSpec",
    "UniformTypeKVCacheSpecs",
    "KVCacheTensor",
    "KVCacheGroupSpec",
    "KVCacheConfig",
]
