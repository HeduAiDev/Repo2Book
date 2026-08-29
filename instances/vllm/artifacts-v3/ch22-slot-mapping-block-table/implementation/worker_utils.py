# SOURCE: vllm/v1/worker/utils.py
# ch22 切面（m9）：后端公共 kernel 块尺寸的选取——select_common_block_size
# （先试管理块尺寸本身、否则从 int 型候选里降序取第一个公共因子）与
# prepare_kernel_block_sizes（逐组装配：attention 组走公共尺寸、mamba 组不
# 拆块）+ AttentionGroup（has_separate_kv_update 裁决读 g.backend 的载体）。
# KVBlockZeroer/_zero_kv_blocks_kernel 归 ch13；mm encoder/sanity 归他域。
from __future__ import annotations

from dataclasses import dataclass

from .backend import MultipleOf
from .kv_cache_interface import (
    AttentionSpec,
    EncoderOnlyAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

# 注：真实文件里 MultipleOf 来自 vllm.v1.attention.backend——本章包内同源
#   （backend.MultipleOf），SUBTRACTED 的只是 vllm.* 顶层 import 归一。


# SOURCE: vllm/v1/worker/utils.py:L217 AttentionGroup —— 同一 KV cache 组的
#   一撮注意力层 + 其后端（has_separate_kv_update 逐组读 .backend）
@dataclass
# SOURCE: vllm/v1/worker/utils.py:L217 AttentionGroup
class AttentionGroup:
    backend: type
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    # When ubatching is enabled we will have a metadata builder for each ubatch
    # so that if they use internal persistent buffers for cudagraphs, and they
    # won't have to worry about conflicting with the other ubatches.
    # SUBTRACTED: metadata_builders 字段（L225-L227 默认工厂）与
    #   create_metadata_builders 方法（L229-L259）——后端 metadata 构建域
    #   → ch21（delete[5] ubatching 亦不进）。


# SOURCE: vllm/v1/worker/utils.py:L266 select_common_block_size —— 公共 kernel
#   块尺寸选取（m9 的算术本体）
def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type],
) -> int:
    """
    Select a block size that is supported by all backends and is a factor of
    kv_manager_block_size.

    If kv_manager_block_size is supported by all backends, return it directly.
    Otherwise, return the max supported size.

    Args:
        kv_manager_block_size: Block size of KV cache.
        backends: List of attention backend classes.

    Returns:
        The selected block size.

    Raises:
        ValueError: If no valid block size found.
    """

    # SOURCE: vllm/v1/worker/utils.py:L288-L305 block_size_is_supported 局部函数
    def block_size_is_supported(
        backends: list[type], block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        for backend in backends:
            is_supported = False
            for supported_size in backend.get_supported_kernel_block_sizes():
                if isinstance(supported_size, int):
                    if block_size == supported_size:
                        is_supported = True
                elif isinstance(supported_size, MultipleOf):
                    if block_size % supported_size.base == 0:
                        is_supported = True
                else:
                    raise ValueError(f"Unknown supported size: {supported_size}")
            if not is_supported:
                return False
        return True

    # SOURCE: vllm/v1/worker/utils.py:L307-L310 Case 1：管理块尺寸本身全后端
    #   支持 → 直接返回
    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # SOURCE: vllm/v1/worker/utils.py:L312-L331 Case 2：int 型候选降序取第一
    #   个公共因子（注释证明原文保留）
    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # Simple proof:
    # If the supported size b is in MultipleOf(x_i) format for all attention
    # backends i, and b a factor of kv_manager_block_size, then
    # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
    # return kv_manager_block_size in case 1.
    all_int_supported_sizes = set(
        supported_size
        for backend in backends
        for supported_size in backend.get_supported_kernel_block_sizes()
        if isinstance(supported_size, int)
    )

    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size != 0:
            continue
        if block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(f"No common block size for {kv_manager_block_size}. ")


# SOURCE: vllm/v1/worker/utils.py:L335 prepare_kernel_block_sizes —— 逐组 kernel
#   块尺寸装配（分配块≠kernel 块的解耦入口）
def prepare_kernel_block_sizes(
    kv_cache_config, attn_groups: list[list[AttentionGroup]]
) -> list[int]:
    """
    Generate kernel_block_sizes that matches each block_size.

    For attention backends that support virtual block splitting,
    use the supported block sizes from the backend.
    For other backends (like Mamba), use the same block size (no splitting).

    Args:
        kv_cache_config: The KV cache configuration.
        attn_groups: Attention groups indexed by KV cache group id.

    Returns:
        List of kernel block sizes for each cache group.
    """
    kernel_block_sizes = []
    # SOURCE: vllm/v1/worker/utils.py:L353-L375 逐组判别装配
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # pick an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
            # SOURCE: vllm/v1/worker/utils.py:L359-L360（encoder-only 组无
            #   paged KV、不参与 kernel 块装配——判别位原样保留）
            continue
        if isinstance(kv_cache_spec, AttentionSpec):
            # This is an attention backend that supports virtual block splitting.
            kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
            group_backends = [g.backend for g in attn_groups[kv_cache_gid]]
            selected_kernel_size = select_common_block_size(
                kv_manager_block_size, group_backends
            )
            kernel_block_sizes.append(selected_kernel_size)
        elif isinstance(kv_cache_spec, MambaSpec):
            # This is likely Mamba or other non-attention cache, no splitting.
            kernel_block_sizes.append(kv_cache_spec.block_size)
        else:
            raise NotImplementedError(
                f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
            )
    return kernel_block_sizes
