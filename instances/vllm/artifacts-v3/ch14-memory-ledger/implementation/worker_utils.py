# SOURCE: vllm/v1/worker/utils.py
# 账本 worker 侧的换算面：request_memory（预算先于一切，m1 第 1 站）+
# prepare_kernel_block_sizes / select_common_block_size（kernel 块细分
# 的后端协商，m14——协商内景深讲 → ch21，函数本体是纯算术保留）。
# MultipleOf 从 vllm/v1/attention/backend.py:L49-L53 折入（支持尺寸的
# 「倍数」声明）。
# SUBTRACTED（dossier.delete 批准项的落点 + 本章链路不用）：mm 嵌入
#   校验族（L350-L406——dossier.delete 第 1 条多模态）；bind_kv_cache /
#   add_kv_sharing_layers_to_kv_cache_groups（L432-L479——跨层 KV 共享，
#   ch13 边界/ch21 前向上下文）；KVBlockZeroer / AttentionGroup 数据类
#   （ch13 精简版消费面——本章 attn_groups 以鸭子类型承载 backend 位）。
import math

from .cache import CacheConfig
from .kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from .mem_utils import MemorySnapshot, format_gib


# SOURCE: vllm/v1/attention/backend.py:L49 MultipleOf（折入：supported
#   sizes 的「倍数」声明，select_common_block_size 消费）
class MultipleOf:
    # SOURCE: vllm/v1/attention/backend.py:L51-L53
    base: int

    # SOURCE: vllm/v1/attention/backend.py:L52-L53 __init__
    def __init__(self, base: int):
        # SOURCE: vllm/v1/attention/backend.py:L53
        self.base = base


# SOURCE: vllm/v1/worker/utils.py:L409 request_memory
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    # SOURCE: vllm/v1/worker/utils.py:L414-L416（ceil(总显存 × util)）
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )

    # SOURCE: vllm/v1/worker/utils.py:L418-L427（free 不足直接 raise）
    if init_snapshot.free_memory < requested_memory:
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            f"({format_gib(init_snapshot.free_memory)}/"
            f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
            f"is less than desired GPU memory utilization "
            f"({cache_config.gpu_memory_utilization}, "
            f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
            f"utilization or reduce GPU memory used by other processes."
        )

    # SOURCE: vllm/v1/worker/utils.py:L429
    return requested_memory


# SOURCE: vllm/v1/worker/utils.py:L266 select_common_block_size
def select_common_block_size(
    kv_manager_block_size: int,
    backends: list,
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

    # SOURCE: vllm/v1/worker/utils.py:L288 block_size_is_supported
    def block_size_is_supported(
        backends: list, block_size: int
    ) -> bool:
        """Check if the block size is supported by all backends."""
        # SOURCE: vllm/v1/worker/utils.py:L292-L305
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

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    # SOURCE: vllm/v1/worker/utils.py:L307-L310
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
    # Simple proof:
    # If the supported size b is in MultipleOf(x_i) format for all attention
    # backends i, and b a factor of kv_manager_block_size, then
    # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
    # return kv_manager_block_size in case 1.
    # SOURCE: vllm/v1/worker/utils.py:L320-L331
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


# SOURCE: vllm/v1/worker/utils.py:L335 prepare_kernel_block_sizes
def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list]
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
    # SOURCE: vllm/v1/worker/utils.py:L352-L375
    kernel_block_sizes = []
    for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to dispatch.
            kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
        # SUBTRACTED: EncoderOnlyAttentionSpec 的 continue（L359-L360
        #   ——encoder-only 族，dossier.delete 第 5 条）
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
