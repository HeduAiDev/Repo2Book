# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""逻辑块尺寸 ↔ kernel 块尺寸的协商。

NIXL 的区域几何按 **kernel 块** 描述：若用户设的逻辑块尺寸不被注意力后端接受，
vLLM 会把块切成 kernel 块的整数倍，并在传输描述符里按块号重映射
（`_physical_blocks_per_logical_kv_block`）。

# SOURCE: vllm/v1/worker/utils.py:L266-L340（select_common_block_size）
# SUBTRACTED: prepare_kernel_block_sizes / 显存分析 / 出块工具等其余函数。
"""

from vllm.v1.attention.backend import AttentionBackend, MultipleOf


# SOURCE: vllm/v1/worker/utils.py:L266-L340
def select_common_block_size(
    kv_manager_block_size: int,
    backends: list[type[AttentionBackend]],
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

    # SOURCE: vllm/v1/worker/utils.py:L288-L305
    def block_size_is_supported(
        backends: list[type[AttentionBackend]], block_size: int
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

    # Case 1: if the block_size of kv cache manager is supported by all backends,
    # return it directly.
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2: otherwise, the block_size must be an `int`-format supported size of
    # at least one backend. Iterate over all `int`-format supported sizes in
    # descending order and return the first one that is supported by all backends.
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


__all__ = ["select_common_block_size"]
