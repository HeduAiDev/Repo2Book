# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU 池的指标名与块描述符（本章保留类型面——减法计划删除项 7 删记录点）。"""

from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


# SOURCE: vllm/v1/kv_offload/cpu/common.py:L6-L11
class CPUOffloadingMetrics:
    # SOURCE: vllm/v1/kv_offload/cpu/common.py:L6-L11
    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


# SOURCE: vllm/v1/kv_offload/cpu/common.py:L14-L17
class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    # SOURCE: vllm/v1/kv_offload/cpu/common.py:L14-L17
    """
    Spec for loading/storing a KV block to CPU memory.
    """
