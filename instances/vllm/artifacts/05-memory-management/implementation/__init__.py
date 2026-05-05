# GPU Memory Management — Ch05 Implementation
# REFERENCE: vllm/v1/core/ (block_pool, kv_cache_manager, kv_cache_coordinator,
#                         kv_cache_utils, single_type_kv_cache_manager)

from .block_pool import (
    BlockPool,
    KVCacheBlock,
    FreeKVCacheBlockQueue,
    BlockHashToBlockMap,
    hash_block_tokens,
    NONE_HASH,
)

from .kv_cache_manager import (
    KVCacheManager,
    KVCacheBlocks,
    Request,
    compute_request_block_hashes,
)

from .memory_profiling import (
    MemoryBudget,
    KVCacheConfig,
    profile_gpu_memory,
    compute_kv_cache_config,
    format_bytes,
    cdiv,
)

__all__ = [
    # block_pool
    "BlockPool",
    "KVCacheBlock",
    "FreeKVCacheBlockQueue",
    "BlockHashToBlockMap",
    "hash_block_tokens",
    "NONE_HASH",
    # kv_cache_manager
    "KVCacheManager",
    "KVCacheBlocks",
    "Request",
    "compute_request_block_hashes",
    # memory_profiling
    "MemoryBudget",
    "KVCacheConfig",
    "profile_gpu_memory",
    "compute_kv_cache_config",
    "format_bytes",
    "cdiv",
]
