# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""链式哈希种子与块尺寸定账（本章消费面：NONE_HASH / init_none_hash /
BlockHash / maybe_convert_block_hash / resolve_kv_cache_block_sizes）。

# SOURCE: vllm/v1/core/kv_cache_utils.py:L44-L115 + L626-L689
# SUBTRACTED: KVCacheBlock/FreeKVCacheBlockQueue/块哈希器（ch13/ch15）、
#   CBOR 哈希函数族（sha256_cbor/xxhash_cbor 的 warning 分支保语义、函数体
#   不进本章）——链式哈希算法归 ch15，本章只立『跨进程共享池押在
#   PYTHONHASHSEED 上』的种子机制（WC6）。
"""

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NewType

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# SOURCE: vllm/v1/core/kv_cache_utils.py:L44 BlockHash
# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# SOURCE: vllm/v1/core/kv_cache_utils.py:L37-L42 ExternalBlockHash
# ExternalBlockHash is the hash type used for external KV cache consumers.
ExternalBlockHash = NewType("ExternalBlockHash", "int | BlockHash")

# SOURCE: vllm/v1/core/kv_cache_utils.py:L79-L82 maybe_convert_block_hash
def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L79-L82
    import vllm.envs as envs

    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


# The hash seed for the first block of any prefix block sequence.
#
# We use a random value to avoid hash collisions or PYTHONHASHSEED environment
# variable if set such that processes can share the seed if needed. This aligns
# with the behavior of Python's hash() function, which also uses a random seed
# if PYTHONHASHSEED is not set.
#
# The function `init_none_hash` initializes this variable globally.
# SOURCE: vllm/v1/core/kv_cache_utils.py:L87-L96
NONE_HASH: BlockHash
_CBOR_HASH_FUNCTIONS: frozenset = frozenset()


# SOURCE: vllm/v1/core/kv_cache_utils.py:L99-L114
def init_none_hash(hash_fn: Callable[[Any], bytes]):
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L99-L114
    global NONE_HASH

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None and hash_fn in _CBOR_HASH_FUNCTIONS:
        logger.warning(
            "PYTHONHASHSEED is not set. This will lead to non-reproducible "
            "block-hashes when using CBOR-based hash functions such as "
            "sha256_cbor or xxhash_cbor. Consider setting PYTHONHASHSEED to a "
            "fixed value for reproducibility."
        )

    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))


# SOURCE: vllm/v1/core/kv_cache_utils.py:L626-L689
def resolve_kv_cache_block_sizes(
    kv_cache_config: "KVCacheConfig",
    vllm_config: "VllmConfig",
) -> tuple[int, int]:
    # SOURCE: vllm/v1/core/kv_cache_utils.py:L626-L689
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler. Single group: ``cache_config.block_size * dcp``. Multiple
      groups: LCM of every group's effective block size.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # Mamba groups with block_size != cache_config.block_size
    # (mamba_cache_mode != "align") break divisibility; back off to the
    # scheduler block size.
    if any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.block_size != cache_config.block_size
        for g in groups
    ):
        return scheduler_block_size, scheduler_block_size

    requested = cache_config.prefix_match_unit
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size


__all__ = [
    "BlockHash",
    "ExternalBlockHash",
    "NONE_HASH",
    "init_none_hash",
    "maybe_convert_block_hash",
    "resolve_kv_cache_block_sizes",
]
