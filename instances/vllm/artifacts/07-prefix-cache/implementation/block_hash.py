# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L40-L78
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L539-L566
"""BlockHash, parent-chained hashing, group-id packing.

Three primitives, each one line of vLLM source:

    BlockHash             ↔ kv_cache_utils.py:L40
    BlockHashWithGroupId  ↔ kv_cache_utils.py:L45 (BlockHash + 4-byte group)
    hash_block_tokens     ↔ kv_cache_utils.py:L539-L566 (parent-chained)
    NONE_HASH             ↔ kv_cache_utils.py:L91-L110 (sentinel for first block)

THE invariant of parent-chained hashing (kv_cache_utils.py:L539-L566):

    H_0 = hash(NONE_HASH,    tokens[0..B])
    H_1 = hash(H_0,          tokens[B..2B])
    H_2 = hash(H_1,          tokens[2B..3B])
    ...
    H_k = hash(H_{k-1},      tokens[kB..(k+1)B])

If H_k matches across two requests, then `tokens[0..(k+1)B]` is byte-identical
across them. Why? Because H_k transitively depends on every token in the
preceding chain. This is what makes `find_longest_cache_hit` a SCAN-AND-STOP
loop (`single_type_kv_cache_manager.py:L473-L483`): the first miss guarantees
no later block can be a hit.

Group-id packing (kv_cache_utils.py:L53-L62) appends a 4-byte big-endian
group id, so the same token content under different KV cache groups (e.g.
sliding-window vs full-attention layers) gets distinct cache entries.
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, NewType, Sequence


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L40
BlockHash = NewType("BlockHash", bytes)

# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L45
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L83-L110
# vLLM uses os.urandom(32) when PYTHONHASHSEED is not set, hashing the seed
# otherwise. We follow the same pattern. Re-init `NONE_HASH` per process.
def init_none_hash() -> BlockHash:
    """Seed the chain. Returns the hash assigned to the (virtual) parent of
    block 0. Same logic as `kv_cache_utils.py:init_none_hash`."""
    seed = os.getenv("PYTHONHASHSEED")
    if seed is None:
        return BlockHash(os.urandom(32))
    return BlockHash(hashlib.sha256(seed.encode()).digest())


NONE_HASH: BlockHash = init_none_hash()


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L53-L62
def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack `(BlockHash, group_id)` into a single bytes blob.

    Same big-endian 4-byte appendix as vLLM. The packed form is what lives in
    `BlockPool.cached_block_hash_to_block`.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L65-L67
def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    return BlockHash(key[:-4])


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L70-L72
def get_group_id(key: BlockHashWithGroupId) -> int:
    return int.from_bytes(key[-4:], "big", signed=False)


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L539-L566
def hash_block_tokens(
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple = (),
) -> BlockHash:
    """One block's chained hash.

    Inputs match vLLM's signature (minus the `hash_function` first arg —
    we hard-code sha256 since the policy is one-line).

    The hashed payload is the 3-tuple `(parent_block_hash, tuple(token_ids),
    extra_keys)`, same as vLLM line L564-L565. We serialize via repr() —
    vLLM uses sha256_cbor by default — but the only PROPERTY we depend on is
    deterministic injection of the parent hash.
    """
    if not parent_block_hash:
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L560-L561
        parent_block_hash = NONE_HASH

    # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L563-L565
    # ((parent_block_hash, curr_block_token_ids_tuple, extra_keys) tuple is
    #  what vLLM hashes; we serialize via repr() instead of CBOR but the
    #  injection ordering is identical)
    payload = repr(
        (bytes(parent_block_hash), tuple(curr_block_token_ids), extra_keys)
    ).encode()
    return BlockHash(hashlib.sha256(payload).digest())


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L501-L536
# (generate_block_hash_extra_keys composes mm/lora/cache_salt/prompt_embeds keys)
# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L635-L686
# (get_request_block_hasher binds extra_keys + hash_function once per request)
def chain_block_hashes(
    token_ids: Sequence[int],
    block_size: int,
    extra_keys: tuple = (),
) -> list[BlockHash]:
    """Compute the entire hash chain for a token sequence.

    REFERENCE: vLLM does this lazily in `Request.update_block_hashes()`
    (request.py) and stores the result in `request.block_hashes`. We expose
    it as a pure function for testability.

    Partial blocks (the trailing < block_size remainder) are NOT hashed —
    you can only cache FULL blocks. This matches vLLM's
    `cache_full_blocks` semantics in `block_pool.py:L211-L320`.
    """
    hashes: list[BlockHash] = []
    parent: BlockHash | None = None
    for i in range(0, len(token_ids), block_size):
        block = token_ids[i : i + block_size]
        if len(block) < block_size:
            break
        h = hash_block_tokens(parent, block, extra_keys)
        hashes.append(h)
        parent = h
    return hashes


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L373-L390
# (need_extra_keys checks whether mm_features / lora / cache_salt are present)
def common_prefix_length(
    chain_a: Iterable[BlockHash], chain_b: Iterable[BlockHash]
) -> int:
    """How many leading hashes match?

    By the chained-hashing invariant, this is also the number of common
    leading blocks (block_size tokens each). Used in the prefix-cache
    benchmarks; not in vLLM's hot path (the cache-hash table makes it
    unnecessary).
    """
    n = 0
    for a, b in zip(chain_a, chain_b):
        if a != b:
            break
        n += 1
    return n
