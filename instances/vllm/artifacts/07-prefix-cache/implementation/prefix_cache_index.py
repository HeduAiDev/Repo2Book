# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127
# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L184-L209
"""BlockHashToBlockMap — vLLM's prefix-cache index.

THIS IS A FLAT HASH TABLE. Not a tree. Not a trie. Not a radix tree.

vLLM stores prefix cache entries in a dict (`block_pool.py:L34-L127`):

    cached_block_hash_to_block: dict[BlockHashWithGroupId, KVCacheBlock | dict]

The chained hash (Ch07.block_hash) gives the dict its key — that's where the
"radix-tree feel" comes from. Two requests with the same prefix produce the
same hash chain; both `H_0` keys collide in the dict, so the second request
finds the first request's blocks via `get_cached_block(H_0)` lookup.

Why this isn't a real radix tree:
    - No path compression: each block is a separate dict entry, not a node.
    - No tree traversal: `find_longest_cache_hit` is a linear scan-and-stop
      over the chain, not a tree walk.
    - O(1) per-block lookup, O(L/B) for a length-L sequence — hash-map
      complexity, not tree complexity.

Why vLLM picked this:
    - Simpler invariant. A dict's get/set/del are atomic; a tree's
      restructuring is not.
    - Cache locality. A dict bucket fits in one cache line; a tree node
      walks pointers.
    - The chained hash already gives us the "tree property" (a miss at level
      k implies miss at level k+1) without paying for an actual tree.

A TRUE radix tree (see ch07.radix_tree) is what you'd build if you wanted to
*enumerate* all cached prefixes (e.g. for visualization). vLLM never enumerates
— it only point-queries — so the hash table dominates.
"""

from __future__ import annotations

from typing import Any

from .block_hash import BlockHash, BlockHashWithGroupId, make_block_hash_with_group_id


# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L34-L127
class BlockHashToBlockMap:
    """Cache of `BlockHashWithGroupId -> KVCacheBlock | dict[block_id, KVCacheBlock]`.

    `KVCacheBlock` here is opaque — we represent it as `int` (block_id) for
    Ch07 since the block-pool internals are Ch05's territory.

    The dict-of-blocks branch (vLLM:L83-L86) handles the rare case of
    multiple physical blocks sharing one hash key. WHY would they? Because
    vLLM does NOT deduplicate at insert time (`block_pool.py:L48-L52`):
    "we don't check if there is already an identical block in the cache."
    That keeps block tables append-only — once a request has its block_id,
    the id never changes underneath it.
    """

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L57-L60
    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L48-L52
    # (NOTE on no-deduplication; explains why the value can be a dict of
    #  multiple blocks under one hash key)
    def __init__(self) -> None:
        # value is `int` (single block) or `dict[int, int]` (collision)
        self._cache: dict[BlockHashWithGroupId, int | dict[int, int]] = {}

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L62-L73
    def get_one_block(self, key: BlockHashWithGroupId) -> int | None:
        """Return any block_id matching `key`, or None."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if isinstance(entry, int):
            return entry
        if isinstance(entry, dict):
            return next(iter(entry.values()))
        raise AssertionError(f"Invalid entry type {type(entry)}")

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L75-L91
    def insert(self, key: BlockHashWithGroupId, block_id: int) -> None:
        """Insert `block_id` under `key`.

        Three branches:
        - Empty bucket: store the int directly.
        - Single int: upgrade to dict-of-two.
        - Dict: insert into the dict.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._cache[key] = block_id
        elif isinstance(entry, int):
            self._cache[key] = {entry: entry, block_id: block_id}
        elif isinstance(entry, dict):
            entry[block_id] = block_id
        else:
            raise AssertionError(f"Invalid entry type {type(entry)}")

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L93-L121
    def pop(self, key: BlockHashWithGroupId, block_id: int) -> int | None:
        """Remove `block_id` from `key`'s bucket.

        Returns the removed block_id, or None if not present. Mirrors vLLM's
        TODO at L101-L105: when a key is found, block_id should always
        match; vLLM keeps the original safety branch and we follow.
        """
        entry = self._cache.pop(key, None)
        if entry is None:
            return None
        if isinstance(entry, int):
            if entry == block_id:
                return entry
            self._cache[key] = entry  # mismatch — restore
            return None
        if isinstance(entry, dict):
            removed = entry.pop(block_id, None)
            if entry:
                self._cache[key] = entry
            return removed
        raise AssertionError(f"Invalid entry type {type(entry)}")

    # REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L123-L124
    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: BlockHashWithGroupId) -> bool:
        return key in self._cache


# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L184-L209
# REFERENCE: instances/vllm/source/vllm/v1/core/single_type_kv_cache_manager.py:L477-L483
# (FullAttentionManager.find_longest_cache_hit's call site for get_cached_block)
def get_cached_block(
    cache: BlockHashToBlockMap,
    block_hash: BlockHash,
    kv_cache_group_ids: list[int],
) -> list[int] | None:
    """Look up `block_hash` for every group in `kv_cache_group_ids`.

    Returns `None` if ANY group misses. This is the "common cache hit
    across all groups" semantic — it matters when a model has hybrid
    attention layers (full + sliding window) sharing a request.

    REFERENCE for the loop pattern: `block_pool.py:L196-L208`.
    """
    cached: list[int] = []
    for group_id in kv_cache_group_ids:
        packed = make_block_hash_with_group_id(block_hash, group_id)
        block_id = cache.get_one_block(packed)
        if block_id is None:
            return None
        cached.append(block_id)
    return cached
