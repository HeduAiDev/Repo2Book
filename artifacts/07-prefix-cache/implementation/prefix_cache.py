"""
Automatic Prefix Caching (APC) — Our Reimplementation.

REFERENCE sources:
    hash_block_tokens():         vllm/v1/core/kv_cache_utils.py:L539
    BlockHashToBlockMap:         vllm/v1/core/block_pool.py:L34
    BlockPool.cache_full_blocks: vllm/v1/core/block_pool.py:L211
    BlockPool.get_cached_block:  vllm/v1/core/block_pool.py:L184
    FullAttentionManager.find_longest_cache_hit:
                                 vllm/v1/core/single_type_kv_cache_manager.py:L448
    NONE_HASH:                   vllm/v1/core/kv_cache_utils.py:L84
    get_request_block_hasher:    vllm/v1/core/kv_cache_utils.py:L636
"""

import hashlib
import os
from typing import Optional, List, Dict, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Chained Hash System
# REFERENCE: vllm/v1/core/kv_cache_utils.py:L539-L566
# ═══════════════════════════════════════════════════════════════════════════

class ChainedBlockHasher:
    """
    Computes chained block hashes — each hash depends on parent hash.

    REFERENCE: vllm/v1/core/kv_cache_utils.py:L539 — hash_block_tokens()
               "hash( (parent_block_hash or NONE_HASH, tuple(token_ids), extra_keys) )"

    The chain property:
        H_0 = hash(NONE_HASH, tokens[0:16])
        H_1 = hash(H_0, tokens[16:32])
        H_2 = hash(H_1, tokens[32:48])

    KEY PROPERTY: If H_k matches for two sequences, then tokens[0:k*16] must
    be identical. Why? Because H_k depends on H_{k-1}, which depends on
    H_{k-2}, ... which depends on NONE_HASH + tokens[0:16].

    This means a cache miss at position K guarantees no later position can
    be a cache hit. The left-to-right scan in find_longest_cache_hit
    exploits this: stop at the first miss.
    """

    def __init__(self, block_size: int = 16, extra_keys: tuple = ()):
        self.block_size = block_size
        self.extra_keys = extra_keys

    def compute_hashes(self, token_ids: List[int]) -> List[bytes]:
        """
        Compute chained hashes for all complete blocks.

        Returns one hash per complete block.
        """
        hashes = []
        prev_hash = None  # First block uses NONE_HASH-equivalent
        for i in range(0, len(token_ids), self.block_size):
            block = token_ids[i:i + self.block_size]
            if len(block) < self.block_size:
                break  # Partial block — don't hash
            h = self._hash_block(prev_hash, block)
            hashes.append(h)
            prev_hash = h
        return hashes

    def _hash_block(self, parent_hash: Optional[bytes], tokens: List[int]) -> bytes:
        """REFERENCE: kv_cache_utils.py:L539 — hash_block_tokens()"""
        h = hashlib.sha256()
        # Parent hash (or sentinel for first block)
        h.update(parent_hash if parent_hash else b'\x00' * 32)
        # Token IDs as bytes
        h.update(bytes(tokens))
        # Extra keys (multi-modal, LoRA, cache salt)
        for key in self.extra_keys:
            h.update(str(key).encode())
        return h.digest()

    @staticmethod
    def make_hash_with_group_id(block_hash: bytes, group_id: int) -> bytes:
        """
        Append 4-byte group ID to block hash.

        REFERENCE: kv_cache_utils.py:L53 — make_block_hash_with_group_id()
        This allows the same token content to have different cache entries
        for different attention layer groups.
        """
        return block_hash + group_id.to_bytes(4, 'big')


# ═══════════════════════════════════════════════════════════════════════════
# BlockHashToBlockMap
# REFERENCE: vllm/v1/core/block_pool.py:L34-L127
# ═══════════════════════════════════════════════════════════════════════════

class PrefixCacheIndex:
    """
    Hash-indexed prefix cache storage.

    REFERENCE: vllm/v1/core/block_pool.py:L34 — BlockHashToBlockMap

    Key design decisions from vLLM:
    1. NO DEDUPLICATION — two blocks with identical content can coexist
       with different physical block_ids. This keeps block tables append-only.
    2. Collision handling: single block → dict[block_id: block] on collision
    3. get_one_block() returns ANY block for a hash — the caller just needs
       *a* cached block, not the specific one.
    """

    def __init__(self):
        self._cache: Dict[bytes, int | Dict[int, int]] = {}
        # Maps hash → block_id (single) or {block_id: block_id} (collision)
        self._block_hashes: Dict[int, bytes] = {}  # block_id → hash

    def insert(self, block_hash: bytes, block_id: int):
        """Store block in cache index. REFERENCE: block_pool.py:L75"""
        self._block_hashes[block_id] = block_hash
        existing = self._cache.get(block_hash)
        if existing is None:
            self._cache[block_hash] = block_id
        elif isinstance(existing, int):
            # Collision: upgrade to dict
            self._cache[block_hash] = {existing: existing, block_id: block_id}
        else:
            existing[block_id] = block_id

    def get(self, block_hash: bytes) -> Optional[int]:
        """Return ANY block_id for this hash. REFERENCE: block_pool.py:L62"""
        entry = self._cache.get(block_hash)
        if entry is None:
            return None
        if isinstance(entry, int):
            return entry
        return next(iter(entry.values()))  # Return any block

    def remove(self, block_hash: bytes, block_id: int):
        """Remove specific block from cache. REFERENCE: block_pool.py:L93"""
        entry = self._cache.get(block_hash)
        if isinstance(entry, dict):
            entry.pop(block_id, None)
            if len(entry) == 1:
                remaining = next(iter(entry.values()))
                self._cache[block_hash] = remaining
            elif len(entry) == 0:
                del self._cache[block_hash]
        elif entry == block_id:
            del self._cache[block_hash]
        self._block_hashes.pop(block_id, None)

    def __contains__(self, block_hash: bytes) -> bool:
        return block_hash in self._cache


# ═══════════════════════════════════════════════════════════════════════════
# Prefix Cache Manager
# ═══════════════════════════════════════════════════════════════════════════

class PrefixCacheManager:
    """
    Manages prefix cache operations over BlockPool-like storage.

    REFERENCE:
        find_longest_cache_hit:  single_type_kv_cache_manager.py:L448
        cache_blocks:            single_type_kv_cache_manager.py:L277
        allocate_new_computed_blocks: single_type_kv_cache_manager.py:L169
    """

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.index = PrefixCacheIndex()
        # REFERENCE: single_type_kv_cache_manager.py:L73, L79
        self.req_to_blocks: Dict[str, List[int]] = {}   # request → [block_ids]
        self.req_cached_count: Dict[str, int] = {}       # how many blocks cached per request

    def find_longest_cache_hit(self, request_id: str,
                                block_hashes: List[bytes]) -> Tuple[List[int], int]:
        """
        Find longest contiguous prefix of cached blocks.

        REFERENCE: single_type_kv_cache_manager.py:L448-L494

        Left-to-right scan. Stops at first miss.
        Because of chained hashing, a miss at K means H_{K+1} would include
        different parent data → can't match either.
        """
        hits = []
        for h in block_hashes:
            bid = self.index.get(h)
            if bid is None:
                break
            hits.append(bid)
        return hits, len(hits) * self.block_size

    def cache_blocks(self, request_id: str, num_computed_tokens: int,
                     block_hashes: List[bytes]):
        """
        Mark full blocks as cached.

        REFERENCE: single_type_kv_cache_manager.py:L277
                   block_pool.py:L211 — BlockPool.cache_full_blocks()
        """
        num_cached = self.req_cached_count.get(request_id, 0)
        num_full = num_computed_tokens // self.block_size
        if num_cached >= num_full:
            return

        blocks = self.req_to_blocks.get(request_id, [])
        for i in range(num_cached, num_full):
            if i < len(blocks) and i < len(block_hashes):
                self.index.insert(block_hashes[i], blocks[i])

        self.req_cached_count[request_id] = num_full

    def touch(self, block_ids: List[int]):
        """
        Prevent eviction of blocks hit by another request.

        REFERENCE: block_pool.py:L391 — BlockPool.touch()
        In production vLLM, this increments ref_cnt and removes from free queue.
        """
        pass  # Our simplified version tracks via req_to_blocks only

    def register_request(self, request_id: str, block_ids: List[int]):
        self.req_to_blocks[request_id] = block_ids
        self.req_cached_count[request_id] = 0

    def free_request(self, request_id: str):
        """Free all blocks registered to a request."""
        self.req_to_blocks.pop(request_id, None)
        self.req_cached_count.pop(request_id, None)


# ═══════════════════════════════════════════════════════════════════════════
# Demonstration
# ═══════════════════════════════════════════════════════════════════════════

def demonstrate_prefix_cache():
    """Show how two requests with shared prompt share KV cache blocks."""
    hasher = ChainedBlockHasher(block_size=4)
    cache = PrefixCacheManager(block_size=4)

    # System prompt shared by both requests
    system_prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens → 2 blocks

    # Request A: system prompt + "what is AI?"
    req_a = system_prompt + [10, 20, 30, 40]  # 12 tokens → 3 blocks
    hashes_a = hasher.compute_hashes(req_a)
    print(f"Request A hashes: {[h[:4].hex() for h in hashes_a]}")

    # Simulate block allocation (simplified)
    cache.register_request("A", [0, 1, 2])
    cache.cache_blocks("A", num_computed_tokens=12, block_hashes=hashes_a)

    # Request B: same system prompt + "what is ML?"
    req_b = system_prompt + [10, 20, 30, 50]  # Different follow-up, same prefix
    hashes_b = hasher.compute_hashes(req_b)
    print(f"Request B hashes: {[h[:4].hex() for h in hashes_b]}")

    hits, num_tokens = cache.find_longest_cache_hit("B", hashes_b)
    print(f"Cache hits for B: {len(hits)} blocks = {num_tokens} tokens")
    print(f"  Block 0 (tokens 0-3): {'HIT' if len(hits) > 0 else 'MISS'}")
    print(f"  Block 1 (tokens 4-7): {'HIT' if len(hits) > 1 else 'MISS'}")
    print(f"  Block 2 (tokens 8-11): {'HIT' if len(hits) > 2 else 'MISS'} — different 4th token!")
    print(f"  Savings: {num_tokens} tokens reused (system prompt shared)")

    return hits, num_tokens


if __name__ == "__main__":
    demonstrate_prefix_cache()
