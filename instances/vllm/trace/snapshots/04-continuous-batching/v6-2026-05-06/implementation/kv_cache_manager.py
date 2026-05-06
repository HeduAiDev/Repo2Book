# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py
"""SimpleKVCacheManager — a pedagogical stand-in for KVCacheManager.

vLLM's KVCacheManager is a 600+ line orchestrator over BlockPool, prefix-cache
hash tables, multi-group attention specs, and copy-on-write for shared blocks.
The scheduler only calls four things on it during the core loop:

    new_step_starts()                         # housekeeping, no-op for us
    allocate_slots(request, num_new_tokens)   # returns block list or None
    free(request)                             # returns blocks to pool
    get_computed_blocks(request)              # prefix-cache hit lookup

We model only `allocate_slots` and `free`, deferring prefix caching to Ch12-13.
This keeps the scheduler the focal point of Ch04 while staying honest about
what is being abstracted.
"""

from __future__ import annotations

from .request import Request


# REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py
class SimpleKVCacheManager:
    """Free-list block allocator. Same external contract as KVCacheManager."""

    def __init__(self, num_gpu_blocks: int, block_size: int = 16) -> None:
        # block_size matches vLLM's default 16 (config.cache_config.block_size).
        # REFERENCE: instances/vllm/source/vllm/v1/core/kv_cache_manager.py
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self._free_blocks: list[int] = list(range(num_gpu_blocks))

    def new_step_starts(self) -> None:
        """No-op. vLLM uses this to reset per-step prefix-cache stats."""

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L466-L471
    # (the scheduler's call site) and kv_cache_manager.py:allocate_slots
    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> list[int] | None:
        """Round up to whole blocks; return new block IDs or None if OOM.

        SIMPLIFIED: the real `allocate_slots` returns a `KVCacheBlocks` object
        and takes additional kwargs (num_new_computed_tokens, new_computed_blocks,
        num_lookahead_tokens, num_external_computed_tokens, delay_cache_blocks,
        num_encoder_tokens, full_sequence_must_fit) — every one of which is
        either prefix-cache or speculative-decode plumbing.
        """
        blocks_needed = (num_new_tokens + self.block_size - 1) // self.block_size
        if blocks_needed > len(self._free_blocks):
            return None
        allocated = self._free_blocks[:blocks_needed]
        del self._free_blocks[:blocks_needed]
        return allocated

    # REFERENCE: instances/vllm/source/vllm/v1/core/sched/scheduler.py:L961
    # (caller) and kv_cache_manager.py:free
    def free(self, request: Request) -> None:
        """Return all of the request's blocks to the pool."""
        self._free_blocks.extend(request.block_ids)
        request.block_ids = []

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_gpu_blocks - len(self._free_blocks)
