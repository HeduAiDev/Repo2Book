# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPUOffloadingManager：单层 CPU 池账本（准入/驱逐/ref_cnt/事件）。"""

from collections import OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory


# SOURCE: vllm/v1/kv_offload/cpu/manager.py:L30-L327
class CPUOffloadingManager(OffloadingManager):
    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L31-L40
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L42-L73
    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L51-L73
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- block pool ---

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L77-L78
    def _get_num_free_blocks(self) -> int:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L77-L78
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L80-L94
    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L81-L94
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L96-L97
    def _free_block(self, block: BlockStatus) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L96-L97
        self._free_list.append(block.block_id)

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L99-L104
    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L99-L104
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L108-L110
    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L109-L110
        return RequestOffloadingContext()

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L112-L127
    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L113-L127
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L129-L146
    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L135-L146
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L148-L150
    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L149-L150
        self._policy.touch(keys, req_context)

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L152-L163
    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L155-L163
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L165-L236
    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L170-L174
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L176-L183
        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        # SUBTRACTED: allocation_sizes_in_current_batch 的直方图记录
        #   （减法计划删除项 7：观测旁路）。
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L185-L186
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L188-L209
        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L211-L218
        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L220-L236
        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L238-L271
    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L245-L271
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)

        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L264-L271
        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L273-L285
    @override
    def reset_cache(self) -> None:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L274-L285
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        self._free_list.clear()
        self._num_allocated_blocks = 0

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L287-L291
    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L288-L291
        if self.events is not None:
            yield from self.events
            self.events.clear()

    # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L293-L327
    def get_stats(self):
        # SOURCE: vllm/v1/kv_offload/cpu/manager.py:L293-L327
        # SUBTRACTED: 使用率/直方图/计数器记录链——减法计划删除项 7
        #   （保留契约签名与默认 None 返回）。
        """Return collected metrics since last call, or None if disabled."""
        return None
