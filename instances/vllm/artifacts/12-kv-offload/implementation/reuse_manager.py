# SPDX-License-Identifier: Apache-2.0
"""
FilterReusedOffloadingManager — store_threshold filter wrapper.

REFERENCE: vllm/v1/kv_offload/reuse_manager.py:L23-L120

Why this exists (HARD GATE design decision 6):
  Not every block deserves to be offloaded. A block that appears once
  is unlikely to be hit again (the "cold long-tail"). Offloading it
  wastes CPU pinned memory and PCIe bandwidth.

  This decorator counts how often each key appears in `lookup()`. Only
  keys that have been seen `>= store_threshold` times are eligible
  for `prepare_store`. The counter LRU is bounded at `max_tracker_size`
  to prevent it from leaking unbounded.

The decorator pattern (wrapping any OffloadingManager) is what lets
us "tap into" the lifecycle without subclassing every manager. Most
of the methods are pure delegation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Optional

from .offload_manager import OffloadingManager
from .offload_spec import (
    LoadStoreSpec,
    OffloadKey,
    OffloadingEvent,
    PrepareStoreOutput,
    ReqContext,
)


class FilterReusedOffloadingManager(OffloadingManager):
    """Decorator: filters out blocks below `store_threshold` reuse count
    before delegating to the backing manager.

    REFERENCE: vllm/v1/kv_offload/reuse_manager.py:L44-L97
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        if store_threshold < 2:
            raise ValueError(
                f"store_threshold must be >= 2, got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                f"max_tracker_size must be >= 1, got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        # REFERENCE: vllm/v1/kv_offload/reuse_manager.py:L62-L63
        self.counts: OrderedDict[OffloadKey, int] = OrderedDict()

    # --- intercepted methods ---

    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> Optional[bool]:
        # REFERENCE: vllm/v1/kv_offload/reuse_manager.py:L70-L79
        if key in self.counts:
            self.counts.move_to_end(key)
            self.counts[key] += 1
        else:
            if len(self.counts) >= self.max_tracker_size:
                self.counts.popitem(last=False)  # evict LRU
            self.counts[key] = 1
        return self._backing.lookup(key, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> Optional[PrepareStoreOutput]:
        # REFERENCE: vllm/v1/kv_offload/reuse_manager.py:L81-L97
        eligible = [
            k for k in keys if self.counts.get(k, 0) >= self.store_threshold
        ]
        return self._backing.prepare_store(eligible, req_context)

    # --- delegated methods ---

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        return self._backing.prepare_load(keys, req_context)

    def touch(self, keys: Collection[OffloadKey]) -> None:
        return self._backing.touch(keys)

    def complete_load(self, keys: Collection[OffloadKey]) -> None:
        return self._backing.complete_load(keys)

    def complete_store(
        self, keys: Collection[OffloadKey], success: bool = True
    ) -> None:
        return self._backing.complete_store(keys, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
