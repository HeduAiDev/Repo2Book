# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV 事件标准流：BlockStored/BlockRemoved（m15 事件面的出口类型）。

# SOURCE: vllm/distributed/kv_events.py:L30-L242
# SUBTRACTED: EventBatch 发布器族（EventPublisher/ZMQ/Kafka，L245 起）与
#   AllBlocksCleared——本章消费面是「事件对象 + 聚合器」。
"""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import msgspec

from vllm.v1.core.kv_cache_utils import ExternalBlockHash


# SOURCE: vllm/distributed/kv_events.py:L20-L32
class EventBatch(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all event batches"""

    ts: float
    events: list[Any]
    data_parallel_rank: int | None = None


# SOURCE: vllm/distributed/kv_events.py:L36-L42
class KVCacheEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


# SOURCE: vllm/distributed/kv_events.py:L45-L47
MEDIUM_GPU = "GPU"
MEDIUM_CPU = "CPU"
MEDIUM_STORAGE = "STORAGE"


# SOURCE: vllm/distributed/kv_events.py:L50-L94
class BlockStored(KVCacheEvent):
    # SOURCE: vllm/distributed/kv_events.py:L50-L94
    """KV cache events for storing blocks"""

    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int

    lora_id: int | None
    """Deprecated: use `lora_name` for KV block key hash.
    Retained for backward compatibility.
    """

    medium: str | None
    lora_name: str | None

    extra_keys: list[tuple[Any, ...] | None] | None = None
    """Extra keys used in block hash computation, one entry per block in
    block_hashes."""

    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None
    """LOCAL or REMOTE relative to the publisher; None means unspecified."""

    # SOURCE: vllm/distributed/kv_events.py:L79-L94
    def __hash__(self) -> int:
        # SOURCE: vllm/distributed/kv_events.py:L79-L94
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
                tuple(self.extra_keys) if self.extra_keys else None,
                self.group_idx,
                self.kv_cache_spec_kind,
                self.kv_cache_spec_sliding_window,
                self.locality,
            )
        )


# SOURCE: vllm/distributed/kv_events.py:L97-L112
class BlockRemoved(KVCacheEvent):
    # SOURCE: vllm/distributed/kv_events.py:L97-L112
    """KV cache events for removing blocks"""

    block_hashes: list[ExternalBlockHash]
    medium: str | None
    group_idx: int | None = None
    locality: str | None = None
    """LOCAL or REMOTE relative to the publisher; None means unspecified."""

    # SOURCE: vllm/distributed/kv_events.py:L104-L112
    def __hash__(self) -> int:
        # SOURCE: vllm/distributed/kv_events.py:L104-L112
        return hash(
            (
                tuple(self.block_hashes),
                self.medium,
                self.group_idx,
                self.locality,
            )
        )


# SOURCE: vllm/distributed/kv_events.py:L123-L207
class KVEventAggregator:
    # SOURCE: vllm/distributed/kv_events.py:L123-L136
    """Aggregates KV events across multiple workers.

    Tracks how many times each event appears and returns only those
    that were emitted by all workers.
    """

    __slots__ = ("_event_counter", "_num_workers")

    # SOURCE: vllm/distributed/kv_events.py:L132-L136
    def __init__(self, num_workers: int) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L132-L136
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers: int = num_workers

    # SOURCE: vllm/distributed/kv_events.py:L138-L147
    def add_events(self, events: list[KVCacheEvent]) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L138-L147
        """Add events from a worker batch."""
        if not isinstance(events, list):
            raise TypeError("events must be a list of KVCacheEvent.")
        self._event_counter.update(events)

    # SOURCE: vllm/distributed/kv_events.py:L149-L160
    def get_common_events(self) -> list[KVCacheEvent]:
        # SOURCE: vllm/distributed/kv_events.py:L149-L160
        """Return events that appeared in all workers."""
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers
        ]

    # SOURCE: vllm/distributed/kv_events.py:L162-L169
    def get_all_events(self) -> list[KVCacheEvent]:
        # SOURCE: vllm/distributed/kv_events.py:L162-L169
        """Return all events for all workers."""
        return list(self._event_counter.elements())

    # SOURCE: vllm/distributed/kv_events.py:L171-L175
    def clear_events(self) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L171-L175
        """Clear the tracked events."""
        self._event_counter.clear()

    # SOURCE: vllm/distributed/kv_events.py:L177-L186
    def increment_workers(self, count: int = 1) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L177-L186
        """Increment the number of workers contributing events."""
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    # SOURCE: vllm/distributed/kv_events.py:L188-L192
    def reset_workers(self) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L188-L192
        """Reset the number of workers to 1."""
        self._num_workers = 1

    # SOURCE: vllm/distributed/kv_events.py:L194-L201
    def get_number_of_workers(self) -> int:
        # SOURCE: vllm/distributed/kv_events.py:L194-L201
        """Return the number of workers."""
        return self._num_workers

    # SOURCE: vllm/distributed/kv_events.py:L203-L207
    def __repr__(self) -> str:
        # SOURCE: vllm/distributed/kv_events.py:L203-L207
        return (
            f"<KVEventAggregator workers={self._num_workers}, "
            f"events={len(self._event_counter)}>"
        )


# SOURCE: vllm/distributed/kv_events.py:L210-L242
class KVConnectorKVEvents(ABC):
    # SOURCE: vllm/distributed/kv_events.py:L210-L215
    """Abstract base class for KV events.

    Acts as a container for KV events from the connector.
    """

    # SOURCE: vllm/distributed/kv_events.py:L216-L218
    @abstractmethod
    def add_events(self, events: list[KVCacheEvent]) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L216-L218
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L220-L222
    @abstractmethod
    def aggregate(self) -> "KVConnectorKVEvents":
        # SOURCE: vllm/distributed/kv_events.py:L220-L222
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L224-L226
    @abstractmethod
    def increment_workers(self, count: int = 1) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L224-L226
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L228-L230
    @abstractmethod
    def get_all_events(self) -> list[KVCacheEvent]:
        # SOURCE: vllm/distributed/kv_events.py:L228-L230
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L232-L234
    @abstractmethod
    def get_number_of_workers(self) -> int:
        # SOURCE: vllm/distributed/kv_events.py:L232-L234
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L236-L238
    @abstractmethod
    def clear_events(self) -> None:
        # SOURCE: vllm/distributed/kv_events.py:L236-L238
        raise NotImplementedError

    # SOURCE: vllm/distributed/kv_events.py:L240-L242
    def merge(self, other: "KVConnectorKVEvents") -> "KVConnectorKVEvents":
        # SOURCE: vllm/distributed/kv_events.py:L240-L242
        self.add_events(other.get_all_events())
        return self


__all__ = [
    "EventBatch",
    "KVCacheEvent",
    "BlockStored",
    "BlockRemoved",
    "MEDIUM_GPU",
    "MEDIUM_CPU",
    "MEDIUM_STORAGE",
    "KVEventAggregator",
    "KVConnectorKVEvents",
]
