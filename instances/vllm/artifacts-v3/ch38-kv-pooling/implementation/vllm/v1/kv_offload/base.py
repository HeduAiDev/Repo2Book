# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Core abstractions for KV cache offloading in vLLM v1.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar, NamedTuple, NewType, TypeVar

import numpy as np
import torch

from vllm.v1.kv_offload.config import OffloadingConfig

# `OffloadKey` identifies an offloaded block. It combines a block hash with
# its KV cache group index, encoded as raw bytes to avoid tuple GC overhead.
# Use the helper functions below to construct / decompose keys.
# SOURCE: vllm/v1/kv_offload/base.py:L23-L26
OffloadKey = NewType("OffloadKey", bytes)


# SOURCE: vllm/v1/kv_offload/base.py:L29-L31
def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    # SOURCE: vllm/v1/kv_offload/base.py:L29-L31
    """Pack a block hash and group index into an `OffloadKey`."""
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


# SOURCE: vllm/v1/kv_offload/base.py:L34-L36
def get_offload_block_hash(key: OffloadKey) -> bytes:
    # SOURCE: vllm/v1/kv_offload/base.py:L34-L36
    """Extract the block hash from an `OffloadKey`."""
    return key[:-4]


# SOURCE: vllm/v1/kv_offload/base.py:L39-L41
def get_offload_group_idx(key: OffloadKey) -> int:
    # SOURCE: vllm/v1/kv_offload/base.py:L39-L41
    """Extract the group index from an `OffloadKey`."""
    return int.from_bytes(key[-4:], "big", signed=False)


_T = TypeVar("_T")


# SOURCE: vllm/v1/kv_offload/base.py:L47-L51
class Medium(Enum):
    # SOURCE: vllm/v1/kv_offload/base.py:L47-L51
    """Storage medium of an offloading tier."""

    CPU = "CPU"
    STORAGE = "STORAGE"


# SOURCE: vllm/v1/kv_offload/base.py:L54-L58
class Locality(Enum):
    # SOURCE: vllm/v1/kv_offload/base.py:L54-L58
    """Locality of a tier's storage relative to the publishing instance."""

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


# SOURCE: vllm/v1/kv_offload/base.py:L61-L70
class TierMatcher(NamedTuple):
    # SOURCE: vllm/v1/kv_offload/base.py:L61-L70
    medium: Medium | None = None
    locality: Locality | None = None

    # SOURCE: vllm/v1/kv_offload/base.py:L65-L70
    def matches(self, medium: Medium | None, locality: Locality | None) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L66-L70
        medium_matches = self.medium is None or medium is None or self.medium == medium
        locality_matches = (
            self.locality is None or locality is None or self.locality == locality
        )
        return medium_matches and locality_matches


# SOURCE: vllm/v1/kv_offload/base.py:L73-L87
@dataclass(frozen=True)
class TierFilter:
    # SOURCE: vllm/v1/kv_offload/base.py:L74-L77
    """Per-request filter controlling which tiers participate."""

    matchers: tuple[TierMatcher, ...] = ()

    ALL: ClassVar["TierFilter"]

    # SOURCE: vllm/v1/kv_offload/base.py:L81-L84
    def allows(self, medium: Medium | None, locality: Locality | None) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L82-L84
        if self is TierFilter.ALL:
            return True
        return any(m.matches(medium, locality) for m in self.matchers)


# SOURCE: vllm/v1/kv_offload/base.py:L87
TierFilter.ALL = TierFilter(matchers=(TierMatcher(),))


# SOURCE: vllm/v1/kv_offload/base.py:L90-L104
@dataclass
class ReqContext:
    # SOURCE: vllm/v1/kv_offload/base.py:L90-L98
    req_id: str
    kv_transfer_params: dict[str, Any] | None = None
    load_tier_filter: TierFilter = TierFilter.ALL
    # Per-request scratch space keyed by value type, so a tier can parse
    # kv_transfer_params once (in on_new_request) and read the result back
    # on later calls for the same request.
    _state: dict[type, Any] = field(default_factory=dict, repr=False, init=False)

    # SOURCE: vllm/v1/kv_offload/base.py:L100-L101
    def set_state(self, val: Any) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L100-L101
        self._state[type(val)] = val

    # SOURCE: vllm/v1/kv_offload/base.py:L103-L104
    def get_state(self, cls: type[_T]) -> _T | None:
        # SOURCE: vllm/v1/kv_offload/base.py:L103-L104
        return self._state.get(cls)


# SOURCE: vllm/v1/kv_offload/base.py:L107-L113
class LookupResult(Enum):
    # SOURCE: vllm/v1/kv_offload/base.py:L107-L113
    """Result of OffloadingManager.lookup()."""

    MISS = auto()
    HIT = auto()
    HIT_PENDING = auto()
    RETRY = auto()


# SOURCE: vllm/v1/kv_offload/base.py:L116-L122
class OffloadPolicy(Enum):
    # SOURCE: vllm/v1/kv_offload/base.py:L116-L122
    # Offload only newly-computed blocks as they arrive; prefix-hit
    # blocks (already offloaded by a prior request) are skipped.
    BLOCK_LEVEL = "block_level"
    # Offload all blocks for the request, including prefix hits.
    # Used by tiers that need the complete KV context for a request.
    REQUEST_LEVEL = "request_level"


# SOURCE: vllm/v1/kv_offload/base.py:L125-L127
@dataclass
class RequestOffloadingContext:
    # SOURCE: vllm/v1/kv_offload/base.py:L125-L127
    policy: OffloadPolicy = OffloadPolicy.BLOCK_LEVEL


# SOURCE: vllm/v1/kv_offload/base.py:L130-L136
class ScheduleEndContext(NamedTuple):
    # SOURCE: vllm/v1/kv_offload/base.py:L130-L136
    """Per-step scheduling info passed to on_schedule_end()."""

    # Request IDs scheduled for the first time this step.
    new_req_ids: Collection[str]
    # Request IDs preempted this step.
    preempted_req_ids: Collection[str]


# SOURCE: vllm/v1/kv_offload/base.py:L139-L143
class LoadStoreSpec:
    # SOURCE: vllm/v1/kv_offload/base.py:L139-L143
    """
    Metadata that encapsulates information allowing a worker
    to load, and optionally also to store, blocks of KV data.
    """


# SOURCE: vllm/v1/kv_offload/base.py:L146-L150
@dataclass
class PrepareStoreOutput:
    # SOURCE: vllm/v1/kv_offload/base.py:L146-L150
    keys_to_store: list[OffloadKey]
    store_spec: LoadStoreSpec
    evicted_keys: list[OffloadKey]


# SOURCE: vllm/v1/kv_offload/base.py:L153-L159
@dataclass
class OffloadingEvent:
    # SOURCE: vllm/v1/kv_offload/base.py:L153-L159
    keys: list[OffloadKey]
    medium: Medium
    # True if blocks are removed, False if stored
    removed: bool
    locality: Locality | None = None


"""
OffloadingManager class for managing KV data offloading in vLLM v1

This class runs in the scheduler, tracks which blocks are offloaded
and their address.

The class provides the following primitives:
    lookup() - check whether a single block is offloaded and ready.
    prepare_load() - prepare given blocks to be read.
        The given blocks will be protected from eviction.
        This function returns a LoadSpec which encapsulates
        information required for performing the load.
    touch() - marks the give blocks as recently used. Can be used
        to track block's LRU. This function is separated from the
        prepare_load function to allow setting block recency even
        for blocks which do not need reading from the cache, such as
        blocks that are cached by the GPU prefix cache.
    complete_load() - mark blocks which were previously prepared to be
        loaded as done loading. This is to re-allow their eviction.
    prepare_store() - prepare the given blocks to be written.
        Returns a StoreSpec encapsulating offloading information,
        as well as a list of blocks that were evicted as a result.
    complete_store() - marks a previous store as completed.
        Following this call, the given blocks will become loadable.
"""
# SOURCE: vllm/v1/kv_offload/base.py:L162-L186（上文契约 docstring 原文）


# SOURCE: vllm/v1/kv_offload/base.py:L189-L207
@dataclass(frozen=True)
class OffloadingMetricMetadata:
    # SOURCE: vllm/v1/kv_offload/base.py:L190-L192
    documentation: str
    labelnames: tuple[str, ...] = ()


# SOURCE: vllm/v1/kv_offload/base.py:L195-L197
@dataclass(frozen=True)
class OffloadingCounterMetadata(OffloadingMetricMetadata):
    # SOURCE: vllm/v1/kv_offload/base.py:L195-L197
    pass


# SOURCE: vllm/v1/kv_offload/base.py:L200-L202
@dataclass(frozen=True)
class OffloadingGaugeMetadata(OffloadingMetricMetadata):
    # SOURCE: vllm/v1/kv_offload/base.py:L200-L202
    pass


# SOURCE: vllm/v1/kv_offload/base.py:L205-L207
@dataclass(frozen=True)
class OffloadingHistogramMetadata(OffloadingMetricMetadata):
    # SOURCE: vllm/v1/kv_offload/base.py:L205-L207
    buckets: tuple[float, ...] | None = None


# SOURCE: vllm/v1/kv_offload/base.py:L210-L217
@dataclass(frozen=True)
class OffloadingKVEventsConfig:
    # SOURCE: vllm/v1/kv_offload/base.py:L210-L217
    # Global vLLM KV event publishing flag. When false, connector-specific
    # event capture must stay inert because take_events() is not drained.
    enable_kv_cache_events: bool
    # OffloadingConnector opt-in for self-describing BlockStored payloads.
    # Effective only when enable_kv_cache_events is true.
    self_describing_kv_events: bool


# SOURCE: vllm/v1/kv_offload/base.py:L220-L394
class OffloadingManager(ABC):
    # SOURCE: vllm/v1/kv_offload/base.py:L221-L235
    @abstractmethod
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        # SOURCE: vllm/v1/kv_offload/base.py:L221-L235
        """
        Checks whether a single block is offloaded and ready to be read.

        Args:
            key: the key identifying the block to lookup.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            HIT if the block is offloaded and ready, MISS if not found,
            HIT_PENDING if found but not yet readable, or RETRY if the
            lookup should be retried later.
        """
        pass

    # SOURCE: vllm/v1/kv_offload/base.py:L237-L257
    @abstractmethod
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        # SOURCE: vllm/v1/kv_offload/base.py:L237-L257
        """
        Prepare the given blocks to be read.
        The given blocks will be protected from eviction until
        complete_load is called.
        It assumes all given blocks are offloaded.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A LoadStoreSpec that can be used by a worker to locate and load
            the actual offloaded KV data.
        """
        pass

    # SOURCE: vllm/v1/kv_offload/base.py:L259-L268
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        # SOURCE: vllm/v1/kv_offload/base.py:L259-L268
        """
        Mark the given blocks as recently used.
        This could in practice mean moving them to the end of an LRU list.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context.
        """
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L270-L278
    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        # SOURCE: vllm/v1/kv_offload/base.py:L270-L278
        """
        Marks previous blocks that were prepared to load as done loading.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context.
        """
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L280-L301
    @abstractmethod
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        # SOURCE: vllm/v1/kv_offload/base.py:L280-L301
        """
        Prepare the given blocks to be offloaded.
        The given blocks will be protected from eviction until
        complete_store is called.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context (e.g. kv_transfer_params).

        Returns:
            A PrepareStoreOutput indicating which blocks need storing,
            where to store them (LoadStoreSpec), and list of blocks that
            were evicted as a result.
            None is returned if the blocks cannot be stored.
        """
        pass

    # SOURCE: vllm/v1/kv_offload/base.py:L303-L320
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        # SOURCE: vllm/v1/kv_offload/base.py:L303-L320
        """
        Marks blocks which were previously prepared to be stored, as stored.
        Following this call, the blocks become loadable.
        If success is False, blocks that were not marked as stored will be
        removed.

        Args:
            keys: the keys identifying the blocks.
            req_context: per-request context.
            success: whether the blocks were stored successfully.
        """
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L322-L333
    @abstractmethod
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # SOURCE: vllm/v1/kv_offload/base.py:L322-L333
        """
        Called when a new request is first seen by the scheduler.

        Returns a RequestOffloadingContext indicating how this request's
        blocks should be offloaded.

        Args:
            req_context: per-request context.
        """
        pass

    # SOURCE: vllm/v1/kv_offload/base.py:L335-L353
    def on_request_finished(self, req_context: ReqContext) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L335-L353
        """
        Called when a request has finished.

        By the time this is called, the scheduler will issue no more
        submit-side calls for this request, such as prepare_store() and
        prepare_load(). Completion callbacks for already-submitted transfers
        (complete_store() and complete_load()) may still arrive afterward.

        This hook does NOT imply the data has been persisted. Asynchronous
        transfers already submitted for this request may still be in flight.
        Managers that cascade to lower tiers should delay those tiers'
        on_request_finished() calls until no more lower-tier submit calls can
        be issued for this request.

        Args:
            req_context: per-request context.
        """
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L355-L366
    def take_events(self) -> Iterable[OffloadingEvent]:
        # SOURCE: vllm/v1/kv_offload/base.py:L355-L366
        """
        Take the offloading events from the manager.

        A tier manager emits only events for storage state it owns. A
        composing manager may aggregate child event streams, but should not
        synthesize events on behalf of a child tier.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        return ()

    # SOURCE: vllm/v1/kv_offload/base.py:L368-L374
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L368-L374
        """Called once at the end of each scheduler step.

        Managers may override this to flush deferred work accumulated
        during the step (e.g., batched promotions).
        """
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L376-L382
    def has_pending_work(self) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L376-L382
        """Whether this manager needs the engine to keep stepping.

        While True, on_schedule_end() and get_finished_jobs() continue
        to be called even when no requests are scheduled.
        """
        return False

    # SOURCE: vllm/v1/kv_offload/base.py:L384-L386
    def reset_cache(self) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L384-L386
        """Evict all tracked blocks and reset internal state."""
        return

    # SOURCE: vllm/v1/kv_offload/base.py:L388-L390
    def get_stats(self) -> "Any | None":
        # SOURCE: vllm/v1/kv_offload/base.py:L388-L390
        # SUBTRACTED: OffloadingConnectorStats 遥测类型——减法计划删除项 7
        #   （保留契约签名与默认 None 返回）。
        """Return collected metrics since last call, or None if disabled."""
        return None

    # SOURCE: vllm/v1/kv_offload/base.py:L392-L394
    def shutdown(self) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L392-L394
        """Shutdown the manager and release any resources."""
        return


# SOURCE: vllm/v1/kv_offload/base.py:L397-L406
class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    # SOURCE: vllm/v1/kv_offload/base.py:L397-L401
    """
    Spec for loading/storing KV blocks from given block numbers.
    """

    # SOURCE: vllm/v1/kv_offload/base.py:L402-L406
    def __init__(self, block_ids: list[int]):
        # SOURCE: vllm/v1/kv_offload/base.py:L402-L406
        self.block_ids = np.array(block_ids, dtype=np.int64)

    # SOURCE: vllm/v1/kv_offload/base.py:L405-L406
    def __repr__(self) -> str:
        # SOURCE: vllm/v1/kv_offload/base.py:L405-L406
        return repr(self.block_ids)


# SOURCE: vllm/v1/kv_offload/base.py:L409-L440
class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    # SOURCE: vllm/v1/kv_offload/base.py:L409-L428
    """
    Spec for loading/storing a KV block to GPU memory.

    If there are multiple KV groups, the blocks are expected to be
    ordered by the group index.
    In that case, group_sizes[i] determines the number of blocks
    per the i-th KV group, and thus sum(group_sizes) == len(block_ids).
    group_sizes=None indicates a single KV group.

    If block_indices is given, each group (determined by group_sizes) of block IDs
    will correspond to logically contiguous blocks, e.g. blocks 5-10 of a some request.
    block_indices[i] will represent the block index of the first block in group #i.
    Thus, len(block_indices) == len(group_sizes) = number of KV cache groups.
    This information is required in order to support off/loading from offloaded blocks
    which are larger than GPU blocks.
    In such cases, the first GPU block per each group may be unaligned to the offloaded
    block size, and so knowing block_indices[i] allows the worker to correctly
    skip part of the first matching offloaded block.
    """

    # SOURCE: vllm/v1/kv_offload/base.py:L430-L440
    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
    ):
        # SOURCE: vllm/v1/kv_offload/base.py:L436-L440
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids)
        assert len(block_indices) == len(group_sizes)
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] = block_indices


# SOURCE: vllm/v1/kv_offload/base.py:L443-L457
@dataclass
class CanonicalKVCacheTensor:
    # SOURCE: vllm/v1/kv_offload/base.py:L443-L457
    """
    A canonicalized KV cache tensor whose first dimension is num_blocks.

    For attention backends where the raw tensor has num_blocks at a
    non-leading physical dimension (e.g. FlashAttention's
    (2, num_blocks, ...) layout), the tensor is split so that each
    resulting CanonicalKVCacheTensor starts with (num_blocks, ...).
    """

    # The KV cache tensor with shape (num_blocks, ...)
    tensor: torch.Tensor
    # The (possibly padded) page size per block in bytes
    page_size_bytes: int


# SOURCE: vllm/v1/kv_offload/base.py:L460-L499
@dataclass(frozen=True)
class CopyRun:
    # SOURCE: vllm/v1/kv_offload/base.py:L460-L472
    """A strided byte correspondence between this worker's physical page and
    a canonical page: for i in range(num_fragments), fragment i spans
    [local_offset + i * local_stride, +fragment_size) in the worker's page and
    [canonical_offset + i * canonical_stride, +fragment_size) canonically."""

    local_offset: int
    canonical_offset: int
    fragment_size: int
    num_fragments: int
    local_stride: int
    canonical_stride: int


# SOURCE: vllm/v1/kv_offload/base.py:L475-L499
@dataclass(frozen=True)
class CanonicalPageMapping:
    # SOURCE: vllm/v1/kv_offload/base.py:L475-L482
    """How this worker's page maps into a canonical (parallelism-free) page.
    In-process only, never serialized. Runs cover the full local page in both
    directions; ranks holding identical bytes take turns writing them.
    """

    # Size of the canonical page in bytes
    canonical_page_size_bytes: int
    # Size of this worker's (un-padded) page in bytes
    local_page_size_bytes: int
    # Byte correspondences between this worker's page and a canonical page
    runs: tuple[CopyRun, ...]
    # Number of ranks holding these exact bytes
    num_writers: int
    # This worker's index among those ranks
    writer_index: int
    # Canonical bytes identical under any parallel config with this block span
    parallelism_agnostic: bool

    # SOURCE: vllm/v1/kv_offload/base.py:L495-L499
    def is_writer(self, block_id: int) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L495-L499
        """Whether this worker stores the canonical page of the given block.
        Rotating by block spreads writes across ranks holding identical bytes.
        """
        return block_id % self.num_writers == self.writer_index


# SOURCE: vllm/v1/kv_offload/base.py:L502-L514
@dataclass
class CanonicalKVCacheRef:
    # SOURCE: vllm/v1/kv_offload/base.py:L502-L514
    """
    Per-layer (or group of layers) reference to a specific (by index)
    CanonicalKVCacheTensor and records the un-padded page size used by that layer.
    """

    # Index into the list of CanonicalKVCacheTensor objects
    tensor_idx: int
    # The un-padded page size per block in bytes
    page_size_bytes: int
    # How this worker's page maps into a canonical page; None = uncertified
    mapping: CanonicalPageMapping | None = None


# SOURCE: vllm/v1/kv_offload/base.py:L517-L534
@dataclass
class CanonicalKVCaches:
    # SOURCE: vllm/v1/kv_offload/base.py:L517-L534
    """
    Canonicalized block-level representation of the KV caches.

    Composed of:
        - Unique list of KV cache data tensors,
          each with shape (num_blocks, page_size_in_bytes) and int8 dtype.
        - Per-group data references of the tensors.
          i.e. how each KV cache group maps to the tensors.
    """

    # Ordered list of unique block tensors, each with shape
    # (num_blocks, ...).
    tensors: list[CanonicalKVCacheTensor]
    # Per-KV-cache-group list of data references that map each layer
    # in the group to the appropriate entry in the tensors list.
    group_data_refs: list[list[CanonicalKVCacheRef]]


# SOURCE: vllm/v1/kv_offload/base.py:L537-L542
@dataclass
class TransferResult:
    # SOURCE: vllm/v1/kv_offload/base.py:L537-L542
    job_id: int
    success: bool
    transfer_size: int | None = None
    transfer_time: float | None = None


# SOURCE: vllm/v1/kv_offload/base.py:L545-L569
class OffloadingWorker(ABC):
    # SOURCE: vllm/v1/kv_offload/base.py:L545-L548
    """Runs in the worker process. Performs async KV transfers for ONE
    offloaded medium (e.g. CPU). Direction is explicit via submit_store /
    submit_load, so there is no (src_medium, dst_medium) routing."""

    # SOURCE: vllm/v1/kv_offload/base.py:L550-L554
    @abstractmethod
    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L550-L554
        """Async GPU -> offloaded medium."""

    # SOURCE: vllm/v1/kv_offload/base.py:L556-L560
    @abstractmethod
    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        # SOURCE: vllm/v1/kv_offload/base.py:L556-L560
        """Async offloaded medium -> GPU."""

    # SOURCE: vllm/v1/kv_offload/base.py:L562-L563
    @abstractmethod
    def get_finished(self) -> list[TransferResult]:  # noqa: D102
        # SOURCE: vllm/v1/kv_offload/base.py:L562-L563
        ...

    # SOURCE: vllm/v1/kv_offload/base.py:L565-L566
    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:  # noqa: D102
        # SOURCE: vllm/v1/kv_offload/base.py:L565-L566
        ...

    # SOURCE: vllm/v1/kv_offload/base.py:L568-L569
    def shutdown(self) -> None:
        # SOURCE: vllm/v1/kv_offload/base.py:L568-L569
        return


# SOURCE: vllm/v1/kv_offload/base.py:L572-L625
class OffloadingSpec(ABC):
    # SOURCE: vllm/v1/kv_offload/base.py:L573
    """Spec for an offloading connector"""

    # SOURCE: vllm/v1/kv_offload/base.py:L575-L580
    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, "OffloadingMetricMetadata"]:
        # SOURCE: vllm/v1/kv_offload/base.py:L575-L580
        # SUBTRACTED: 各 spec 的指标定义——减法计划删除项 7。
        """Return Prometheus metric definitions emitted by this spec."""
        return {}

    # SOURCE: vllm/v1/kv_offload/base.py:L582-L603
    def __init__(self, config: OffloadingConfig):
        # SOURCE: vllm/v1/kv_offload/base.py:L582-L603
        self.config = config
        self.extra_config = config.extra_config
        self.replicated_layout: bool = False
        self.kv_events_config = OffloadingKVEventsConfig(
            enable_kv_cache_events=config.enable_kv_cache_events,
            self_describing_kv_events=bool(
                self.extra_config.get("self_describing_kv_events", False)
            ),
        )

        # When True, only prompt (prefill) blocks are offloaded; decode-phase
        # blocks (KV generated after the prompt) are skipped. Useful when prior
        # turns' generated tokens are dropped before the next turn (e.g.
        # reasoning models that strip thinking).
        self.offload_prompt_only: bool = bool(
            self.extra_config.get("offload_prompt_only", True)
        )

        self.tokens_per_block = tuple(group.tokens_per_block for group in config.groups)
        self.tokens_per_hash = config.cache.tokens_per_hash
        self.blocks_per_chunk = config.cache.blocks_per_chunk

    # SOURCE: vllm/v1/kv_offload/base.py:L605-L612
    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        # SOURCE: vllm/v1/kv_offload/base.py:L605-L612
        """
        Get an OffloadingManager that will be used
        by the scheduler-side offloading connector to track
        offloaded blocks and manage evictions.
        """
        pass

    # SOURCE: vllm/v1/kv_offload/base.py:L614-L625
    @abstractmethod
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        # SOURCE: vllm/v1/kv_offload/base.py:L614-L625
        """
        Get an OffloadingWorker that handles async KV transfers for this spec.

        Args:
            kv_caches: Canonicalized KV caches.

        Returns:
            An OffloadingWorker instance for this medium.
        """
        pass
