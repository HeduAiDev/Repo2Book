"""Tests for FilterReusedOffloadingManager — store_threshold filter wrapper."""

from __future__ import annotations

import pytest

from implementation.offload_manager import CPUOffloadingManager
from implementation.offload_spec import (
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from implementation.reuse_manager import FilterReusedOffloadingManager


def _key(i: int) -> OffloadKey:
    return make_offload_key(i.to_bytes(28, "big"), 0)


@pytest.fixture
def ctx():
    return ReqContext()


@pytest.fixture
def base_mgr():
    return CPUOffloadingManager(num_blocks=16, enable_events=True)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_threshold_below_2_raises(self, base_mgr):
        """store_threshold must be >= 2."""
        with pytest.raises(ValueError, match="store_threshold"):
            FilterReusedOffloadingManager(backing=base_mgr, store_threshold=1)

    def test_zero_threshold_raises(self, base_mgr):
        with pytest.raises(ValueError):
            FilterReusedOffloadingManager(backing=base_mgr, store_threshold=0)

    def test_max_tracker_size_below_one_raises(self, base_mgr):
        with pytest.raises(ValueError, match="max_tracker_size"):
            FilterReusedOffloadingManager(
                backing=base_mgr, store_threshold=2, max_tracker_size=0,
            )

    def test_default_threshold_2(self, base_mgr):
        f = FilterReusedOffloadingManager(backing=base_mgr)
        assert f.store_threshold == 2


# ---------------------------------------------------------------------------
# Lookup increments counter
# ---------------------------------------------------------------------------
class TestLookupCounting:
    def test_first_lookup_count_1(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        f.lookup(_key(0), ctx)
        assert f.counts[_key(0)] == 1

    def test_repeated_lookups_increment(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        for _ in range(5):
            f.lookup(_key(0), ctx)
        assert f.counts[_key(0)] == 5

    def test_lookup_delegates_to_backing(self, base_mgr, ctx):
        """lookup() return value is whatever the backing manager returns."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        # Empty backing → False
        assert f.lookup(_key(0), ctx) is False
        # Insert into backing
        base_mgr.prepare_store([_key(0)], ctx)
        base_mgr.complete_store([_key(0)], success=True)
        assert f.lookup(_key(0), ctx) is True

    def test_tracker_evicts_lru_at_max_size(self, base_mgr, ctx):
        """When tracker hits max_size, LRU entry is evicted."""
        f = FilterReusedOffloadingManager(
            backing=base_mgr, store_threshold=2, max_tracker_size=3,
        )
        for i in range(5):
            f.lookup(_key(i), ctx)
        # Only the last 3 keys should be tracked
        assert len(f.counts) == 3
        # Keys 0 and 1 should be evicted (LRU)
        assert _key(0) not in f.counts
        assert _key(1) not in f.counts


# ---------------------------------------------------------------------------
# prepare_store filters
# ---------------------------------------------------------------------------
class TestPrepareStoreFilter:
    def test_below_threshold_filtered_out(self, base_mgr, ctx):
        """Keys seen only once with threshold=2 are filtered out of prepare_store."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        f.lookup(_key(0), ctx)  # count=1, below threshold 2

        out = f.prepare_store([_key(0)], ctx)
        assert out is not None
        assert out.keys_to_store == []  # filtered

    def test_at_threshold_passes_through(self, base_mgr, ctx):
        """Once count reaches threshold, key is eligible for store."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        f.lookup(_key(0), ctx)  # 1
        f.lookup(_key(0), ctx)  # 2 — at threshold

        out = f.prepare_store([_key(0)], ctx)
        assert out is not None
        assert _key(0) in out.keys_to_store

    def test_above_threshold_passes_through(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=3)
        for _ in range(3):
            f.lookup(_key(0), ctx)
        out = f.prepare_store([_key(0)], ctx)
        assert _key(0) in out.keys_to_store

    def test_partial_filter(self, base_mgr, ctx):
        """Keys above threshold pass; those below are dropped."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        f.lookup(_key(0), ctx)
        f.lookup(_key(0), ctx)  # 2: passes
        f.lookup(_key(1), ctx)  # 1: filtered

        out = f.prepare_store([_key(0), _key(1)], ctx)
        assert _key(0) in out.keys_to_store
        assert _key(1) not in out.keys_to_store


# ---------------------------------------------------------------------------
# Delegation of other verbs
# ---------------------------------------------------------------------------
class TestDelegation:
    def test_prepare_load_delegates(self, base_mgr, ctx):
        """prepare_load passes straight through to backing manager."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        # Set up a stored key in the backing manager directly
        base_mgr.prepare_store([_key(0)], ctx)
        base_mgr.complete_store([_key(0)], success=True)

        spec = f.prepare_load([_key(0)], ctx)
        assert spec is not None

    def test_complete_store_delegates(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        # Lookup twice to allow store
        f.lookup(_key(0), ctx)
        f.lookup(_key(0), ctx)
        f.prepare_store([_key(0)], ctx)
        f.complete_store([_key(0)], success=True)
        # The backing manager now has _key(0) ready
        assert base_mgr.lookup(_key(0), ctx) is True

    def test_complete_load_delegates(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        # Set up via backing
        base_mgr.prepare_store([_key(0)], ctx)
        base_mgr.complete_store([_key(0)], success=True)
        # Use filter for prepare_load + complete_load
        f.prepare_load([_key(0)], ctx)
        f.complete_load([_key(0)])
        block = base_mgr._policy.get(_key(0))
        assert block.ref_cnt == 0

    def test_take_events_delegates(self, base_mgr, ctx):
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        f.lookup(_key(0), ctx)
        f.lookup(_key(0), ctx)
        f.prepare_store([_key(0)], ctx)
        f.complete_store([_key(0)], success=True)
        events = list(f.take_events())
        assert len(events) >= 1

    def test_touch_delegates(self, base_mgr, ctx):
        """touch is a delegated no-op-default → forwards to backing."""
        f = FilterReusedOffloadingManager(backing=base_mgr, store_threshold=2)
        # Should not raise
        f.touch([_key(0)])
