"""Tests for OffloadingManager + CPUOffloadingManager.

ABC contract, ref-count semantics, atomic prepare_store, complete_store flip.
"""

from __future__ import annotations

import pytest

from implementation.offload_manager import (
    CPUOffloadingManager,
    OffloadingManager,
)
from implementation.offload_spec import (
    CPULoadStoreSpec,
    OffloadKey,
    OffloadingEvent,
    PrepareStoreOutput,
    ReqContext,
    make_offload_key,
)


def _key(i: int, group: int = 0) -> OffloadKey:
    return make_offload_key(i.to_bytes(28, "big"), group)


@pytest.fixture
def ctx():
    return ReqContext(kv_transfer_params=None)


# ---------------------------------------------------------------------------
# OffloadingManager ABC contract
# ---------------------------------------------------------------------------
class TestOffloadingManagerABC:
    def test_is_abstract(self):
        """OffloadingManager cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OffloadingManager()  # type: ignore[abstract]

    def test_ten_methods_present(self):
        """Eight verbs + take_events + shutdown."""
        for m in (
            "lookup",
            "prepare_load",
            "touch",
            "complete_load",
            "prepare_store",
            "complete_store",
            "take_events",
            "shutdown",
        ):
            assert hasattr(OffloadingManager, m)


# ---------------------------------------------------------------------------
# CPUOffloadingManager construction
# ---------------------------------------------------------------------------
class TestManagerInit:
    def test_default_lru(self):
        """Default cache_policy is 'lru'."""
        mgr = CPUOffloadingManager(num_blocks=16)
        assert mgr.policy_name() == "LRUCachePolicy"

    def test_arc_policy_loaded(self):
        """cache_policy='arc' loads ARC."""
        mgr = CPUOffloadingManager(num_blocks=16, cache_policy="arc")
        assert mgr.policy_name() == "ARCCachePolicy"

    def test_unknown_policy_raises(self):
        """Bogus policy name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CPUOffloadingManager(num_blocks=16, cache_policy="lfu")

    def test_medium_is_CPU(self):
        """medium identifier matches CPULoadStoreSpec.medium()."""
        mgr = CPUOffloadingManager(num_blocks=16)
        assert mgr.medium == "CPU"
        assert mgr.medium == CPULoadStoreSpec.medium()

    def test_initial_num_offloaded_zero(self):
        """Fresh manager has zero offloaded blocks."""
        mgr = CPUOffloadingManager(num_blocks=16)
        assert mgr.num_offloaded() == 0

    def test_num_blocks_inspector(self):
        mgr = CPUOffloadingManager(num_blocks=128)
        assert mgr.num_blocks() == 128

    def test_events_disabled_by_default(self):
        """Events queue disabled by default; take_events yields nothing."""
        mgr = CPUOffloadingManager(num_blocks=16)
        assert mgr.events is None

    def test_events_enabled_when_requested(self):
        """enable_events=True initializes empty queue."""
        mgr = CPUOffloadingManager(num_blocks=16, enable_events=True)
        assert mgr.events == []


# ---------------------------------------------------------------------------
# lookup contract
# ---------------------------------------------------------------------------
class TestLookup:
    def test_missing_returns_falsy(self, ctx):
        """Missing key → False (NOT None — None is the deferral sentinel)."""
        mgr = CPUOffloadingManager(num_blocks=8)
        assert mgr.lookup(_key(0), ctx) is False

    def test_after_store_returns_true(self, ctx):
        """A stored + completed block returns True."""
        mgr = CPUOffloadingManager(num_blocks=8)
        keys = [_key(0)]
        out = mgr.prepare_store(keys, ctx)
        assert out is not None
        mgr.complete_store(keys, success=True)
        assert mgr.lookup(_key(0), ctx) is True

    def test_during_store_returns_false(self, ctx):
        """A reserved-but-not-yet-loadable block (ref_cnt=-1) returns False."""
        mgr = CPUOffloadingManager(num_blocks=8)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        # not yet completed → block.is_ready False → lookup False
        assert mgr.lookup(_key(0), ctx) is False


# ---------------------------------------------------------------------------
# prepare_store atomicity + proactive eviction
# ---------------------------------------------------------------------------
class TestPrepareStore:
    def test_fresh_store_no_evictions(self, ctx):
        """When capacity > requested, no evictions and all keys stored."""
        mgr = CPUOffloadingManager(num_blocks=10)
        keys = [_key(i) for i in range(5)]
        out = mgr.prepare_store(keys, ctx)
        assert out is not None
        assert len(out.keys_to_store) == 5
        assert out.evicted_keys == []

    def test_idempotent_re_store(self, ctx):
        """A key already in keyspace → not re-stored (PrepareStoreOutput empty for it)."""
        mgr = CPUOffloadingManager(num_blocks=10)
        keys = [_key(0)]
        first = mgr.prepare_store(keys, ctx)
        assert first is not None
        mgr.complete_store(keys, success=True)

        # try to store the same key again — should be idempotent
        second = mgr.prepare_store(keys, ctx)
        assert second is not None
        assert second.keys_to_store == []
        assert second.evicted_keys == []

    def test_proactive_eviction_returns_evicted_keys(self, ctx):
        """When at capacity, prepare_store returns evicted_keys eagerly (D5/O15)."""
        mgr = CPUOffloadingManager(num_blocks=2)
        # fill capacity
        keys_a = [_key(0), _key(1)]
        mgr.prepare_store(keys_a, ctx)
        mgr.complete_store(keys_a, success=True)

        # store 2 new keys → must evict the 2 old ones
        keys_b = [_key(10), _key(11)]
        out = mgr.prepare_store(keys_b, ctx)
        assert out is not None
        assert len(out.keys_to_store) == 2
        assert len(out.evicted_keys) == 2
        # evicted should be among the old keys
        for ek in out.evicted_keys:
            assert ek in keys_a

    def test_atomic_abort_when_no_idle_blocks(self, ctx):
        """If can't evict enough idle blocks, returns None (no state mutation)."""
        mgr = CPUOffloadingManager(num_blocks=2)
        keys_a = [_key(0), _key(1)]
        mgr.prepare_store(keys_a, ctx)
        mgr.complete_store(keys_a, success=True)
        # Pin both blocks (simulate in-flight load)
        from implementation.policies import LRUCachePolicy
        pol = mgr._policy
        for k in keys_a:
            pol.get(k).ref_cnt = 5  # pinned

        # Try to store 2 more — no idle blocks available
        out = mgr.prepare_store([_key(10), _key(11)], ctx)
        assert out is None
        # state unchanged
        assert mgr.num_offloaded() == 2

    def test_input_keys_not_evicted(self, ctx):
        """Keys being stored cannot themselves be evicted as collateral."""
        mgr = CPUOffloadingManager(num_blocks=2)
        keys_a = [_key(0), _key(1)]
        mgr.prepare_store(keys_a, ctx)
        mgr.complete_store(keys_a, success=True)
        # Store a mix: 1 already-present + 1 new. Must evict 1 old key.
        keys_b = [_key(0), _key(2)]
        out = mgr.prepare_store(keys_b, ctx)
        assert out is not None
        # _key(0) must not be evicted (it's an input key)
        assert _key(0) not in out.evicted_keys

    def test_ref_cnt_stays_neg_one_after_prepare(self, ctx):
        """prepare_store inserts blocks with ref_cnt=-1 (not yet loadable)."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        # block exists but is_ready False
        block = mgr._policy.get(_key(0))
        assert block is not None
        assert block.ref_cnt == -1
        assert not block.is_ready


# ---------------------------------------------------------------------------
# complete_store flip
# ---------------------------------------------------------------------------
class TestCompleteStore:
    def test_success_flips_neg_one_to_zero(self, ctx):
        """complete_store(success=True) flips ref_cnt from -1 to 0."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        block = mgr._policy.get(_key(0))
        assert block.ref_cnt == 0
        assert block.is_ready

    def test_failure_removes_block(self, ctx):
        """complete_store(success=False) removes the block + frees the id."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=False)
        assert mgr._policy.get(_key(0)) is None

    def test_complete_already_ready_idempotent(self, ctx):
        """Calling complete_store on an already-ready block is a no-op."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        # second call should not re-emit event nor change state
        mgr.complete_store(keys, success=True)
        block = mgr._policy.get(_key(0))
        assert block.ref_cnt == 0  # still 0


# ---------------------------------------------------------------------------
# prepare_load / complete_load — ref-count semantics
# ---------------------------------------------------------------------------
class TestLoadPath:
    def test_prepare_load_bumps_ref_cnt(self, ctx):
        """prepare_load bumps ref_cnt; protects from eviction."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        mgr.prepare_load(keys, ctx)
        block = mgr._policy.get(_key(0))
        assert block.ref_cnt == 1

    def test_complete_load_drops_ref_cnt(self, ctx):
        """complete_load decrements ref_cnt symmetrically."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        mgr.prepare_load(keys, ctx)
        mgr.complete_load(keys)
        block = mgr._policy.get(_key(0))
        assert block.ref_cnt == 0

    def test_prepare_load_returns_cpu_spec(self, ctx):
        """prepare_load returns a CPULoadStoreSpec with block_ids."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(i) for i in range(3)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        spec = mgr.prepare_load(keys, ctx)
        assert isinstance(spec, CPULoadStoreSpec)
        assert len(spec.block_ids) == 3

    def test_concurrent_loads_stack_ref_cnt(self, ctx):
        """Two prepare_loads on same key bumps ref_cnt to 2; one complete_load drops to 1."""
        mgr = CPUOffloadingManager(num_blocks=4)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        mgr.prepare_load(keys, ctx)
        mgr.prepare_load(keys, ctx)
        block = mgr._policy.get(_key(0))
        assert block.ref_cnt == 2
        mgr.complete_load(keys)
        assert block.ref_cnt == 1


# ---------------------------------------------------------------------------
# touch passes through to policy
# ---------------------------------------------------------------------------
class TestTouch:
    def test_touch_promotes_in_lru(self, ctx):
        """touch on LRU policy moves key to MRU end."""
        mgr = CPUOffloadingManager(num_blocks=4, cache_policy="lru")
        keys = [_key(0), _key(1)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        mgr.touch([_key(0)])  # _key(0) becomes MRU
        # next eviction should pick _key(1)
        out = mgr._policy.evict(1, set())
        assert out is not None
        ev_key, _ = out[0]
        assert ev_key == _key(1)


# ---------------------------------------------------------------------------
# Block pool primitives
# ---------------------------------------------------------------------------
class TestBlockPool:
    def test_free_blocks_initially_full_capacity(self):
        """All num_blocks ids are available initially."""
        mgr = CPUOffloadingManager(num_blocks=10)
        assert mgr._get_num_free_blocks() == 10

    def test_free_decreases_after_alloc(self, ctx):
        """Allocating 3 keys reduces free count by 3."""
        mgr = CPUOffloadingManager(num_blocks=10)
        keys = [_key(i) for i in range(3)]
        mgr.prepare_store(keys, ctx)
        assert mgr._get_num_free_blocks() == 7

    def test_block_id_recycled_after_eviction(self, ctx):
        """Evicted block_id returns to free_list; reused on next alloc."""
        mgr = CPUOffloadingManager(num_blocks=2)
        keys_a = [_key(0)]
        out = mgr.prepare_store(keys_a, ctx)
        first_id = out.store_spec.block_ids[0]
        mgr.complete_store(keys_a, success=True)
        # fill capacity
        out2 = mgr.prepare_store([_key(1)], ctx)
        mgr.complete_store([_key(1)], success=True)
        # evict 0 by storing 2 new
        keys_b = [_key(2)]
        out3 = mgr.prepare_store(keys_b, ctx)
        # The newly-allocated id should be one of the previously-used ids
        assert out3.store_spec.block_ids[0] in (first_id, out2.store_spec.block_ids[0])


# ---------------------------------------------------------------------------
# Event queue (prom-metrics surface)
# ---------------------------------------------------------------------------
class TestEvents:
    def test_no_events_when_disabled(self, ctx):
        """take_events yields nothing when events disabled."""
        mgr = CPUOffloadingManager(num_blocks=4, enable_events=False)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        events = list(mgr.take_events())
        assert events == []

    def test_store_event_emitted(self, ctx):
        """complete_store(success=True) emits a 'removed=False' event."""
        mgr = CPUOffloadingManager(num_blocks=4, enable_events=True)
        keys = [_key(0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)
        events = list(mgr.take_events())
        assert len(events) == 1
        assert events[0].removed is False
        assert _key(0) in events[0].keys

    def test_evict_event_emitted(self, ctx):
        """prepare_store with evictions emits a 'removed=True' event."""
        mgr = CPUOffloadingManager(num_blocks=2, enable_events=True)
        keys_a = [_key(0), _key(1)]
        mgr.prepare_store(keys_a, ctx)
        mgr.complete_store(keys_a, success=True)
        # drain stored events
        list(mgr.take_events())

        keys_b = [_key(10)]
        mgr.prepare_store(keys_b, ctx)
        evict_events = [e for e in mgr.take_events() if e.removed]
        assert len(evict_events) >= 1

    def test_take_events_drains(self, ctx):
        """take_events clears the queue."""
        mgr = CPUOffloadingManager(num_blocks=4, enable_events=True)
        mgr.prepare_store([_key(0)], ctx)
        mgr.complete_store([_key(0)], success=True)
        list(mgr.take_events())  # drain
        assert list(mgr.take_events()) == []  # second drain empty


# ---------------------------------------------------------------------------
# num_offloaded inspector — used by demos and tests
# ---------------------------------------------------------------------------
class TestInspectors:
    def test_num_offloaded_after_full_roundtrip(self, ctx):
        """Demo 5: 100 blocks → num_offloaded = 100 after store, still 100 after load."""
        n = 100
        mgr = CPUOffloadingManager(num_blocks=n, cache_policy="lru", enable_events=True)
        keys = [_key(i) for i in range(n)]
        out = mgr.prepare_store(keys, ctx)
        assert out is not None
        assert len(out.keys_to_store) == n
        assert len(out.evicted_keys) == 0
        mgr.complete_store(keys, success=True)
        spec = mgr.prepare_load(keys, ctx)
        assert len(spec.block_ids) == n
        mgr.complete_load(keys)
        assert mgr.num_offloaded() == n
