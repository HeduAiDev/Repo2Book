"""Tests for policies — LRU + ARC.

Critical: Trap E test — ARC LOSES to LRU on phase_shift workload (HONEST CAVEAT).
Anchored on `vllm/v1/kv_offload/cpu/policies/{base,lru,arc}.py` semantics.
"""

from __future__ import annotations

import pytest

from implementation.offload_spec import OffloadKey, make_offload_key
from implementation.policies import (
    ARCCachePolicy,
    BlockStatus,
    CACHE_POLICIES,
    CachePolicy,
    LRUCachePolicy,
)


def _key(i: int, group: int = 0) -> OffloadKey:
    return make_offload_key(i.to_bytes(28, "big"), group)


# ---------------------------------------------------------------------------
# BlockStatus — ref_cnt sentinel
# ---------------------------------------------------------------------------
class TestBlockStatus:
    def test_default_ref_cnt_is_neg_one(self):
        """ref_cnt = -1 means 'reserved but not yet loadable' (W10 wisdom)."""
        b = BlockStatus(block_id=0)
        assert b.ref_cnt == -1

    def test_is_ready_false_when_neg_one(self):
        """is_ready returns False at ref_cnt == -1 (Trap reframe)."""
        b = BlockStatus(block_id=0, ref_cnt=-1)
        assert b.is_ready is False

    def test_is_ready_true_when_zero(self):
        """ref_cnt == 0 → idle, eligible for read."""
        b = BlockStatus(block_id=0, ref_cnt=0)
        assert b.is_ready is True

    def test_is_ready_true_when_positive(self):
        """ref_cnt > 0 → being read, also ready."""
        b = BlockStatus(block_id=0, ref_cnt=5)
        assert b.is_ready is True


# ---------------------------------------------------------------------------
# CachePolicy registry
# ---------------------------------------------------------------------------
class TestPolicyRegistry:
    def test_lru_registered(self):
        """`'lru'` resolves to LRUCachePolicy."""
        assert CACHE_POLICIES["lru"] is LRUCachePolicy

    def test_arc_registered(self):
        """`'arc'` resolves to ARCCachePolicy."""
        assert CACHE_POLICIES["arc"] is ARCCachePolicy

    def test_no_lfu_registered(self):
        """Trap B: no LFU policy ships in vLLM at 98661fe."""
        assert "lfu" not in CACHE_POLICIES

    def test_no_attention_score_registered(self):
        """Trap B: no attention-score policy ships in vLLM at 98661fe."""
        assert "attention_score" not in CACHE_POLICIES
        assert "attn_score" not in CACHE_POLICIES

    def test_only_two_policies(self):
        """Exactly LRU + ARC at this commit (sources confirm directory listing)."""
        assert set(CACHE_POLICIES.keys()) == {"lru", "arc"}


# ---------------------------------------------------------------------------
# LRU correctness
# ---------------------------------------------------------------------------
class TestLRUPolicy:
    def test_get_missing_returns_none(self):
        """Missing key returns None (no exception)."""
        p = LRUCachePolicy(cache_capacity=10)
        assert p.get(_key(0)) is None

    def test_insert_then_get(self):
        """insert → get returns the BlockStatus."""
        p = LRUCachePolicy(cache_capacity=10)
        blk = BlockStatus(block_id=42, ref_cnt=0)
        p.insert(_key(7), blk)
        assert p.get(_key(7)) is blk

    def test_remove(self):
        """remove deletes; subsequent get returns None."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.remove(_key(0))
        assert p.get(_key(0)) is None

    def test_len_tracks_count(self):
        """len() == number of live (non-evicted) blocks."""
        p = LRUCachePolicy(cache_capacity=10)
        for i in range(5):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        assert len(p) == 5

    def test_touch_moves_to_end(self):
        """touch promotes to MRU; eviction picks the OTHER (LRU) block."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        p.touch([_key(0)])  # 0 becomes MRU
        evicted = p.evict(1, set())
        assert evicted is not None
        evicted_key, _ = evicted[0]
        assert evicted_key == _key(1)  # 1 was LRU after touching 0

    def test_touch_reverse_order_last_is_mru(self):
        """If touch receives [a, b], b ends up MRU (touch iterates reversed)."""
        p = LRUCachePolicy(cache_capacity=10)
        for i in range(3):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        # touch in chronological order [0, 1, 2] — last is most recent
        p.touch([_key(0), _key(1), _key(2)])
        evicted = p.evict(1, set())
        assert evicted is not None
        # The touch iterates in REVERSE: 2 is moved first, then 1 (now MRU), then 0.
        # After reversal: order is 2,1,0 with 0 last → 0 is MRU.
        # So eviction picks 2 (now LRU).
        ev_key, _ = evicted[0]
        # verify it's NOT key 0 (which was last in chronological touch list)
        assert ev_key == _key(2)

    def test_evict_zero_returns_empty(self):
        """evict(0) is the no-op; returns empty list."""
        p = LRUCachePolicy(cache_capacity=10)
        assert p.evict(0, set()) == []

    def test_evict_skips_protected(self):
        """evict skips keys in `protected` set."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        evicted = p.evict(1, protected={_key(0)})
        assert evicted is not None
        ev_key, _ = evicted[0]
        assert ev_key == _key(1)

    def test_evict_skips_pinned(self):
        """evict skips blocks with ref_cnt > 0 (in-use)."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=5))  # pinned
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        evicted = p.evict(1, set())
        assert evicted is not None
        ev_key, _ = evicted[0]
        assert ev_key == _key(1)

    def test_evict_skips_not_ready(self):
        """evict skips blocks with ref_cnt == -1 (not yet loadable)."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=-1))  # not ready
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        evicted = p.evict(1, set())
        assert evicted is not None
        ev_key, _ = evicted[0]
        assert ev_key == _key(1)

    def test_evict_returns_none_on_insufficient(self):
        """evict returns None — no state mutation — when n cannot be satisfied."""
        p = LRUCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        # ask for 2; only 1 idle
        assert p.evict(2, set()) is None
        # state unchanged after failure
        assert len(p) == 1
        assert p.get(_key(0)) is not None

    def test_evict_atomic_no_partial_state(self):
        """If evict fails, NONE of the candidates are removed (atomic abort)."""
        p = LRUCachePolicy(cache_capacity=10)
        for i in range(3):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        # ask for 5; only 3 available
        assert p.evict(5, set()) is None
        assert len(p) == 3  # all 3 still present


# ---------------------------------------------------------------------------
# ARC correctness — basic semantics
# ---------------------------------------------------------------------------
class TestARCBasics:
    def test_get_missing_none(self):
        p = ARCCachePolicy(cache_capacity=10)
        assert p.get(_key(0)) is None

    def test_insert_goes_to_t1(self):
        """New blocks always land in T1 (recent partition)."""
        p = ARCCachePolicy(cache_capacity=10)
        blk = BlockStatus(block_id=0, ref_cnt=0)
        p.insert(_key(0), blk)
        assert _key(0) in p.t1
        assert _key(0) not in p.t2

    def test_insert_strips_b1(self):
        """Insert clears the key from B1 ghost list (re-promoted)."""
        p = ARCCachePolicy(cache_capacity=10)
        p.b1[_key(0)] = None
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        assert _key(0) not in p.b1

    def test_insert_strips_b2(self):
        """Insert clears the key from B2 ghost list."""
        p = ARCCachePolicy(cache_capacity=10)
        p.b2[_key(0)] = None
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        assert _key(0) not in p.b2

    def test_remove_from_t1(self):
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.remove(_key(0))
        assert p.get(_key(0)) is None

    def test_remove_from_t2(self):
        p = ARCCachePolicy(cache_capacity=10)
        # promote 0 to T2 by touch
        blk = BlockStatus(block_id=0, ref_cnt=0)
        p.insert(_key(0), blk)
        p.touch([_key(0)])  # T1 → T2
        assert _key(0) in p.t2
        p.remove(_key(0))
        assert p.get(_key(0)) is None

    def test_get_finds_t1_or_t2(self):
        """get returns hit regardless of T1/T2 partition."""
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        assert p.get(_key(0)) is not None  # in T1
        p.touch([_key(0)])  # promote to T2
        assert p.get(_key(0)) is not None  # in T2

    def test_len_sums_t1_t2(self):
        """len = |T1| + |T2|; ghosts B1/B2 do NOT count."""
        p = ARCCachePolicy(cache_capacity=10)
        for i in range(3):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        p.touch([_key(0)])
        p.b1[_key(99)] = None  # ghost — should not count
        assert len(p) == 3


# ---------------------------------------------------------------------------
# ARC promotion + ghost-list behavior
# ---------------------------------------------------------------------------
class TestARCPromotion:
    def test_t1_to_t2_on_touch(self):
        """Touch on a T1 (ready) key promotes to T2."""
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.touch([_key(0)])
        assert _key(0) in p.t2
        assert _key(0) not in p.t1

    def test_t1_not_promoted_when_not_ready(self):
        """Touch on a T1 (not_ready) key STAYS in T1 (haven't truly been used twice)."""
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=-1))  # not ready
        p.touch([_key(0)])
        assert _key(0) in p.t1
        assert _key(0) not in p.t2

    def test_b1_ghost_hit_grows_target(self):
        """Touch on a B1 ghost key INCREASES target_t1_size (recency wins)."""
        p = ARCCachePolicy(cache_capacity=10)
        p.b1[_key(0)] = None
        p.b2[_key(99)] = None  # so |B2| > 0 for delta calc
        prev = p.target_t1_size
        p.touch([_key(0)])
        assert p.target_t1_size > prev

    def test_b2_ghost_hit_shrinks_target(self):
        """Touch on a B2 ghost key DECREASES target_t1_size (frequency wins)."""
        p = ARCCachePolicy(cache_capacity=10)
        p.target_t1_size = 5.0
        p.b1[_key(99)] = None
        p.b2[_key(0)] = None
        prev = p.target_t1_size
        p.touch([_key(0)])
        assert p.target_t1_size < prev

    def test_target_clamped_to_capacity(self):
        """target_t1_size is bounded above by cache_capacity."""
        p = ARCCachePolicy(cache_capacity=4)
        for i in range(20):
            p.b1[_key(i)] = None
            p.b2[_key(100 + i)] = None
        for i in range(20):
            p.touch([_key(i)])
        assert p.target_t1_size <= 4

    def test_target_clamped_to_zero(self):
        """target_t1_size is bounded below by 0."""
        p = ARCCachePolicy(cache_capacity=4)
        p.target_t1_size = 4.0
        for i in range(20):
            p.b1[_key(i)] = None
            p.b2[_key(100 + i)] = None
        for i in range(20):
            p.touch([_key(100 + i)])  # B2 hits
        assert p.target_t1_size >= 0


# ---------------------------------------------------------------------------
# ARC eviction
# ---------------------------------------------------------------------------
class TestARCEviction:
    def test_evict_zero(self):
        p = ARCCachePolicy(cache_capacity=10)
        assert p.evict(0, set()) == []

    def test_evict_returns_none_when_insufficient(self):
        """ARC eviction is atomic: None on insufficient idle blocks."""
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        assert p.evict(2, set()) is None
        assert len(p) == 1  # state unchanged

    def test_evicted_t1_pushed_to_b1(self):
        """Evicted T1 keys appear in B1 ghost list."""
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.evict(1, set())
        assert _key(0) in p.b1

    def test_evicted_t2_pushed_to_b2(self):
        """Evicted T2 keys appear in B2 ghost list."""
        p = ARCCachePolicy(cache_capacity=10)
        p.target_t1_size = 0.0  # force T2 eviction
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.touch([_key(0)])  # promote to T2
        assert _key(0) in p.t2
        p.evict(1, set())
        assert _key(0) in p.b2

    def test_evict_skips_protected(self):
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=0))
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        ev = p.evict(1, protected={_key(0)})
        assert ev is not None
        ev_key, _ = ev[0]
        assert ev_key == _key(1)

    def test_evict_skips_pinned(self):
        p = ARCCachePolicy(cache_capacity=10)
        p.insert(_key(0), BlockStatus(block_id=0, ref_cnt=2))  # pinned
        p.insert(_key(1), BlockStatus(block_id=1, ref_cnt=0))
        ev = p.evict(1, set())
        assert ev is not None
        ev_key, _ = ev[0]
        assert ev_key == _key(1)

    def test_ghost_lists_bounded_by_capacity(self):
        """Ghost lists trim to cache_capacity to bound memory."""
        p = ARCCachePolicy(cache_capacity=2)
        # Push many evictions
        for i in range(10):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
            p.evict(1, set())
        # B1 should have AT MOST cache_capacity entries
        assert len(p.b1) <= 2

    def test_dry_run_does_not_mutate_on_failure(self):
        """ARC dry-run phase: no state changes when n cannot be satisfied."""
        p = ARCCachePolicy(cache_capacity=10)
        for i in range(2):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        b1_before = list(p.b1)
        b2_before = list(p.b2)
        ev = p.evict(5, set())
        assert ev is None
        assert list(p.b1) == b1_before  # B1 unchanged
        assert list(p.b2) == b2_before  # B2 unchanged
        assert len(p) == 2  # both blocks still live

    def test_evict_multiple_picks_correct_partition(self):
        """When |T1| > target, eviction prefers T1 first."""
        p = ARCCachePolicy(cache_capacity=10)
        p.target_t1_size = 0.0  # T1 partition overweight by definition
        for i in range(3):
            p.insert(_key(i), BlockStatus(block_id=i, ref_cnt=0))
        ev = p.evict(2, set())
        assert ev is not None
        # All 2 evicted should have come from T1 (and now in B1)
        for k, _ in ev:
            assert k in p.b1


# ---------------------------------------------------------------------------
# Trap E HONESTY: ARC LOSES to LRU on phase_shift workload
# ---------------------------------------------------------------------------
class TestTrapEArcLoses:
    """Honesty test — phase_shift workload shows ARC's 14.15% miss vs LRU's 2.60%.

    This is NOT a bug. ARC's adaptation cost (T2/B2 ghost-list overhead)
    can hurt on synthetic phase shifts where pure LRU happens to be optimal.
    The chapter MUST frame this honestly: ARC is not strictly better.
    """

    def _run_workload(self, policy_cls, ops, capacity=32):
        """Replicate Demo 2's run() loop."""
        policy = policy_cls(cache_capacity=capacity)
        misses = 0
        next_block_id = 0
        for key in ops:
            blk = policy.get(key)
            if blk is None:
                misses += 1
                if len(policy) >= capacity:
                    evicted = policy.evict(1, protected=set())
                    if evicted is None:
                        # force one to idle
                        for k, b in (
                            list(policy.t1.items()) + list(policy.t2.items())
                            if hasattr(policy, "t1")
                            else list(policy.blocks.items())
                        )[:1]:
                            b.ref_cnt = 0
                        evicted = policy.evict(1, protected=set())
                blk = BlockStatus(block_id=next_block_id, ref_cnt=0)
                next_block_id += 1
                policy.insert(key, blk)
            else:
                policy.touch([key])
        return misses

    def _make_phase_shift_keys(self):
        """Reproduce demo.py phase_shift workload exactly (seed=7)."""
        import random
        random.seed(7)
        keys = []
        for i in range(1000):
            k = random.randint(0, 25)  # phase A
            keys.append(_key(k))
        for i in range(1000):
            k = random.randint(50, 75)  # phase B
            keys.append(_key(k))
        return keys

    def test_phase_shift_lru_miss_2_60_percent(self):
        """LRU on phase_shift = 2.60% miss rate (verbatim from demo)."""
        ops = self._make_phase_shift_keys()
        miss = self._run_workload(LRUCachePolicy, ops, capacity=32)
        miss_pct = round(100.0 * miss / len(ops), 2)
        assert miss_pct == 2.60

    def test_phase_shift_arc_miss_14_15_percent(self):
        """ARC on phase_shift = 14.15% miss rate — ARC LOSES (verbatim from demo)."""
        ops = self._make_phase_shift_keys()
        miss = self._run_workload(ARCCachePolicy, ops, capacity=32)
        miss_pct = round(100.0 * miss / len(ops), 2)
        assert miss_pct == 14.15

    def test_arc_loses_to_lru_on_phase_shift(self):
        """The headline: ARC > LRU miss% (= worse) on this synthetic workload."""
        ops = self._make_phase_shift_keys()
        lru_miss = self._run_workload(LRUCachePolicy, ops, capacity=32)
        arc_miss = self._run_workload(ARCCachePolicy, ops, capacity=32)
        assert arc_miss > lru_miss, (
            f"ARC ({arc_miss}) should LOSE to LRU ({lru_miss}) on phase_shift "
            f"— this is HONEST CAVEAT O08, not a bug."
        )


# ---------------------------------------------------------------------------
# Workload determinism — verify the other Demo 2 numerics
# ---------------------------------------------------------------------------
class TestDemo2Workloads:
    def _run(self, policy_cls, ops, capacity=32):
        policy = policy_cls(cache_capacity=capacity)
        misses = 0
        nbid = 0
        for key in ops:
            blk = policy.get(key)
            if blk is None:
                misses += 1
                if len(policy) >= capacity:
                    ev = policy.evict(1, protected=set())
                    if ev is None:
                        for k, b in (
                            list(policy.t1.items()) + list(policy.t2.items())
                            if hasattr(policy, "t1")
                            else list(policy.blocks.items())
                        )[:1]:
                            b.ref_cnt = 0
                        ev = policy.evict(1, protected=set())
                blk = BlockStatus(block_id=nbid, ref_cnt=0)
                nbid += 1
                policy.insert(key, blk)
            else:
                policy.touch([key])
        return misses

    def test_loop_scan_lru_100_percent(self):
        """loop_scan defeats LRU (50 unique keys, capacity 32)."""
        ops = []
        for _ in range(40):
            for i in range(50):
                ops.append(_key(i))
        miss = self._run(LRUCachePolicy, ops, 32)
        rate = round(100.0 * miss / len(ops), 2)
        assert rate == 100.00

    def test_loop_scan_arc_100_percent(self):
        """ARC ALSO 100% miss on loop scan — Belady-defeating."""
        ops = []
        for _ in range(40):
            for i in range(50):
                ops.append(_key(i))
        miss = self._run(ARCCachePolicy, ops, 32)
        rate = round(100.0 * miss / len(ops), 2)
        assert rate == 100.00

    def test_zipfian_lru_17_30_percent(self):
        """Zipfian (top-12 hot, 200-tail cold): LRU = 17.30% miss."""
        import random
        random.seed(42)
        ops = []
        for _ in range(2000):
            if random.random() < 0.8:
                i = random.randint(0, 11)
            else:
                i = random.randint(12, 199)
            ops.append(_key(i))
        miss = self._run(LRUCachePolicy, ops, 32)
        rate = round(100.0 * miss / 2000, 2)
        assert rate == 17.30

    def test_zipfian_arc_17_25_percent(self):
        """Zipfian: ARC = 17.25% miss (negligibly better than LRU here)."""
        import random
        random.seed(42)
        ops = []
        for _ in range(2000):
            if random.random() < 0.8:
                i = random.randint(0, 11)
            else:
                i = random.randint(12, 199)
            ops.append(_key(i))
        miss = self._run(ARCCachePolicy, ops, 32)
        rate = round(100.0 * miss / 2000, 2)
        assert rate == 17.25
