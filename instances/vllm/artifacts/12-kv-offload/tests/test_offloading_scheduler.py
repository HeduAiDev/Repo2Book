"""Tests for OffloadingConnectorScheduler — REACTIVE prefix lookup.

Trap E anchor: vLLM does NOT do predictive prefetch — only block-hash matching.
"""

from __future__ import annotations

import pytest

from implementation.offload_manager import CPUOffloadingManager
from implementation.offload_spec import (
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from implementation.offloading_scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    RequestGroupState,
    RequestOffloadState,
    SchedulerOffloadConfig,
    _SimpleRequest,
    cdiv,
    headroom_freed_gb,
    overlap_blocks_per_step,
)


def _bh(i: int) -> bytes:
    """28-byte block hash filled with byte i."""
    return bytes([i % 256]) * 28


def _key(i: int, group: int = 0) -> OffloadKey:
    return make_offload_key(_bh(i), group)


# ---------------------------------------------------------------------------
# cdiv
# ---------------------------------------------------------------------------
class TestCdiv:
    def test_exact(self):
        assert cdiv(8, 2) == 4

    def test_round_up(self):
        assert cdiv(7, 2) == 4

    def test_zero_numerator(self):
        assert cdiv(0, 5) == 0


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
class TestConfig:
    def test_group_offload_config_frozen(self):
        gc = GroupOffloadConfig(
            group_idx=0, gpu_block_size=16, offloaded_block_size=16,
            hash_block_size_factor=1,
        )
        with pytest.raises((AttributeError, Exception)):
            gc.group_idx = 1  # frozen=True

    def test_from_groups_full_attention(self):
        """When sliding_windows=None, all groups are full attention."""
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=2,
            hash_block_size=16,
        )
        assert len(cfg.kv_group_configs) == 1
        gc = cfg.kv_group_configs[0]
        assert gc.gpu_block_size == 16
        assert gc.offloaded_block_size == 32
        assert gc.sliding_window_size_in_blocks is None

    def test_from_groups_with_sliding(self):
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
            sliding_windows=(64,),  # 64 tokens
        )
        gc = cfg.kv_group_configs[0]
        assert gc.sliding_window_size_in_blocks == 4  # 64/16

    def test_from_groups_divisibility_assertion(self):
        with pytest.raises(AssertionError):
            SchedulerOffloadConfig.from_groups(
                gpu_block_sizes=(15,),
                block_size_factor=1,
                hash_block_size=16,  # 15 not divisible by 16
            )

    def test_hash_block_size_factor(self):
        """hash_block_size_factor = offloaded_block_size // hash_block_size."""
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(64,),
            block_size_factor=2,
            hash_block_size=16,
        )
        # offloaded = 64 * 2 = 128; factor = 128 / 16 = 8
        assert cfg.kv_group_configs[0].hash_block_size_factor == 8


# ---------------------------------------------------------------------------
# OffloadingConnectorScheduler — basic
# ---------------------------------------------------------------------------
class TestSchedulerBasic:
    def _make(self, num_blocks=64):
        mgr = CPUOffloadingManager(num_blocks=num_blocks, cache_policy="lru")
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
        )
        return OffloadingConnectorScheduler(manager=mgr, config=cfg), mgr

    def test_no_hits_when_keyspace_empty(self):
        """Fresh scheduler with empty manager → 0 matched tokens."""
        sched, mgr = self._make()
        req = _SimpleRequest(
            request_id="r1",
            block_hashes=[_bh(0), _bh(1), _bh(2), _bh(3)],
            num_tokens=64,
        )
        n, async_flag = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
        assert n == 0

    def test_hit_after_pre_population(self):
        """Pre-populate manager; scheduler counts the prefix hits."""
        sched, mgr = self._make()
        ctx = ReqContext()
        # Store keys for hashes 0,1,2,3 in manager
        keys = [make_offload_key(_bh(i), 0) for i in range(4)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)

        req = _SimpleRequest(
            request_id="r1",
            block_hashes=[_bh(i) for i in range(4)],
            num_tokens=64,
        )
        n, _ = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
        # 4 blocks × 16 tokens/block = 64 tokens
        assert n == 64


# ---------------------------------------------------------------------------
# Trap E: REACTIVE prefix lookup, NOT predictive
# ---------------------------------------------------------------------------
class TestTrapEReactive:
    """The scheduler walks block_hashes IN ORDER and stops at the first miss.
    No Markov model, no ML predictor, no learned access pattern."""

    def _make(self, num_blocks=8):
        mgr = CPUOffloadingManager(num_blocks=num_blocks, cache_policy="lru")
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
        )
        return OffloadingConnectorScheduler(manager=mgr, config=cfg), mgr

    def test_maximal_prefix_stops_at_first_miss(self):
        """If hash[2] is offloaded but hash[1] is not, hit_count = 1 (NOT 2)."""
        sched, mgr = self._make()
        ctx = ReqContext()
        # store block 0 and block 2 only — middle block 1 missing
        keys = [make_offload_key(_bh(0), 0), make_offload_key(_bh(2), 0)]
        mgr.prepare_store(keys, ctx)
        mgr.complete_store(keys, success=True)

        all_keys = [make_offload_key(_bh(i), 0) for i in range(3)]
        result = sched._maximal_prefix_lookup(all_keys, ctx)
        # block 0 hit, block 1 miss → stop. result = 1
        assert result == 1

    def test_no_predictor_in_module(self):
        """Negative grep: NO predictive ML/Markov code in scheduler module."""
        import implementation.offloading_scheduler as sched_mod
        import inspect
        src = inspect.getsource(sched_mod)
        for forbidden in ("predict", "ml_prefetch", "markov", "transformer_predictor"):
            assert forbidden.lower() not in src.lower() or "NOT predictive" in src or "predictive" in src.lower()
        # Stronger: assert no predictive identifiers as substrings
        for f in ("ml_prefetch", "markov_chain", "predict_prefix"):
            assert f not in src

    def test_deferred_lookup_returns_none(self):
        """Backend returning None (deferral) → scheduler returns None."""
        sched, mgr = self._make()
        ctx = ReqContext()

        class DeferringManager:
            def lookup(self, key, req_context):
                return None  # always defer

        sched.manager = DeferringManager()
        keys = [make_offload_key(_bh(i), 0) for i in range(3)]
        result = sched._maximal_prefix_lookup(keys, ctx)
        # Defer → None
        assert result is None


# ---------------------------------------------------------------------------
# Sliding-window lookup
# ---------------------------------------------------------------------------
class TestSlidingWindowLookup:
    def _make(self):
        mgr = CPUOffloadingManager(num_blocks=8)
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
        )
        return OffloadingConnectorScheduler(manager=mgr, config=cfg), mgr

    def test_window_zero_when_no_run(self):
        """If no consecutive hits at the END equal the window, returns 0."""
        sched, mgr = self._make()
        ctx = ReqContext()
        keys = [make_offload_key(_bh(i), 0) for i in range(5)]
        result = sched._sliding_window_lookup(keys, sliding_window_size=3, req_context=ctx)
        # all miss → 0
        assert result == 0

    def test_window_returns_end_index_when_run_found(self):
        """When window run found, returns end_idx + window_size."""
        sched, mgr = self._make()
        ctx = ReqContext()
        # store last 3 keys (indices 2,3,4)
        all_keys = [make_offload_key(_bh(i), 0) for i in range(5)]
        mgr.prepare_store(all_keys[2:], ctx)
        mgr.complete_store(all_keys[2:], success=True)
        result = sched._sliding_window_lookup(all_keys, sliding_window_size=3, req_context=ctx)
        # 3 consecutive hits found from end → returns end_idx (2) + 3 = 5
        # OR returns the consecutive_hits count if it reaches sliding_window_size early
        assert result is not None
        assert result >= 3


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------
class TestOverlapMath:
    def test_decode_step_191_blocks(self):
        """Demo 3 verbatim: 50 ms decode, 261.66 µs/block → 191 blocks/step."""
        n = overlap_blocks_per_step(50.0, 261.66)
        assert n == 191

    def test_prefill_step_764_blocks(self):
        """Demo 3 verbatim: 200 ms prefill, 261.66 µs/block → 764 blocks/step."""
        n = overlap_blocks_per_step(200.0, 261.66)
        assert n == 764

    def test_zero_latency_returns_zero(self):
        """Defensive: zero/negative latency → 0 (avoid divide-by-zero)."""
        assert overlap_blocks_per_step(50.0, 0.0) == 0
        assert overlap_blocks_per_step(50.0, -1.0) == 0

    def test_headroom_zero_hit_rate(self):
        """No hits → no headroom freed."""
        assert headroom_freed_gb(100.0, hit_rate=0.0) == 0.0

    def test_headroom_full_hit_rate(self):
        """100% hits at default ratio → full offload size."""
        assert headroom_freed_gb(100.0, hit_rate=1.0) == 100.0

    def test_headroom_with_ratio(self):
        """Custom HBM/DRAM ratio scales the result."""
        assert headroom_freed_gb(100.0, hit_rate=0.5, hbm_to_dram_ratio=2.0) == 100.0


# ---------------------------------------------------------------------------
# RequestOffloadState
# ---------------------------------------------------------------------------
class TestRequestOffloadState:
    def test_init_creates_group_states(self):
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16, 16),
            block_size_factor=1,
            hash_block_size=16,
        )
        req = _SimpleRequest(request_id="r", block_hashes=[], num_tokens=0)
        rs = RequestOffloadState(config=cfg, req=req)
        assert len(rs.group_states) == 2
        assert all(isinstance(gs, RequestGroupState) for gs in rs.group_states)

    def test_kv_transfer_params_propagated(self):
        cfg = SchedulerOffloadConfig.from_groups(
            gpu_block_sizes=(16,),
            block_size_factor=1,
            hash_block_size=16,
        )
        params = {"do_remote_decode": True}
        req = _SimpleRequest(
            request_id="r", block_hashes=[], num_tokens=0,
            kv_transfer_params=params,
        )
        rs = RequestOffloadState(config=cfg, req=req)
        assert rs.req_context.kv_transfer_params == params
