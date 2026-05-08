"""Tests for SimpleCPUOffloadScheduler — 742-LOC educational variant."""

from __future__ import annotations

import pytest

from implementation.simple_offload_manager import (
    LoadRequestState,
    OffloadMode,
    SimpleCPUOffloadScheduler,
    StoreRequestState,
    TransferMeta,
    estimate_lazy_target_blocks,
)


# ---------------------------------------------------------------------------
# OffloadMode enum
# ---------------------------------------------------------------------------
class TestOffloadMode:
    def test_lazy_value(self):
        assert OffloadMode.LAZY.value == "lazy"

    def test_eager_value(self):
        assert OffloadMode.EAGER.value == "eager"


# ---------------------------------------------------------------------------
# Scheduler init
# ---------------------------------------------------------------------------
class TestInit:
    def _mk(self, **kw):
        defaults = dict(
            block_size=16,
            num_gpu_blocks=64,
            gpu_kv_bytes_per_block=1024,
            cpu_capacity_bytes=8192,
        )
        defaults.update(kw)
        return SimpleCPUOffloadScheduler(**defaults)

    def test_eager_default(self):
        """Default mode is EAGER."""
        s = self._mk()
        assert s.mode == OffloadMode.EAGER

    def test_num_cpu_blocks_derived(self):
        """num_cpu_blocks derived from cpu_capacity_bytes / per-block bytes ratio."""
        s = self._mk(num_gpu_blocks=64, gpu_kv_bytes_per_block=1024,
                     cpu_capacity_bytes=8192)
        # gpu_total = 64 * 1024 = 65536; ratio = 8192/65536 = 0.125; num_cpu = 64 * 0.125 = 8
        assert s.num_cpu_blocks == 8

    def test_num_cpu_blocks_minimum_one(self):
        """num_cpu_blocks at LEAST 1 — defensive for tiny configs."""
        s = self._mk(num_gpu_blocks=64, gpu_kv_bytes_per_block=1024,
                     cpu_capacity_bytes=10)  # tiny
        assert s.num_cpu_blocks >= 1

    def test_lazy_target_free_set(self):
        """In LAZY mode, target_free is set to (num_gpu_blocks * watermark_ratio)."""
        s = self._mk(mode=OffloadMode.LAZY, num_gpu_blocks=100, watermark_ratio=0.5)
        assert s.target_free == 50

    def test_eager_target_free_zero(self):
        """In EAGER mode, target_free is 0 (offload always)."""
        s = self._mk(mode=OffloadMode.EAGER, watermark_ratio=0.5)
        assert s.target_free == 0


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------
class TestLookup:
    def _mk(self):
        return SimpleCPUOffloadScheduler(
            block_size=16, num_gpu_blocks=64,
            gpu_kv_bytes_per_block=1024, cpu_capacity_bytes=8192,
        )

    def test_lookup_miss_returns_none(self):
        """Missing hash returns None."""
        s = self._mk()
        assert s.lookup(b"\x00" * 32) is None

    def test_lookup_after_store_returns_block_id(self):
        """After store, lookup returns the cpu_block_id."""
        s = self._mk()
        h = b"\xAA" * 32
        meta = s.queue_store("req-1", [h], [42])
        assert meta is not None
        cid = s.lookup(h)
        assert cid is not None

    def test_lookup_refreshes_lru(self):
        """lookup moves the entry to MRU end."""
        s = self._mk()
        h0 = b"\x00" * 32
        h1 = b"\x11" * 32
        s.queue_store("r1", [h0], [0])
        s.queue_store("r2", [h1], [1])
        # now h0 is LRU. Lookup h0 to refresh.
        s.lookup(h0)
        # OrderedDict's last key is the MRU
        last_key = next(reversed(s.cpu_block_pool))
        assert last_key == h0


# ---------------------------------------------------------------------------
# queue_store
# ---------------------------------------------------------------------------
class TestQueueStore:
    def _mk(self):
        return SimpleCPUOffloadScheduler(
            block_size=16, num_gpu_blocks=64,
            gpu_kv_bytes_per_block=1024, cpu_capacity_bytes=8192,
        )

    def test_basic_store_returns_meta(self):
        s = self._mk()
        h = b"\xAA" * 32
        meta = s.queue_store("r1", [h], [0])
        assert meta is not None
        assert isinstance(meta, TransferMeta)
        assert meta.gpu_block_ids == [0]
        assert len(meta.cpu_block_ids) == 1

    def test_idempotent_re_store(self):
        """Re-storing same hash returns empty meta (already stored)."""
        s = self._mk()
        h = b"\xAA" * 32
        s.queue_store("r1", [h], [0])
        meta = s.queue_store("r2", [h], [1])
        assert meta is not None
        assert meta.gpu_block_ids == []
        assert meta.cpu_block_ids == []

    def test_assert_lengths_match(self):
        """block_hashes and gpu_block_ids must be the same length."""
        s = self._mk()
        with pytest.raises(AssertionError):
            s.queue_store("r1", [b"a" * 32, b"b" * 32], [0])  # 2 vs 1

    def test_evict_when_full(self):
        """When cpu_block_pool is full, _allocate_cpu_blocks evicts to make room."""
        s = SimpleCPUOffloadScheduler(
            block_size=16, num_gpu_blocks=4,
            gpu_kv_bytes_per_block=1024, cpu_capacity_bytes=2048,  # 2 cpu blocks
        )
        # Fill capacity (2)
        s.queue_store("r1", [b"\x00" * 32], [0])
        s.queue_store("r2", [b"\x11" * 32], [1])
        assert s.num_cpu_used() == 2
        # store a third — should evict the oldest
        meta = s.queue_store("r3", [b"\x22" * 32], [2])
        assert meta is not None


# ---------------------------------------------------------------------------
# queue_load
# ---------------------------------------------------------------------------
class TestQueueLoad:
    def _mk(self):
        return SimpleCPUOffloadScheduler(
            block_size=16, num_gpu_blocks=64,
            gpu_kv_bytes_per_block=1024, cpu_capacity_bytes=8192,
        )

    def test_load_after_store(self):
        """A request can load a previously-stored block."""
        s = self._mk()
        h = b"\xAA" * 32
        s.queue_store("r1", [h], [0])
        meta = s.queue_load("r2", [h], [99])
        assert meta is not None
        assert meta.gpu_block_ids == [99]

    def test_load_miss_returns_none(self):
        """Loading an unstored hash returns None."""
        s = self._mk()
        assert s.queue_load("r1", [b"\x00" * 32], [0]) is None

    def test_prefix_lookup_stops_at_first_miss(self):
        """Loader scans hashes in order; stops at first miss."""
        s = self._mk()
        h0 = b"\x00" * 32
        h1 = b"\x11" * 32
        s.queue_store("r1", [h0], [0])  # only h0 stored
        meta = s.queue_load("r2", [h0, h1], [10, 11])
        assert meta is not None
        # Only the prefix of confirmed hits — h0 — should be loaded
        assert meta.gpu_block_ids == [10]


# ---------------------------------------------------------------------------
# Inspectors
# ---------------------------------------------------------------------------
class TestInspectors:
    def _mk(self):
        return SimpleCPUOffloadScheduler(
            block_size=16, num_gpu_blocks=64,
            gpu_kv_bytes_per_block=1024, cpu_capacity_bytes=8192,
        )

    def test_initial_state(self):
        s = self._mk()
        assert s.num_cpu_used() == 0
        assert s.num_cpu_free() == s.num_cpu_blocks

    def test_used_grows_with_store(self):
        s = self._mk()
        for i in range(3):
            s.queue_store(f"r{i}", [bytes([i]) * 32], [i])
        assert s.num_cpu_used() == 3

    def test_evict_oldest_returns_count(self):
        """Test helper: evict_oldest(n) pops n oldest entries."""
        s = self._mk()
        for i in range(3):
            s.queue_store(f"r{i}", [bytes([i]) * 32], [i])
        n = s.evict_oldest(2)
        assert n == 2


# ---------------------------------------------------------------------------
# estimate_lazy_target_blocks
# ---------------------------------------------------------------------------
class TestEstimateLazyTarget:
    def test_attention_only(self):
        """num_attention * cdiv(max_batched, block_size)."""
        # 8 attention groups, max_batched=512, block_size=16 → 8 * 32 = 256
        assert estimate_lazy_target_blocks(
            num_attention_groups=8,
            num_mamba_groups=0,
            num_sliding_window_groups=0,
            sliding_window_blocks=0,
            max_num_batched_tokens=512,
            block_size=16,
        ) == 256

    def test_mamba_contributes_two_per_group(self):
        """Each mamba group adds 2 blocks."""
        result = estimate_lazy_target_blocks(
            num_attention_groups=0,
            num_mamba_groups=4,
            num_sliding_window_groups=0,
            sliding_window_blocks=0,
            max_num_batched_tokens=0,
            block_size=16,
        )
        assert result == 8

    def test_sliding_window_contribution(self):
        """Each sliding-window group adds sliding_window_blocks."""
        result = estimate_lazy_target_blocks(
            num_attention_groups=0,
            num_mamba_groups=0,
            num_sliding_window_groups=2,
            sliding_window_blocks=10,
            max_num_batched_tokens=0,
            block_size=16,
        )
        assert result == 20


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------
class TestStateDataclasses:
    def test_transfer_meta(self):
        meta = TransferMeta(gpu_block_ids=[1, 2], cpu_block_ids=[10, 11])
        assert meta.gpu_block_ids == [1, 2]
        assert meta.cpu_block_ids == [10, 11]

    def test_load_request_state(self):
        st = LoadRequestState(
            request_id="r1",
            transfer_meta=TransferMeta([], []),
        )
        assert st.request_id == "r1"
        assert st.finished is False

    def test_store_request_state_cursor(self):
        """num_stored_blocks is a CURSOR (not a count) — list of cursors per group."""
        st = StoreRequestState(
            request_id="r1",
            block_ids=([1, 2, 3],),
            num_stored_blocks=[2],
        )
        assert st.num_stored_blocks == [2]
