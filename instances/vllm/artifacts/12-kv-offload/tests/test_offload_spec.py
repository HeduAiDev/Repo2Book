"""Tests for offload_spec — OffloadKey packing, LoadStoreSpec, PrepareStoreOutput.

Anchored on `vllm/v1/kv_offload/base.py:L24-L398` semantics.
"""

from __future__ import annotations

import pytest

from implementation.offload_spec import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    CPULoadStoreSpec,
    CPUOffloadingSpec,
    DDR5_BANDWIDTH_GB_PER_S,
    DDR5_CAPACITY_GB,
    GPULoadStoreSpec,
    HBM3_BANDWIDTH_GB_PER_S,
    HBM_CAPACITY_GB,
    KV_BLOCK_BYTES,
    LoadStoreSpec,
    NVME_CAPACITY_GB,
    NVME_GEN5_BANDWIDTH_GB_PER_S,
    OffloadingEvent,
    OffloadingSpec,
    OffloadKey,
    PCIE_GEN5_BANDWIDTH_GB_PER_S,
    PrepareStoreOutput,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)


# ---------------------------------------------------------------------------
# OffloadKey packing — round-trip
# ---------------------------------------------------------------------------
class TestOffloadKeyPacking:
    def test_make_offload_key_returns_bytes(self):
        """OffloadKey is a NewType wrapping bytes; instance is a bytes object."""
        key = make_offload_key(b"\x01" * 28, 7)
        assert isinstance(key, bytes)

    def test_packed_length_block_hash_plus_4(self):
        """Packed key length = len(block_hash) + 4 bytes for big-endian uint32."""
        block_hash = b"\xAB" * 28
        key = make_offload_key(block_hash, 0)
        assert len(key) == 32

    def test_block_hash_round_trip(self):
        """get_offload_block_hash recovers the prefix block hash."""
        bh = b"\xDE\xAD\xBE\xEF" + b"\x00" * 24
        key = make_offload_key(bh, 3)
        assert get_offload_block_hash(key) == bh

    def test_group_idx_round_trip(self):
        """get_offload_group_idx recovers the 4-byte big-endian suffix."""
        for gidx in (0, 1, 7, 255, 65535, 16777215, 2**32 - 1):
            key = make_offload_key(b"\x00" * 28, gidx)
            assert get_offload_group_idx(key) == gidx

    def test_group_idx_endianness_big(self):
        """Group idx is encoded big-endian — last 4 bytes spell the integer."""
        key = make_offload_key(b"\x00" * 28, 0x01020304)
        assert bytes(key[-4:]) == b"\x01\x02\x03\x04"

    def test_distinct_keys_for_distinct_group_idx(self):
        """Same block_hash but different group_idx → different keys."""
        bh = b"\xAB" * 28
        k0 = make_offload_key(bh, 0)
        k1 = make_offload_key(bh, 1)
        assert k0 != k1

    def test_keys_hashable_in_dict(self):
        """OffloadKey must be usable as dict keys (load-bearing for keyspace)."""
        d = {}
        for i in range(10):
            d[make_offload_key(i.to_bytes(28, "big"), 0)] = i
        assert len(d) == 10
        # Same key constructed twice should hit the same dict slot.
        same = make_offload_key((3).to_bytes(28, "big"), 0)
        assert d[same] == 3


# ---------------------------------------------------------------------------
# LoadStoreSpec ABC + concretes
# ---------------------------------------------------------------------------
class TestLoadStoreSpec:
    def test_loadstorespec_is_abstract(self):
        """Abstract base — direct instantiation fails."""
        with pytest.raises(TypeError):
            LoadStoreSpec()  # type: ignore[abstract]

    def test_cpu_medium_is_CPU(self):
        """CPULoadStoreSpec.medium() returns the literal 'CPU'."""
        assert CPULoadStoreSpec.medium() == "CPU"

    def test_gpu_medium_is_GPU(self):
        """GPULoadStoreSpec.medium() returns the literal 'GPU'."""
        assert GPULoadStoreSpec.medium() == "GPU"

    def test_block_ids_stored_as_list(self):
        """BlockIDsLoadStoreSpec accepts an iterable and stores a list copy."""
        spec = CPULoadStoreSpec([3, 1, 4, 1, 5])
        assert spec.block_ids == [3, 1, 4, 1, 5]

    def test_block_ids_copy_not_alias(self):
        """Constructor copies list — caller cannot mutate the spec by side."""
        ids = [1, 2, 3]
        spec = CPULoadStoreSpec(ids)
        ids.append(99)
        assert spec.block_ids == [1, 2, 3]

    def test_gpu_default_group_sizes_full(self):
        """GPULoadStoreSpec with no group_sizes defaults to single-group spanning all blocks."""
        spec = GPULoadStoreSpec(block_ids=[0, 1, 2, 3])
        assert spec.group_sizes == [4]
        assert spec.block_indices == [0]

    def test_gpu_group_size_invariant(self):
        """sum(group_sizes) MUST equal len(block_ids) (worker loop invariant)."""
        with pytest.raises(AssertionError):
            GPULoadStoreSpec(
                block_ids=[0, 1, 2],
                group_sizes=[2, 2],  # sums to 4 != 3
                block_indices=[0, 0],
            )

    def test_gpu_group_indices_invariant(self):
        """len(block_indices) MUST equal len(group_sizes)."""
        with pytest.raises(AssertionError):
            GPULoadStoreSpec(
                block_ids=[0, 1, 2, 3],
                group_sizes=[2, 2],
                block_indices=[0],  # len 1 != 2
            )

    def test_gpu_repr_shows_block_ids(self):
        """__repr__ delegates to the block_ids list."""
        spec = GPULoadStoreSpec(block_ids=[7, 8, 9])
        assert repr(spec) == "[7, 8, 9]"


# ---------------------------------------------------------------------------
# PrepareStoreOutput
# ---------------------------------------------------------------------------
class TestPrepareStoreOutput:
    def test_three_fields(self):
        """PrepareStoreOutput holds (keys_to_store, store_spec, evicted_keys)."""
        out = PrepareStoreOutput(
            keys_to_store=[],
            store_spec=CPULoadStoreSpec([]),
            evicted_keys=[],
        )
        assert out.keys_to_store == []
        assert isinstance(out.store_spec, LoadStoreSpec)
        assert out.evicted_keys == []

    def test_proactive_eviction_field_exists(self):
        """evicted_keys is the proactive-eviction return path (Trap C anchor)."""
        ek = [make_offload_key(b"\x00" * 28, 0)]
        out = PrepareStoreOutput(
            keys_to_store=[],
            store_spec=CPULoadStoreSpec([]),
            evicted_keys=ek,
        )
        assert out.evicted_keys == ek


# ---------------------------------------------------------------------------
# OffloadingEvent
# ---------------------------------------------------------------------------
class TestOffloadingEvent:
    def test_removed_true_means_eviction(self):
        """removed=True is the eviction signal for prom-metrics."""
        evt = OffloadingEvent(keys=[], medium="CPU", removed=True)
        assert evt.removed is True

    def test_removed_false_means_store(self):
        """removed=False is the new-store signal."""
        evt = OffloadingEvent(keys=[], medium="CPU", removed=False)
        assert evt.removed is False


# ---------------------------------------------------------------------------
# ReqContext
# ---------------------------------------------------------------------------
class TestReqContext:
    def test_default_kv_transfer_params_none(self):
        """Default kv_transfer_params is None (no cross-process metadata)."""
        ctx = ReqContext()
        assert ctx.kv_transfer_params is None

    def test_kv_transfer_params_passthrough(self):
        """LMCache/Mooncake attach do_remote_decode flags here."""
        ctx = ReqContext(kv_transfer_params={"do_remote_decode": True})
        assert ctx.kv_transfer_params == {"do_remote_decode": True}


# ---------------------------------------------------------------------------
# OffloadingSpec ABC + CPUOffloadingSpec
# ---------------------------------------------------------------------------
class TestOffloadingSpec:
    def test_block_size_divisibility_required(self):
        """gpu_block_size must be divisible by hash_block_size (hybrid models)."""
        with pytest.raises(AssertionError):
            CPUOffloadingSpec(
                hash_block_size=16,
                gpu_block_size=(15,),  # not divisible by 16
                kv_bytes_per_block=1024,
                cpu_bytes_to_use=1024 * 100,
            )

    def test_num_blocks_calculation(self):
        """num_blocks = cpu_bytes // (kv_bytes_per_block * block_size_factor)."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 100,
        )
        assert spec.num_blocks == 100

    def test_num_blocks_with_factor(self):
        """block_size_factor scales the per-block byte count."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 100,
            block_size_factor=2,
        )
        # 100 * 1024 / (1024 * 2) = 50
        assert spec.num_blocks == 50

    def test_zero_kv_bytes_yields_zero_blocks(self):
        """Edge: kv_bytes_per_block=0 → num_blocks=0 (defensive)."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=0,
            cpu_bytes_to_use=1024 * 100,
        )
        assert spec.num_blocks == 0

    def test_get_manager_returns_cpu_offloading_manager(self):
        """get_manager builds a CPUOffloadingManager wired to the policy."""
        from implementation.offload_manager import CPUOffloadingManager
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
            eviction_policy="lru",
        )
        mgr = spec.get_manager()
        assert isinstance(mgr, CPUOffloadingManager)

    def test_get_manager_idempotent(self):
        """get_manager caches the manager instance."""
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
        )
        m1 = spec.get_manager()
        m2 = spec.get_manager()
        assert m1 is m2

    def test_store_threshold_wraps_in_filter(self):
        """When store_threshold >= 2, get_manager wraps in FilterReusedOffloadingManager."""
        from implementation.reuse_manager import FilterReusedOffloadingManager
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
            store_threshold=3,
        )
        mgr = spec.get_manager()
        assert isinstance(mgr, FilterReusedOffloadingManager)

    def test_store_threshold_below_2_unwrapped(self):
        """store_threshold < 2 means no filtering; bare manager returned."""
        from implementation.reuse_manager import FilterReusedOffloadingManager
        spec = CPUOffloadingSpec(
            hash_block_size=16,
            gpu_block_size=(16,),
            kv_bytes_per_block=1024,
            cpu_bytes_to_use=1024 * 50,
            store_threshold=0,
        )
        mgr = spec.get_manager()
        assert not isinstance(mgr, FilterReusedOffloadingManager)


# ---------------------------------------------------------------------------
# Constants — verify the exact values used in demos
# ---------------------------------------------------------------------------
class TestSpecConstants:
    def test_hbm_capacity_80(self):
        """H100 80 GB SKU — used by Demo 1 latency stair."""
        assert HBM_CAPACITY_GB == 80.0

    def test_hbm_bandwidth_3000(self):
        """HBM3 ~3 TB/s — Demo 1 numeric."""
        assert HBM3_BANDWIDTH_GB_PER_S == 3000.0

    def test_ddr5_capacity_512(self):
        """Typical 2-socket server DDR5 — Demo 1 numeric."""
        assert DDR5_CAPACITY_GB == 512.0

    def test_ddr5_bandwidth_96(self):
        """DDR5 96 GB/s aggregate — Demo 1 numeric."""
        assert DDR5_BANDWIDTH_GB_PER_S == 96.0

    def test_pcie_gen5_64(self):
        """PCIe Gen5 ×16 = 64 GB/s — Trap C anchor + Demo 4 base."""
        assert PCIE_GEN5_BANDWIDTH_GB_PER_S == 64.0

    def test_nvme_gen5_14(self):
        """NVMe Gen5 sequential ~14 GB/s — Demo 1 NVMe row."""
        assert NVME_GEN5_BANDWIDTH_GB_PER_S == 14.0

    def test_nvme_capacity_4000(self):
        """4 TB consumer SSD — Demo 1 NVMe row."""
        assert NVME_CAPACITY_GB == 4000.0

    def test_kv_block_bytes_16mb(self):
        """16 MB conservative KV block size — Trap C / Demo 3 anchor."""
        assert KV_BLOCK_BYTES == 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Canonical KV-cache wrappers
# ---------------------------------------------------------------------------
class TestCanonicalKVCaches:
    def test_canonical_tensor_holds_bytes_per_page(self):
        """CanonicalKVCacheTensor(tensor, page_size_bytes)."""
        ct = CanonicalKVCacheTensor(tensor=None, page_size_bytes=4096)
        assert ct.page_size_bytes == 4096

    def test_canonical_ref_holds_index(self):
        """CanonicalKVCacheRef indexes into tensors list."""
        ref = CanonicalKVCacheRef(tensor_idx=2, page_size_bytes=8192)
        assert ref.tensor_idx == 2
        assert ref.page_size_bytes == 8192

    def test_canonical_caches_aggregates(self):
        """CanonicalKVCaches bundles tensors + group_data_refs."""
        caches = CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(None, 4096)],
            group_data_refs=[[CanonicalKVCacheRef(0, 4096)]],
        )
        assert len(caches.tensors) == 1
        assert len(caches.group_data_refs) == 1
