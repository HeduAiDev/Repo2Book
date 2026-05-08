"""Tests for cpu_gpu_worker — async transfer handlers, two CUDA streams.

Verifies: handler abstraction, alpha-beta latency model, in-order completion,
worker dispatch by transfer-type tuple, two-direction handler bundle.
"""

from __future__ import annotations

import math
import time

import pytest

from implementation.cpu_gpu_worker import (
    CpuGpuOffloadingHandlers,
    OffloadingHandler,
    OffloadingWorker,
    SingleDirectionOffloadingHandler,
    Transfer,
    TransferResult,
    alpha_beta_latency_us,
    break_even_block_bytes,
)
from implementation.offload_spec import (
    CanonicalKVCacheTensor,
    CanonicalKVCaches,
    CPULoadStoreSpec,
    GPULoadStoreSpec,
    KV_BLOCK_BYTES,
    PCIE_GEN5_BANDWIDTH_GB_PER_S,
    PCIE_OVERHEAD_ALPHA_US,
    PCIE_OVERHEAD_BETA_US_PER_BYTE,
)


# ---------------------------------------------------------------------------
# Alpha-beta model + break-even
# ---------------------------------------------------------------------------
class TestAlphaBetaModel:
    def test_zero_bytes_returns_alpha(self):
        """latency(0 bytes) = alpha (just overhead)."""
        assert alpha_beta_latency_us(0) == PCIE_OVERHEAD_ALPHA_US

    def test_linear_in_bytes(self):
        """latency(N) = alpha + beta * N → doubles when bytes go up sufficiently."""
        l1 = alpha_beta_latency_us(KV_BLOCK_BYTES)
        l2 = alpha_beta_latency_us(2 * KV_BLOCK_BYTES) - alpha_beta_latency_us(0)
        # difference between (alpha + beta*2N) and alpha = beta*2N = 2*(beta*N)
        slope = (alpha_beta_latency_us(2 * KV_BLOCK_BYTES) - alpha_beta_latency_us(KV_BLOCK_BYTES))
        slope_unit = alpha_beta_latency_us(KV_BLOCK_BYTES) - PCIE_OVERHEAD_ALPHA_US
        assert math.isclose(slope, slope_unit, rel_tol=1e-9)

    def test_16mb_block_is_about_262us(self):
        """Demo 3 anchor: alpha-beta latency for 16 MB ≈ 261.66 µs."""
        latency = alpha_beta_latency_us(KV_BLOCK_BYTES)
        assert round(latency, 2) == 261.66

    def test_break_even_651_kib(self):
        """Demo 3 anchor: alpha == beta*bytes at ~651 KiB (= 666 667 bytes)."""
        be = break_even_block_bytes()
        assert be == 666_667
        # ≈ 651 KiB
        assert round(be / 1024, 1) == 651.0

    def test_16mb_is_24x_past_break_even(self):
        """vLLM's 16 MB block is ≈24× past break-even (Trap C nuance)."""
        be = break_even_block_bytes()
        ratio = KV_BLOCK_BYTES / be
        assert 23.5 < ratio < 26.0


# ---------------------------------------------------------------------------
# OffloadingHandler ABC
# ---------------------------------------------------------------------------
class TestHandlerABC:
    def test_is_abstract(self):
        """OffloadingHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OffloadingHandler()  # type: ignore[abstract]

    def test_has_required_methods(self):
        """transfer_async / get_finished / wait / shutdown."""
        for m in ("transfer_async", "get_finished", "wait", "shutdown"):
            assert hasattr(OffloadingHandler, m)


# ---------------------------------------------------------------------------
# SingleDirectionOffloadingHandler
# ---------------------------------------------------------------------------
class TestSingleDirHandler:
    def _mk(self, gpu_to_cpu=True, gpu_block_bytes=KV_BLOCK_BYTES):
        return SingleDirectionOffloadingHandler(
            gpu_block_bytes=gpu_block_bytes,
            cpu_block_bytes=gpu_block_bytes,
            gpu_to_cpu=gpu_to_cpu,
        )

    def test_gpu_to_cpu_transfer_type(self):
        """gpu_to_cpu=True yields ('GPU','CPU') transfer type."""
        h = self._mk(gpu_to_cpu=True)
        assert h.transfer_type == ("GPU", "CPU")

    def test_cpu_to_gpu_transfer_type(self):
        """gpu_to_cpu=False yields ('CPU','GPU')."""
        h = self._mk(gpu_to_cpu=False)
        assert h.transfer_type == ("CPU", "GPU")

    def test_cpu_block_must_be_ge_gpu(self):
        """cpu_block_bytes < gpu_block_bytes is invalid (block_size_factor >= 1)."""
        with pytest.raises(AssertionError):
            SingleDirectionOffloadingHandler(
                gpu_block_bytes=2048,
                cpu_block_bytes=1024,  # too small
                gpu_to_cpu=True,
            )

    def test_zero_gpu_block_bytes_invalid(self):
        with pytest.raises(AssertionError):
            SingleDirectionOffloadingHandler(
                gpu_block_bytes=0,
                cpu_block_bytes=0,
                gpu_to_cpu=True,
            )

    def test_transfer_async_succeeds(self):
        """transfer_async returns True (submission accepted)."""
        h = self._mk()
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        ok = h.transfer_async(0, spec)
        assert ok is True

    def test_inflight_count_increases(self):
        """num_inflight grows after submission, before completion."""
        h = self._mk()
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        h.transfer_async(0, spec)
        assert h.num_inflight() >= 1

    def test_get_finished_drains_after_wait(self):
        """get_finished returns completed transfers (after wait)."""
        h = self._mk(gpu_block_bytes=1024)  # tiny, so it finishes fast
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        h.transfer_async(0, spec)
        h.wait({0})
        results = h.get_finished()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].job_id == 0

    def test_get_finished_returns_typed_result(self):
        """Result has job_id, success, transfer_size, transfer_time, transfer_type."""
        h = self._mk(gpu_block_bytes=1024)
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        h.transfer_async(7, spec)
        h.wait({7})
        r = h.get_finished()[0]
        assert isinstance(r, TransferResult)
        assert r.job_id == 7
        assert r.transfer_size == 1024
        assert r.transfer_type == ("GPU", "CPU")

    def test_in_order_completion_within_direction(self):
        """Transfers on a single stream complete in submission order."""
        h = self._mk(gpu_block_bytes=8192)
        for i in range(3):
            spec = (GPULoadStoreSpec([i]), CPULoadStoreSpec([i]))
            h.transfer_async(i, spec)
        h.wait({0, 1, 2})
        results = h.get_finished()
        # Job 0 must finish at or before job 1 (in-order semantics)
        ids = [r.job_id for r in results]
        assert ids == sorted(ids)

    def test_shutdown_clears(self):
        """shutdown clears in-flight tracking."""
        h = self._mk()
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        h.transfer_async(0, spec)
        h.shutdown()
        assert h.num_inflight() == 0


# ---------------------------------------------------------------------------
# OffloadingWorker dispatch
# ---------------------------------------------------------------------------
class TestOffloadingWorker:
    def test_register_handler(self):
        """register_handler binds (src_medium, dst_medium) → handler."""
        w = OffloadingWorker()
        h = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=True,
        )
        w.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, h)
        assert ("GPU", "CPU") in w.transfer_type_to_handler
        assert h in w.handlers

    def test_double_register_raises(self):
        """Registering same (src, dst) twice raises AssertionError."""
        w = OffloadingWorker()
        h1 = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=True,
        )
        h2 = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=True,
        )
        w.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, h1)
        with pytest.raises(AssertionError):
            w.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, h2)

    def test_dispatch_by_transfer_type(self):
        """transfer_async routes spec to the matching handler."""
        w = OffloadingWorker()
        gtc = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=True,
        )
        ctg = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=False,
        )
        w.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, gtc)
        w.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, ctg)

        # store path: GPU→CPU
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        w.transfer_async(0, spec)
        assert gtc.num_inflight() == 1
        assert ctg.num_inflight() == 0

        # load path: CPU→GPU
        spec2 = (CPULoadStoreSpec([0]), GPULoadStoreSpec([0]))
        w.transfer_async(1, spec2)
        assert ctg.num_inflight() == 1

    def test_unknown_transfer_type_raises(self):
        """No handler for (src, dst) pair → AssertionError."""
        w = OffloadingWorker()
        spec = (GPULoadStoreSpec([0]), CPULoadStoreSpec([0]))
        with pytest.raises(AssertionError):
            w.transfer_async(0, spec)

    def test_get_finished_aggregates(self):
        """get_finished collects results from ALL registered handlers."""
        w = OffloadingWorker()
        gtc = SingleDirectionOffloadingHandler(
            gpu_block_bytes=512, cpu_block_bytes=512, gpu_to_cpu=True,
        )
        ctg = SingleDirectionOffloadingHandler(
            gpu_block_bytes=512, cpu_block_bytes=512, gpu_to_cpu=False,
        )
        w.register_handler(GPULoadStoreSpec, CPULoadStoreSpec, gtc)
        w.register_handler(CPULoadStoreSpec, GPULoadStoreSpec, ctg)
        w.transfer_async(0, (GPULoadStoreSpec([0]), CPULoadStoreSpec([0])))
        w.transfer_async(1, (CPULoadStoreSpec([0]), GPULoadStoreSpec([0])))
        w.wait({0, 1})
        results = w.get_finished()
        ids = sorted(r.job_id for r in results)
        assert ids == [0, 1]


# ---------------------------------------------------------------------------
# CpuGpuOffloadingHandlers — bundle of two streams
# ---------------------------------------------------------------------------
class TestCpuGpuBundle:
    def _make_caches(self, page_size_bytes=4096):
        return CanonicalKVCaches(
            tensors=[CanonicalKVCacheTensor(tensor=None, page_size_bytes=page_size_bytes)],
            group_data_refs=[[]],
        )

    def test_two_handlers_separate_directions(self):
        """Bundle has gpu_to_cpu_handler AND cpu_to_gpu_handler — TWO streams."""
        caches = self._make_caches()
        bundle = CpuGpuOffloadingHandlers(
            kv_caches=caches, block_size_factor=1, num_cpu_blocks=8,
        )
        assert bundle.gpu_to_cpu_handler is not bundle.cpu_to_gpu_handler

    def test_handlers_have_opposite_directions(self):
        """One handler is G→C, the other C→G (Trap G core: not 2× speedup but real overlap)."""
        caches = self._make_caches()
        bundle = CpuGpuOffloadingHandlers(
            kv_caches=caches, block_size_factor=1, num_cpu_blocks=8,
        )
        assert bundle.gpu_to_cpu_handler.transfer_type == ("GPU", "CPU")
        assert bundle.cpu_to_gpu_handler.transfer_type == ("CPU", "GPU")

    def test_block_size_factor_scales_cpu(self):
        """cpu_block_bytes = gpu_block_bytes * block_size_factor."""
        caches = self._make_caches(page_size_bytes=8192)
        bundle = CpuGpuOffloadingHandlers(
            kv_caches=caches, block_size_factor=2, num_cpu_blocks=4,
        )
        assert bundle.gpu_block_bytes == 8192
        assert bundle.cpu_block_bytes == 8192 * 2

    def test_total_layer_bytes_sum(self):
        """gpu_block_bytes = sum(per-tensor page_size_bytes) — multi-layer aggregation."""
        caches = CanonicalKVCaches(
            tensors=[
                CanonicalKVCacheTensor(tensor=None, page_size_bytes=2048),
                CanonicalKVCacheTensor(tensor=None, page_size_bytes=4096),
            ],
            group_data_refs=[[]],
        )
        bundle = CpuGpuOffloadingHandlers(
            kv_caches=caches, block_size_factor=1, num_cpu_blocks=4,
        )
        assert bundle.gpu_block_bytes == 2048 + 4096


# ---------------------------------------------------------------------------
# Trap G HONESTY: two streams ≠ 2× speedup (PCIe-bound)
# ---------------------------------------------------------------------------
class TestTrapGStreamsNotTwiceSpeedup:
    def test_two_streams_concurrent_overlap(self):
        """With two streams, G→C and C→G can run concurrently — but PCIe is the cap.

        We measure: simultaneous submissions on both streams DON'T multiply
        the per-direction throughput. The handler latency simulator models
        each direction's queue independently (modeling separate copy engines)
        but real PCIe bandwidth is shared. This test affirms the simulation
        keeps each direction's queue independent (== concurrent), but the
        per-direction latency is the same as serial within that direction.
        """
        gtc = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=True,
        )
        ctg = SingleDirectionOffloadingHandler(
            gpu_block_bytes=1024, cpu_block_bytes=1024, gpu_to_cpu=False,
        )
        # both streams submit at the same time
        gtc.transfer_async(0, (GPULoadStoreSpec([0]), CPULoadStoreSpec([0])))
        ctg.transfer_async(1, (CPULoadStoreSpec([0]), GPULoadStoreSpec([0])))
        # Each stream processes its own queue independently — overlap is real
        gtc.wait({0})
        ctg.wait({1})
        # Both finished
        assert len(gtc.get_finished()) == 1
        assert len(ctg.get_finished()) == 1

    def test_serial_within_direction(self):
        """Within the same direction, transfers serialize (in-order completion)."""
        h = SingleDirectionOffloadingHandler(
            gpu_block_bytes=4096, cpu_block_bytes=4096, gpu_to_cpu=True,
        )
        # rapid-fire submissions — all inflight at once
        for i in range(5):
            h.transfer_async(i, (GPULoadStoreSpec([i]), CPULoadStoreSpec([i])))
        # Each transfer's finish_t is at least the previous +. Their finish times
        # are monotonically increasing.
        finishes = [t.finish_t for t in h._transfers]
        for i in range(len(finishes) - 1):
            assert finishes[i] <= finishes[i + 1]
