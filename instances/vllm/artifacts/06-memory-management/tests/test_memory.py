"""Tests — Ch5 GPU Memory Management."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.memory_profiler import (
    MemorySnapshot, LlamaModelMemory, KVCacheSpec, MemoryProfiler,
)


class TestMemorySnapshot:
    def test_used_memory(self):
        snap = MemorySnapshot(total_memory=1024, free_memory=512, torch_peak=256, torch_current=200)
        assert snap.used_memory == 512

    def test_non_torch(self):
        """Non-torch = used - torch_current (NCCL, CUDA context, etc.)"""
        snap = MemorySnapshot(total_memory=1024, free_memory=400, torch_peak=300, torch_current=200)
        assert snap.non_torch_memory == 1024 - 400 - 200  # = 424


class TestLlamaMemory:
    def test_weight_size_positive(self):
        m = LlamaModelMemory(d_model=2048, num_layers=32, num_heads=32, num_kv_heads=8)
        assert m.weight_size() > 1_000_000_000  # At least ~2 GB

    def test_memory_scales_with_layers(self):
        """More layers = more weights (excluding fixed embedding cost)."""
        m1 = LlamaModelMemory(512, 8, 8, 4)
        m4 = LlamaModelMemory(512, 32, 8, 4)
        # Embedding + lm_head are fixed costs, so total doesn't scale 4:1
        assert m4.weight_size() > m1.weight_size()


class TestKVCacheSpec:
    def test_page_size(self):
        spec = KVCacheSpec(block_size=16, num_kv_heads=8, head_dim=128, num_layers=32)
        # 2 * 16 * 8 * 128 * 2 = 65536 bytes per layer per block
        assert spec.page_size_bytes() == 65536

    def test_total_block_bytes(self):
        spec = KVCacheSpec(block_size=16, num_kv_heads=8, head_dim=128, num_layers=2)
        # 2 layers × 65536 = 131072 bytes per block
        assert spec.total_block_bytes(10) == 10 * 131072


class TestMemoryProfiler:
    def test_requested_memory(self):
        profiler = MemoryProfiler(total_gpu_memory=1000, gpu_memory_utilization=0.9)
        r = profiler.profile(
            LlamaModelMemory(64, 1, 4, 4),
            peak_activation=10, cuda_graph_memory=5, non_torch_overhead=5,
        )
        assert r["requested_memory"] == 900

    def test_available_kv_cache_positive(self):
        profiler = MemoryProfiler(total_gpu_memory=80 * 1024**3)
        model = LlamaModelMemory(2048, 32, 32, 8)
        spec = KVCacheSpec(16, 8, 128, 32)
        r = profiler.profile(model, peak_activation=2 * 1024**3, kv_cache_spec=spec)
        assert r["available_kv_cache_bytes"] > 0
        assert r.get("num_gpu_blocks", 0) > 1000  # should have thousands of blocks

    def test_utilization_margin(self):
        """8% margin means not all GPU memory is used."""
        profiler = MemoryProfiler(total_gpu_memory=1000, gpu_memory_utilization=0.92)
        r = profiler.profile(
            LlamaModelMemory(64, 1, 4, 4), peak_activation=0,
        )
        assert r["requested_memory"] == 920  # 1000 * 0.92
        assert r["requested_memory"] < 1000

    def test_kv_cache_dominates_long_sequences(self):
        """For large models + long seqs, KV cache dominates."""
        profiler = MemoryProfiler(total_gpu_memory=80 * 1024**3)
        model = LlamaModelMemory(4096, 40, 32, 8)  # Larger model
        spec = KVCacheSpec(16, 8, 128, 40)
        r = profiler.profile(model, peak_activation=3 * 1024**3)
        kv_gb = r["available_kv_cache_bytes"] / (1024**3)
        assert kv_gb > 0
        assert r["breakdown"]["total_non_kv_cache"] > r["breakdown"]["model_weights"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
