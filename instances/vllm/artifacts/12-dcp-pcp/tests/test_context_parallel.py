"""Tests — Ch11 DCP/PCP."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.context_parallel import (
    analyze_kv_cache_replication, simulate_dcp_attention,
    lse_weighted_merge, recommend_cp_config,
)


class TestKVCacheReplication:
    def test_deepseek_style_replication(self):
        """1 KV head × TP=8 = 8× replication."""
        r = analyze_kv_cache_replication(8, 1, seq_len=4096, head_dim=128, num_layers=32)
        assert r["replication_factor"] == 8.0
        assert r["optimal_dcp_size"] == 8

    def test_no_replication_when_kv_heads_match_tp(self):
        """8 KV heads × TP=8 = no replication."""
        r = analyze_kv_cache_replication(8, 8, seq_len=4096, head_dim=128, num_layers=32)
        assert r["replication_factor"] == 1.0

    def test_wasted_memory_scales_with_replication(self):
        r2x = analyze_kv_cache_replication(8, 4, seq_len=4096, head_dim=128, num_layers=32)
        r4x = analyze_kv_cache_replication(8, 2, seq_len=4096, head_dim=128, num_layers=32)
        assert r4x["total_wasted_gb"] > r2x["total_wasted_gb"]


class TestDCPAttention:
    def test_local_seq_lens_sum_to_total(self):
        s = simulate_dcp_attention(4096, 32, 128, dcp_size=4, cp_interleave=1)
        assert sum(s["local_seq_lens"]) == 4096

    def test_interleave_balance(self):
        """Token-level interleave should be nearly perfectly balanced."""
        s = simulate_dcp_attention(4096, 32, 128, dcp_size=4, cp_interleave=1)
        assert s["imbalance_ratio"] <= 1.01

    def test_block_interleave(self):
        s = simulate_dcp_attention(4096, 32, 128, dcp_size=4, cp_interleave=16)
        assert sum(s["local_seq_lens"]) == 4096


class TestLSEMerge:
    def test_merge_two_ranks(self):
        """Merging equal partials should give same output."""
        out0 = torch.ones(2, 4, 8)
        out1 = torch.ones(2, 4, 8) * 2
        lse0 = torch.zeros(2, 4)
        lse1 = torch.zeros(2, 4)

        merged = lse_weighted_merge([out0, out1], [lse0, lse1])
        # With equal LSE, weights = [0.5, 0.5]
        assert merged.shape == (2, 4, 8)

    def test_dominant_lse(self):
        """Much larger LSE → that rank's output dominates."""
        out0 = torch.ones(2, 4, 8)
        out1 = torch.ones(2, 4, 8) * 100
        lse0 = torch.zeros(2, 4)
        lse1 = torch.ones(2, 4) * 100  # Much larger LSE

        merged = lse_weighted_merge([out0, out1], [lse0, lse1])
        # Rank 1 should dominate → output ≈ 100
        assert torch.allclose(merged, out1, atol=1e-3)


class TestRecommendation:
    def test_recommends_dcp_when_replication(self):
        r = recommend_cp_config("DeepSeek-R1", 1, 8, max_seq_len=32768,
                                head_dim=128, num_layers=64)
        assert r["recommended_dcp"] > 1

    def test_no_dcp_when_no_replication(self):
        r = recommend_cp_config("Llama-70B", 8, 8, max_seq_len=32768,
                                head_dim=128, num_layers=80)
        assert r["recommended_dcp"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
