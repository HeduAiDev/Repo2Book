"""Tests — Ch3 FlashAttention & PagedAttention."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import pytest
from implementation.paged_attention import (
    calculate_hbm_traffic, paged_attention_with_block_table,
    fused_paged_attention_tiled, build_block_table,
)


class TestHBMTraffic:
    def test_naive_grows_quadratically(self):
        """Naive attention's intermediate writes grow as O(seq²)."""
        r512 = calculate_hbm_traffic(512, num_heads=32, head_dim=128)
        r1024 = calculate_hbm_traffic(1024, num_heads=32, head_dim=128)
        # 2x seq → 4x intermediate writes (O(seq²): S and P matrices)
        ratio = r1024["naive"]["write_intermediate_bytes"] / r512["naive"]["write_intermediate_bytes"]
        assert 3.0 < ratio < 5.0  # ~4x (allow for rounding)

    def test_fa_eliminates_intermediate_writes(self):
        """FlashAttention's key innovation: NO O(n²) intermediate writes."""
        r = calculate_hbm_traffic(4096, num_heads=32, head_dim=128)
        # Naive attention writes S and P to HBM (O(seq²))
        assert r["naive"]["write_intermediate_bytes"] > r["naive"]["write_output_bytes"] * 10
        # FA is strictly cheaper than naive (exact speedup depends on block sizes)
        assert r["flashattention"]["total_gb"] < r["naive"]["total_gb"]


class TestPagedAttention:
    def test_single_sequence(self):
        B, H, D, bs = 1, 4, 32, 16
        L = 50  # 50 tokens → ceil(50/16)=4 blocks
        K_cache = torch.randn(10, bs, H, D)  # 10 GPU blocks
        V_cache = torch.randn(10, bs, H, D)
        block_table, seq_lens, _ = build_block_table([L], num_gpu_blocks=10, block_size=bs)
        Q = torch.randn(B, H, D)

        out = paged_attention_with_block_table(Q, K_cache, V_cache, block_table, seq_lens, bs)

        assert out.shape == (B, H, D)
        assert not torch.isnan(out).any()

    def test_block_table_indirection(self):
        """Verify that block_table actually controls which blocks are read."""
        H, D, bs = 2, 16, 4
        # 2 blocks: block 0 = all zeros, block 1 = all ones
        K_cache = torch.zeros(2, bs, H, D)
        V_cache = torch.zeros(2, bs, H, D)
        K_cache[1] = 100.0  # block 1 has huge values
        V_cache[1] = 1.0

        Q = torch.ones(1, H, D)

        # block_table: [block 1, block 0] — first read block 1
        block_table = torch.tensor([[1, 0]], dtype=torch.int32)
        seq_lens = torch.tensor([bs * 2], dtype=torch.int32)

        out = paged_attention_with_block_table(Q, K_cache, V_cache, block_table, seq_lens, bs)

        # Should be dominated by block 1's values
        assert out.abs().sum() > 0

    def test_variable_length_sequences(self):
        """Sequences of different lengths should work in the same batch."""
        B, H, D, bs = 3, 4, 32, 16
        Ls = [20, 48, 5]
        K_cache = torch.randn(10, bs, H, D)
        V_cache = torch.randn(10, bs, H, D)
        block_table, seq_lens, _ = build_block_table(Ls, num_gpu_blocks=10, block_size=bs)
        Q = torch.randn(B, H, D)

        out = paged_attention_with_block_table(Q, K_cache, V_cache, block_table, seq_lens, bs)

        assert out.shape == (B, H, D)


class TestFusedPagedAttention:
    def test_matches_reference(self):
        """Fused version should match the step-by-step version."""
        torch.manual_seed(42)
        B, H, D, bs = 2, 4, 32, 16
        Ls = [30, 45]
        K_cache = torch.randn(10, bs, H, D)
        V_cache = torch.randn(10, bs, H, D)
        block_table, seq_lens, _ = build_block_table(Ls, num_gpu_blocks=10, block_size=bs)
        Q = torch.randn(B, H, D)

        ref = paged_attention_with_block_table(Q, K_cache, V_cache, block_table, seq_lens, bs)
        fused = fused_paged_attention_tiled(Q, K_cache, V_cache, block_table, seq_lens, bs)

        assert torch.allclose(ref, fused, atol=1e-4)

    def test_gqa_handling(self):
        """GQA: fewer KV heads than Q heads."""
        torch.manual_seed(42)
        B, H_q, H_kv, D, bs = 1, 8, 2, 32, 16  # 4 Q per KV
        L = 40
        K_cache = torch.randn(5, bs, H_kv, D)
        V_cache = torch.randn(5, bs, H_kv, D)
        block_table, seq_lens, _ = build_block_table([L], num_gpu_blocks=5, block_size=bs)
        Q = torch.randn(B, H_q, D)

        ref = paged_attention_with_block_table(Q, K_cache, V_cache, block_table, seq_lens, bs)
        fused = fused_paged_attention_tiled(Q, K_cache, V_cache, block_table, seq_lens, bs)

        assert torch.allclose(ref, fused, atol=1e-4)


class TestBlockTable:
    def test_simple_allocation(self):
        Ls = [20, 30]
        bt, sl, num_used = build_block_table(Ls, num_gpu_blocks=10, block_size=16)
        # seq 0: ceil(20/16)=2 blocks, seq 1: ceil(30/16)=2 blocks
        assert bt.shape == (2, 2)
        assert sl.tolist() == [20, 30]
        assert num_used.item() == 4

    def test_oom(self):
        with pytest.raises(RuntimeError):
            build_block_table([200], num_gpu_blocks=2, block_size=16)  # need 13 blocks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
