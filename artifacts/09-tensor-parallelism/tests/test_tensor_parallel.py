"""Tests — Ch8 Tensor Parallelism."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import pytest
from implementation.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear,
    TPTransformerBlock, tp_communication_analysis,
)


class TestColumnParallelLinear:
    def test_output_shape(self):
        m = ColumnParallelLinear(64, 128, tp_size=4, tp_rank=0, gather_output=True)
        x = torch.randn(2, 8, 64)
        out, _ = m(x)
        # 128/4 = 32 per rank (gather simulated, so each rank sees its shard)
        assert out.shape == (2, 8, 32)

    def test_gather_disabled(self):
        m = ColumnParallelLinear(64, 128, tp_size=4, tp_rank=0, gather_output=False)
        x = torch.randn(2, 8, 64)
        out, _ = m(x)
        assert out.shape == (2, 8, 32)  # Same — each rank keeps its shard

    def test_weight_shard_size(self):
        m = ColumnParallelLinear(64, 128, tp_size=4, tp_rank=2)
        assert m.weight.shape == (32, 64)  # [out/tp, in] = [128/4, 64]

    def test_different_ranks_different_weights(self):
        m0 = ColumnParallelLinear(32, 64, tp_size=2, tp_rank=0)
        m1 = ColumnParallelLinear(32, 64, tp_size=2, tp_rank=1)
        # Weights should be different (each rank has its own shard)
        assert not torch.allclose(m0.weight, m1.weight)


class TestRowParallelLinear:
    def test_output_shape(self):
        m = RowParallelLinear(128, 64, tp_size=4, tp_rank=0,
                              input_is_parallel=True)
        x = torch.randn(2, 8, 32)  # input is already partitioned: 128/4=32
        out = m(x)
        # F.linear([B,L,32], [64,32]^T) = [B,L,64] ✓
        assert out.shape == (2, 8, 64)

    def test_input_is_parallel_auto_splits(self):
        m = RowParallelLinear(128, 64, tp_size=4, tp_rank=0,
                              input_is_parallel=False)
        x = torch.randn(2, 8, 128)  # Full input → auto-split to [2, 8, 32]
        out = m(x)
        assert out.shape == (2, 8, 64)  # All-reduce restores full output

    def test_weight_shard_size(self):
        m = RowParallelLinear(128, 64, tp_size=4, tp_rank=0)
        assert m.weight.shape == (64, 32)  # [out, in/tp] = [64, 128/4]


class TestTPTransformerBlock:
    def test_forward(self):
        block = TPTransformerBlock(d_model=256, num_heads=8, tp_size=4, tp_rank=0)
        x = torch.randn(2, 16, 256)
        y = block(x)
        assert y.shape == x.shape  # Output shape = input shape

    def test_no_nan(self):
        block = TPTransformerBlock(d_model=256, num_heads=8, tp_size=4, tp_rank=0)
        x = torch.randn(2, 16, 256)
        y = block(x)
        assert not torch.isnan(y).any()


class TestCommunicationAnalysis:
    def test_basic(self):
        config = dict(d_model=4096, tp_size=8, num_layers=32,
                      seq_len=2048, batch_size=4, dtype_bytes=2)
        result = tp_communication_analysis(config)
        assert result["num_all_reduces_per_forward"] > 0
        assert result["total_communication_bytes"] > 0
        # For 32 layers: 1 (embedding) + 32×2 (per-layer) = 65 all-reduces
        assert result["num_all_reduces_per_forward"] == 65


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
