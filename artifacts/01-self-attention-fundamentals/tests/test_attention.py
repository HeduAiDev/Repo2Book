"""
Tests for Chapter 1 — Self-Attention 算子深度解析.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import pytest
from implementation.reference_attention import (
    scaled_dot_product_attention, MultiHeadAttention,
    GroupedQueryAttention, create_causal_mask,
    create_padding_mask, create_sliding_window_mask,
)


class TestScaledDotProductAttention:
    def test_basic_shape(self):
        Q = torch.randn(2, 10, 64)
        K = torch.randn(2, 10, 64)
        V = torch.randn(2, 10, 64)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 10, 64)

    def test_causal_mask_correctness(self):
        Q = K = V = torch.randn(1, 4, 8)
        mask = create_causal_mask(4)
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        assert not torch.isnan(out).any()

    def test_no_nan_with_large_values(self):
        Q = torch.randn(1, 32, 128) * 10
        K = torch.randn(1, 32, 128) * 10
        V = torch.randn(1, 32, 128) * 10
        out = scaled_dot_product_attention(Q, K, V)
        assert not torch.isnan(out).any()


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        out, attn = mha(x)
        assert out.shape == (2, 8, 64)
        assert attn.shape == (2, 4, 8, 8)

    def test_attention_sums_to_one(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 6, 64)
        _, attn = mha(x)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(1, 4, 6), atol=1e-5)

    def test_causal_mask(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(1, 6, 64)
        mask = create_causal_mask(6)
        _, attn = mha(x, attention_mask=mask)
        # Upper triangle should all be 0
        for i in range(6):
            for j in range(i + 1, 6):
                assert torch.all(attn[:, :, i, j] == 0.0)

    def test_scale_factor(self):
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        expected = 1.0 / math.sqrt(16)  # 64/4 = 16
        assert abs(mha.scale - expected) < 1e-6


class TestGroupedQueryAttention:
    def test_output_shape(self):
        gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 8, 64)
        out, attn = gqa(x)
        assert out.shape == (2, 8, 64)
        assert attn.shape == (2, 4, 8, 8)

    def test_equals_mha_when_kv_heads_equal(self):
        torch.manual_seed(42)
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        torch.manual_seed(42)
        gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=4)
        x = torch.randn(2, 8, 64)
        out_mha, _ = mha(x)
        out_gqa, _ = gqa(x)
        assert torch.allclose(out_mha, out_gqa, atol=1e-5)


class TestAttentionMasks:
    def test_causal_mask_upper_tri_zero(self):
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[0, 0, i, j] == 0

    def test_padding_mask(self):
        lengths = torch.tensor([3, 5])
        mask = create_padding_mask(lengths, 6)
        assert mask[0, 0, 0, 0:3].all()
        assert not mask[0, 0, 0, 3:].any()
        assert mask[1, 0, 0, 0:5].all()
        assert not mask[1, 0, 0, 5:].any()

    def test_sliding_window(self):
        mask = create_sliding_window_mask(6, window_size=3)
        # position 0: only sees itself (dist 0 < 3, but j > i is causal violation)
        # position 5: sees positions 3,4,5 (dist = 2,1,0)
        for i in range(6):
            for j in range(6):
                dist = i - j
                if 0 <= dist < 3:
                    assert mask[0, 0, i, j] == 1, f"({i},{j}) dist={dist} should be visible"
                else:
                    assert mask[0, 0, i, j] == 0, f"({i},{j}) dist={dist} should be masked"


class TestVarianceAnalysis:
    def test_import(self):
        from implementation import variance_analysis
        r = variance_analysis.analyze_variance_empirically(64, num_samples=500)
        assert abs(r["empirical_variance_scaled"] - 1.0) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
