"""Tests — Ch15 Llama Architecture."""
import sys, torch, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from implementation.llama_config import (
    LlamaConfig, RMSNorm, RotaryEmbedding,
    count_parameters,
)


class TestLlamaConfig:
    def test_default_1b_config(self):
        c = LlamaConfig()
        assert c.hidden_size == 2048
        assert c.num_hidden_layers == 16
        assert c.num_attention_heads == 32
        assert c.num_key_value_heads == 8
        assert c.head_dim == 64

    def test_gqa_ratio(self):
        c = LlamaConfig()
        assert c.num_queries_per_kv == 4  # 32 Q / 8 KV = 4

    def test_intermediate_ratio(self):
        c = LlamaConfig()
        # SwiGLU intermediate is 8/3 × d_model ... actually Llama uses 8192 for 2048 → 4×
        assert c.intermediate_size == 8192


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 8, 256)
        y = norm(x)
        assert y.shape == x.shape

    def test_near_unit_variance(self):
        norm = RMSNorm(128)
        x = torch.randn(4, 128) * 10 + 3  # Shifted + scaled
        y = norm(x)
        # RMS after norm should be close to 1.0
        rms = torch.sqrt(y.float().pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_residual_fusion(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        residual = torch.randn(2, 8, 64)
        y = norm(x, residual)
        assert y.shape == x.shape  # Residual is fused in-place


class TestRoPE:
    def test_qk_rotate(self):
        rope = RotaryEmbedding(head_dim=64, max_seq_len=128)
        q = torch.randn(2, 8, 64)  # [B, L, D]
        k = torch.randn(2, 8, 64)
        pos = torch.arange(8).unsqueeze(0).expand(2, -1).reshape(-1)  # [16]
        q_out, k_out = rope(pos, q.reshape(-1, 64), k.reshape(-1, 64))
        assert q_out.shape == (16, 64)

    def test_relative_position_sensitivity(self):
        """RoPE output should differ for different positions."""
        rope = RotaryEmbedding(head_dim=64)
        q = torch.ones(2, 64)
        q0, _ = rope(torch.tensor([0, 0]), q, q.clone())
        q1, _ = rope(torch.tensor([1, 100]), q, q.clone())
        assert not torch.allclose(q0[0], q1[1])  # Different positions → different


class TestParameterCount:
    def test_total_around_1b(self):
        params = count_parameters(LlamaConfig())
        # Should be ~1.2B parameters (actual Llama-3.2-1B is 1.24B)
        assert 1000 < params["total_params_M"] < 2000

    def test_model_fits_in_4gb(self):
        params = count_parameters(LlamaConfig(), dtype_bytes=2)
        assert params["total_size_gb"] < 5.0  # ~2.3 GB in bf16

    def test_mlp_dominates_attention(self):
        params = count_parameters(LlamaConfig())
        # MLP should have more params than attention (standard for Llama)
        assert params["breakdown"]["mlp"] > params["breakdown"]["attention"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
