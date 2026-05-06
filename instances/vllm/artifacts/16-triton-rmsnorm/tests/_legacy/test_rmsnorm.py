"""
Tests for Triton RMSNorm Implementation.

Test levels:
    - unit: core logic correctness (vs PyTorch reference)
    - integration: cross-chapter compatibility
    - teaching: runnable examples for the book
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from rmsnorm import (
    rms_norm,
    rms_norm_ref,
    fused_add_rms_norm,
    fused_add_rms_norm_ref,
    HAS_TRITON,
)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
needs_triton = pytest.mark.skipif(
    not HAS_TRITON,
    reason="Triton not available",
)


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Basic RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N,D", [
    (1, 256),
    (1, 2048),
    (32, 2048),
    (128, 4096),
    (7, 1536),   # odd batch size
    (1, 8192),
])
@needs_cuda
@needs_triton
def test_rms_norm_correctness(N, D):
    """Triton RMSNorm must match pure-PyTorch reference within 0.01 tolerance."""
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)

    out_triton = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    max_err = (out_triton.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.01, f"Max error {max_err:.6f} exceeds 0.01 tolerance"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@needs_cuda
@needs_triton
def test_rms_norm_dtypes(dtype):
    """RMSNorm must work with both fp16 and fp32 inputs."""
    x = torch.randn(16, 2048, device='cuda', dtype=dtype)
    w = torch.randn(2048, device='cuda', dtype=dtype)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    # fp32 should be near-exact
    tol = 1e-4 if dtype == torch.float32 else 0.01
    max_err = (out.float() - out_ref.float()).abs().max().item()
    assert max_err < tol, f"dtype={dtype}: max error {max_err:.6f} exceeds {tol}"


@needs_cuda
@needs_triton
def test_rms_norm_unit_variance():
    """After RMSNorm, the output should have RMS ≈ 1 (before weight scaling).

    With weight=ones, RMSNorm should produce output with RMS ≈ 1.0.
    """
    x = torch.randn(128, 4096, device='cuda', dtype=torch.float32)
    w = torch.ones(4096, device='cuda', dtype=torch.float32)

    out = rms_norm(x, w)
    # RMS of output = sqrt(mean(out^2))
    rms_out = torch.sqrt((out ** 2).mean(dim=-1))

    # Should be close to 1.0
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=0.01), \
        f"RMS after normalization: {rms_out[:3].tolist()} (expected ~1.0)"


@needs_cuda
@needs_triton
def test_rms_norm_weight_scaling():
    """RMSNorm weight should scale the output proportionally."""
    D = 512
    x = torch.randn(32, D, device='cuda', dtype=torch.float32)
    w = torch.linspace(0.5, 2.0, D, device='cuda', dtype=torch.float32)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    max_err = (out - out_ref).abs().max().item()
    assert max_err < 1e-4, f"Max error {max_err:.6f} exceeds 1e-4"


@needs_cuda
@needs_triton
def test_rms_norm_3d_input():
    """RMSNorm must handle 3D inputs [batch, seq, hidden] correctly."""
    x = torch.randn(4, 32, 1024, device='cuda', dtype=torch.float16)
    w = torch.randn(1024, device='cuda', dtype=torch.float16)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    max_err = (out.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.01, f"3D max error {max_err:.6f} exceeds 0.01"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Fused Residual + RMSNorm
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("N,D", [
    (1, 2048),
    (32, 2048),
    (128, 4096),
])
@needs_cuda
@needs_triton
def test_fused_add_rmsnorm_correctness(N, D):
    """Fused residual+RMSNorm must match the reference."""
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    residual = torch.randn(N, D, device='cuda', dtype=torch.float16)

    w = torch.randn(D, device='cuda', dtype=torch.float16)
    out_fused, new_res = fused_add_rms_norm(x.clone(), residual, w)
    out_ref, new_res_ref = fused_add_rms_norm_ref(x.clone(), residual, w)

    max_err = (out_fused.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.01, f"Fused max error {max_err:.6f} exceeds 0.01"


@needs_cuda
@needs_triton
def test_fused_add_rmsnorm_identity_residual():
    """With zero residual, fused RMSNorm should equal regular RMSNorm."""
    D = 2048
    x = torch.randn(16, D, device='cuda', dtype=torch.float32)
    residual = torch.zeros(16, D, device='cuda', dtype=torch.float32)
    w = torch.randn(D, device='cuda', dtype=torch.float32)

    out_fused, _ = fused_add_rms_norm(x.clone(), residual, w)
    out_regular = rms_norm(x, w)

    max_err = (out_fused - out_regular).abs().max().item()
    assert max_err < 1e-4, f"Identity residual error {max_err:.6f} exceeds 1e-4"


@needs_cuda
@needs_triton
def test_fused_add_rmsnorm_residual_preserved():
    """The returned residual must be x + residual (the pre-norm sum).

    vLLM convention: residual is modified in-place to hold the pre-norm sum
    (x + residual), not the original x alone.
    REFERENCE: vllm/vllm/model_executor/layers/layernorm.py:L196
    """
    D = 512
    x = torch.randn(8, D, device='cuda', dtype=torch.float32)
    residual = torch.randn(8, D, device='cuda', dtype=torch.float32)
    w = torch.randn(D, device='cuda', dtype=torch.float32)

    # Save original x before the in-place modification
    x_original = x.clone()
    _, new_residual = fused_add_rms_norm(x, residual, w)

    # The returned residual is the pre-norm sum: x + residual (fp32 exact)
    expected_sum = x_original + residual
    assert torch.equal(new_residual, expected_sum), \
        f"Returned residual must equal x + residual (pre-norm sum)"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests — Cross-Chapter Compatibility
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_compatible_with_llama_architecture():
    """
    Integration test: RMSNorm must work with dimensions matching
    Chapter 16's Llama-3.2-1B config (hidden_size=2048).
    """
    D = 2048  # Llama-3.2-1B hidden_size
    N = 4     # batch=4

    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)

    out = rms_norm(x, w)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device


@needs_cuda
@needs_triton
def test_compatible_with_triton_primer():
    """
    Integration test: RMSNorm must follow the same Triton patterns taught
    in Chapter 15 (Triton Primer): grid, program_id, tl.load/store, masks.
    """
    D = 1024
    x = torch.randn(16, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    # The kernel should produce valid output with the same pattern
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    max_err = (out.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.01


@needs_cuda
@needs_triton
def test_residual_pattern_for_llama_layer():
    """
    Integration test: The fused RMSNorm must support the residual pattern
    used in LlamaDecoderLayer:
        x = rms_norm(x)          # pre-norm
        x = attention(x) + x     # attention + residual
        x = rms_norm(x)          # pre-norm
        x = mlp(x) + x           # MLP + residual
    """
    D = 2048
    N = 2

    # Simulate one Llama layer
    x = torch.randn(N, D, device='cuda', dtype=torch.float16)
    w1 = torch.randn(D, device='cuda', dtype=torch.float16)  # attention norm
    w2 = torch.randn(D, device='cuda', dtype=torch.float16)  # mlp norm

    # Pre-attention RMSNorm
    x_norm = rms_norm(x, w1)

    # Simulate attention output
    attn_out = torch.randn(N, D, device='cuda', dtype=torch.float16) * 0.1

    # Fused residual + RMSNorm for MLP
    x_fused, residual = fused_add_rms_norm(x_norm, attn_out, w2)

    # Verify shapes and no NaN
    assert x_fused.shape == x.shape
    assert not torch.isnan(x_fused).any()
    assert not torch.isinf(x_fused).any()


# ═══════════════════════════════════════════════════════════════════════════
# Teaching Example Tests
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_teaching_example_traceable():
    """
    Teaching test: Small enough that a reader can trace through manually.
    D=8, N=1 — the reader can compute RMS by hand and verify.
    """
    D = 8
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        device='cuda', dtype=torch.float32
    )
    w = torch.ones(D, device='cuda', dtype=torch.float32)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    # Manual computation: mean(x^2) = (1+4+9+16+25+36+49+64)/8 = 204/8 = 25.5
    # rms = sqrt(25.5 + 1e-6) ≈ 5.04975
    # output = x / 5.04975
    expected = x / torch.tensor(5.04975, device='cuda')
    max_err = (out - expected).abs().max().item()
    assert max_err < 0.05, f"Teaching example error {max_err:.6f} exceeds 0.05"


@needs_cuda
@needs_triton
def test_teaching_numerical_stability():
    """
    Teaching test: Show that fp32 accumulation prevents overflow.
    Large values in fp16 would overflow if accumulated in fp16.
    """
    D = 4096
    x = torch.randn(1, D, device='cuda', dtype=torch.float16) * 100.0  # large values
    w = torch.ones(D, device='cuda', dtype=torch.float16)

    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)

    assert not torch.isnan(out).any(), "Output contains NaN (overflow in accumulation?)"
    assert not torch.isinf(out).any(), "Output contains Inf (overflow?)"
    max_err = (out.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.1, f"Large values error {max_err:.6f} exceeds 0.1"


# ═══════════════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_shape_mismatch_raises():
    """Weight shape must match input's last dimension."""
    x = torch.randn(8, 512, device='cuda')
    w = torch.randn(256, device='cuda')  # wrong size

    with pytest.raises(AssertionError):
        rms_norm(x, w)


@needs_cuda
@needs_triton
@pytest.mark.parametrize("D", [64, 128, 1023, 1025, 4096])
def test_various_hidden_sizes(D):
    """RMSNorm must work with hidden sizes both below and above BLOCK_SIZE=1024."""
    x = torch.randn(8, D, device='cuda', dtype=torch.float16)
    w = torch.randn(D, device='cuda', dtype=torch.float16)
    out = rms_norm(x, w)
    out_ref = rms_norm_ref(x, w)
    max_err = (out.float() - out_ref.float()).abs().max().item()
    assert max_err < 0.02, f"D={D}: max error {max_err:.6f} exceeds 0.02"


@needs_cuda
@needs_triton
def test_determinism():
    """Same input must produce identical output across multiple runs."""
    D = 2048
    x = torch.randn(16, D, device='cuda', dtype=torch.float32)
    w = torch.randn(D, device='cuda', dtype=torch.float32)
    results = [rms_norm(x.clone(), w) for _ in range(5)]
    for i in range(1, 5):
        assert torch.equal(results[0], results[i]), f"Run {i} differs from run 0"


@needs_cuda
@needs_triton
def test_weight_not_1d_raises():
    """Weight must be 1-dimensional."""
    x = torch.randn(8, 512, device='cuda')
    w = torch.randn(1, 512, device='cuda')  # 2D

    with pytest.raises(AssertionError):
        rms_norm(x, w)
