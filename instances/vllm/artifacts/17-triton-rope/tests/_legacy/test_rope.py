"""
Tests for Triton RoPE Implementation (Chapter 18).

Test levels:
    - unit: core logic correctness (vs PyTorch reference)
    - integration: cross-chapter compatibility (GQA, partial rot_dim)
    - teaching: runnable examples for the book
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))
from rope import (
    rotary_embedding,
    rotary_embedding_ref,
    apply_rotary_emb,
    apply_rotary_emb_ref,
    compute_cos_sin_cache,
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

# Default RoPE config matching Llama-3.2-1B
LLAMA1B_CONFIG = {
    "head_size": 128,
    "num_heads": 8,
    "rotary_dim": 128,
    "max_position": 4096,
}


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Neox-style RoPE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("num_tokens,num_heads,head_size", [
    (1, 4, 64),
    (1, 8, 128),
    (32, 8, 128),   # Llama-3.2-1B sized
    (128, 8, 128),
    (7, 4, 64),     # odd token count
])
@needs_cuda
@needs_triton
def test_rope_neox_correctness(num_tokens, num_heads, head_size):
    """Triton RoPE (Neox) must match pure-PyTorch reference within tolerance."""
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"Q max_error={q_err:.6f} exceeds 0.02"
    assert k_err < 0.02, f"K max_error={k_err:.6f} exceeds 0.02"


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@needs_cuda
@needs_triton
def test_rope_neox_dtypes(dtype):
    """RoPE must work with both fp16 and fp32 inputs."""
    num_tokens, num_heads, head_size = 16, 4, 64
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size, device='cuda', dtype=dtype)
    key = torch.randn(num_tokens, num_heads, head_size, device='cuda', dtype=dtype)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda().to(dtype)

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    tol = 1e-4 if dtype == torch.float32 else 0.02
    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < tol, f"dtype={dtype}: Q max_err={q_err:.6f} exceeds {tol}"
    assert k_err < tol, f"dtype={dtype}: K max_err={k_err:.6f} exceeds {tol}"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — GPT-J-style RoPE
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("num_tokens,num_heads,head_size", [
    (1, 4, 64),
    (16, 8, 128),
    (32, 4, 64),
])
@needs_cuda
@needs_triton
def test_rope_gptj_correctness(num_tokens, num_heads, head_size):
    """Triton RoPE (GPT-J) must match reference within tolerance."""
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=False,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=False,
    )

    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"GPT-J Q max_error={q_err:.6f} exceeds 0.02"
    assert k_err < 0.02, f"GPT-J K max_error={k_err:.6f} exceeds 0.02"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Partial Rotary Dimension
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("head_size,rotary_dim", [
    (128, 64),   # half rotation
    (128, 32),   # quarter rotation
    (256, 128),  # half of larger head
])
@needs_cuda
@needs_triton
def test_rope_partial_rotary_dim(head_size, rotary_dim):
    """RoPE with rot_dim < head_size: must only rotate first rot_dim elements."""
    num_tokens, num_heads = 8, 4
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(rotary_dim, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"Partial rot_dim: Q max_err={q_err:.6f}"
    assert k_err < 0.02, f"Partial rot_dim: K max_err={k_err:.6f}"

    # Verify pass-through elements are unchanged (only first rot_dim elements rotate)
    q_orig = query.clone()
    q_rotated, _ = rotary_embedding(
        positions, query, key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    pass_through = q_rotated[..., rotary_dim:]
    orig_pass = q_orig[..., rotary_dim:]
    assert torch.equal(pass_through, orig_pass), \
        f"Pass-through elements changed after RoPE (rot_dim={rotary_dim}, head_size={head_size})"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — GQA (Grouped Query Attention)
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_rope_gqa():
    """RoPE must handle num_kv_heads < num_q_heads (GQA)."""
    num_tokens, num_q_heads, num_kv_heads, head_size = 16, 8, 2, 128
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_q_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_kv_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"GQA Q max_error={q_err:.6f}"
    assert k_err < 0.02, f"GQA K max_error={k_err:.6f}"

    # Verify shapes preserved
    assert q_triton.shape == (num_tokens, num_q_heads, head_size)
    assert k_triton.shape == (num_tokens, num_kv_heads, head_size)


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Key = None
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_rope_key_none():
    """RoPE must work when key is None (cross-layer KV sharing)."""
    num_tokens, num_heads, head_size = 8, 4, 64
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), None, head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), None, head_size,
        cos_sin_cache, is_neox_style=True,
    )

    assert k_triton is None and k_ref is None, "key should be None"
    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    assert q_err < 0.02, f"Key=None: Q max_error={q_err:.6f}"

    # Query shape preserved
    assert q_triton.shape == (num_tokens, num_heads, head_size)


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Flat (2D) Input
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_rope_flat_input():
    """RoPE must handle flat 2D input [num_tokens, num_heads * head_size]."""
    num_tokens, num_heads, head_size = 8, 4, 64
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)

    # 3D input
    query_3d = torch.randn(num_tokens, num_heads, head_size,
                           device='cuda', dtype=torch.float16)
    key_3d = torch.randn(num_tokens, num_heads, head_size,
                         device='cuda', dtype=torch.float16)

    # 2D flat input
    query_2d = query_3d.reshape(num_tokens, num_heads * head_size)
    key_2d = key_3d.reshape(num_tokens, num_heads * head_size)

    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    # Process 3D
    q_3d, k_3d = rotary_embedding(
        positions, query_3d.clone(), key_3d.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    # Process 2D (flat)
    q_2d, k_2d = rotary_embedding(
        positions, query_2d.clone(), key_2d.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    # Reshape 2D results back to 3D and compare
    q_2d_reshaped = q_2d.reshape(num_tokens, num_heads, head_size)
    k_2d_reshaped = k_2d.reshape(num_tokens, num_heads, head_size)

    q_err = (q_3d.float() - q_2d_reshaped.float()).abs().max().item()
    k_err = (k_3d.float() - k_2d_reshaped.float()).abs().max().item()
    assert q_err < 1e-5, f"Flat input Q mismatch: {q_err:.6f}"
    assert k_err < 1e-5, f"Flat input K mismatch: {k_err:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Determinism
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_rope_determinism():
    """Same input must produce identical output across multiple runs."""
    num_tokens, num_heads, head_size = 16, 4, 64
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    # Run 5 times with same input
    results = []
    for _ in range(5):
        query = torch.randn(num_tokens, num_heads, head_size,
                            device='cuda', dtype=torch.float16)
        key = torch.randn(num_tokens, num_heads, head_size,
                          device='cuda', dtype=torch.float16)
        q_out, k_out = rotary_embedding(
            positions, query, key, head_size,
            cos_sin_cache, is_neox_style=True,
        )
        results.append((q_out.clone(), k_out.clone()))

    for i in range(1, 5):
        assert torch.equal(results[0][0], results[i][0]), \
            f"Query run {i} differs from run 0"
        assert torch.equal(results[0][1], results[i][1]), \
            f"Key run {i} differs from run 0"


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests — Rotation Orthogonality (R^T * R = I)
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_rope_orthogonality():
    """RoPE is a 2D rotation: applying it with negated angle recovers original.

    RoPE applies rotation matrix R = [[cos, -sin], [sin, cos]].
    R is orthogonal: R^T * R = I. So applying the same rotation with
    negative sin (inverse) should recover the original vector.
    """
    num_tokens, num_heads, head_size = 4, 2, 128
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    # Gather cos/sin for positions
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos_gathered, sin_gathered = cos_sin.chunk(2, dim=-1)

    x = torch.randn(num_tokens, num_heads, head_size,
                    device='cuda', dtype=torch.float32)

    # Apply forward rotation
    x_rot = apply_rotary_emb(x.clone(), cos_gathered, sin_gathered, True)

    # Apply inverse rotation (negated sin)
    sin_neg = -sin_gathered
    x_restored = apply_rotary_emb(x_rot.clone(), cos_gathered, sin_neg, True)

    err = (x.float() - x_restored.float()).abs().max().item()
    assert err < 1e-4, f"Orthogonality error: {err:.6f} (should be ~0)"


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests — Cross-Chapter Compatibility
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_compatible_with_llama_architecture():
    """Integration test: RoPE must work with Llama-3.2-1B dimensions."""
    cfg = LLAMA1B_CONFIG
    num_tokens = 16

    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, cfg["num_heads"], cfg["head_size"],
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, cfg["num_heads"], cfg["head_size"],
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(cfg["rotary_dim"], cfg["max_position"]).cuda()

    q_out, k_out = rotary_embedding(
        positions, query, key, cfg["head_size"],
        cos_sin_cache, is_neox_style=True,
    )

    # Verify shapes preserved
    assert q_out.shape == (num_tokens, cfg["num_heads"], cfg["head_size"])
    assert k_out.shape == (num_tokens, cfg["num_heads"], cfg["head_size"])
    assert not torch.isnan(q_out).any(), "Q contains NaN"
    assert not torch.isnan(k_out).any(), "K contains NaN"
    assert not torch.isinf(q_out).any(), "Q contains Inf"
    assert not torch.isinf(k_out).any(), "K contains Inf"


@needs_cuda
@needs_triton
def test_compatible_with_triton_primer():
    """Integration test: RoPE must follow the same Triton patterns as Chapter 15."""
    num_tokens, num_heads, head_size = 32, 4, 64
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_out, k_out = rotary_embedding(
        positions, query, key, head_size,
        cos_sin_cache, is_neox_style=True,
    )

    # Reference
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    assert not torch.isnan(q_out).any(), "Output contains NaN"
    assert not torch.isinf(q_out).any(), "Output contains Inf"
    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02
    assert k_err < 0.02


@needs_cuda
@needs_triton
def test_rope_with_random_positions():
    """RoPE must work with non-contiguous position indices (continuous batching)."""
    num_tokens, num_heads, head_size = 8, 4, 64

    # Non-contiguous positions (e.g., after prefill with chunking)
    positions = torch.tensor([0, 0, 1, 5, 10, 15, 20, 100],
                             device='cuda', dtype=torch.long)

    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_triton, k_triton = rotary_embedding(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_triton.float() - q_ref.float()).abs().max().item()
    k_err = (k_triton.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"Random positions Q error: {q_err:.6f}"
    assert k_err < 0.02, f"Random positions K error: {k_err:.6f}"


# ═══════════════════════════════════════════════════════════════════════════
# Teaching Example Tests
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_teaching_example_traceable():
    """
    Teaching test: Small enough to trace manually.
    D=4, single token, single head — verify the rotation formula by hand.
    """
    head_size = 4
    rotary_dim = 4

    # Simple known values for manual verification
    # RoPE: freq_i = 1/10000^(2i/4) = 1/10000^(i/2) for i in [0,1]
    # i=0: theta_0 = 1/10000^0 = 1.0
    # i=1: theta_0 = 1/10000^0.5 = 1/100 = 0.01
    # For position 1: phi_0 = 1.0, phi_1 = 0.01
    # cos = [cos(1.0), cos(0.01)], sin = [sin(1.0), sin(0.01)]

    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device='cuda', dtype=torch.float32)
    cos_sin_cache = compute_cos_sin_cache(rotary_dim, 8, base=10000.0).cuda()

    positions = torch.tensor([1], device='cuda', dtype=torch.long)

    q_out, _ = rotary_embedding(
        positions, x.clone(), None, head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, _ = rotary_embedding_ref(
        positions, x.clone(), None, head_size,
        cos_sin_cache, is_neox_style=True,
    )

    err = (q_out - q_ref).abs().max().item()
    assert err < 1e-5, f"Teaching example error: {err:.6f}"

    # Also verify that Neox style splits [x1, x2, x3, x4] into
    # pairs (x1, x3) and (x2, x4), then rotates each pair.
    # After rotation, each pair satisfies:
    #   o1 = x1*cos - x2*sin
    #   o2 = x2*cos + x1*sin
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos_val, sin_val = cos_sin.chunk(2, dim=-1)

    # Manual check: pair 0 = (x[0]=1, x[2]=3), phi = 1*1.0
    c0, s0 = cos_val[0, 0].item(), sin_val[0, 0].item()
    expected_o0 = 1.0 * c0 - 3.0 * s0
    expected_o2 = 3.0 * c0 + 1.0 * s0
    assert abs(q_out[0, 0, 0].item() - expected_o0) < 1e-5, \
        f"Manual verification of pair 0, elem 0 failed: {q_out[0,0,0].item()} != {expected_o0}"
    assert abs(q_out[0, 0, 2].item() - expected_o2) < 1e-5, \
        f"Manual verification of pair 0, elem 2 failed: {q_out[0,0,2].item()} != {expected_o2}"


@needs_cuda
@needs_triton
def test_teaching_numerical_stability():
    """
    Teaching test: fp32 accumulation prevents overflow with large values.
    """
    num_tokens, num_heads, head_size = 4, 2, 128
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    # Large values that could overflow in fp16
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16) * 100.0
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16) * 100.0

    q_out, k_out = rotary_embedding(
        positions, query, key, head_size,
        cos_sin_cache, is_neox_style=True,
    )

    assert not torch.isnan(q_out).any(), "Q contains NaN (overflow in Triton kernel?)"
    assert not torch.isnan(k_out).any(), "K contains NaN (overflow in Triton kernel?)"
    assert not torch.isinf(q_out).any(), "Q contains Inf (overflow?)"
    assert not torch.isinf(k_out).any(), "K contains Inf (overflow?)"


# ═══════════════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════

@needs_cuda
@needs_triton
def test_apply_rotary_emb_3d_required():
    """apply_rotary_emb must raise on non-3D input."""
    x_2d = torch.randn(8, 64, device='cuda')
    cos_2d = torch.randn(8, 32, device='cuda')
    sin_2d = torch.randn(8, 32, device='cuda')

    with pytest.raises(Exception):
        apply_rotary_emb(x_2d, cos_2d, sin_2d, True)


@needs_cuda
@needs_triton
@pytest.mark.parametrize("head_size", [32, 64, 128, 256])
def test_various_head_sizes(head_size):
    """RoPE must work with various head sizes."""
    num_tokens, num_heads = 8, 4
    positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
    query = torch.randn(num_tokens, num_heads, head_size,
                        device='cuda', dtype=torch.float16)
    key = torch.randn(num_tokens, num_heads, head_size,
                      device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_out, k_out = rotary_embedding(
        positions, query, key, head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()
    assert q_err < 0.02, f"head_size={head_size}: Q error {q_err:.6f}"
    assert k_err < 0.02, f"head_size={head_size}: K error {k_err:.6f}"


@needs_cuda
@needs_triton
def test_single_token_single_head():
    """Edge case: 1 token, 1 head."""
    head_size = 64
    positions = torch.tensor([0], device='cuda', dtype=torch.long)
    query = torch.randn(1, 1, head_size, device='cuda', dtype=torch.float16)
    key = torch.randn(1, 1, head_size, device='cuda', dtype=torch.float16)
    cos_sin_cache = compute_cos_sin_cache(head_size, 4096).cuda()

    q_out, k_out = rotary_embedding(
        positions, query, key, head_size,
        cos_sin_cache, is_neox_style=True,
    )
    q_ref, k_ref = rotary_embedding_ref(
        positions, query.clone(), key.clone(), head_size,
        cos_sin_cache, is_neox_style=True,
    )

    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()
    assert q_err < 1e-5, f"Single token Q error: {q_err:.6f}"
    assert k_err < 1e-5, f"Single token K error: {k_err:.6f}"
