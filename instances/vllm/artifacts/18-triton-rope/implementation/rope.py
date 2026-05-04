"""
Triton Rotary Position Embedding (RoPE) Implementation — Our Simplified Version.

REFERENCE sources in vLLM:
    RotaryEmbeddingBase class:    vllm/vllm/model_executor/layers/rotary_embedding/base.py:L15
    RotaryEmbedding class:        vllm/vllm/model_executor/layers/rotary_embedding/base.py:L118
    forward_static (PyTorch):     vllm/vllm/model_executor/layers/rotary_embedding/base.py:L139
    ApplyRotaryEmb class:         vllm/vllm/model_executor/layers/rotary_embedding/common.py:L123
    ApplyRotaryEmb.forward_static: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143
    CUDA roty kernel:             vllm/vllm/csrc/pos_encoding_kernels.cu:L79
    get_rope factory:             vllm/vllm/model_executor/layers/rotary_embedding/__init__.py:L33
    aiter Triton rope:            vllm/vllm/_aiter_ops.py:L1125

This chapter covers:
    1. RoPE math: 2D rotation of query/key pairs by position-dependent angles
    2. Triton kernel: one program per (token, head), tiled pair rotation
    3. Two rotation styles: Neox (half-split) and GPT-J (interleaved)
    4. Partial rotary dimension (rot_dim < head_size)
    5. Benchmark: Triton vs pure PyTorch reference

RoPE formula:
    For each pair (x1, x2) at position p and frequency theta_i:
        o1 = x1 * cos(p * theta_i) - x2 * sin(p * theta_i)
        o2 = x2 * cos(p * theta_i) + x1 * sin(p * theta_i)

    This is exactly a 2D rotation by angle phi = p * theta_i.
    Different frequency bands rotate at different speeds, encoding
    relative position information into the attention computation.

Neox-style:  x = [x1_0..x1_{d/2-1}, x2_0..x2_{d/2-1}] → pairs (x1_i, x2_i)
GPT-J-style: x = [x0, x1, x2, x3, ...] → pairs (x_{2i}, x_{2i+1})
"""

import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════════════
# Kernel: Rotary Position Embedding
# REFERENCE: vllm/vllm/csrc/pos_encoding_kernels.cu:L79
#            rotary_embedding_kernel (CUDA)
#
# vLLM's CUDA kernel uses grid=(num_tokens,) with one block iterating over
# all (head, pair) combinations. Our Triton kernel uses grid=(num_tokens,
# num_heads) with one program per (token, head) pair.
#
# REFERENCE: vllm/vllm/_aiter_ops.py:L1125
#            _triton_rotary_embedding_impl — the aiter Triton path
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _rotary_embedding_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        x_token_stride,
        x_head_stride,
        cos_stride,
        half_rot_dim,
        IS_NEOX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Apply RoPE to one (token, head) pair — in-place.

        Grid: (num_tokens, num_heads) — one program per (token, head).

        Each program:
        1. Loads `half_rot_dim` elements as pairs (x1, x2)
        2. Loads corresponding cos and sin values for this token's position
        3. Applies the 2D rotation: o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin
        4. Stores back in-place

        fp32 accumulation is used inside the kernel to match vLLM's
        ApplyRotaryEmb.forward_static() which promotes to fp32 internally.

        REFERENCE: vllm/vllm/csrc/pos_encoding_kernels.cu:L10-L35
                   apply_token_rotary_embedding — the per-element rotation
        """
        token_idx = tl.program_id(0).to(tl.int64)
        head_idx = tl.program_id(1).to(tl.int64)

        # Pointer to the start of this (token, head) slice
        x_base = x_ptr + token_idx * x_token_stride + head_idx * x_head_stride
        cos_base = cos_ptr + token_idx * cos_stride
        sin_base = sin_ptr + token_idx * cos_stride

        offs = tl.arange(0, BLOCK_SIZE)

        for start in range(0, half_rot_dim, BLOCK_SIZE):
            idx = start + offs
            mask = idx < half_rot_dim

            if IS_NEOX:
                # Neox: first half = x1, second half = x2
                x1 = tl.load(x_base + idx, mask=mask, other=0.0)
                x2 = tl.load(x_base + idx + half_rot_dim, mask=mask, other=0.0)
            else:
                # GPT-J: even indices = x1, odd indices = x2
                x1 = tl.load(x_base + 2 * idx, mask=mask, other=0.0)
                x2 = tl.load(x_base + 2 * idx + 1, mask=mask, other=0.0)

            c = tl.load(cos_base + idx, mask=mask, other=0.0)
            s = tl.load(sin_base + idx, mask=mask, other=0.0)

            # fp32 accumulation for numerical stability
            # REFERENCE: vllm/common.py:L163-L164 — enable_fp32_compute
            x1_f32 = x1.to(tl.float32)
            x2_f32 = x2.to(tl.float32)
            c_f32 = c.to(tl.float32)
            s_f32 = s.to(tl.float32)

            o1 = x1_f32 * c_f32 - x2_f32 * s_f32
            o2 = x2_f32 * c_f32 + x1_f32 * s_f32

            if IS_NEOX:
                tl.store(x_base + idx, o1.to(x1.dtype), mask=mask)
                tl.store(x_base + idx + half_rot_dim, o2.to(x2.dtype), mask=mask)
            else:
                tl.store(x_base + 2 * idx, o1.to(x1.dtype), mask=mask)
                tl.store(x_base + 2 * idx + 1, o2.to(x2.dtype), mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
# Python API: Triton-based apply_rotary_emb
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143
#            ApplyRotaryEmb.forward_static()
# ═══════════════════════════════════════════════════════════════════════════

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """
    Apply rotary position embedding via Triton kernel.

    Args:
        x: [num_tokens, num_heads, head_size] input tensor (modified in-place).
        cos: [num_tokens, half_rot_dim] cos values at each position.
        sin: [num_tokens, half_rot_dim] sin values at each position.
        is_neox_style: If True, use Neox-style (half-split pairs).
                       If False, use GPT-J-style (interleaved pairs).

    Returns:
        x modified in-place (use return value for clarity in call chains).

    Raises:
        AssertionError: If shapes are incompatible or Triton/CUDA unavailable.

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143
    """
    if not HAS_TRITON or not torch.cuda.is_available():
        raise RuntimeError(
            "Triton or CUDA not available. Use apply_rotary_emb_ref() instead."
        )

    assert x.ndim == 3, f"Expected 3D x, got {x.ndim}D"
    assert cos.ndim == 2 and sin.ndim == 2
    assert x.shape[0] == cos.shape[0] == sin.shape[0], \
        f"token dim mismatch: x={x.shape[0]}, cos={cos.shape[0]}"

    num_tokens, num_heads, head_size = x.shape
    half_rot_dim = cos.shape[-1]
    assert sin.shape[-1] == half_rot_dim
    assert 2 * half_rot_dim <= head_size, \
        f"2 * half_rot_dim ({2 * half_rot_dim}) > head_size ({head_size})"

    # Ensure contiguity for correct pointer arithmetic
    if not x.is_contiguous():
        x = x.contiguous()
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    BLOCK_SIZE = min(64, triton.next_power_of_2(half_rot_dim))
    grid = (num_tokens, num_heads)

    _rotary_embedding_kernel[grid](
        x,
        cos,
        sin,
        x.stride(0),   # token stride
        x.stride(1),   # head stride
        cos.stride(0),  # cos/sin stride (same)
        half_rot_dim,
        IS_NEOX=is_neox_style,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return x


# ═══════════════════════════════════════════════════════════════════════════
# Main API: rotary_embedding (positions, query, key)
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L139
#            RotaryEmbedding.forward_static()
# ═══════════════════════════════════════════════════════════════════════════

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Apply rotary position embedding to query and key.

    This is the main API, matching vLLM's RotaryEmbedding.forward_static().

    Steps:
        1. Flatten positions to 1D
        2. Index-select cos/sin from cache based on positions
        3. Split query/key into rot_part and pass_part
        4. Apply Triton rotation to rot_part
        5. Concatenate rot_part and pass_part back

    Args:
        positions: [num_tokens] or [batch_size, seq_len] integer positions.
        query: [num_tokens, num_heads, head_size] or
               [num_tokens, num_heads * head_size] query tensor.
        key: [num_tokens, num_kv_heads, head_size] or
             [num_tokens, num_kv_heads * head_size] key tensor, or None.
        head_size: Dimension per attention head.
        cos_sin_cache: [max_seqlen, rot_dim] pre-computed cos/sin cache.
                       First half = cos, second half = sin.
        is_neox_style: Neox-style (half-split) vs GPT-J-style (interleaved).

    Returns:
        (query, key) with RoPE applied. Tensors are modified in-place
        if they were contiguous; otherwise copies are returned.

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L139-L180
    """
    # --- Step 1: Flatten positions ---
    # REFERENCE: base.py:L150 — positions = positions.flatten()
    positions = positions.flatten()
    num_tokens = positions.shape[0]

    # --- Step 2: Gather cos/sin from cache ---
    # REFERENCE: base.py:L152-L153
    rotary_dim = cos_sin_cache.shape[-1]
    half_rot_dim = rotary_dim // 2

    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)   # each [num_tokens, half_rot_dim]

    # --- Step 3: Process query ---
    # REFERENCE: base.py:L155-L165
    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    num_q_heads = query.shape[1]

    query_rot = query[..., :rotary_dim]       # [num_tokens, num_q_heads, rotary_dim]
    query_pass = query[..., rotary_dim:]      # pass-through, unchanged

    apply_rotary_emb(query_rot, cos, sin, is_neox_style)

    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    # --- Step 4: Process key (if provided) ---
    # REFERENCE: base.py:L168-L179
    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        num_kv_heads = key.shape[1]

        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]

        apply_rotary_emb(key_rot, cos, sin, is_neox_style)

        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


# ═══════════════════════════════════════════════════════════════════════════
# Pure PyTorch Reference: apply_rotary_emb_ref
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143
#            ApplyRotaryEmb.forward_static() — EXACT match
# ═══════════════════════════════════════════════════════════════════════════

def apply_rotary_emb_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
    enable_fp32_compute: bool = True,
) -> torch.Tensor:
    """
    Pure PyTorch RoPE reference — matches vLLM's ApplyRotaryEmb.forward_static().

    Args:
        x: [batch_size (optional), seq_len, num_heads, head_size]
        cos: [seq_len, head_size // 2]
        sin: [seq_len, head_size // 2]
        is_neox_style: Neox vs GPT-J style.
        enable_fp32_compute: Use fp32 for intermediate computation.

    Returns:
        Rotated tensor, same shape as x.

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143-L182
    """
    origin_dtype = x.dtype
    if enable_fp32_compute:
        x = x.float()

    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        output = torch.cat((o1, o2), dim=-1)
    else:
        output = torch.stack((o1, o2), dim=-1).flatten(-2)

    if enable_fp32_compute:
        output = output.to(origin_dtype)
    return output


def rotary_embedding_ref(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Pure PyTorch RoPE reference — matches vLLM's RotaryEmbedding.forward_static().

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L139-L180
    """
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :cos_sin_cache.shape[-1]]
    query_pass = query[..., cos_sin_cache.shape[-1]:]
    query_rot = apply_rotary_emb_ref(query_rot, cos, sin, is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    if key is not None:
        key_shape = key.shape
        key = key.view(num_tokens, -1, head_size)
        key_rot = key[..., :cos_sin_cache.shape[-1]]
        key_pass = key[..., cos_sin_cache.shape[-1]:]
        key_rot = apply_rotary_emb_ref(key_rot, cos, sin, is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

    return query, key


# ═══════════════════════════════════════════════════════════════════════════
# Utility Functions (standalone versions of vLLM's internal helpers)
# ═══════════════════════════════════════════════════════════════════════════

def compute_inv_freq(
    rotary_dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Compute inverse frequencies for RoPE.

    theta_i = 1 / base^(2i / rotary_dim)   for i = 0, 1, ..., rotary_dim/2 - 1

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L69-L81
               _compute_inv_freq()
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    return inv_freq


def compute_cos_sin_cache(
    rotary_dim: int,
    max_position: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Pre-compute cos/sin cache for all positions up to max_position.

    Returns: [max_position, rotary_dim] tensor where:
        cache[p, :rotary_dim//2] = cos(p * theta_i)
        cache[p, rotary_dim//2:] = sin(p * theta_i)

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L83-L92
               _compute_cos_sin_cache()
    """
    inv_freq = compute_inv_freq(rotary_dim, base)
    t = torch.arange(max_position, dtype=torch.float)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# Helper: rotate_neox and rotate_gptj (standalone, for debugging)
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L17-L27
# ═══════════════════════════════════════════════════════════════════════════

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """
    Neox-style pair splitting: [x1, x2] → [-x2, x1].

    Used in rotary embedding for the Neox-style pair formation.

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L17-L20
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """
    GPT-J-style pair splitting: interleaved even/odd → [-odd, even].

    REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L23-L27
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_rope():
    """
    Compare Triton RoPE vs pure PyTorch reference.

    Measures throughput (tokens/second) for various batch sizes and
    head configurations matching Llama model sizes.
    """
    configs = [
        # (num_tokens, num_heads, head_size) — matching Llama variants
        (1, 4, 64),      # Single token, tiny head
        (32, 4, 64),     # Small batch, 256 hidden
        (128, 8, 128),   # Llama-3.2-1B: num_heads=8, head_size=128
        (256, 32, 128),  # Llama-8B scale: 32 heads
        (512, 8, 128),   # Many tokens, Llama-1B heads
        (1, 8, 128),     # Single token, Llama-1B head size
    ]

    print("Triton RoPE vs PyTorch Reference Benchmark")
    print(f"{'Shape (T×H×D)':>22} | {'Triton':>10} | {'Torch':>10} | {'Ratio':>8}")
    print("-" * 65)

    for num_tokens, num_heads, head_size in configs:
        rotary_dim = head_size
        half_rot_dim = rotary_dim // 2

        positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
        query = torch.randn(num_tokens, num_heads, head_size,
                            device='cuda', dtype=torch.float16)
        key = torch.randn(num_tokens, num_heads, head_size,
                          device='cuda', dtype=torch.float16)
        cos_sin_cache = compute_cos_sin_cache(rotary_dim, 4096).cuda()

        # Warmup
        for _ in range(20):
            rotary_embedding(positions, query.clone(), key.clone(),
                             head_size, cos_sin_cache, True)
            rotary_embedding_ref(positions, query.clone(), key.clone(),
                                 head_size, cos_sin_cache, True)

        torch.cuda.synchronize()

        # Benchmark Triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(500):
            rotary_embedding(positions, query.clone(), key.clone(),
                             head_size, cos_sin_cache, True)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 500

        # Benchmark Torch
        start.record()
        for _ in range(500):
            rotary_embedding_ref(positions, query.clone(), key.clone(),
                                 head_size, cos_sin_cache, True)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 500

        ratio = torch_ms / triton_ms
        shape_str = f"{num_tokens}×{num_heads}×{head_size}"
        print(f"  {shape_str:>20} | {triton_ms:7.3f}ms | "
              f"{torch_ms:7.3f}ms | {ratio:6.2f}x")


def demonstrate():
    """Run correctness + benchmark demo."""
    if not HAS_TRITON or not torch.cuda.is_available():
        print("Triton or CUDA not available. Read the source above.")
        print("Key concepts: tiled 2D rotation, Neox/GPT-J pair splitting, "
              "pre-computed cos/sin cache")
        print("REFERENCE: vllm/vllm/csrc/pos_encoding_kernels.cu:L79")
        return

    print("=" * 65)
    print("Triton RoPE — Correctness + Benchmark")
    print("=" * 65)

    # Test configurations
    test_cases = [
        (1, 4, 64, 64, True, "Neox"),
        (1, 4, 64, 64, False, "GPT-J"),
        (32, 8, 128, 128, True, "Neox, Llama-1B"),
        (7, 8, 128, 64, True, "Neox, partial rot_dim"),
        (32, 8, 128, 128, True, "GQA style (same heads for simplicity)"),
    ]

    for num_tokens, num_heads, head_size, rotary_dim, is_neox, label in test_cases:
        positions = torch.arange(num_tokens, device='cuda', dtype=torch.long)
        query = torch.randn(num_tokens, num_heads, head_size,
                            device='cuda', dtype=torch.float16)
        key = torch.randn(num_tokens, num_heads, head_size,
                          device='cuda', dtype=torch.float16)
        cos_sin_cache = compute_cos_sin_cache(rotary_dim, 4096).cuda()

        # Triton version
        q_triton, k_triton = rotary_embedding(
            positions, query.clone(), key.clone(),
            head_size, cos_sin_cache, is_neox,
        )

        # Reference version
        q_ref, k_ref = rotary_embedding_ref(
            positions, query.clone(), key.clone(),
            head_size, cos_sin_cache, is_neox,
        )

        q_err = (q_triton.float() - q_ref.float()).abs().max().item()
        k_err = (k_triton.float() - k_ref.float()).abs().max().item()

        q_status = "PASS" if q_err < 0.02 else "FAIL"
        k_status = "PASS" if k_err < 0.02 else "FAIL"
        print(f"[{q_status}/{k_status}] {label}: "
              f"Q max_err={q_err:.6f}, K max_err={k_err:.6f}")

    print()
    benchmark_rope()


if __name__ == "__main__":
    demonstrate()


# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L15
#            — RotaryEmbeddingBase class
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L118
#            — RotaryEmbedding class
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/base.py:L139
#            — forward_static (PyTorch path)
# REFERENCE: vllm/vllm/model_executor/layers/rotary_embedding/common.py:L143
#            — ApplyRotaryEmb.forward_static (core rotation math)
# REFERENCE: vllm/vllm/csrc/pos_encoding_kernels.cu:L79
#            — CUDA rotary_embedding_kernel
# REFERENCE: vllm/vllm/csrc/pos_encoding_kernels.cu:L10
#            — apply_token_rotary_embedding (per-element rotation)
