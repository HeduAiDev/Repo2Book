"""
Triton RMSNorm / LayerNorm Implementation — Our Simplified Version.

REFERENCE sources in vLLM:
    RMSNorm class:           vllm/vllm/model_executor/layers/layernorm.py:L103
    RMSNorm.forward_static:  vllm/vllm/model_executor/layers/layernorm.py:L188
    _rms_norm_kernel:        vllm/vllm/model_executor/layers/batch_invariant.py:L786
    fused_add_rms_norm:      vllm/vllm/_custom_ops.py:L434
    fused_qk_rmsnorm:        vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py:L9

This chapter covers:
    1. RMSNorm math: y = x / sqrt(mean(x^2) + eps) * weight
    2. Triton kernel: tiled sum-of-squares + rsqrt + scale
    3. Fused residual add + RMSNorm kernel
    4. Benchmark: Triton vs torch.nn.RMSNorm

RMSNorm vs LayerNorm:
    LayerNorm: y = (x - mean(x)) / std(x) * w + b
    RMSNorm:   y = x / rms(x) * w   (no mean subtraction, no bias)
    RMSNorm is ~15-20% faster because it skips the mean computation and subtraction.
    Llama, Mistral, DeepSeek all use RMSNorm.
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
# Kernel 1: Tiled RMSNorm — One Program Per Row
# REFERENCE: vllm/vllm/model_executor/layers/batch_invariant.py:L786
#            _rms_norm_kernel(input_ptr, weight_ptr, output_ptr, ...)
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(
        input_ptr,            # [N, D] input tensor
        weight_ptr,           # [D] learned weight (gamma)
        output_ptr,           # [N, D] output tensor
        input_row_stride,     # stride(0) of input
        output_row_stride,    # stride(0) of output
        n_cols,               # D = hidden_size
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        RMSNorm: y = x / sqrt(mean(x^2) + eps) * w

        Grid: (N,) — one program per row.
        Each program:
            1. Pass 1: accumulate sum(x_i^2) across tiles of the row
            2. Compute rms = sqrt(sum_sq / D + eps), inv_rms = 1 / rms
            3. Pass 2: normalize each element and multiply by weight

        REFERENCE: vllm/vllm/model_executor/layers/batch_invariant.py:L786
        vLLM's _rms_norm_kernel uses this exact two-pass pattern.
        The first pass ensures numerically stable fp32 accumulation; the second
        pass applies the normalization.
        """
        row_idx = tl.program_id(0).to(tl.int64)
        row_start = input_ptr + row_idx * input_row_stride
        out_row_start = output_ptr + row_idx * output_row_stride

        # --- Pass 1: accumulate sum of squares in fp32 ---
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start + col_idx, mask=mask, other=0.0)
            vals_f32 = vals.to(tl.float32)
            sum_sq += tl.sum(tl.where(mask, vals_f32 * vals_f32, 0.0))

        # --- Compute inv_rms ---
        mean_sq = sum_sq / n_cols
        inv_rms = tl.rsqrt(mean_sq + eps)

        # --- Pass 2: normalize + scale ---
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            vals = tl.load(row_start + col_idx, mask=mask, other=0.0)
            w = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            vals_f32 = vals.to(tl.float32)
            w_f32 = w.to(tl.float32)
            out_f32 = vals_f32 * inv_rms * w_f32
            tl.store(out_row_start + col_idx, out_f32.to(vals.dtype), mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
# Kernel 2: Fused Residual + RMSNorm
# REFERENCE: vllm/vllm/_custom_ops.py:L434  fused_add_rms_norm()
#            vllm/vllm/model_executor/layers/layernorm.py:L196  forward_static()
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_add_rms_norm_kernel(
        x_ptr,                # [N, D] input (written in-place!)
        residual_ptr,         # [N, D] residual (read-only)
        weight_ptr,           # [D] weight
        x_row_stride,
        residual_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused residual add + RMSNorm: x = rmsnorm(x + residual) * weight

        Grid: (N,) — one program per row.
        This fuses two operations into one kernel:
            1. x = x + residual (element-wise add)
            2. x = rmsnorm(x) * weight

        REFERENCE: vllm/vllm/_custom_ops.py:L434  fused_add_rms_norm()
        vLLM does this via torch.ops._C.fused_add_rms_norm (a CUDA kernel).
        Our Triton version achieves the same fusion — avoiding an intermediate
        tensor allocation and a second kernel launch.

        The key optimization: the sum-of-squares pass reads (x + residual)
        directly without materializing it, then the normalization pass writes
        the final result back to x in-place.
        """
        row_idx = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row_idx * x_row_stride
        res_row = residual_ptr + row_idx * residual_row_stride

        # --- Pass 1: accumulate sum_sq of (x + residual) in fp32 ---
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            x_vals = tl.load(x_row + col_idx, mask=mask, other=0.0)
            r_vals = tl.load(res_row + col_idx, mask=mask, other=0.0)
            added = (x_vals.to(tl.float32) + r_vals.to(tl.float32))
            sum_sq += tl.sum(tl.where(mask, added * added, 0.0))

        # --- Compute inv_rms ---
        mean_sq = sum_sq / n_cols
        inv_rms = tl.rsqrt(mean_sq + eps)

        # --- Pass 2: normalize + scale, write back to x in-place ---
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols
            x_vals = tl.load(x_row + col_idx, mask=mask, other=0.0)
            r_vals = tl.load(res_row + col_idx, mask=mask, other=0.0)
            w = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
            added_f32 = x_vals.to(tl.float32) + r_vals.to(tl.float32)
            out_f32 = added_f32 * inv_rms * w.to(tl.float32)
            tl.store(x_row + col_idx, out_f32.to(x_vals.dtype), mask=mask)


# ═══════════════════════════════════════════════════════════════════════════
# Python API
# ═══════════════════════════════════════════════════════════════════════════

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight

    Args:
        x: [..., D] input tensor
        weight: [D] learned scale parameter
        eps: numerical stability constant

    Returns:
        [..., D] normalized tensor (same shape as x)
    """
    assert weight.ndim == 1
    assert x.shape[-1] == weight.shape[0], (
        f"x.shape[-1]={x.shape[-1]} != weight.shape[0]={weight.shape[0]}"
    )

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    w = weight.contiguous()

    N, D = x_2d.shape
    out = torch.empty_like(x_2d)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(D))

    _rms_norm_kernel[(N,)](
        x_2d, w, out,
        x_2d.stride(0), out.stride(0),
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.reshape(original_shape)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual add + RMSNorm (in-place on x).

    Computes: x = rmsnorm(x + residual, weight)
    Returns new_residual = x + residual (pre-norm sum).

    REFERENCE: vllm/vllm/model_executor/layers/layernorm.py:L56  fused_add_rms_norm()
               vllm/vllm/model_executor/layers/layernorm.py:L196 forward_static()
    In vLLM: x and residual are modified in-place by CUDA kernel.
    x gets rmsnorm(x + residual) * weight; residual gets x + residual.
    Our Triton version matches this convention.

    Args:
        x: [..., D] input (modified in-place to normalized output)
        residual: [..., D] residual to add before normalization
        weight: [D] weight parameter
        eps: numerical stability constant

    Returns:
        (x, new_residual) where new_residual = old_x + old_residual (pre-norm)
    """
    assert x.shape == residual.shape
    assert x.shape[-1] == weight.shape[0]

    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    r_2d = residual.reshape(-1, residual.shape[-1])

    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    if not r_2d.is_contiguous():
        r_2d = r_2d.contiguous()
    w = weight.contiguous()

    N, D = x_2d.shape
    BLOCK_SIZE = min(1024, triton.next_power_of_2(D))

    # vLLM convention: new_residual = x + residual (pre-norm sum)
    new_residual = (x_2d + r_2d).contiguous()

    _fused_add_rms_norm_kernel[(N,)](
        x_2d, r_2d, w,
        x_2d.stride(0), r_2d.stride(0),
        D, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return x_2d.reshape(original_shape), new_residual.reshape(original_shape)


# ═══════════════════════════════════════════════════════════════════════════
# Pure PyTorch Reference (for correctness checking)
# REFERENCE: vllm/vllm/model_executor/layers/layernorm.py:L188
#            RMSNorm.forward_static() — the pure-PyTorch reference path
# ═══════════════════════════════════════════════════════════════════════════

def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Pure PyTorch RMSNorm (fp32 accumulation for numerical stability).

    Matches vLLM's RMSNorm.forward_static() at layernorm.py:L196 exactly:
    x_f32.pow(2).mean(-1) → rsqrt → scale → cast back.
    """
    orig_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(variance + eps)
    return (x_norm * weight.to(torch.float32)).to(orig_dtype)


def fused_add_rms_norm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch fused residual + RMSNorm (reference for correctness).

    REFERENCE: vllm/vllm/model_executor/layers/layernorm.py:L196 forward_static()
    """
    new_residual = (x + residual).contiguous()
    added = x + residual
    orig_dtype = added.dtype
    x_f32 = added.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(variance + eps)
    out = (x_norm * weight.to(torch.float32)).to(orig_dtype)
    return out, new_residual


# ═══════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_rmsnorm():
    """
    Compare Triton RMSNorm vs torch.nn.RMSNorm.

    torch.nn.RMSNorm (PyTorch 2.0+) uses CUDA fused kernels under the hood.
    Our Triton kernel should be competitive or faster for typical LLM hidden sizes.
    """
    configs = [
        # (N, D) — N tokens, D hidden dim
        (1, 2048),       # Single token, Llama-3.2-1B
        (32, 2048),      # Small batch
        (128, 4096),     # Medium batch, Llama-8B hidden size
        (256, 8192),     # Large batch, Llama-70B hidden size
    ]

    print("Triton RMSNorm vs torch.nn.RMSNorm Benchmark")
    print(f"{'Shape':>20} | {'Triton':>10} | {'Torch':>10} | {'Ratio':>8} | {'Fused':>10}")
    print("-" * 70)

    for N, D in configs:
        x = torch.randn(N, D, device='cuda', dtype=torch.float16)
        w = torch.randn(D, device='cuda', dtype=torch.float16)
        residual = torch.randn(N, D, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(20):
            rms_norm(x, w)
            torch.nn.functional.rms_norm(x, (D,), weight=w)
            fused_add_rms_norm(x.clone(), residual, w)

        torch.cuda.synchronize()

        # Benchmark Triton
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(500):
            rms_norm(x, w)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 500

        # Benchmark Torch
        start.record()
        for _ in range(500):
            torch.nn.functional.rms_norm(x, (D,), weight=w)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / 500

        # Benchmark Fused
        x_copy = x.clone()
        start.record()
        for _ in range(500):
            fused_add_rms_norm(x_copy, residual, w)
        end.record()
        torch.cuda.synchronize()
        fused_ms = start.elapsed_time(end) / 500

        ratio = torch_ms / triton_ms
        print(f"  {N}×{D:>5}        | {triton_ms:7.3f}ms | "
              f"{torch_ms:7.3f}ms | {ratio:6.2f}× | {fused_ms:7.3f}ms")


def demonstrate():
    """Run correctness + benchmark demo."""
    if not HAS_TRITON or not torch.cuda.is_available():
        print("Triton or CUDA not available. Read the source above.")
        print("Key concepts: tiled sum-of-squares, rsqrt, fused residual+norm")
        print("REFERENCE: vllm/vllm/model_executor/layers/batch_invariant.py:L786")
        return

    print("=" * 60)
    print("Triton RMSNorm — Correctness + Benchmark")
    print("=" * 60)

    D = 2048

    for N in [1, 32, 128]:
        x = torch.randn(N, D, device='cuda', dtype=torch.float16)
        w = torch.randn(D, device='cuda', dtype=torch.float16)

        # Correctness: basic RMSNorm
        out_triton = rms_norm(x, w)
        out_ref = rms_norm_ref(x, w)
        max_err = (out_triton.float() - out_ref.float()).abs().max().item()
        status = "PASS" if max_err < 0.01 else "FAIL"
        print(f"\n[{status}] RMSNorm N={N}, D={D}: max error = {max_err:.6f}")

        # Correctness: fused residual + RMSNorm
        residual = torch.randn(N, D, device='cuda', dtype=torch.float16)
        x_copy = x.clone()
        out_fused, new_res = fused_add_rms_norm(x_copy, residual, w)
        out_fused_ref, _ = fused_add_rms_norm_ref(x, residual, w)
        max_err_fused = (out_fused.float() - out_fused_ref.float()).abs().max().item()
        status_f = "PASS" if max_err_fused < 0.01 else "FAIL"
        print(f"[{status_f}] Fused RMSNorm N={N}, D={D}: max error = {max_err_fused:.6f}")

    print()
    benchmark_rmsnorm()


if __name__ == "__main__":
    demonstrate()


# REFERENCE: vllm/vllm/model_executor/layers/layernorm.py:L103 — RMSNorm(CustomOp)
# REFERENCE: vllm/vllm/model_executor/layers/batch_invariant.py:L786 — _rms_norm_kernel
# REFERENCE: vllm/vllm/_custom_ops.py:L434 — fused_add_rms_norm()
