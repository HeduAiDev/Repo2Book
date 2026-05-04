# RMSNorm Implementation Notes

## Source Analysis

### 5-Item Source Analysis (vLLM files studied before writing)

1. **vllm/vllm/model_executor/layers/layernorm.py:L103-L120** — RMSNorm class definition
   - CustomOp-based design with multiple dispatch paths (CUDA, ROCm, Oink, Native)
   - Constructor handles `hidden_size`, `eps`, `var_hidden_size`, `has_weight`, `dtype`
   - Weight is `nn.Parameter(torch.ones(hidden_size))` when `has_weight=True`

2. **vllm/vllm/model_executor/layers/layernorm.py:L188-L231** — `forward_static()`
   - Pure PyTorch reference: x→fp32, pow(2).mean(-1), rsqrt, scale, cast back
   - Handles optional residual add and variance_size_override
   - This is the mathematical ground truth we validate against

3. **vllm/vllm/model_executor/layers/batch_invariant.py:L786-L833** — `_rms_norm_kernel`
   - Triton kernel: one program per row, two-pass (sum_sq then normalize)
   - fp32 accumulation for sum_sq to avoid overflow
   - `tl.rsqrt(mean_sq + eps)` for inverse RMS
   - Weight applied in second pass alongside normalization

4. **vllm/vllm/_custom_ops.py:L434-L438** — `fused_add_rms_norm()`
   - Calls `torch.ops._C.fused_add_rms_norm` — a compiled CUDA kernel
   - Fuses residual add + RMSNorm into a single kernel launch
   - In-place on input tensor

5. **vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py:L9-L55** — DeepSeek V4 fused Q/KV RMSNorm
   - Same RMSNorm math but specialized for Q and KV projections fused into one kernel
   - Uses 2D grid: (num_tokens, 2) where pid_task selects Q vs KV path
   - Demonstrates vLLM's pattern of fusing RMSNorm with neighboring operations

## Design Decisions

### Our Implementation vs vLLM

| Aspect | Our Implementation | vLLM Production |
|--------|-------------------|-----------------|
| Kernel language | Pure Triton | CUDA (torch.ops._C) + Triton fallback (batch_invariant) |
| Dispatch | Single path | Multi-backend (CUDA/ROCm/Oink/Native) |
| Fused residual | Separate Triton kernel | CUDA fused_add_rms_norm |
| var_hidden_size | Not implemented | Supported for Q/K norm with different sizes |
| Batch invariance | Not implemented | Batch-invariant mode for deterministic results |

### Simplifications

1. **No multi-backend dispatch.** We implement only the Triton path. vLLM has CUDA, ROCm/aiter, Oink (Blackwell), and native PyTorch paths.
2. **No variance_size_override.** vLLM uses this for Q/K RMSNorm where the variance is computed over a subset of dimensions. Our implementation always normalizes over the full last dimension.
3. **No batch-invariant mode.** vLLM has a special mode that ensures deterministic results regardless of batch composition. We skip this for simplicity.

## Source Mapping Table

| Our Code | vLLM Source | What We Changed & Why |
|----------|-------------|----------------------|
| `_rms_norm_kernel()` | `batch_invariant.py:L786 _rms_norm_kernel()` | Same two-pass pattern. Added BLOCK_SIZE as constexpr instead of hardcoded. |
| `_fused_add_rms_norm_kernel()` | `_custom_ops.py:L434 fused_add_rms_norm()` | Triton kernel instead of CUDA. Same fusion: add then rmsnorm in one kernel. |
| `rms_norm()` | `batch_invariant.py:L836 rms_norm()` | Same reshape→contiguous→launch→reshape pattern. BLOCK_SIZE adapts to D. |
| `fused_add_rms_norm()` | `layernorm.py:L56 fused_add_rms_norm()` | Our version returns (output, new_residual). vLLM modifies input in-place via CUDA. |
| `rms_norm_ref()` | `layernorm.py:L188 forward_static()` | Exact match: fp32→pow(2).mean→rsqrt→scale→cast. Skipped var_hidden_size and residual paths. |

## Key Files Referenced

1. `vllm/vllm/model_executor/layers/layernorm.py` — RMSNorm class (103 lines of class + dispatch logic)
2. `vllm/vllm/model_executor/layers/batch_invariant.py` — Triton RMSNorm kernel (lines 786-880)
3. `vllm/vllm/_custom_ops.py` — CUDA fused_add_rms_norm dispatch (lines 434-438)
4. `vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py` — Advanced fused pattern (full file, 97 lines)
5. `vllm/vllm/model_executor/custom_op.py` — CustomOp base class used by RMSNorm
