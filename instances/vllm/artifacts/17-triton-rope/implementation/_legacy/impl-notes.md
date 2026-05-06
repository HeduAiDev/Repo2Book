# Rotary Position Embedding — Implementation Notes

## Source Analysis

### 1. What vLLM files implement this feature?

```
vllm/vllm/model_executor/layers/rotary_embedding/base.py        — RotaryEmbeddingBase, RotaryEmbedding class
vllm/vllm/model_executor/layers/rotary_embedding/common.py       — ApplyRotaryEmb, rotate_neox, rotate_gptj
vllm/vllm/model_executor/layers/rotary_embedding/__init__.py     — Factory: get_rope()
vllm/vllm/csrc/pos_encoding_kernels.cu                           — CUDA rotary_embedding_kernel
vllm/vllm/csrc/cpu/pos_encoding.cpp                              — CPU fallback
vllm/vllm/csrc/ops.h                                             — Op declaration (line 159)
vllm/vllm/csrc/torch_bindings.cpp                                — Torch binding (line 260)
vllm/vllm/_custom_ops.py                                         — Python-level custom op wrapper
```

### 2. What are the key classes and their responsibilities?

**RotaryEmbeddingBase** (base.py:L15)
- Owns: `head_size`, `rotary_dim`, `max_position_embeddings`, `base`, `is_neox_style`, `cos_sin_cache`
- Key methods:
  - `_compute_inv_freq()` — computes inverse frequencies: `1.0 / base**(arange(0,rotary_dim,2) / rotary_dim)`
  - `_compute_cos_sin_cache()` — pre-computes all cos/sin values up to max_position
  - `get_cos_sin(seqlen)` — slices cache to given sequence length
- Delegates to: `ApplyRotaryEmb` for the actual rotation

**RotaryEmbedding** (base.py:L118)
- Extends `RotaryEmbeddingBase`
- `forward_static()` — pure PyTorch path: index_select cos/sin by position, split rot/pass, apply rotation, concat
- `forward_cuda()` — dispatch to `ops.rotary_embedding` (CUDA, in-place)
- `forward_hip()` — ROCm Triton path via aiter

**ApplyRotaryEmb** (common.py:L123)
- `forward_static(x, cos, sin, is_neox_style)` — the core rotation formula
  - Neox: split x into halves, `o1 = x1*cos - x2*sin`, `o2 = x2*cos + x1*sin`
  - GPT-J: `x1 = x[..., ::2]`, `x2 = x[..., 1::2]`, same formulas, then stack+flatten

**CUDA Kernel** (pos_encoding_kernels.cu)
- Grid: `(num_tokens,)`  — one block per token
- Block: `min(num_heads * rot_dim / 2, 512)` threads
- `apply_token_rotary_embedding()` — per-element pair rotation
- `rotary_embedding_kernel()` — top-level kernel, iterates heads within a block

### 3. What is the data flow?

```
Positions (int64) ──┐
                    ├──→ index_select cos_sin_cache → cos, sin per token
cos_sin_cache  ─────┘

query [num_tokens, num_heads, head_size] ──→ reshape ──→ split rot/pass
  rot_part → ApplyRotaryEmb(x_rot, cos, sin) → rotate in-place
  pass_part → unchanged
  concat(rot_part, pass_part) → reshape back

key (same flow, optional, may have fewer heads for GQA)
```

CUDA path does this entirely in-place with `rope_dim_offset` to handle the rot/pass boundary inside the kernel.

### 4. What design decisions did vLLM make and WHY?

**Decision 1: Pre-computed cos/sin cache (not on-the-fly)**
- Why: cos/sin at position p are `cos(p * theta_i)` where `theta_i = 1/base^(2i/d)`. Computing this per-position is expensive. Pre-computing `[max_position, rot_dim]` table and index_select-ing is much faster.
- Trade-off: Uses `max_position * rot_dim * dtype_size` bytes of memory (e.g., 128K * 64 * 2 = 16MB for Llama-8B).
- Source: base.py:L83-L92

**Decision 2: Neox-style by default (GPT-J as opt-in)**
- Why: Neox-style splits the tensor into two contiguous halves, which maps well to GPU memory access. GPT-J-style interleaves even/odd elements, requiring strided loads.
- Trade-off: Neox needs rot_dim to be even (always true since rotary_dim is typically head_size). GPT-J interleaving is required by some models (GPT-J, CodeGen).
- Source: common.py:L17-L27

**Decision 3: In-place operation in CUDA path**
- Why: Avoids allocating new tensors for rotated output. The rotation is `x1, x2 → o1, o2` which can be computed and stored back to the same memory locations.
- Source: pos_encoding_kernels.cu:L33-L35 (read x, write o back to same arr)

**Decision 4: rot_dim can be less than head_size**
- Why: Partial rotary (e.g., GPT-NeoX rotates only first `head_size/2` dimensions) is supported. The remaining dimensions pass through unchanged.
- Source: base.py:L157-L159 (query_rot = query[..., :rotary_dim], query_pass = query[..., rotary_dim:])

### 5. What complexity must our implementation preserve?

- **Two rotation styles**: Neox (half-split) and GPT-J (interleaved). Both formulas must be correct.
- **Rotary dim vs head size**: The kernel must handle rot_dim < head_size, passing through remaining dimensions.
- **Position-based cos/sin lookup**: The kernel must correctly index the cos_sin_cache at each token's position.
- **In-place operation**: Like vLLM's CUDA kernel, our Triton kernel modifies query/key in-place.
- **GQA support**: Key may have fewer heads than query (num_kv_heads <= num_heads).

## Source Mapping Table

| Our Code | vLLM Source | What We Changed & Why |
|----------|-------------|----------------------|
| `_rotary_embedding_kernel()` | `pos_encoding_kernels.cu:L79` `rotary_embedding_kernel` | Triton instead of CUDA. One program per (token, head) instead of one block per token. Used pre-gathered cos/sin for simplicity. |
| `apply_rotary_emb_triton()` | `common.py:L143` `ApplyRotaryEmb.forward_static()` | Triton kernel call instead of PyTorch loop. In-place modification. |
| `rotary_embedding()` | `base.py:L140` `RotaryEmbedding.forward_static()` | Same reshape → split rot/pass → apply → concat pattern. Uses Triton kernel for the applying step. |
| `apply_rotary_emb_ref()` | `common.py:L143` `ApplyRotaryEmb.forward_static()` | Exact match: Neox half-split + rotate and GPT-J interleave + rotate. |
| `rotary_embedding_ref()` | `base.py:L140` `RotaryEmbedding.forward_static()` | Exact match: index_select → chunk → split rot/pass → apply → concat. |

## Key Files Referenced

1. `vllm/vllm/model_executor/layers/rotary_embedding/base.py` — RotaryEmbedding class (304 lines)
2. `vllm/vllm/model_executor/layers/rotary_embedding/common.py` — ApplyRotaryEmb (290 lines)
3. `vllm/vllm/csrc/pos_encoding_kernels.cu` — CUDA RoPE kernel (196 lines)
4. `vllm/vllm/csrc/torch_bindings.cpp` — Torch custom op binding (line 260)
5. `vllm/vllm/csrc/ops.h` — Op declaration (line 159)
