# Implementation Notes — Self-Attention (v2: Source-Grounded)

## Source Analysis — vLLM's Attention Architecture

### Files (with line numbers)
| File | Lines | Content |
|------|-------|---------|
| `vllm/model_executor/layers/attention/attention.py` | L177-L582 | `Attention` class — the layer that wraps backends |
| `vllm/v1/attention/backend.py` | Full file | `AttentionBackend`, `AttentionImpl`, `AttentionMetadataBuilder` ABCs |
| `vllm/v1/attention/backends/flash_attn.py` | Full file | `FlashAttentionBackend` + `FlashAttentionImpl` |
| `vllm/v1/attention/selector.py` | Full file | `get_attn_backend()` — auto-selects optimal backend |
| `vllm/v1/attention/backends/registry.py` | Full file | `AttentionBackendEnum` — all registered backends |
| `vllm/v1/attention/backends/triton_attn.py` | Full file | vLLM's own Triton attention kernels |

### Key Classes
| Class | File:Line | Responsibility |
|-------|-----------|---------------|
| `Attention` | attention.py:L177 | Top-level layer. Owns `self.impl` (backend). Delegates computation. |
| `AttentionLayerBase` | attention_layer_base.py | Abstract base. `get_attn_backend()`, `get_kv_cache_spec()` |
| `AttentionBackend` | backend.py | ABC. `get_name()`, `get_impl_cls()`, `get_kv_cache_shape()` |
| `AttentionImpl` | backend.py | ABC. `forward(layer, query, key, value, kv_cache, attn_metadata, output)` |
| `FlashAttentionImpl` | flash_attn.py | Concrete. Calls `flash_attn_varlen_func()` from Dao-AILab |
| `TritonAttentionImpl` | triton_attn.py | Concrete. vLLM's custom Triton kernels |

### Key Design Decisions

**Decision 1: Backend abstraction separates "what" from "how"**
- The `Attention` class defines WHAT attention does (QKV → output).
- The backend defines HOW it's computed (FlashAttention CUDA vs Triton vs FlexAttention).
- This lets vLLM auto-select the best kernel for the available GPU and workload.
- Source: `attention.py:L177` creates `self.impl` from `get_attn_backend()`.

**Decision 2: Combined QKV projection in model files, not in Attention**
- vLLM's `Attention` does NOT own `q_proj/k_proj/v_proj`.
- These live in model files (e.g., `vllm/model_executor/models/llama.py`).
- Reason: different models have different QKV arrangements (MLA, GQA, etc.)
- We include them in our implementation for self-contained clarity.

**Decision 3: Opaque custom ops for torch.compile**
- `torch.ops.vllm.unified_attention_with_output` wraps the backend call.
- This prevents torch.compile from graph-breaking on the attention kernel.
- Source: `attention.py:L410-L450`

**Decision 4: GQA handled implicitly in the kernel**
- FlashAttention natively handles `num_kv_heads < num_heads`.
- K,V are NOT expanded in HBM — the kernel reads with stride.
- Source: `flash_attn.py → FlashAttentionImpl.forward()` passes `num_kv_heads` to `flash_attn_varlen_func()`.

## Source Mapping Table

| Our Implementation | vLLM Source | What We Changed & Why |
|---|---|---|
| `MultiHeadAttention.__init__()` | `attention.py:L177` `Attention.__init__()` | No backend abstraction, no KV cache spec, no quantization. Simplified for chapter scope. |
| `MultiHeadAttention.forward()` | `attention.py:L410` `Attention.forward()` | Explicit attention computation instead of delegating to `self.impl.forward()`. Readers see the math. |
| `self.W_q, self.W_k, self.W_v` | Model files e.g. `llama.py` → `LlamaAttention.qkv_proj` | vLLM uses combined QKV; we separate for pedagogical clarity. |
| `self.scale = 1/sqrt(head_dim)` | `attention.py:L200` | Identical. Pre-computed for efficiency. |
| `_reshape_for_heads()` | `attention.py:L410-L450` | vLLM reshapes to `[tokens, heads, dim]`; we use `[B, h, L, d]` for readability. |
| `GroupedQueryAttention` | `attention.py:L177` + `flash_attn.py` | vLLM handles GQA in the kernel; we expand K,V for visualization. |
| `create_causal_mask()` | Inside FlashAttention kernel (not materialized) | vLLM never creates mask tensors; we do for visualization. |
| `scaled_dot_product_attention()` | All backends compute this math internally | No vLLM function does this explicitly; it's split across backends. |
