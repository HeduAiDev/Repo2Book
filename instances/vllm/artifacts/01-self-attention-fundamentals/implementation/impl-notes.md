# Implementation Notes — Self-Attention (v2: Source-Grounded)

## Source Analysis — vLLM's Attention Architecture

### Files (with line numbers)
| File | Lines | Content |
|------|-------|---------|
| `vllm/model_executor/layers/attention/attention.py` | L177-L519 | `Attention` class — the layer that wraps backends |
| `vllm/model_executor/layers/attention/attention.py` | L189-L384 | `Attention.__init__()` — backend selection, impl creation |
| `vllm/model_executor/layers/attention/attention.py` | L409-L501 | `Attention.forward()` — reshape + dispatch to backend |
| `vllm/v1/attention/backend.py` | Full file | `AttentionBackend`, `AttentionImpl`, `AttentionMetadataBuilder` ABCs |
| `vllm/v1/attention/backends/flash_attn.py` | L594-L681 | `FlashAttentionImpl.__init__()` — scale, num_kv_heads, alibi |
| `vllm/v1/attention/backends/flash_attn.py` | L682-L703+ | `FlashAttentionImpl.forward()` — calls flash_attn_varlen_func() |
| `vllm/v1/attention/selector.py` | Full file | `get_attn_backend()` — auto-selects optimal backend |
| `vllm/v1/attention/backends/registry.py` | Full file | `AttentionBackendEnum` — all registered backends |
| `vllm/v1/attention/backends/triton_attn.py` | Full file | vLLM's Triton attention backend (uses ops/triton_prefill_attention.py) |
| `vllm/v1/attention/ops/triton_prefill_attention.py` | L37-L177 | `_fwd_kernel` — vLLM's actual Triton kernel (var-len, GQA, causal) |

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
- Source: `flash_attn.py:L682-L703` — `FlashAttentionImpl.forward()` receives `key=[num_tokens, num_kv_heads, head_size]` directly.

**Decision 5: scale is a constructor parameter, not a local computation**
- `scale = 1/√head_size` is computed once in model config and passed through to `Attention.__init__(scale=...)` and then to `impl_cls(scale=...)`.
- It is NEVER recomputed inside the Attention class. This signals: the scale is mathematically derived, not a tunable hyperparameter.
- Source: `attention.py:L193` (constructor takes `scale: float`), `attention.py:L345` (passed to `impl_cls`).

**Decision 6: head_size_v — separate V head dimension**
- vLLM supports `head_size_v != head_size` for architectures where V has a different dimension than Q/K (e.g., MLA in DeepSeek).
- For standard MHA/GQA: `head_size_v = head_size`. For Ch01 scope, we keep them equal.
- Source: `attention.py:L286` — `self.head_size_v = self.head_size if head_size_v is None else head_size_v`

## Source Mapping Table

| Our Implementation | vLLM Source | What We Changed & Why |
|---|---|---|
| `MultiHeadAttention.__init__()` | `attention.py:L189-L384` `Attention.__init__()` | Takes `(d_model, num_heads)` vs vLLM's `(num_heads, head_size, scale, ...)`. No backend abstraction, no KV cache spec, no quantization. Simplified for chapter scope. |
| `MultiHeadAttention.forward()` | `attention.py:L409-L501` `Attention.forward()` | Explicit attention computation instead of delegating to `self.impl.forward()`. Readers see the math. Returns `(output, attn_weights)` — vLLM returns output only. |
| `self.W_q, self.W_k, self.W_v` | Model files e.g. `llama.py` → `LlamaAttention.qkv_proj` | vLLM uses combined QKV; we separate for pedagogical clarity. |
| `self.scale = 1/sqrt(head_dim)` | `attention.py:L193` — `scale` constructor param, `L345` — passed to `impl_cls` | vLLM receives scale as a pre-computed float; we compute it locally since we own `head_dim`. |
| `_reshape_for_heads()` | `attention.py:L455-L460` — inline reshape in `forward()` | vLLM reshapes to `[num_tokens, heads, dim]` (3D, sequence-pack); we use `[B, h, L, d]` (4D, batch-aware) for readability. No such method exists in vLLM. |
| `GroupedQueryAttention` | `attention.py:L276-L280` (GQA in same class) + `flash_attn.py:L682-L703` (kernel native GQA) | vLLM handles GQA in the kernel via stride-based K,V reads; we expand K,V with `repeat_interleave` for visualization. |
| `create_causal_mask()` | `flash_attn.py:L256` — `causal: bool = True` flag, applied inside FA kernel | vLLM never materializes mask tensors — it's a boolean flag passed to the CUDA/Triton kernel. We materialize for testing and visualization. |
| `scaled_dot_product_attention()` | All backends compute this math internally | No single vLLM function is this pure — it's split across `FlashAttentionImpl`, `TritonAttentionImpl`, etc. |
| `_fused_attention_kernel` (Triton) | `triton_prefill_attention.py:L37-L177` `_fwd_kernel` | vLLM's kernel handles variable-length sequences (B_Start_Loc, B_Seqlen), GQA grouping, bidirectional sliding window, and uses `tl.math.exp2`. Our kernel is fixed-length, MHA-only, with `tl.exp`, and shows IS_CAUSAL as a constexpr flag. |
| `fused_attention_triton()` (wrapper) | `triton_prefill_attention.py:L37-L177` + `flash_attn.py:L797-L819` | vLLM has no exact equivalent — `Attention.forward()` dispatches via `unified_attention_with_output()` which ultimately calls the backend kernel. Our wrapper is a pedagogical simplification. |
| `validate_triton_vs_pytorch()` | `tests/kernels/attention/test_flash_attn.py` → `ref_paged_attn()` | Same pattern: generate inputs, run kernel + PyTorch reference, compare with tolerance. vLLM's version covers variable-length, GQA, sliding window, and multiple dtypes. |
| `create_padding_mask()` | `flash_attn.py:L276-L298` (metadata builder) + `triton_prefill_attention.py:L120` | vLLM handles variable-length via `cu_seqlens_q` and `seqused_k` — padding is implicit in the metadata, never a materialized mask tensor. The kernel masks with `pos_k < cur_batch_seq_len`. |
| `create_sliding_window_mask()` | `flash_attn.py:L618-L623` (impl init) + `triton_prefill_attention.py:L126-L135` | vLLM converts sliding_window to `(left, right)` tuple and applies it inside the kernel. Never materializes a mask tensor. Our version materializes for visualization. |
| `analyze_variance_empirically()` | `attention.py:L193, L348` (scale param) + `flash_attn.py:L613, L806` (scale stored & applied) | vLLM receives `scale = 1/√head_size` as a constructor parameter. This function empirically verifies WHY: Var(q·k) = d_k, and without scaling softmax collapses. |
| `demonstrate_variance_problem()` + `manual_softmax_example()` + `entropy()` | `attention.py:L193` (scale param) + `flash_attn.py:L806` (softmax_scale applied) | Pedagogical functions — no direct vLLM equivalent. Demonstrate the mathematical necessity behind vLLM's scale parameter choice using concrete examples. |
