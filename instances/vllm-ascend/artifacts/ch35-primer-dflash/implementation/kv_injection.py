"""
A small, paper-faithful reference implementation of DFlash's KV injection.

PAPER: arXiv:2602.06036 §4.1 ("Conditioning via KV injection enables
acceptance scaling") + Appendix A.3 (exact operator form). The target
model's hidden features from a fixed set of shallow-to-deep layers are
concatenated and projected once into a shared "target context feature":

    H_t = RMSNorm(W_c [H^(l1); ...; H^(l5)])                       (A.3)

Unlike EAGLE-3, which fuses target features with token embeddings only at
the draft model's *input* layer (diluting them as depth grows), DFlash
treats H_t as persistent contextual information and injects it into the Key
and Value projections of *every* draft layer. At layer i, the draft's own
hidden states H_d produce queries; both H_t and H_d are projected into keys
and values:

    Q_i = W_i^Q H_d
    K_i = [W_i^K H_t ; W_i^K H_d]_seq
    V_i = [W_i^V H_t ; W_i^V H_d]_seq

so target features only ever contribute additional Key/Value entries -- they
bypass the draft model's Q projection, output projection, self-attention
update, and FFN entirely.

This module also reproduces the *engineering* realization documented for
precompute_and_store_context_kv (§4.1: "projected features are stored in the
draft model's KV cache and reused across drafting iterations"): rather than
looping over draft layers one at a time, all L layers' K/V projection
weights are stacked into one matrix so a single fused GEMM computes every
layer's K/V at once, then reshaped into a layer-major layout. This is a pure
optimization of the same per-layer formula above (fewer kernel launches),
which is exactly what this module's test suite checks numerically.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# PAPER: Appendix A.3 (RMSNorm applied when forming H_t)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm(x) = x / rms(x) * weight, applied over the last dimension."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight


# PAPER: Appendix A.3, H_t = RMSNorm(W_c [H^(l1);...;H^(l5)])
def fuse_target_context_features(
    selected_layer_hidden_states: list[torch.Tensor],
    w_c: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    selected_layer_hidden_states: list of 5 tensors [num_ctx, target_hidden],
        the hidden states extracted from 5 layers "uniformly sampled from
        shallow to deep" (§4.1).
    w_c: [hidden_size, 5 * target_hidden] shared projection ("the only extra
        parameterized component", §Appendix A.3).

    Concatenates the 5 selected layers along the feature dimension, projects
    with the shared W_c (no bias -- the paper's W_c is a bare linear map),
    and RMSNorms the result into the shared target context feature H_t,
    reused unchanged by every draft layer.
    """
    concatenated = torch.cat(selected_layer_hidden_states, dim=-1)
    projected = F.linear(concatenated, w_c)
    return rms_norm(projected, norm_weight, eps)


# PAPER: Appendix A.3 (engineering realization -- stack per-layer K/V weights for one fused GEMM)
def build_fused_kv_weight(k_weights: list[torch.Tensor], v_weights: list[torch.Tensor]) -> torch.Tensor:
    """
    k_weights[i], v_weights[i]: [kv_size, hidden_size] per-layer K/V
    projection weights (K_i = W_i^K H_t, V_i = W_i^V H_t in Appendix A.3).

    Stacks them, [K_0; V_0; K_1; V_1; ...; K_{L-1}; V_{L-1}], into one
    [L * 2 * kv_size, hidden_size] matrix so a single GEMM computes every
    layer's K and V in one shot -- numerically identical to projecting each
    layer separately, just fewer kernel launches (mirrors
    precompute_and_store_context_kv's "fused GEMM" design).
    """
    interleaved = []
    for wk, wv in zip(k_weights, v_weights):
        interleaved.append(wk)
        interleaved.append(wv)
    return torch.cat(interleaved, dim=0)


# PAPER: Appendix A.3, K_i = W_i^K H_t / V_i = W_i^V H_t (per-layer reference form)
def precompute_layer_kv_looped(
    h_t: torch.Tensor,
    k_weights: list[torch.Tensor],
    v_weights: list[torch.Tensor],
    k_norm_weights: list[torch.Tensor],
    positions: torch.Tensor,
    num_kv_heads: int,
    head_dim: int,
    rms_eps: float = 1e-6,
    rope_base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects the shared context feature H_t into K/V for every draft layer
    one layer at a time -- the straightforward, unfused reading of Appendix
    A.3's per-layer K_i/V_i formulas. Only K is RMSNorm'd and RoPE'd here
    (per-head RMSNorm-K and rotary position encoding are the standard Qwen3
    attention preprocessing applied uniformly to every K projection, so that
    injected context K carries the same positional convention as the query
    tokens' own K at attention time); V passes straight through.

    Returns (all_k, all_v), each [num_layers, num_ctx, num_kv_heads, head_dim].
    """
    num_layers = len(k_weights)
    num_ctx = h_t.shape[0]
    all_k, all_v = [], []
    for i in range(num_layers):
        k_i = F.linear(h_t, k_weights[i]).view(num_ctx, num_kv_heads, head_dim)
        v_i = F.linear(h_t, v_weights[i]).view(num_ctx, num_kv_heads, head_dim)
        k_i = rms_norm(k_i, k_norm_weights[i], rms_eps)
        k_i = apply_rope(k_i, positions, rope_base)
        all_k.append(k_i)
        all_v.append(v_i)
    return torch.stack(all_k, dim=0), torch.stack(all_v, dim=0)


# PAPER: Appendix A.3 (engineering realization of the same K_i/V_i formulas via one fused GEMM)
def precompute_layer_kv_fused(
    h_t: torch.Tensor,
    fused_kv_weight: torch.Tensor,
    k_norm_weights: list[torch.Tensor],
    positions: torch.Tensor,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    rms_eps: float = 1e-6,
    rope_base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One fused GEMM (using build_fused_kv_weight's stacked matrix) computes
    K/V for every draft layer at once; a single view+permute then puts them
    in layer-major [num_layers, num_ctx, num_kv_heads, head_dim] layout so
    all_k[i]/all_v[i] are contiguous per-layer slices -- mirrors
    precompute_and_store_context_kv's "one GEMM for all layers" design.
    Must produce identical numbers to precompute_layer_kv_looped.
    """
    num_ctx = h_t.shape[0]
    kv_size = num_kv_heads * head_dim

    all_kv_flat = F.linear(h_t, fused_kv_weight)  # [num_ctx, num_layers * 2 * kv_size]
    all_kv = (
        all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )
    all_k, all_v = all_kv[0], all_kv[1]  # [num_layers, num_ctx, num_kv_heads, head_dim]

    all_k_normed = torch.empty_like(all_k)
    for i in range(num_layers):
        all_k_normed[i] = rms_norm(all_k[i], k_norm_weights[i], rms_eps)

    all_k_flat = all_k_normed.view(num_layers * num_ctx, num_kv_heads, head_dim)
    positions_repeated = positions.repeat(num_layers)
    all_k_flat = apply_rope(all_k_flat, positions_repeated, rope_base)
    all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)
    return all_k_final, all_v


# PAPER: Appendix A.3 (engineering support -- consistent positional encoding for injected K)
def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Standard NeoX-style rotary position embedding, applied per head over the
    last dimension of x: [num_tokens, num_heads, head_dim]. Not itself a
    distinct equation in Appendix A.3 (which only gives the bare K_i/V_i
    weight-projection formulas), but a necessary detail for the injected
    context K to carry a position axis consistent with the query tokens'
    own K -- otherwise attention over the concatenated K would have no
    positional information for the context half.
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    angles = positions.unsqueeze(-1).to(torch.float32) * freqs  # [num_tokens, half]
    cos = torch.cos(angles).unsqueeze(1)  # [num_tokens, 1, half]
    sin = torch.sin(angles).unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated.to(x.dtype)


# PAPER: Appendix A.3, Q_i = W_i^Q H_d; K_i=[W_i^K H_t;W_i^K H_d]; V_i=[W_i^V H_t;W_i^V H_d]
def dflash_layer_attention(
    h_d: torch.Tensor,
    context_k: torch.Tensor,
    context_v: torch.Tensor,
    positions: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rms_eps: float = 1e-6,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """
    One draft layer's cross-attention: query tokens (h_d, e.g. bonus + mask
    tokens) produce Q; K/V are the concatenation of the already-injected
    context_k/context_v (produced once by precompute_layer_kv_fused/_looped
    and shared by every layer's own attention call) with this layer's own
    projection of h_d. Note h_t/target hidden states never appear here as an
    argument used to form Q -- only their *already-projected* K/V do,
    exactly as Appendix A.3 specifies ("target features only serve as
    additional KV entries [...] They bypass the draft model's Q
    projection...").

    Attention is non-causal (full visibility) over the concatenated
    sequence: every query token sees the whole injected context and every
    other query token, matching the real proposer's `cad.causal = False`
    for the DFlash first pass.

    h_d: [num_query, hidden]; context_k/context_v: [num_ctx, num_kv_heads, head_dim].
    Returns [num_query, hidden].
    """
    num_query, hidden = h_d.shape
    num_ctx = context_k.shape[0]

    q = F.linear(h_d, w_q).view(num_query, num_heads, head_dim)
    k_d = F.linear(h_d, w_k).view(num_query, num_kv_heads, head_dim)
    v_d = F.linear(h_d, w_v).view(num_query, num_kv_heads, head_dim)

    q = rms_norm(q, q_norm_weight, rms_eps)
    k_d = rms_norm(k_d, k_norm_weight, rms_eps)
    q = apply_rope(q, positions, rope_base)
    k_d = apply_rope(k_d, positions, rope_base)

    # K_i = [W_i^K H_t ; W_i^K H_d], V_i = [W_i^V H_t ; W_i^V H_d] -- concat along the
    # sequence (token) axis, matching Appendix A.3's "_seq" subscript.
    k = torch.cat([context_k, k_d], dim=0)  # [num_ctx + num_query, num_kv_heads, head_dim]
    v = torch.cat([context_v, v_d], dim=0)

    # Repeat KV heads up to num_heads if grouped-query attention (num_heads > num_kv_heads).
    group = num_heads // num_kv_heads
    k = k.repeat_interleave(group, dim=1)  # [num_ctx+num_query, num_heads, head_dim]
    v = v.repeat_interleave(group, dim=1)

    q_t = q.transpose(0, 1)  # [num_heads, num_query, head_dim]
    k_t = k.transpose(0, 1)  # [num_heads, num_ctx+num_query, head_dim]
    v_t = v.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / (head_dim**0.5)
    weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(weights, v_t)  # [num_heads, num_query, head_dim]
    attn_out = attn_out.transpose(0, 1).reshape(num_query, num_heads * head_dim)

    return F.linear(attn_out, w_o)
