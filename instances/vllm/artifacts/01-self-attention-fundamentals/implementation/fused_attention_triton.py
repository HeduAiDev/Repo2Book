"""
Triton Fused Attention Kernel — Tiled MatMul + Online Softmax + Tiled Reduction.

This is the HPC core of Chapter 1. We implement the same algorithm as
FlashAttention (Dao et al., 2022) — tiled attention with online softmax —
in Triton, so you can see exactly how it works.

KEY INSIGHT: Why "fused" attention?
    Naive PyTorch attention:
        S = Q @ K^T          → writes [seq²] to HBM
        P = softmax(S)       → reads [seq²] from HBM, writes [seq²] to HBM
        O = P @ V           → reads [seq²] from HBM

    Problem: The [seq²] attention matrix never fits in SRAM for long sequences.
             Each read/write to HBM costs 100-1000× more than SRAM access.

    FlashAttention solution:
        - Split Q into blocks
        - For each Q block, iterate over K,V blocks
        - Compute softmax incrementally (online softmax)
        - Accumulate output without ever writing the full attention matrix

    Result: O(seq²·d) compute, but only O(seq·d) HBM writes. No [seq²] materialized!

This kernel is a simplified version of:
    Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness", NeurIPS 2022.

For vLLM's production attention, see:
    vllm/v1/attention/backends/flash_attn.py → FlashAttentionImpl (L594-L681)
    vllm/v1/attention/backends/triton_attn.py → TritonAttentionImpl
    vllm/v1/attention/ops/triton_prefill_attention.py → _fwd_kernel (L36-L177)
        — vLLM's actual Triton kernel: handles variable-length sequences,
           GQA grouping (cur_kv_head = cur_head // kv_group_num),
           bidirectional sliding window, and uses tl.math.exp2 for speed.
"""

import math
import torch
import torch.nn.functional as F

# Triton may not be available outside a GPU environment.
# The code is designed to be READ and UNDERSTOOD even without a GPU.
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════════════
# Triton Kernel: Fused Attention Forward
# ═══════════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _fused_attention_kernel(
        # Inputs
        Q_ptr,           # [batch, seq_len, num_heads, head_dim]
        K_ptr,           # [batch, seq_len, num_heads, head_dim]
        V_ptr,           # [batch, seq_len, num_heads, head_dim]
        # Outputs
        O_ptr,           # [batch, seq_len, num_heads, head_dim]
        # Scalars
        stride_qb, stride_qs, stride_qh, stride_qd,  # Q strides
        stride_kb, stride_ks, stride_kh, stride_kd,  # K strides
        stride_vb, stride_vs, stride_vh, stride_vd,  # V strides
        stride_ob, stride_os, stride_oh, stride_od,  # O strides
        B: tl.constexpr,       # batch size
        SEQ_LEN: tl.constexpr, # sequence length
        N_HEADS: tl.constexpr, # number of heads
        HEAD_DIM: tl.constexpr,# head dimension
        SCALE: tl.constexpr,   # 1/sqrt(head_dim) — the scale factor
        IS_CAUSAL: tl.constexpr, # whether to apply causal mask
        # Tuning parameters
        BLOCK_Q: tl.constexpr, # Q block size (rows)
        BLOCK_KV: tl.constexpr,# KV block size (rows)
    ):
        """
        TRITON FUSED ATTENTION — Single forward pass.

        This kernel computes:
            O = softmax(Q @ K^T / sqrt(d_k)) @ V

        WITHOUT materializing the full [seq, seq] attention matrix.

        HOW IT WORKS (the "online softmax" algorithm):

        For each Q block (size BLOCK_Q):
            1. Initialize:  O_acc = 0,  l_acc = 0,  m_prev = -inf
            2. For each KV block (size BLOCK_KV):
                a. Load Q_block [BLOCK_Q, HEAD_DIM] into SRAM
                b. Load K_block [BLOCK_KV, HEAD_DIM] into SRAM
                c. Compute S = Q_block @ K_block^T  →  [BLOCK_Q, BLOCK_KV]
                d. Online softmax update:
                   m_new = max(m_prev, row_max(S))
                   P = exp(S - m_new)                # numerically stable exp
                   l_new = exp(m_prev - m_new) * l_acc + row_sum(P)
                   P_normalized = P / l_new
                e. Load V_block [BLOCK_KV, HEAD_DIM]
                f. Update output: O_acc = diag(exp(m_prev - m_new)) * O_acc + P_normalized @ V_block
                g. m_prev = m_new, l_acc = l_new
            3. Write O_acc to output

        MEMORY ACCESS ANALYSIS:
            - Q: loaded once per Q block (SEQ/BLOCK_Q times)
            - K,V: loaded once per (Q block × KV block) pair
            - Attention matrix: NEVER stored to HBM
            - Output: written once (tiled accumulation in SRAM)

            Total HBM reads/writes:  O(SEQ²·HEAD_DIM / BLOCK_Q/BLOCK_KV) compute
                                    O(SEQ·HEAD_DIM) HBM traffic

        WHY THIS MATTERS FOR vLLM:
            In the prefill phase, we have long sequences (often 4K-128K tokens).
            The naive O(seq²) attention matrix would be:
                128K² × 2 bytes (bf16) = 32 GB — doesn't fit in any GPU!

            With tiled attention: we use O(BLOCK_Q × BLOCK_KV × HEAD_DIM) SRAM,
            which is ~128 × 128 × 128 × 2 bytes = 4 MB for typical block sizes.
        """

        # Program ID: which Q block and which (batch, head) this program handles
        pid_batch = tl.program_id(0)
        pid_head = tl.program_id(1)
        pid_q_block = tl.program_id(2)

        # Current Q block range
        q_start = pid_q_block * BLOCK_Q
        q_end = min(q_start + BLOCK_Q, SEQ_LEN)
        q_len = q_end - q_start

        # Pointers to Q, O for this batch/head
        Q_ptr_block = Q_ptr + pid_batch * stride_qb + pid_head * stride_qh
        O_ptr_block = O_ptr + pid_batch * stride_ob + pid_head * stride_oh

        # Initialize accumulators for online softmax
        # m: row-wise max (for numerical stability)
        # l: row-wise sum of exp(scores - m) (for normalization)
        m_i = tl.full([q_len], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([q_len], dtype=tl.float32)
        O_acc = tl.zeros([q_len, HEAD_DIM], dtype=tl.float32)

        # Load Q block once (reused across all KV blocks)
        Q_offs = (
            (q_start + tl.arange(0, BLOCK_Q))[:, None] * stride_qs
            + tl.arange(0, HEAD_DIM)[None, :] * stride_qd
        )
        Q_block = tl.load(Q_ptr_block + Q_offs, mask=(tl.arange(0, BLOCK_Q)[:, None] < q_len))
        Q_block = Q_block * SCALE

        # Loop over KV blocks
        for kv_start in range(0, SEQ_LEN, BLOCK_KV):
            kv_end = min(kv_start + BLOCK_KV, SEQ_LEN)

            # --- Load K block ---
            K_offs = (
                (kv_start + tl.arange(0, BLOCK_KV))[:, None] * stride_ks
                + tl.arange(0, HEAD_DIM)[None, :] * stride_kd
            )
            K_block = tl.load(
                K_ptr + pid_batch * stride_kb + pid_head * stride_kh + K_offs,
                mask=(tl.arange(0, BLOCK_KV)[:, None] < (kv_end - kv_start)),
            )  # [BLOCK_KV, HEAD_DIM]

            # --- Compute S = Q @ K^T ---
            # Q_block: [BLOCK_Q, HEAD_DIM],  K_block^T: [HEAD_DIM, BLOCK_KV]
            # S: [BLOCK_Q, BLOCK_KV]
            S = tl.dot(Q_block, tl.trans(K_block))

            # --- Apply causal mask ---
            # For causal attention: position i can only attend to positions ≤ i.
            # REFERENCE: triton_prefill_attention.py:L122-L123
            #   pos_q = offs_m[:, None]; pos_k = start_n + offs_n[None, :]
            #   mask &= pos_q >= pos_k   (IS_CAUSAL)
            if IS_CAUSAL:
                # Q positions: q_start + [0..BLOCK_Q), K positions: kv_start + [0..BLOCK_KV)
                q_pos = (q_start + tl.arange(0, BLOCK_Q))[:, None]  # [BLOCK_Q, 1]
                k_pos = (kv_start + tl.arange(0, BLOCK_KV))[None, :]  # [1, BLOCK_KV]
                causal_mask = q_pos >= k_pos  # [BLOCK_Q, BLOCK_KV]
                S = tl.where(causal_mask, S, float("-inf"))

            # --- Online Softmax Update ---
            # m_new = max(m_prev, row_max(S))
            m_new = tl.maximum(m_i, tl.max(S, axis=1))

            # P = exp(S - m_new) — numerically stable
            S_adj = S - m_new[:, None]
            P = tl.exp(S_adj)

            # Correction factor for old accumulator
            # Why? Previous output was computed with old max → need to rescale
            correction = tl.exp(m_i - m_new)

            # l_new = correction * l_old + sum(P)
            l_new = correction * l_i + tl.sum(P, axis=1)

            # --- Update Output: O_acc ---
            # Load V block
            V_offs = (
                (kv_start + tl.arange(0, BLOCK_KV))[:, None] * stride_vs
                + tl.arange(0, HEAD_DIM)[None, :] * stride_vd
            )
            V_block = tl.load(
                V_ptr + pid_batch * stride_vb + pid_head * stride_vh + V_offs,
                mask=(tl.arange(0, BLOCK_KV)[:, None] < (kv_end - kv_start)),
            )

            # O_acc_new = correction * O_acc_old + P @ V_block
            O_acc = (
                correction[:, None] * O_acc
                + tl.dot(P.to(V_block.dtype), V_block)
            )

            # Update state for next iteration
            m_i = m_new
            l_i = l_new

        # --- Normalize output ---
        # Final divide by l_i (the sum of exp scores)
        O_final = O_acc / l_i[:, None]

        # --- Write output ---
        O_offs = (
            (q_start + tl.arange(0, BLOCK_Q))[:, None] * stride_os
            + tl.arange(0, HEAD_DIM)[None, :] * stride_od
        )
        tl.store(O_ptr_block + O_offs, O_final, mask=(tl.arange(0, BLOCK_Q)[:, None] < q_len))


    def fused_attention_triton(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float = None,
        causal: bool = False,
        BLOCK_Q: int = 64,
        BLOCK_KV: int = 64,
    ) -> torch.Tensor:
        """
        User-facing function: run the Triton fused attention kernel.

        Args:
            Q: [batch, seq_len, num_heads, head_dim] — float16/bfloat16
            K: [batch, seq_len, num_heads, head_dim]
            V: [batch, seq_len, num_heads, head_dim]
            scale: 1/sqrt(head_dim). If None, computed automatically.
            causal: apply causal mask (position i sees only positions ≤ i).
            BLOCK_Q: Q tile size (rows). Tune this for your GPU.
            BLOCK_KV: KV tile size (rows).

        Returns:
            O: [batch, seq_len, num_heads, head_dim]

        REFERENCE: triton_prefill_attention.py:L36-L177 — vLLM's _fwd_kernel
        Differences from vLLM's kernel:
          - Ours is simplified: fixed-length sequences, no GQA grouping,
            no sliding window, uses tl.exp (not tl.math.exp2).
          - vLLM handles variable-length per batch (B_Start_Loc, B_Seqlen),
            GQA (cur_kv_head = cur_head // kv_group_num), and uses
            tl.math.exp2 for a small speedup on supported hardware.
          - vLLM's grid is (batch, heads, M_blocks) — same as ours.

        Tuning notes:
            - Larger BLOCK_Q/BLOCK_KV → more SRAM usage, fewer HBM passes
            - BLOCK_Q × BLOCK_KV × HEAD_DIM × sizeof(dtype) must fit in L1 cache
            - Typical: BLOCK_Q=64, BLOCK_KV=64 for H100 (228KB L1 per SM)
            - H100 has 228 KB L1 + shared memory per SM
            - Our tiles: 64×64×128×2 bytes ≈ 1 MB (fits easily)
        """
        B, SEQ_LEN, N_HEADS, HEAD_DIM = Q.shape

        if scale is None:
            scale = 1.0 / math.sqrt(HEAD_DIM)

        O = torch.empty_like(Q)

        # Grid: (batch, heads, Q_blocks)
        grid = (B, N_HEADS, triton.cdiv(SEQ_LEN, BLOCK_Q))

        _fused_attention_kernel[grid](
            Q, K, V, O,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B=B,
            SEQ_LEN=SEQ_LEN,
            N_HEADS=N_HEADS,
            HEAD_DIM=HEAD_DIM,
            SCALE=scale,
            IS_CAUSAL=causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
        )

        return O


else:
    # No Triton available — provide a pure-PyTorch fallback with the same API
    def fused_attention_triton(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: float = None,
        causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fallback when Triton is not available. Uses PyTorch's scaled_dot_product_attention
        which internally dispatches to FlashAttention when available.

        This lets you run the code to learn, even without a GPU.
        """
        return F.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=causal
        )


# ═══════════════════════════════════════════════════════════════════════════
# Validation & Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def validate_triton_vs_pytorch(d_model: int = 256, num_heads: int = 8, seq_len: int = 256):
    """
    Verify our Triton kernel produces the same output as PyTorch reference.
    This is the correctness oracle for our implementation.
    """
    from reference_attention import MultiHeadAttention, create_causal_mask

    head_dim = d_model // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Same input for both
    torch.manual_seed(42)
    x = torch.randn(2, seq_len, d_model, device="cuda", dtype=torch.float16)

    # Reference: PyTorch MHA
    mha = MultiHeadAttention(d_model, num_heads).to("cuda").to(torch.float16)
    with torch.no_grad():
        ref_out, _ = mha(x)

    # Triton: fused kernel
    # First, manually compute Q,K,V using the SAME weights
    with torch.no_grad():
        Q = mha._reshape_for_heads(mha.W_q(x))
        K = mha._reshape_for_heads(mha.W_k(x))
        V = mha._reshape_for_heads(mha.W_v(x))

        tri_out_heads = fused_attention_triton(Q, K, V, scale=scale)

        # Reshape back and apply output projection
        tri_out = tri_out_heads.transpose(1, 2).contiguous().view(2, seq_len, d_model)
        tri_out = mha.W_o(tri_out)

    # Compare
    max_err = (ref_out - tri_out).abs().max().item()
    rel_err = ((ref_out - tri_out).abs() / (ref_out.abs() + 1e-6)).mean().item()

    print(f"Max absolute error: {max_err:.6f}")
    print(f"Mean relative error: {rel_err:.6f}")
    print(f"Match: {'PASS' if max_err < 0.1 else 'FAIL — check kernel'}")

    if max_err < 0.1:
        print("✓ Triton kernel matches PyTorch reference (within fp16 tolerance)")
    else:
        print("✗ Kernel diverges — check online softmax logic")

    # Also test causal masking
    print("\n--- Causal variant ---")
    with torch.no_grad():
        tri_out_causal = fused_attention_triton(Q, K, V, scale=scale, causal=True)
        tri_out_causal = tri_out_causal.transpose(1, 2).contiguous().view(2, seq_len, d_model)
        tri_out_causal = mha.W_o(tri_out_causal)

    # Reference with causal mask
    causal_mask = create_causal_mask(seq_len, device=x.device).to(torch.bool)
    with torch.no_grad():
        ref_out_causal, _ = mha(x, attention_mask=causal_mask)

    max_err_causal = (ref_out_causal - tri_out_causal).abs().max().item()
    print(f"Max absolute error (causal): {max_err_causal:.6f}")
    print(f"Causal match: {'PASS' if max_err_causal < 0.1 else 'FAIL — check causal mask logic'}")

    return max(max_err, max_err_causal)


if __name__ == "__main__":
    if torch.cuda.is_available() and HAS_TRITON:
        print("CUDA + Triton available — running validation...")
        validate_triton_vs_pytorch()
    else:
        print("No GPU or Triton not installed.")
        print("Read the kernel source above — the logic is fully documented.")
        print("To run: pip install triton && python fused_attention_triton.py")
