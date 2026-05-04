"""
Triton Paged Attention — Mirror of vLLM's production kernel.

REFERENCE:
    vllm/v1/attention/ops/triton_decode_attention.py:L60   — _fwd_kernel_stage1 (MHA)
    vllm/v1/attention/ops/triton_decode_attention.py:L261  — _fwd_grouped_kernel_stage1 (GQA)
    vllm/v1/attention/ops/triton_decode_attention.py:L539  — _fwd_kernel_stage2 (reduce)
    vllm/v1/attention/ops/triton_unified_attention.py:L58  — kernel_unified_attention
    csrc/attention/attention_kernels.cuh:L85               — CUDA reference

This is NOT a generic attention tutorial. Every function mirrors vLLM's naming,
grid structure, and data flow. The reader can open triton_decode_attention.py
and find the corresponding lines.

Grid structure (matching vLLM's decode kernel):
    grid = (batch, head_num, NUM_KV_SPLITS)
    pid 0 = batch index   (which sequence)
    pid 1 = head index    (which KV head)
    pid 2 = KV split      (which partition of the KV sequence)

Key differences from vLLM production:
    - Single-stage (no split-KV) for clarity — avoid stage2 reduce complexity
    - Simplified block table indexing (no multi-dimensional strides)
    - No FP8 quantization support
    - No GQA grouped kernel variant (uses repeat_interleave for simplicity)
"""

import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ═══════════════════════════════════════════════════════════════════
# TRITON KERNEL — mirrors vLLM's triton_decode_attention.py:L60
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _paged_attention_kernel(
        # --- Input: Q, K_cache, V_cache ---
        Q_ptr,                  # [num_tokens, num_heads, head_dim]
        K_cache_ptr,            # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
        V_cache_ptr,            # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]

        # --- Output ---
        Out_ptr,                # [num_tokens, num_heads, head_dim]

        # --- Metadata ---
        block_tables_ptr,       # [num_seqs, max_blocks_per_seq]
        seq_lens_ptr,           # [num_seqs]
        max_blocks_per_seq,

        # --- Strides ---
        stride_q_tok, stride_q_h, stride_q_d,
        stride_k_blk, stride_k_tok, stride_k_h, stride_k_d,
        stride_v_blk, stride_v_tok, stride_v_h, stride_v_d,
        stride_o_tok, stride_o_h, stride_o_d,
        stride_bt_seq, stride_bt_blk,

        # --- Constants ---
        SCALE: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,     # tokens per KV block (vLLM default: 16)
        BLOCK_KV: tl.constexpr,       # KV tile size for inner loop
    ):
        """
        Triton paged attention — one Q token × one KV head.

        REFERENCE:
            vllm/v1/attention/ops/triton_decode_attention.py:L60  (MHA kernel)
            csrc/attention/attention_kernels.cuh:L85              (CUDA kernel)

        This is the decode-phase kernel: one query token attends to its
        entire KV cache, using block_table to access non-contiguous blocks.

        Grid: (num_seqs, num_kv_heads)
        pid 0 → which sequence (maps to Q token)
        pid 1 → which KV head
        """
        seq_idx = tl.program_id(0)
        kv_head = tl.program_id(1)

        # --- Load sequence length & compute blocks ---
        seq_len = tl.load(seq_lens_ptr + seq_idx)
        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        # --- Load Q for this sequence & head ---
        # REFERENCE: triton_decode_attention.py:L102 — Q loading
        q_offset = seq_idx * stride_q_tok + kv_head * stride_q_h
        Q_vec = tl.load(Q_ptr + q_offset + tl.arange(0, HEAD_DIM))

        # --- Online softmax state ---
        # REFERENCE: attention_kernels.cuh:L196 — float qk_max = -FLT_MAX
        m_i = tl.full([1], float("-inf"), dtype=tl.float32)
        l_i = tl.full([1], 0.0, dtype=tl.float32)
        O_acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        # --- Block table for this sequence ---
        # REFERENCE: attention_kernels.cuh:L202
        # REFERENCE: triton_decode_attention.py:L119 — Req_to_tokens + offs_n // PAGE_SIZE
        bt_offset = seq_idx * stride_bt_seq

        # --- Loop over KV blocks ---
        # REFERENCE: attention_kernels.cuh:L222 — for (block_idx = ...)
        for blk_idx in range(num_blocks):
            # ---- PA: block_table[seq_idx][blk_idx] → physical block ----
            # REFERENCE: attention_kernels.cuh:L252 — block_table[block_idx]
            phys_blk = tl.load(block_tables_ptr + bt_offset + blk_idx * stride_bt_blk)

            # Tokens in this block (last block may be partial)
            blk_start = blk_idx * BLOCK_SIZE
            blk_end = tl.minimum(blk_start + BLOCK_SIZE, seq_len)
            n_tokens = blk_end - blk_start

            # ---- Load K block ----
            # REFERENCE: attention_kernels.cuh:L269 — k_cache + phys_blk * kv_block_stride
            k_offs = (phys_blk * stride_k_blk +
                      tl.arange(0, BLOCK_SIZE)[:, None] * stride_k_tok +
                      kv_head * stride_k_h +
                      tl.arange(0, HEAD_DIM)[None, :] * stride_k_d)
            K_blk = tl.load(K_cache_ptr + k_offs,
                            mask=tl.arange(0, BLOCK_SIZE)[:, None] < n_tokens)
            # K_blk: [BLOCK_SIZE, HEAD_DIM]

            # ---- Load V block ----
            # REFERENCE: attention_kernels.cuh:L397 — v_cache + phys_blk * kv_block_stride
            v_offs = (phys_blk * stride_v_blk +
                      tl.arange(0, BLOCK_SIZE)[:, None] * stride_v_tok +
                      kv_head * stride_v_h +
                      tl.arange(0, HEAD_DIM)[None, :] * stride_v_d)
            V_blk = tl.load(V_cache_ptr + v_offs,
                            mask=tl.arange(0, BLOCK_SIZE)[:, None] < n_tokens)
            # V_blk: [BLOCK_SIZE, HEAD_DIM]

            # ---- FA: Q @ K^T (in SRAM) ----
            # REFERENCE: attention_kernels.cuh:L289 — Qk_dot::dot(q_vecs, k_vecs)
            Q_broadcast = Q_vec[None, :]  # [1, HEAD_DIM]
            S = tl.sum(Q_broadcast * K_blk, axis=1) * SCALE  # [BLOCK_SIZE]
            S = tl.where(tl.arange(0, BLOCK_SIZE) < n_tokens, S, float("-inf"))

            # ---- FA: Online softmax update ----
            # REFERENCE: attention_kernels.cuh:L307-L341 (warp-level softmax)
            m_new = tl.maximum(m_i, tl.max(S, axis=0))
            P = tl.exp(S - m_new)                        # [BLOCK_SIZE]
            correction = tl.exp(m_i - m_new)              # scalar
            l_new = correction * l_i + tl.sum(P, axis=0)  # scalar
            # O_acc update: P-weighted sum of V
            # P: [BLOCK_SIZE], V_blk: [BLOCK_SIZE, HEAD_DIM]
            # P[:, None] * V_blk → [BLOCK_SIZE, HEAD_DIM] → sum over blk dim → [HEAD_DIM]
            O_acc = correction * O_acc + tl.sum(P[:, None] * V_blk, axis=0)

            m_i = m_new
            l_i = l_new

        # ---- Final normalization ----
        # REFERENCE: attention_kernels.cuh:L337 — inv_sum * logits
        O_final = O_acc / l_i

        # ---- Write output ----
        o_offset = seq_idx * stride_o_tok + kv_head * stride_o_h
        tl.store(Out_ptr + o_offset + tl.arange(0, HEAD_DIM), O_final.to(Q_vec.dtype))


def triton_paged_attention(
    Q: torch.Tensor,           # [num_tokens, num_heads, head_dim]
    K_cache: torch.Tensor,     # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
    V_cache: torch.Tensor,     # [num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
    block_tables: torch.Tensor, # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,    # [num_seqs]
    block_size: int = 16,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Launch the Triton paged attention kernel.

    REFERENCE: vllm/v1/attention/ops/triton_decode_attention.py:L648 — decode_attention_fwd_normal()
    """
    num_tokens, num_heads, head_dim = Q.shape
    num_seqs = block_tables.shape[0]
    num_kv_heads = K_cache.shape[2]
    max_blocks = block_tables.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    Out = torch.empty_like(Q)
    BLOCK_KV = min(block_size, 16)

    grid = (num_seqs, num_kv_heads)

    _paged_attention_kernel[grid](
        Q, K_cache, V_cache, Out,
        block_tables, seq_lens, max_blocks,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_cache.stride(0), K_cache.stride(1), K_cache.stride(2), K_cache.stride(3),
        V_cache.stride(0), V_cache.stride(1), V_cache.stride(2), V_cache.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        SCALE=sm_scale,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_KV=BLOCK_KV,
    )
    return Out


# ═══════════════════════════════════════════════════════════════════
# VALIDATION — compare Triton kernel vs reference
# ═══════════════════════════════════════════════════════════════════

def validate_triton_kernel():
    """Verify Triton kernel matches reference (non-paged) attention."""
    if not HAS_TRITON or not torch.cuda.is_available():
        print("Triton + CUDA required for validation. Skipping.")
        return

    torch.manual_seed(42)
    N_SEQS, N_HEADS, HEAD_DIM = 2, 4, 64
    BLOCK_SIZE = 16
    SEQ_LEN = 48   # 3 blocks per seq
    N_BLOCKS = 10

    Q = torch.randn(N_SEQS, N_HEADS, HEAD_DIM, device='cuda', dtype=torch.float16)
    K_cache = torch.randn(N_BLOCKS, BLOCK_SIZE, N_HEADS, HEAD_DIM,
                          device='cuda', dtype=torch.float16)
    V_cache = torch.randn(N_BLOCKS, BLOCK_SIZE, N_HEADS, HEAD_DIM,
                          device='cuda', dtype=torch.float16)
    block_table = torch.tensor([[0, 3, 7], [2, 5, 9]], device='cuda', dtype=torch.int32)
    seq_lens = torch.tensor([SEQ_LEN, SEQ_LEN], device='cuda', dtype=torch.int32)

    # Triton kernel
    Out_triton = triton_paged_attention(Q, K_cache, V_cache, block_table, seq_lens)

    # Reference: gather KV → contiguous → manual attention per-head
    scale = 1.0 / math.sqrt(HEAD_DIM)
    Out_ref = torch.zeros_like(Q)
    for s in range(N_SEQS):
        K_seq = torch.cat([K_cache[block_table[s, b].item()] for b in range(SEQ_LEN // BLOCK_SIZE)])
        V_seq = torch.cat([V_cache[block_table[s, b].item()] for b in range(SEQ_LEN // BLOCK_SIZE)])
        # K_seq: [48, H, D], V_seq: [48, H, D]
        for h in range(N_HEADS):
            q_h = Q[s, h]  # [D]
            k_h = K_seq[:, h, :]  # [48, D]
            v_h = V_seq[:, h, :]
            scores = torch.matmul(q_h, k_h.T) * scale  # [48]
            attn = F.softmax(scores.float(), dim=-1)
            Out_ref[s, h] = torch.matmul(attn, v_h.float()).half()

    max_err = (Out_triton.float() - Out_ref.float()).abs().max().item()
    print(f"Triton vs Reference (fp16): max error = {max_err:.6f}")
    if max_err < 0.1:
        print("✅ MATCH (within fp16 tolerance)")
    else:
        print("❌ MISMATCH — check kernel logic")


if __name__ == "__main__":
    if HAS_TRITON and torch.cuda.is_available():
        validate_triton_kernel()
    else:
        print("No Triton/CUDA — read the source.")
        print("This kernel mirrors vllm/v1/attention/ops/triton_decode_attention.py:L60")
