#!/usr/bin/env python3
"""
Fused FlashAttention + PagedAttention — Runnable Demo.

REFERENCE:
    vllm/csrc/attention/attention_kernels.cuh:L85-L490  (paged_attention_kernel)
    vllm/v1/attention/ops/triton_decode_attention.py:L60 (Triton decode kernel)
    vllm/v1/attention/backends/flash_attn.py:L884       (_forward_with_dcp)

Run: python3 fused_attention_demo.py
Output: annotated trace of online softmax state across 3 KV blocks

This is the educational implementation that bridges Ch3 theory → vLLM source.
"""

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# CONFIG — small numbers for traceability
# ═══════════════════════════════════════════════════════════════════
NUM_TOKENS = 4        # Q tokens
NUM_KV = 12           # total KV tokens
HEAD_DIM = 8          # head dimension (small for clarity)
BLOCK_Q = 4           # Q tile size
BLOCK_KV = 4          # KV tile size
NUM_KV_BLOCKS = NUM_KV // BLOCK_KV  # 3 KV blocks
GQA_RATIO = 1         # num_heads = num_kv_heads for simplicity

SCALE = 1.0 / math.sqrt(HEAD_DIM)


# ═══════════════════════════════════════════════════════════════════
# SIMULATED KV CACHE — non-contiguous physical blocks
# ═══════════════════════════════════════════════════════════════════
# Physical KV cache: [num_physical_blocks, BLOCK_KV, HEAD_DIM]
# Blocks scattered non-contiguously in physical memory
torch.manual_seed(42)
K_physical = torch.randn(6, BLOCK_KV, HEAD_DIM)  # 6 physical blocks total
V_physical = torch.randn(6, BLOCK_KV, HEAD_DIM)

# --- Block table: logical → physical mapping ---
# REFERENCE: csrc/attention/attention_kernels.cuh:L202,L252
# block_tables: [num_seqs, max_blocks_per_seq]
# Sequence "A" has 3 logical blocks mapped to physical blocks 3, 1, 5
block_table = torch.tensor([[3, 1, 5]], dtype=torch.int32)  # seq 0's mapping
seq_len = torch.tensor([12], dtype=torch.int32)              # 12 tokens

# Q matrix: [1 query token, HEAD_DIM]
Q = torch.randn(1, HEAD_DIM)
Q = Q * SCALE  # pre-scale Q

print("=" * 65)
print("Fused FlashAttention + PagedAttention — Runnable Trace")
print("=" * 65)
print(f"Q: [1 token, d={HEAD_DIM}]  KV: [{NUM_KV} tokens, d={HEAD_DIM}]")
print(f"BLOCK_KV={BLOCK_KV} → {NUM_KV_BLOCKS} KV blocks")
print(f"Block table: logical 0→physical {block_table[0,0].item()}, "
      f"1→{block_table[0,1].item()}, 2→{block_table[0,2].item()}")
print()


# ═══════════════════════════════════════════════════════════════════
# ONLINE SOFTMAX + PAGED ATTENTION LOOP
# ═══════════════════════════════════════════════════════════════════
# REFERENCE: attention_kernels.cuh:L85 — paged_attention_kernel()
#
# This loop is what vLLM's CUDA kernel does in one pass:
#   for each logical KV block:
#       1. block_table[blk] → physical block ID  (PA indirection)
#       2. Load K, V from physical block          (PA non-contiguous load)
#       3. Q @ K^T * scale                        (FA compute in SRAM)
#       4. Online softmax update (m, l, correction, O_acc)  (FA)
#       5. Accumulate into O_acc

print("ONLINE SOFTMAX TRACE — 3 KV blocks, 1 Q token")
print("-" * 65)

# Running state (REFERENCE: attention_kernels.cuh:L85 — online softmax state)
m = torch.tensor([float("-inf")])  # running max  (L85: m_i = -inf)
l = torch.tensor([0.0])            # running exp sum (L85: l_i = 0)
O_acc = torch.zeros(1, HEAD_DIM)   # running output accumulator (fp32)

# --- Iterate over logical KV blocks ---
for blk_idx in range(NUM_KV_BLOCKS):
    # ---- STEP 1: PagedAttention — logical→physical indirection ----
    # REFERENCE: attention_kernels.cuh:L252
    physical_blk = block_table[0, blk_idx].item()  # ← THE KEY LINE in vLLM

    # ---- STEP 2: Load K, V from non-contiguous physical block ----
    # REFERENCE: attention_kernels.cuh:L269, L397
    K_blk = K_physical[physical_blk]  # [BLOCK_KV, HEAD_DIM]
    V_blk = V_physical[physical_blk]

    # ---- STEP 3: FlashAttention — Q @ K^T in "SRAM" ----
    # REFERENCE: attention_kernels.cuh:L222-L300 (QK dot product loop)
    S = Q @ K_blk.T  # [1, BLOCK_KV] — in real FA, this is in SRAM only
    S_flat = S.squeeze(0)

    # ---- STEP 4: Online Softmax Update ----
    # REFERENCE: attention_kernels.cuh:L307-L341 (warp-level softmax)
    m_new = torch.max(m, S_flat.max())                    # update running max
    P = torch.exp(S_flat - m_new)                         # stable exp
    correction = torch.exp(m - m_new)                     # rescale factor
    l_new = correction * l + P.sum()                      # update exp sum
    # P: [4], V_blk: [4, 8] → P @ V_blk = [8] → unsqueeze to [1, 8]
    O_acc = correction * O_acc + (P @ V_blk).unsqueeze(0)   # accumulate weighted V

    # ---- Print trace for this iteration ----
    print(f"Iteration {blk_idx+1} (logical block {blk_idx} → physical block {physical_blk}):")
    print(f"  S = {S_flat.tolist()}")
    print(f"  m: {m.item():>8.4f} → {m_new.item():>8.4f}  {'(max updated!)' if m_new > m else '(same)'}")
    print(f"  P = [{', '.join(f'{x:.4f}' for x in P.tolist())}]")
    print(f"  correction = exp({m.item():.4f} - {m_new.item():.4f}) = {correction.item():.6f}")
    print(f"  l: {l.item():.6f} → {l_new.item():.6f}")
    print(f"  O_acc (first 4 dims): [{', '.join(f'{x:.4f}' for x in O_acc[0,:4].tolist())}]")

    # Update state for next iteration
    m, l = m_new, l_new

# ---- STEP 5: Final normalization ----
# REFERENCE: attention_kernels.cuh:L432-L476 (warp-level reduction + normalize)
O_final = O_acc / l
print(f"\n{'=' * 65}")
print("FINAL: O = O_acc / l")
print(f"  l = {l.item():.6f}")
print(f"  O (normalized, first 4 dims): [{', '.join(f'{x:.4f}' for x in O_final[0,:4].tolist())}]")
print()

# ═══════════════════════════════════════════════════════════════════
# VERIFICATION: Compare against naive reference
# ═══════════════════════════════════════════════════════════════════
print("VERIFICATION: Compare against naive (full KV, contiguous, standard softmax)")

# Gather all KV into contiguous tensors (what naive attention does)
K_contig = torch.cat([K_physical[block_table[0, i].item()] for i in range(NUM_KV_BLOCKS)], dim=0)
V_contig = torch.cat([V_physical[block_table[0, i].item()] for i in range(NUM_KV_BLOCKS)], dim=0)

# Standard attention (Q is already pre-scaled on line 63)
S_ref = Q @ K_contig.T
P_ref = F.softmax(S_ref, dim=-1)
O_ref = P_ref @ V_contig

max_err = (O_final - O_ref).abs().max().item()
print(f"  Max error (fp32): {max_err:.10f}")
print(f"  {'✅ EXACT MATCH' if max_err < 1e-6 else '❌ MISMATCH — check online softmax logic'}")
print(f"\n{'=' * 65}")
print("This loop corresponds to vLLM's CUDA kernel:")
print("  csrc/attention/attention_kernels.cuh:L85  — paged_attention_kernel()")
print("  L202:  block_table = block_tables + seq_idx * max_num_blocks_per_seq")
print("  L252:  physical_block_number = block_table[block_idx]")
print("  L269:  k_ptr = k_cache + physical_block_number * kv_block_stride")
print("  L307:  warp-level max reduction (m_new)")
print("  L341:  warp-level sum reduction (l_new)")
print("  L432:  warp-level output reduction + normalize (O / l)")
