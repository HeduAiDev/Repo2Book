# Knowledge: Attention (vLLM)

Repo-specific facts about attention kernels and FlashAttention integration.

---

## K10: FlashAttention kernel location

**Module**: attention
**Chapter**: 03-flashattention-pagedattention
**Discovered by**: implementer
**TTL**: permanent
**Access count**: 4

vLLM's FlashAttention integration: `vllm/v1/attention/backends/flash_attn.py`.
The key method is `FlashAttentionImpl.forward()` which dispatches to the appropriate
FlashAttention version based on the input shape and available hardware.

---

## K11: Triton decode attention kernel

**Module**: attention
**Chapter**: 03-flashattention-pagedattention
**Discovered by**: implementer
**TTL**: permanent
**Access count**: 3

`csrc/attention/triton_decode_attention.py:L60` — vLLM's custom Triton kernel for
PagedAttention during decode. Uses block_table indirection for KV cache lookup.
The grid structure is: (num_queries, num_kv_heads) with each program handling
one query head against all KV blocks.

---

## K12: Online softmax correctness

**Module**: attention
**Chapter**: 03-flashattention-pagedattention
**Discovered by**: writer (proof)
**TTL**: 60 days
**Access count**: 2

The online softmax algorithm is EXACT (not approximate). Proof by induction:
- Base case: m₀ = -∞, l₀ = 0 → O₀ = 0 (correct for empty set)
- Inductive step: correction = exp(m_old - m_new) rescales old accumulation
- The correction factor ensures old values are properly weighted in the new sum

**For writers**: MUST include this proof with intuition scaffolding (numerical trace first,
then algebraic proof). Without proof, the algorithm seems like magic.

**For reviewers**: If this proof is missing from a chapter that covers attention → auto-REJECT.

---

## K13: GQA/MQA KV head compression

**Module**: attention
**Chapter**: 01-self-attention-fundamentals
**Discovered by**: implementer
**TTL**: 60 days
**Access count**: 2

GQA (Grouped Query Attention): num_kv_heads < num_query_heads, KV heads are shared
across groups of query heads. MQA (Multi-Query Attention): num_kv_heads = 1 (extreme case).
The compression ratio = num_query_heads / num_kv_heads.

**For implementers**: When reimplementing attention, verify head count assumptions.
Mistaking query_heads for kv_heads causes silent shape bugs.
