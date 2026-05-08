# Test Report — Ch11 DCP/PCP

**Tester**: tester-3@book-factory
**Date**: 2026-05-08
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer

## Summary

```
402 passed in 0.94s
```

| Module | Tests | Status |
|---|---|---|
| `test_parallel_state.py` | 48 | PASS |
| `test_world_topology.py` | 50 | PASS |
| `test_lse_combine.py` | 37 | PASS |
| `test_dcp_alltoall.py` | 32 | PASS |
| `test_seq_sharding.py` | 66 | PASS |
| `test_kv_cache_per_rank.py` | 36 | PASS |
| `test_attention_backend.py` | 34 | PASS |
| `test_dcp_vs_pcp_separability.py` | 24 | PASS |
| `test_fidelity.py` | 39 | PASS |
| `test_integration.py` | 36 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/11-dcp-pcp
/home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ -q
```

Floor target ≥150; achieved **402** (Ch10 baseline 311; Ch09 was 204; Ch08
was 144). Breadth comes from parametrising (dcp_size, pcp_size, tp_size)
combinations across attention-backend / parallel-state / sharding modules,
and reproducing Demo §1-§5 numerics bit-for-bit.

## Demo numerics — every headline number reproduced (writer quotes verbatim)

### §1 HBM-per-rank capacity sweep (Llama-70B at 128K)

```
Naive total KV bytes (no CP): 42,949,672,960 = 40.0 GB

(dcp, pcp)   per_rank_len    per_rank_bytes     as GB
(1,1)             131,072    42,949,672,960   40.0 GB
(1,2)              65,536    21,474,836,480   20.0 GB
(2,1)              65,536    21,474,836,480   20.0 GB
(2,2)              32,768    10,737,418,240   10.0 GB
(1,4)              32,768    10,737,418,240   10.0 GB
(4,1)              32,768    10,737,418,240   10.0 GB
(2,4)              16,384     5,368,709,120    5.0 GB
(4,4)               8,192     2,684,354,560    2.5 GB
```

Pinning tests: `test_section_1_naive_total_kv_verbatim`,
`test_section_1_full_table_verbatim`, `test_section_1_16x_reduction_at_total_cp_16`.

### §2 LSE-weighted combine (4 ranks)

```
Per-rank LSE (token 0, head 0):
  rank 0: 1.448093
  rank 1: 0.996192
  rank 2: 2.106473  ← lse_max
  rank 3: 1.629767

Per-rank normalized weights (token 0, head 0):
  rank 0: 0.209762
  rank 1: 0.133496
  rank 2: 0.405190
  rank 3: 0.251552

max abs error vs single-process FlashAttention = 3.33e-16 (theorem holds)
associativity error ((rank01)+(rank23) vs flat) = 2.22e-16
```

Pinning tests: `test_demo2_lse_for_rank2_token0_head0`,
`test_demo2_lse_max_token0_head0`, `test_demo2_per_rank_normalized_weights`,
`test_demo2_max_error_bound`, `test_section_2_associativity_pairwise`,
`test_section_2_a2a_equals_ag_rs_combine`.

### §3 AG+RS vs A2A NCCL ops + α-β bandwidth model (P=8, NVLink)

```
dcp_size  AG+RS ops  A2A ops  AG+RS bytes  A2A bytes   T_AG+RS μs  T_A2A μs  speedup
       2          3        2   67,108,864 34,078,720      1036.6     360.8    2.87x
       4          3        2   67,108,864 17,039,360      1036.6     190.4    5.44x
       8          3        2   67,108,864  8,519,680      1036.6     105.2    9.85x

A2A reduces NCCL ops by 33% per layer (3 → 2).
```

Pinning tests: `test_section_3_ag_rs_op_count_3`, `test_section_3_a2a_op_count_2`,
`test_section_3_a2a_reduces_ops_by_33pct`, `test_section_3_ag_rs_bytes_at_dcp_2`,
`test_section_3_a2a_bytes_dcp_2/4/8`, `test_section_3_alpha_beta_t_ag_rs_dcp_2`,
`test_section_3_alpha_beta_speedup_dcp_2/4/8`.

### §4 Striped vs contiguous causal-mask imbalance (cp=8, seq_len=64)

```
scheme                 interleave per-rank work (KV-attends)
contiguous                      8 [36, 100, 164, 228, 292, 356, 420, 484]
block-striped                   2 [204, 220, 236, 252, 268, 284, 300, 316]
striped (interleave=1)          1 [232, 240, 248, 256, 264, 272, 280, 288]

imbalance ratio (max/min):
  contiguous           = 13.44x  (rank-7=484, rank-0=36)
  block-striped (K=2)  = 1.55x
  striped (interleave=1) = 1.24x
```

Pinning tests: `test_demo4_contiguous_per_rank_work_verbatim`,
`test_demo4_block_striped_K2_per_rank_work_verbatim`,
`test_demo4_striped_per_rank_work_verbatim`,
`test_section_4_contiguous_imbalance_13_44x`,
`test_section_4_block_striped_imbalance_1_55x`,
`test_section_4_striped_imbalance_1_24x`.

### §5 5D mesh groups (world_size=16, tp=4, pcp=2, pp=2, dp=1, dcp=2)

```
world_size = ext_dp * dp * pp * pcp * tp = 1 * 1 * 2 * 2 * 4 = 16
total_cp_world_size = pcp * dcp = 2 * 2 = 4
num_dcp_subgroups per TP-group = tp/dcp = 2

TP groups (4): [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
DCP sub-groups (8, folded inside TP):
  [0,1] [2,3] [4,5] [6,7] [8,9] [10,11] [12,13] [14,15]
PCP groups (8, independent): [0,4] [1,5] [2,6] [3,7] [8,12] [9,13] [10,14] [11,15]
PP groups (8): [0,8] [1,9] [2,10] [3,11] [4,12] [5,13] [6,14] [7,15]
```

Pinning tests: `test_demo5_*` (parallel_state.py), `test_section_5_*` (integration),
`test_world_size_5d_full_product`.

## 7-trap fidelity verification — all pinned

### Trap A: "DCP doubles decode throughput at dcp_size=2" — WRONG

**Method**: `test_section_1_*` — HBM is the win axis.

DCP shards KV cache memory per rank by dcp_size, but per-rank attention
compute against KV is still ``cdiv(seq_len, dcp_size) × num_heads ×
head_dim`` plus 1 extra collective per layer. Demo §1 verbatim: total_cp=16
cuts HBM 16×, but throughput is workload-dependent. At short seq_len, comm
overhead can dominate.

### Trap B: "PCP halves prefill latency at pcp_size=2" — PARTIALLY TRUE

**Method**: `test_section_3_alpha_beta_*` — speedup depends on bandwidth.

Demo §3 α-β model: at dcp_size=2 with H100 NVLink (β=200 GB/s, α=10 μs)
A2A speedup is 2.87×. With slower PCIe networks and shorter prefills, comm
overhead can exceed compute. **Operator must measure prefill length × bandwidth before enabling.**

### Trap C: "Context parallel is just sequence parallel renamed" — WRONG

**Method**: `test_dcp_vs_pcp_separability` — distinct mechanisms.

Sequence parallel (Megatron) shards activations within a TP group's MLP/LN.
DCP shards stored KV cache. PCP shards prefill input sequence. The two are
ORTHOGONAL and coexist. ``parallel_state.py`` exposes both
``is_sequence_parallel`` AND ``_DCP``/``_PCP`` GroupCoordinators.

### Trap D: "DCP and PCP must match" — WRONG

**Method**: `test_dcp_vs_pcp_separability`, `test_section_5_*`.

They are separable axes. Production runs ``(tp=8, dcp=2, pcp=4)``. The
ONLY hard constraint is ``tp_size % dcp_size == 0`` (parallel.py:L474-L478).
PCP is independent of DCP. Test
``test_tp_must_be_divisible_by_dcp`` pins the constraint; multiple
``test_section_5_*`` tests pin (dcp, pcp) = (2, 2) and other valid combos.

### Trap E: "Context parallel is the same as TP for the attention layer" — WRONG

**Method**: `test_section_3_*` — different code paths.

TP shards the HEAD axis (each rank owns ``num_heads / tp_size`` heads,
full sequence per head); CP shards the SEQUENCE axis. Demo §3 shows CP's
α-β model is fundamentally different from TP's (CP uses LSE-weighted combine,
TP uses all_reduce). Different communication, different math, different
layer placement.

### Trap F: "Ring Attention is the canonical implementation in vLLM" — WRONG

**Method**: `test_section_2_a2a_equals_ag_rs_combine`,
`test_section_3_ag_rs_op_count_3`, `test_section_3_a2a_op_count_2`.

vLLM ships AllGather+ReduceScatter (default) or All-to-All (advanced). Both
are NCCL collectives, not P2P send/recv ring. Demo §2 verifies that A2A
and AG+RS produce IDENTICAL outputs when fed the same partials — the
difference is transport/buffer-packing, not algebra. Source
``dist.all_to_all_single`` at ``dcp_alltoall.py:L448`` confirms.

### Trap G: "Striped Attention is just renamed Ring Attention" — WRONG

**Method**: `test_section_4_*`, `test_demo4_*`.

Striped is a TOKEN-PARTITIONING scheme (token i → rank ``i % cp_size``,
controlled by ``cp_kv_cache_interleave_size``), independent of communication
pattern. Demo §4 verifies: contiguous=13.44× imbalance, striped=1.24×. The
imbalance is a LOAD-BALANCING property under causal masking, NOT a
communication choice.

## ep=1 vs ep=N forward equivalence — chain-break invariant

LSE-weighted combine is associative + commutative. Tests verify:
- Combined output bit-equivalent (≤ 1e-10) to single-process FlashAttention.
- Reordering ranks (full permutation) produces same output.
- Pairwise fold ((01)+(23)) equals flat 4-rank combine.

This is the Ch11 chain-break invariant.

## Coverage by module

1. **`parallel_state.py`** (48 tests): _DCP/_PCP singleton lifecycle;
   AssertionError on uninitialized accessor; backward-compat alias;
   tp_size % dcp_size == 0 hard constraint with parametric (tp,dcp) matrix;
   world_size = tp×pp×pcp×dp (DCP excluded); 5D mesh group construction
   for tp/dcp/pcp/pp/dp; partition-of-rank invariants; rank-in-group
   correctness for ranks 0,1,2,3,5; demo §5 verbatim TP/DCP/PCP/PP groups.

2. **`world_topology.py`** (50 tests): MeshConfig dataclass invariants;
   tp%dcp==0 validator; world_size formula excludes DCP; total_cp = pcp*dcp;
   num_dcp_subgroups = tp/dcp; per_rank_kv_fraction = 1/total_cp;
   process_name_for_rank conditional axis appending; 5D mesh frozen.

3. **`lse_combine.py`** (37 tests): NamedTuple return; shape contract;
   ground-truth equivalence vs reference_attention at N ∈ {2, 4, 8};
   global_lse matches reference; associativity (pairwise + left-fold);
   commutativity under all permutations; demo §2 verbatim numerics
   (rank LSEs, weights, max error 3.33e-16); single-rank fast path;
   NaN/+inf sanitization; base-2 path; uniform LSE → mean of partials;
   dominant-rank limit; reference_attention shape; split_attention contract.

4. **`dcp_alltoall.py`** (32 tests): ag_rs_op_count=3, a2a_op_count=2;
   33% reduction; payload formulas; α-β cost model; speedup verbatim at
   dcp ∈ {2, 4, 8}; A2A == AG+RS combine output (same algebra);
   CommCost dataclass.

5. **`seq_sharding.py`** (66 tests): get_dcp_local_seq_lens shape contract;
   dtype int32; per-rank sums equal global seq_lens at I ∈ {1, 4, 16};
   demo §4 verbatim per-rank length tables (4 cells × 3 interleaves);
   dcp_rank parameter behavior; causal_attention_work formula
   (sum=L(L+1)/2); demo §4 verbatim work tables (3 schemes); imbalance
   ratio 13.44/1.55/1.24; striped vs block-striped vs contiguous
   token→rank assignment.

6. **`kv_cache_per_rank.py`** (36 tests): KVCacheSpec dataclass; cdiv
   helper; max_memory_usage_bytes formula matches kv_cache_interface.py;
   page_size_bytes = 2 × block_size × num_kv_heads × head_size × bytes;
   per-rank memory inverse-proportional to dcp×pcp; LLAMA_70B_KV_SPEC
   constants; hbm_naive_total = 42,949,672,960 = 40.0 GB; demo §1 cells
   verbatim.

7. **`attention_backend.py`** (34 tests): __new__ discovery via try/except
   AssertionError → fallback to size-1/rank-0; total_cp = pcp*dcp;
   total_cp_rank = pcp_rank*dcp_world + dcp_rank; need_to_return_lse_for_decode
   gate (dcp>1 AND can_return_lse); subclass class-level flags
   (FlashAttn3MlaBackend, FlashAttnBackend, FlashInferBackend,
   RocmAiterMlaBackend); FlashAttn3Mla num_heads_q = num_heads × dcp;
   kernel_call_signature keys; subclass inherits __new__ discovery.

8. **`dcp_vs_pcp_separability.py`** (24 tests): CPRoles dataclass;
   both_match_required = False (Trap D); world_size_for excludes DCP;
   per_rank_kv_chunk = seq_len / (dcp × pcp); explain_separability message
   contains canonical (tp=8, dcp=2, pcp=4) example; explain_axis_difference
   keys; production-config validity matrix; constraint enforcement.

9. **`test_fidelity.py`** (39 tests): redundant deep checks across all
   modules — partition coverage, axis orthogonality, formula equivalence.

10. **`test_integration.py`** (36 tests): demo §1 HBM table verbatim;
    demo §2 LSE-weighted combine round-trip vs reference; demo §3 AG+RS
    vs A2A op count + α-β verbatim (4 dcp values); demo §4 imbalance
    ratios; demo §5 mesh groups; cross-module composition (parallel_state
    init → AttentionImplBase reads → seq_sharding partitions → lse_combine
    reconstructs); trap roster meta-test; demo-output.txt presence.

## Knowledge applied

- **D01-D15** (implementer-supplied) — every claim pinned by a test.
- **D16-D20** (tester-added, appended to
  `instances/vllm/knowledge/modules/dcp-pcp.md`) — composition-test
  witness requirements; HBM 40.0 GB headline; combine-takes-pre-summed
  contract; striped 1.24× (not 1.0×); A2A payload shrinks with dcp_size.

## Wisdom applied

- **W02** (`wisdom/testing.md`, "don't pass for the wrong reason"): every
  demo number is asserted exactly (per-rank work `[36, 100, 164, ..., 484]`
  not "approximately"; `physical_to_logical[-4:]=[5, 2, 0, 4]` exactly).
  Tests that check `total_cp_world_size == 4` are paired with assertions
  that BOTH `pcp_world_size > 1` AND `dcp_world_size > 1` (D16) so the
  composition isn't satisfied trivially.

## 5 framing tips for writer (surgical guidance)

These come from places where a paraphrase risks subtly misleading the
reader.

### Tip 1 — Lead Trap A with the (4,4)→2.5 GB datapoint, NOT the 16× ratio

A common framing mistake is "DCP × PCP scales HBM linearly". That's true
but ABSTRACT. Demo §1 has a punch: a 70B model serving 128K sequence
**costs 40.0 GB per rank without CP, 2.5 GB per rank with (dcp=4, pcp=4)**.
That 16× reduction crosses the H100 80GB barrier — the same hardware now
serves 128K instead of OOM-ing. Lead §11.1 with the absolute MEMORY savings
(40 → 2.5 GB), not the ratio. The ratio is the math; the absolute number
is the WHY.

### Tip 2 — Trap F needs the "same algebra, different transport" explanation

A naive reading would say "AG+RS and A2A produce different outputs". They
DON'T — test ``test_section_2_a2a_equals_ag_rs_combine`` proves it.
The difference is in BUFFER PACKING and NCCL OP COUNT, not in the math. Tell
the reader: "Both transports invoke the same LSE-weighted combine. A2A
bundles output+LSE into one buffer and uses 1 NCCL op + 2 Triton kernels.
AG+RS uses 2 NCCL ops with separate buffers. The output bytes on every rank
are bit-identical." Then quote ``dist.all_to_all_single`` at line 448 as
the production-side proof. This prevents readers from concluding "Ring vs
non-Ring" — vLLM ships neither Ring NOR a single-canonical implementation.

### Tip 3 — Striped 1.24× is "near-balanced" NOT "perfectly balanced"

Demo §4 says "striped (interleave=1) = 1.24x  (perfectly balanced)" but
the 1.24× ratio is NOT 1.0× — token 0 has 1 KV-attend, token 63 has 64,
and even round-robin gives rank 0 sum(1,9,17,...,57) = 232 vs rank 7
sum(8,16,...,64) = 288. Ratio 288/232 = 1.241. The phrase "perfectly
balanced" is a relative claim ("vs contiguous's 13.44×") not absolute.
Frame it carefully: "Striped achieves near-balance (1.24×) — an order of
magnitude better than contiguous (13.44×) but not perfect, because causal
mask still favors late-position tokens." Knowledge D19 captures this.

### Tip 4 — Total cp = pcp × dcp; total_cp_rank uses dcp_world_size as the multiplier

The composed CP rank formula `total_cp_rank = pcp_rank * dcp_world_size +
dcp_rank` (NOT `pcp_rank * total_cp_world_size + dcp_rank`) is subtle. The
multiplier is `dcp_world_size`, not the total. Why? Because in the
flattened total-CP-rank space, you "stride" by dcp_world_size when stepping
across PCP. Show the worked example for (pcp_rank=1, dcp_rank=0,
dcp_world=2): total = 2. Then (pcp_rank=1, dcp_rank=1, dcp_world=2): total
= 3. Walking through this in §11.5 with the (tp=4, pcp=2, dcp=2) demo
prevents the off-by-one that would silently corrupt every backend's slot
mapping.

### Tip 5 — DCP folds inside TP: world_size does NOT multiply by dcp

The most-cited Trap D evidence is `(tp=8, dcp=2, pcp=4)` is valid with
world_size = 8×4 = 32 (NOT 8×2×4 = 64). vLLM's
`multiproc_executor.py:L116-L121` enforces `world_size == tp * pp * pcp`.
DCP is folded inside TP — when you go from dcp=1 to dcp=2 in a fixed
world, you're NOT doubling GPUs; you're SUB-DIVIDING each TP group. Use
the §11.5 demo to walk through:
- 16 GPUs, TP=4, PCP=2, PP=2 → 4 TP groups of 4 each.
- DCP=2 inside TP=4 → each TP group of 4 splits into 2 DCP sub-groups of 2.
- 8 DCP sub-groups total; world_size unchanged.

This is the "DCP doesn't add hardware" framing that distinguishes it from
PCP, EP, DP — Trap D's clearest illustration.

## Backpressure gate

OPEN. Writer is clear to start.
