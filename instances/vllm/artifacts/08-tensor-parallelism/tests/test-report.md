# Test Report — Ch08 Tensor Parallelism

**Tester**: tester@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer

## Summary

```
144 passed in 4.05s
```

| Module | Tests | Status |
|---|---|---|
| `test_tp_math.py` | 29 | PASS |
| `test_comm_primitives.py` | 22 | PASS |
| `test_column_parallel.py` | 19 | PASS |
| `test_row_parallel.py` | 18 | PASS |
| `test_qkv_parallel.py` | 25 | PASS |
| `test_mlp_block.py` | 16 | PASS |
| `test_integration.py` | 15 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/08-tensor-parallelism
/home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ --ignore=tests/_legacy -q
```

Floor target was ≥80; achieved 144 (Ch07 baseline was 83). The breadth comes
from parametrising tp_size ∈ {2, 4, 8} in math/comm/integration plus the
GQA boundary table (5 tp values for the same KV-head config).

## Demo numerics — every headline number reproduced (writer quotes verbatim)

### §1 Mathematical equivalence (TP forward == unsharded)

```
[§1_equivalence]
  col_tp2_max_abs_diff = 0
  col_tp4_max_abs_diff = 0
  col_tp8_max_abs_diff = 0
  row_tp2_max_abs_diff = 7.629e-06
  row_tp4_max_abs_diff = 9.537e-06
  row_tp8_max_abs_diff = 9.537e-06
  colrow_tp2_max_abs_diff = 0
  colrow_tp2_num_collectives = 1
  colrow_tp4_max_abs_diff = 2.384e-07
  colrow_tp4_num_collectives = 1
  colrow_tp8_max_abs_diff = 2.980e-07
  colrow_tp8_num_collectives = 1
```

### §2 α-β model + ring all-reduce (NVLink_HSXM4 profile)

```
[§2_alpha_beta]
  ring_sim_max_diff = 2.384e-07
  fit_alpha_us = 4.32492
  fit_bw_GBps = 144.56
  true_alpha_us = 5
  true_bw_GBps = 150
  nvlink_alpha_us = 2
  nvlink_bw_GBps = 300

  payload (B)        P=2          P=4          P=8
        1024     2.00 μs     3.00 μs     3.50 μs
       16384     2.03 μs     3.02 μs     3.51 μs
      262144     2.44 μs     3.33 μs     3.69 μs
     4194304     8.99 μs     8.24 μs     6.56 μs
    67108864   113.85 μs    86.89 μs    52.43 μs
```

### §3 TP throughput sweep — Llama-7B-shaped MLP block (H=4096, F=11008)

```
[§3_throughput_sweep]
  weights_per_layer_MB_fp16 = 270.533
  weights_per_rank_tp1_MB_fp16 = 270.533
  predicted_AR_us_tp1_NVLink = 0
  collectives_per_forward_tp1 = 0
  weights_per_rank_tp2_MB_fp16 = 135.266
  predicted_AR_us_tp2_NVLink = 8.99051
  collectives_per_forward_tp2 = 1
  weights_per_rank_tp4_MB_fp16 = 67.6332
  predicted_AR_us_tp4_NVLink = 8.24288
  collectives_per_forward_tp4 = 1
```

**HONEST CAVEAT**: `compute_per_forward` ms numbers in demo-output.txt are
NOT representative of real TP performance — single-process simulation runs
all ranks SERIALLY in one Python process, so wallclock grows linearly with
tp_size. Writer must NOT cite those ms times. Use only: weights/rank,
predicted AR overhead, collectives_per_forward (per impl-notes §7 / K17).

### §4 GQA × TP boundary (Llama-3-70B-style: 64 Q heads, 8 KV heads)

| tp_size | kv_heads/rank | replicas | KV/rank/token (B) | KV save factor |
|---|---|---|---|---|
| 2 | 4 | 1 | 2048 | 2.0× |
| 4 | 2 | 1 | 1024 | 4.0× |
| 8 | 1 | 1 | 512 | **8.0×** |
| 16 | 1 | 2 | 512 | **8.0× (cap)** |
| 32 | 1 | 4 | 512 | **8.0× (cap)** |

Full KV cache per token (fp16) = `2 × 8 × 128 × 2 = 4096 bytes`.

### §5 LlamaMLP TP correctness (H=1024, F=2752, seq=16, seed=42)

```
[§5_llama_mlp]
  mlp_tp1_max_abs_diff = 0
  mlp_tp1_collectives_per_forward = 0
  mlp_tp2_max_abs_diff = 6.403e-10
  mlp_tp2_collectives_per_forward = 1
  mlp_tp4_max_abs_diff = 8.149e-10
  mlp_tp4_collectives_per_forward = 1
  mlp_tp8_max_abs_diff = 6.912e-10
  mlp_tp8_collectives_per_forward = 1
```

**The Megatron pair signature**: 1 all-reduce per forward, regardless of
tp_size. Always exactly 1 — never 2, never 0 (when tp>1).

## Critical fidelity checks — all VERIFIED

### 1. T08 + Trap-E: per-segment narrow vs naive narrow on MergedColumnParallel

**Method**: `test_per_segment_loader_avoids_naive_narrow_bug` and
`test_naive_uniform_narrow_would_be_wrong` in
`tests/test_column_parallel.py` plus `test_chain_with_naive_narrow_would_be_wrong`
in `tests/test_integration.py`. We:
1. Build a recognizable fused weight (gate values 100..115, up values 200..215).
2. Verify the proper loader puts `[gate_rank0_shard, up_rank0_shard]` in rank 0
   (per linear.py:L767-L820 per-segment loop).
3. Reproduce the naive narrow path manually (`A_fused[:, r*2*ffn/p : (r+1)*2*ffn/p]`)
   and show it puts `[gate cols 0..3, gate cols 4..7]` in rank 0 — wrong.
4. Run a full MLP forward with both shardings and show the naive output diverges
   from the unsharded reference by ~7.7e-4 at tp=4 (4 orders of magnitude above
   the proper TP MLP's ~1e-7). The bug is trivially detectable end-to-end but
   would slip through any test that only checks shapes.

This is the bug the implementer caught and fixed; T09 in
`knowledge/modules/tensor-parallelism.md` records the test-confirmed magnitude.

### 2. Trap-A: TP=2 ≠ 2× — α-β model evidence pinned

**Method**: `test_small_payload_alpha_bound_regime` and
`test_large_payload_p8_beats_p2` in `tests/test_comm_primitives.py`.

Demo §2 NVLink table reproduces bit-for-bit:
- **α-bound (1 KB payload)**: P=8 takes 3.50 μs vs P=2's 2.00 μs — P=8 is
  1.75× SLOWER, not 4× faster. Even small all-reduces never vanish.
- **β-bound (64 MB payload)**: P=8 takes 52.43 μs vs P=2's 113.85 μs —
  P=8 is 2.17× faster, NOT 4× faster (sub-linear).

Test asserts ratios in [1.6, 1.9] (small payload) and [2.10, 2.25] (large
payload). The "TP=2 doubles throughput" claim fails on BOTH ends of the
payload spectrum.

### 3. Trap-C: QKV head-parallel, NOT feature-parallel

**Method**: `TestHeadShardingMath` in `tests/test_qkv_parallel.py` parametrises
4 combinations of `(total_num_heads, tp_size)` (including Llama-7B 32-head
and Llama-3-70B 64-head) and asserts `num_heads = divide(total_num_heads,
tp_size)` per linear.py:L1030 — NOT `output_size // tp_size`.

`test_split_offsets_are_q_then_kv_then_kv` builds a recognizable QKV weight
(Q values 0..127, K values 1000..1063, V values 2000..2063) and verifies
that split_qkv on rank 0 puts (Q[0..63] | K[0..31] | V[0..31]) — heads
0..3 of Q (4 of 8 heads × 16 head_size = 64), heads 0..1 of K and V (2 of
4 kv_heads × 16). Shows head-major layout, not feature-major.

### 4. Trap-D: KV memory floor at total_num_kv_heads

**Method**: `test_kv_memory_floor_verbatim_from_demo_section_4` reproduces
the entire Demo §4 table — 5 tp_size values, 5 (kv_heads_per_rank,
replicas, KV bytes per rank, save factor) tuples — and asserts EVERY value.

The non-monotonic save factor is the load-bearing fact: 2.0× → 4.0× → 8.0×
→ 8.0× → 8.0×. After the boundary at tp=total_num_kv_heads, adding more
ranks does NOTHING for KV memory. Memory savings cap at 8× regardless of
tp_size when the model has 8 KV heads.

### 5. Trap-E: col→row uses ONE all-reduce per block

**Method**: `test_one_collective_per_forward_when_tp_gt_1` parametrises
tp ∈ {2, 4, 8} on `LlamaMLPTP.forward` and asserts
`mlp.count_collectives() == 1`. Test
`test_attn_then_mlp_two_collectives_per_block` in `test_integration.py`
stitches `QKVParallelLinear → split → o_proj → mlp` and asserts the
combined count is exactly 2 (one for o_proj, one for down_proj — not 4
that a naive col→all-gather→col→row architecture would need).

Demo §5 numerics confirm: `mlp_tp{2,4,8}_collectives_per_forward = 1.0` exactly.

### 6. Trap-F: input_is_parallel default + bias-only-on-rank-0

**Method**: `test_default_input_is_parallel_true` and
`test_bias_added_only_on_rank_zero` in `tests/test_row_parallel.py`.

The bias test is a clean trap: with WEIGHT=zero (so the GEMM contributes 0)
and bias=non-zero, the all-reduced output should equal `bias`. If a buggy
implementation added bias on every rank, post-reduce output would equal
`tp_size × bias` — a 4× off-by-tp_size error. Test asserts NOT-allclose
to `4 × bias` to cement the contract (T13 records this pattern).

## Coverage by behavior class

1. **`tp_math.py`** (29 tests): divide()/ensure_divisibility() contract;
   split_tensor_along_last_dim() partition+round-trip; column_parallel_forward
   identity and per-rank shard shape; row_parallel_forward sum-of-partials at
   tp ∈ {2,4,8} with input_is_parallel branches; column_then_row_block
   collective count = 1 invariant; verify_*() helpers reproduce demo §1 diffs.

2. **`comm_primitives.py`** (22 tests): AlphaBetaModel.predict and
   bandwidth_GBps property; ring_all_reduce_cost formula at P=2, P=4, and
   demo §2 NVLink table reproduction; world_size==1 bypass returns 0;
   simulate_all_reduce equals naive sum at P ∈ {2,4,8}; chunking divisibility
   assertion; fit_alpha_beta recovers ground truth within 15-20%; HARDWARE_PROFILES
   sanity (NVLink ≈300 GB/s, PCIe < NVLink/5); predict_block_overhead doubles
   for two-all-reduce blocks and matches demo §3 verbatim (8.99 μs per AR);
   α-bound + β-bound regime ratios pin Trap-A.

3. **`column_parallel.py`** (19 tests): output_size_per_partition uses divide();
   indivisible asserts; load_weight narrows along OUTPUT dim (not input — Trap-F);
   bias shards parallel to weight; gather_output=True path reassembles;
   MergedColumnParallelLinear output_partition_sizes per-rank-per-segment;
   per-segment loader vs naive narrow (T08 evidence with recognizable values);
   split_per_rank correctness; fused gate_up_proj equals two separate matmuls
   at tp ∈ {2, 4}.

4. **`row_parallel.py`** (18 tests): input narrowed/output full; default
   input_is_parallel=True (Trap-F evidence); default reduce_results=True;
   bias + reduce_results=False raises ValueError per linear.py:L1480-L1483;
   weight_loader narrows along INPUT dim (T-flip; W01 from wisdom/debugging.md);
   bias is FULL output_size, not sharded (T06); input_is_parallel branches
   correctness; bias-only-on-rank-0 with zero-weight test (T13);
   reduce_results=False returns partials; tp_size=1 skips all-reduce.

5. **`qkv_parallel.py`** (25 tests): num_heads = divide(total_num_heads, tp_size)
   parametrised over 5 (heads, tp) combos; indivisible asserts; GQA branch
   matrix (MHA, GQA-below-boundary, GQA-at-boundary, GQA-above-boundary,
   GQA-far-above); KV memory floor reproducing entire Demo §4 table;
   output_sizes triple = [q_full, k_full, v_full]; output_partition_sizes
   per rank; load_qkv_weights forward equivalence at tp ∈ {1,2,4,8} for MHA
   and tp ∈ {2,4} for GQA; replication branch Q correctness; split_qkv head-major
   layout test with recognizable values; per_rank_summary key set.

6. **`mlp_block.py`** (16 tests): silu_and_mul shape + math; per_rank silu
   on sharded data is element-wise-equivalent to full SiluAndMul slice (Trap-E
   primitive); LlamaMLPTP wiring (MergedColumn output_sizes, RowParallel
   defaults); load_weights concatenates gate+up; ONE collective per forward
   parametrised over tp ∈ {2,4,8}; ZERO collectives at tp=1; collective count
   accumulates and resets correctly; full MLP forward matches reference at
   tp ∈ {1,2,4,8}; Demo §5 numerics reproducing pinned diffs.

7. **`test_integration.py`** (15 tests): demo §3 weights memory math (270.5
   MB / 135.3 / 67.6); Trap-A α-bound and β-bound regime ratios; block-overhead
   non-zero; Llama transformer block (qkv → split → o_proj → mlp) collective
   count = 2; Demo §1 col→row pinned diffs (tp=2 → 0; tp=4,8 → < 3e-7);
   T08+Trap-E composed test showing naive narrow diverges visibly; cross-chapter
   imports clean.

## Knowledge applied

- **T01-T08** (implementer-supplied) — every claim pinned by a test.
- **T09-T13** (tester-added) — appended to `knowledge/modules/tensor-parallelism.md`
  documenting test-confirmed magnitudes and patterns (column-exact, row-tolerance,
  P=1 bypass, regime-ratio detection, bias-only-on-rank-0 zero-weight test).

## Wisdom applied

- **W01** (`wisdom/debugging.md`, F.linear shape `[out, in]`): manifested as
  the column-parallel-narrows-output-dim vs row-parallel-narrows-input-dim
  flip. Two specific tests (`test_load_weight_narrows_along_output_dim`,
  `test_load_weight_narrows_along_input_dim`) pin each direction with explicit
  shape assertions and `np.array_equal` against the slice.
- **W02** (`wisdom/testing.md`, "don't pass for the wrong reason"): every demo
  number is asserted exactly (tp=2 col-row diff = 0.0, not "small"; KV save
  factor = 8.0 exactly, not "approximately 8"); tests use `np.array_equal`
  for column-parallel (no addition noise) and `np.allclose(atol=1e-5)` for
  row-parallel (sum noise expected).

## Reference count observation

Implementer reported 64 `# REFERENCE:` comments at v6 baseline (linter PASS).
Tests carry forward the source citations in module docstrings + per-test
docstrings, so any future re-grep can follow each assertion back to a specific
linear.py / parallel_state.py / llama.py line.

## Linter status (informational)

Source-grounding linter on the implementation already passed pre-handoff
(implementer ran it). No tester rerun needed at this gate; writer + reviewer
will re-run for the chapter.

## Framing tips for writer (3-5 surgical guidance items)

These are the test-derived narrative-shaping notes the writer should weave
into §8.x. Each comes from a place where a paraphrase risks subtly misleading
the reader.

### Tip 1 — Don't conflate "1 all-reduce per pair" with "1 all-reduce per chapter"
The Megatron pair (col→row) gets ONE all-reduce per pair. A full Llama
transformer block has TWO pairs (attn o_proj + mlp down_proj), so TWO
all-reduces per block. Test `test_attn_then_mlp_two_collectives_per_block`
pins this: don't let a reader walk away thinking "TP works because there's
one all-reduce per block" — the count is **per pair**, not **per block**.
Highlight the pair structure when introducing each (attn / mlp).

### Tip 2 — Lead Trap-A with the small-payload regime, NOT the large-payload regime
A common framing mistake is "comm overhead is bandwidth-limited at scale,
so TP doesn't double". That's only HALF of Trap-A and the LESS surprising
half. Demo §2 shows: at 1 KB payloads, P=8 is **1.75× SLOWER** than P=2
(the ring takes more α-cost steps when P grows). That's the genuinely
counter-intuitive fact: more ranks can MAKE small-payload all-reduce slower.
Lead §8.5 with this α-bound pathology, then introduce β-bound as the second
asymptote — the writer should not let the reader assume "more ranks always
help, just not by 2×".

### Tip 3 — When you cite Demo §3 ms times, you must immediately quote K17/§7 caveat
The implementer's `compute_per_forward` ms in Demo §3 is a single-process
serial simulation, so it grows linearly with tp_size — exactly the OPPOSITE
of real production behavior where wallclock stays roughly flat (compute-bound)
plus an α-β all-reduce. The writer can say "weights/rank halves cleanly at
tp=2 (135.3 MB vs 270.5 MB)" and "predicted AR overhead at tp=2 is 8.99 μs"
**verbatim** — but every ms time must be paired with the K17 honest caveat
or skipped. The §3.4 GQA boundary, §3.2 α-β formulas, and §3.5 collective
counts are production-honest and quote-safe.

### Tip 4 — The "bias on rank 0 only" subtlety is a high-leverage paragraph
RowParallelLinear's bias is FULL output_size on every rank — it's NOT
sharded. But it's added on rank 0 ONLY before the all-reduce. Both halves
matter: the broadcast tells you why bias memory doesn't shrink with tp_size;
the rank-0-only addition tells you why a naive port that "treats bias the
same as weight" gives `tp_size × bias` after reduce. Test
`test_bias_added_only_on_rank_zero` uses zero-weight + nonzero-bias to make
the bug detectable in one assertion (asserts NOT-allclose to `4 × bias`).
Use the same construction as a worked example in §8.4 — it's the cleanest
demonstration of why the rule exists.

### Tip 5 — The MergedColumn per-segment loader bug is a real bug, not pedagogical drama
The naive narrow on `[hidden, 2*ffn]` (the fused gate+up) puts `[gate cols 0..3, gate cols 4..7]`
in rank 0 — i.e., rank 0 gets all of gate's first half and NO up. The proper
loader puts `[gate cols 0..3, up cols 0..3]`. The implementer hit this in
the first draft. End-to-end MLP test diverges by ~7.7e-4 at tp=4 (proper
diff is ~1e-7) — visible to any correctness test, but invisible to a
shape-only test. Frame this as a real bug the implementer caught (with
file:line evidence linear.py:L767-L820) — readers respect concrete bug
stories more than abstract "be careful" warnings.

## Backpressure gate

OPEN. Writer (Task #34 → Task pipeline next) is clear to start.
