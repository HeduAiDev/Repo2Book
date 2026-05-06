# Tensor Parallelism Knowledge

T-prefix IDs to avoid collision with K-prefix scheduler entries and P-prefix
preemption entries (per Ch07 lesson, learn.py append-mode is brittle on
ID collision — write the file directly).

Source pin: vLLM `98661fe`.

---

## T01–T05: [COMPACTED 2026-05-07] TP foundation — divide(), _TP singleton, output_partition_sizes LIST, GQA replication, all-reduce wrapper

**Module**: tensor-parallelism
**Chapter**: 08-tensor-parallelism
**Role**: implementer
**Compacted by**: book-editor (manual; `learn.py compact` non-functional — `_parse_module_file` returns `[]`)
**Archive**: `knowledge/archive/tensor-parallelism-20260507-t01-t05.json` (full original text, 5 facts)
**Citation anchors preserved**: All five `### T01:` … `### T05:` headings below survive heading-level grep. Tests/narrative/brief references to T01-T05 by ID remain stable.

### T01: divide() is the universal asserting divisor for every TP shard size
Source: `vllm/distributed/utils.py:L60-L64`. `def divide(num, den): assert num % den == 0; return num // den`. Every shard size in vLLM passes through `divide()`: `output_size_per_partition` (linear.py:L454), `input_size_per_partition` (linear.py:L1447), `num_heads` (linear.py:L1030), `num_kv_heads` (linear.py:L1035), `num_kv_head_replicas` (linear.py:L1033). It is THE contract — `tp_size | total_heads`, `tp_size | hidden_size`, etc. — enforced at construction, not deep in the GEMM. If this assert fires, model config is incompatible with chosen tp_size.

### T02: TP group is a module-level singleton _TP, retrieved via get_tp_group()
Source: `parallel_state.py:L1494-L1592` (`initialize_model_parallel` sets `_TP`); `L1229-L1235` (`get_tp_group` retrieves). `_TP` is a `GroupCoordinator` wrapping a `torch.distributed.ProcessGroup`. No per-layer TP context — layers call `get_tp_group()` directly. `tensor_model_parallel_all_reduce` (`communication_op.py:L12-L14`) is literally `return get_tp_group().all_reduce(input_)`. For testing: `ensure_model_parallel_initialized` (`parallel_state.py:L1738-L1781`) asserts the world is set up before any layer is constructed. In our single-process simulation we hold all ranks' state in `self.rank_states[r]`; production has one process per rank.

### T03: ColumnParallel's output_partition_sizes is a LIST when fused
Source: `linear.py:L455-L460`. `self.output_partition_sizes = [self.output_size_per_partition]; if hasattr(self, 'output_sizes'): self.output_partition_sizes = [divide(s, tp_size) for s in self.output_sizes]`. Subclasses (`MergedColumnParallelLinear`, `QKVParallelLinear`) set `output_sizes` BEFORE calling `super().__init__()` so the parent reads it via `hasattr` (MRO trick). One matmul produces multiple sharded outputs. **Critical for the loader**: each segment is sharded INDEPENDENTLY along the output dim (`linear.py:L767-L820`); a naive narrow on the fused output puts `[gate_rank0, gate_rank1, …]` in rank 0 instead of `[gate_rank0_shard, up_rank0_shard]`. T09 (tester) confirmed: naive narrow shows ~7.7e-4 max-abs-diff at tp=4 vs proper loader's ~1e-7 — 4-5 orders of magnitude. Bug is LATENT if you only check forward shapes, not cross-rank correctness.

### T04: GQA × TP — KV head replication branch when tp_size >= total_num_kv_heads
Source: `linear.py:L1031-L1036`. `if tp_size >= self.total_num_kv_heads: self.num_kv_heads = 1; self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads); else: self.num_kv_heads = divide(total_num_kv_heads, tp_size); self.num_kv_head_replicas = 1`. Llama-3-70B (8 KV heads): tp=8 → one KV head per rank, clean 8× memory savings; tp=16 → one KV head per rank still, replicated 2× across rank-pairs. KV memory savings cap at 8× regardless of tp_size. **Trap-D**. Same logic also in `llama.py:L147-L155`: `assert tp_size % self.total_num_kv_heads == 0` when `total_num_kv_heads < tp_size`.

### T05: vLLM's all-reduce is a 1-line wrapper; algorithm choice is in NCCL
Source: `communication_op.py:L12-L14` + `parallel_state.py:L502-L530`. `tensor_model_parallel_all_reduce(input_)` calls `get_tp_group().all_reduce(input_)`, which (after `world_size==1` bypass) dispatches to `device_communicator.all_reduce(input_)`. CUDA device communicator (`device_communicators/cuda_communicator.py`) uses NCCL — selects ring / tree / double-binary-tree internally by payload size + topology. For α-β reasoning: NCCL ring is asymptotically `2(P-1)/P × (α + (S/P) × β)` at large payloads; small payloads use `custom_all_reduce.py` (NVLink P2P) when available — the "fast path" that beats α-bound regime.

---

## T06: RowParallelLinear bias is FULL output_size, added on rank 0 ONLY

**Source**: `instances/vllm/source/vllm/model_executor/layers/linear.py:L1486-L1487`,
`L1557-L1559`.

Bias for RowParallelLinear is NOT sharded — every rank holds the full
`[output_size]` bias. But only rank 0 adds it before the all-reduce; otherwise
the post-reduce output would have `tp_size × bias` instead of `bias`.

```python
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)
```

For the implementer: this is easy to miss. If you "shard the bias the same
way you shard the weight", you'll get `bias / p` after reduce — silently
wrong outputs.

---

## T07: Trap inventory — load-bearing TP misconceptions

For Ch09 implementer / Ch11 implementer: these traps recur.

- **Trap A**: TP=2 ≠ 2× throughput (col-parallel saves no comm; row-parallel
  pays α+β all-reduce; sub-linear).
- **Trap C**: QKV is column-parallel along the HEAD dim, not arbitrary
  feature columns (heads are independent; arbitrary column slicing breaks
  attention).
- **Trap D**: TP halves KV cache only when `total_num_kv_heads >= tp_size`.
- **Trap E**: MLP TP = ONE all-reduce per col→row pair, NOT one all-gather
  + one all-reduce. SiLU is element-wise so works on sharded data.
- **Trap F**: RowParallelLinear's `input_is_parallel=True` is the default
  (because the previous layer is column-parallel). Mis-setting silently
  doubles communication.

---

## T08: Per-segment narrow loop in MergedColumnParallelLinear

**Source**: `instances/vllm/source/vllm/model_executor/layers/linear.py:L767-L820`

```python
shard_offsets = []
for i, output_size in enumerate(output_sizes):
    shard_offsets.append((i, current_shard_offset, output_size))
    current_shard_offset += output_size
for shard_id, shard_offset, shard_size in shard_offsets:
    ...
```

Each output segment in `output_sizes` is sharded INDEPENDENTLY along the
output dim. Per-rank weight is the concatenation of (segment_0_rank_r,
segment_1_rank_r, …). This is what makes `gate_up_proj` actually correct
under TP — the implementer who naively narrows the fused output uniformly
gets wrong outputs (we hit this and fixed it).

---

## T09: Column-parallel forward concatenated equals unsharded matmul EXACTLY

**Module**: tensor-parallelism
**Chapter**: 08-tensor-parallelism
**Role**: tester
**Source**: `instances/vllm/source/vllm/model_executor/layers/linear.py:L579-L607`
**Date learned**: 2026-05-06

Column-parallel does NO addition during forward — each rank computes
`Y_i = X @ A_i` independently, and concatenation is bit-for-bit identity to
`X @ A`. Demo §1 numerics confirm: `col_tp{2,4,8}_max_abs_diff = 0` (zero,
not "small"). This contrasts with row-parallel where the sum-of-partials
introduces fp32 noise (~7.6e-6 to 9.5e-6 in the demo).

**Test pattern**: assert with `np.array_equal`, NOT `np.allclose`, when
testing column-parallel concatenation. If a test author uses `allclose` they
miss a regression that introduces tiny addition noise (e.g. accidentally
running an all-reduce on already-correct shards).

---

## T10: Row-parallel sum-of-partials is fp32-tolerance-bound, not exact

**Source**: `instances/vllm/source/vllm/model_executor/layers/linear.py:L1562-L1563`

Demo §1 row_parallel tolerance pinned at 1e-4 (observed ≈7.6e-6 to 9.5e-6).
The diff is purely the difference between `(X_1 @ A_1) + (X_2 @ A_2) + ...`
in chunked order vs `X @ A` in one BLAS call: float32 addition is
non-associative, so different summation orders can produce slightly different
results. Tolerance 1e-4 is generous; 1e-5 would still pass at all tested tp
sizes for 32×32 inputs.

For Llama-7B-shaped inputs (4096 hidden), the row-parallel diff stays in
single-digit micros — the chunked sum noise grows slowly with input size.

---

## T11: ring_all_reduce_cost(P=1) returns 0 (world_size==1 bypass)

**Source**: `instances/vllm/source/vllm/distributed/parallel_state.py:L518-L519`

The cost model honors the same bypass that production GroupCoordinator does:
when `world_size == 1`, no collective runs. This is critical for tests that
sweep tp ∈ {1, 2, 4, 8} — at tp=1, predicted overhead is 0, so any test
asserting "predicted_AR > 0" must skip tp=1.

The simulate_all_reduce function returns the input identity at P=1 (a single
copy of the input tensor).

---

## T12: α-bound vs β-bound regimes are detectable via P=2 vs P=8 ratio

**Source**: derived from the formula `T = 2(P-1)/P × (α + S/P × β)`.

- **α-bound (small payloads, e.g. 1 KB on NVLink)**: P=8 is SLOWER than P=2.
  The 2(P-1)/P term dominates, growing from 1.0 (P=2) to 1.75 (P=8). Demo §2
  pinned ratio: P=8/P=2 ≈ 1.75 (3.50/2.00).
- **β-bound (large payloads, e.g. 64 MB on NVLink)**: P=8 is FASTER than P=2,
  but only sub-linearly. Demo §2 pinned ratio: P=2/P=8 ≈ 2.17 (113.85/52.43),
  far short of the 4× one might naïvely expect.

This is THE evidence for Trap-A ("TP=2 doubles throughput" is wrong). A test
that asserts these ratios pins the chapter's main numerical claim.

---

## T13: bias_only_on_rank_0 is detectable via zero-weight test

**Source**: `instances/vllm/source/vllm/model_executor/layers/linear.py:L1557-L1559`

Test pattern: build a RowParallelLinear with the WEIGHT matrix set to zero
and a non-zero bias. The all-reduced output should equal `bias` (added once),
NOT `tp_size × bias`. If a buggy implementation adds bias on every rank
before the all-reduce, the test catches a 4× off-by-tp_size silently.

This is a high-leverage one-line test that pins the bias semantics
unambiguously without depending on numerical tolerance.

---

## T14: 1 all-reduce per Megatron pair vs 2 per transformer block — narrative consequence

**Module**: tensor-parallelism
**Chapter**: 08-tensor-parallelism
**Role**: writer
**Source**: `instances/vllm/source/vllm/model_executor/models/llama.py:L94-L121` (LlamaMLP)
+ `L164-L179` (LlamaAttention)
**Date learned**: 2026-05-06

A reader who only sees the demo §5 number `mlp_tp{2,4,8}_collectives_per_forward = 1.0`
naturally rounds it to "Megatron TP needs 1 all-reduce per block". This is wrong
twice over: (a) the count is per col→row PAIR, not per block; (b) a Llama
transformer block has TWO pairs (qkv_proj + o_proj as the attention pair,
gate_up_proj + down_proj as the mlp pair), so TWO all-reduces per block.

Narrative pattern that works: introduce the pair structure when each pair is
introduced (§8.3 ColRow + §8.4 QKV+OProj), then in the closing §8.6 explicitly
quote the integration test result `test_attn_then_mlp_two_collectives_per_block`
asserts collective count = 2 — anchor the disambiguation on Tester's pinned
number, not on author claim.

Same pattern applicable to Ch09 (EP+TP composition: each TP pair adds 1 all-reduce,
so an EP-MoE-block with 2 experts × 2 TP-pairs has 4 collectives), Ch11 (RingAttention
inside attention pair adds P2P sends but the o_proj all-reduce remains).

---

## T15: α-bound regime first, β-bound second — pedagogical ordering

**Module**: tensor-parallelism
**Chapter**: 08-tensor-parallelism
**Role**: writer
**Source**: `instances/vllm/artifacts/08-tensor-parallelism/tests/test-report.md` Tip 2
+ Demo §2 NVLink table (1 KB row: P=2 → 2.00 μs, P=8 → 3.50 μs)
**Date learned**: 2026-05-06

When teaching the α-β model and Trap-A ("TP=2 ≠ 2× throughput"), the natural
instinct is to lead with β-bound ("comm overhead is bandwidth-limited, so TP=2
doesn't double") because that's the more familiar framing. But that's only HALF
the trap and the LESS surprising half — readers nod and forget.

Lead with α-bound: at 1 KB payload on NVLink, P=8 takes 3.50 μs vs P=2's 2.00 μs
— P=8 is 1.75× SLOWER. "More ranks make small all-reduces SLOWER" — that's the
genuinely counter-intuitive fact, and the one readers need to internalize for
production decisions (when batch is small, like decode at low concurrency).
β-bound (more ranks help, sub-linearly) is the polite second half.

Anchored in the demo §2 NVLink table; the (P-1)/P factor in the latency term
explains the slowdown asymptote (1.75× = 2(P-1)/P at P=8 / P=2).

---

## T16: K17 honest demo caveat — single-process simulation wallclock is misleading

**Module**: tensor-parallelism
**Chapter**: 08-tensor-parallelism
**Role**: writer
**Source**: `instances/vllm/artifacts/08-tensor-parallelism/implementation/demo.py` §3
+ `tests/test-report.md` Tip 3
**Date learned**: 2026-05-06

The single-process Python simulation that runs all `tp_size` ranks SERIALLY in
one process produces `compute_per_forward` ms numbers that grow LINEARLY with
tp_size — exactly the OPPOSITE of real production TP (where wallclock stays
roughly flat compute-bound, plus α-β all-reduce overhead).

Quote-safe demo numbers (production-honest):
- weights/rank (cleanly halves; demo §3 270.5 / 135.3 / 67.6 MB)
- predicted AR overhead from α-β model (demo §3 8.99 / 8.24 μs at NVLink)
- collectives_per_forward (always 1 for tp>1; demo §5)
- GQA boundary table (demo §4 — pure memory math)
- max-abs-diff equivalence numbers (demo §1, §5 — fp32 fidelity)

NOT-quote-safe: any `compute_per_forward` ms (must be paired with K17 caveat
or skipped). This pattern recurs in every chapter that uses single-process
multi-rank simulation — the writer's instinct "let me show wallclock to make
the perf story concrete" must be resisted unless the simulation has real
multi-process parallelism.

---

## T17: Reviewer pattern — "no class X" reframe is now a Ch07/Ch08 series convention

**Source**:
+ Ch07 §7.2 ("vLLM 没有 radix tree, chain hash + flat dict 替代")
+ Ch08 §8.2 ("vLLM 没有 class TensorParallel, 5 文件协同")
**Date learned**: 2026-05-06 (during Ch08 review)

When the outline subsection name describes a textbook concept (radix tree, TP
framework class) but the source has NO such class, the chapter MUST:
1. Open with a `grep -rE "class TensorParallel|..."` "(zero matches)" evidence
   block in the chapter opener (NOT just inside §X.2)
2. State the meta-callout to the prior "no X" instance (Ch07 → Ch08 explicit
   parallel at L11 + L237)
3. Use §X.2 as the reframe section, breaking the file/class composition into
   N labeled members (Ch08 has 5)
4. Echo the reframe in the closing summary (Ch08 L1027)
5. Mark `## X.6.5 N 个语言陷阱回顾（Ch07 §7.6.4 风格）` so future chapters can
   trace the lineage

For the reviewer this is a **pass/fail gate**: if the writer accepts an
outline-vs-source mismatch and "硬讲" the textbook framing, REVISE.

## T18: Reviewer pattern — framing tips from tester are load-bearing, not decorative

**Source**: `tests/test-report.md` Tips 1-5; Ch08 chapter weaving (8 cite-sites for Tip 1)
**Date learned**: 2026-05-06

When the tester provides framing tips (in `tests/test-report.md` "Framing tips
for writer"), the reviewer must verify each tip is **structurally** woven, not
just appended:

- "Cited at multiple sections" → grep for ≥3 cite-sites (Tip 1 in Ch08 has 8)
- "Lead with X, then Y" → check the SECTION TITLE encodes the ordering (Ch08
  §8.5.3 title literally says "先讲 α-bound，再讲 β-bound")
- "Worked example with construction Y" → grep for the exact construction
  (Tip 4 zero-weight construction at L482; Tip 5 file:line linear.py:L767-L820
  cited 4×)
- "Concrete bug story" → check there's a dedicated subsection (§8.3.3 entire
  subsection L496-549 for Tip 5)

If any tip is mentioned only once at the bottom recap, REVISE — it means the
writer treated it as decoration not as scaffolding.

## T19: Reviewer pattern — honest-demo-caveat OR-skip discipline

**Source**: impl-notes §7 K17; Ch08 application
**Date learned**: 2026-05-06

When a demo number is flagged "not production-honest" by impl-notes:

- The chapter must EITHER (a) cite verbatim with the caveat **immediately**
  paired (within same paragraph), OR (b) skip the number entirely.
- The chapter MUST NOT cite the number with the caveat in a different section
  (e.g., quoting `compute_per_forward` ms in §8.5.5 then disclosing K17 in
  §8.6.1) — that's "buried caveat" and reads as dishonest.
- Verify by searching for any forbidden substring (e.g., "ms", "ms wallclock")
  outside the caveat-bearing paragraphs.

Ch08 satisfies this perfectly — `compute_per_forward` is mentioned only in
§8.6.1 (K17 dedicated subsection) and the recap with "**不引用**——K17 caveat".
This pattern is mandatory for any chapter using single-process multi-rank
simulation.
