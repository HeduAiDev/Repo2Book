# Rehydration Brief — Ch08 Tensor Parallelism (Implementer)

- **Chapter**: `08-tensor-parallelism`
- **Title**: Tensor Parallelism 张量并行
- **Outline level**: core (Part 2)
- **Status**: `needs_rewrite` — full v6-grade rewrite per Ch04/Ch05/Ch06/Ch07 baseline (cadence holds at N=4)
- **Dependencies (per outline)**: `01-self-attention-fundamentals` (head structure, QKV projection); also touches Ch03 (PagedAttention) and Ch15+ (model architecture, where TP-wrapped layers actually plug in)
- **Dependents downstream**: `09-expert-parallelism` (EP+TP composition), `11-dcp-pcp` (DCP/PCP as TP siblings), `26-28` (frontier-model parallelism plans)
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Brief generated**: 2026-05-06 by archivist
- **Recipient**: implementer-2 (direct dispatch by team-lead, no book-editor relay)

---

## §1 — Chapter scope (what Ch08 actually covers — and what it does NOT)

**Core question**: when an LLM is too large for one GPU's HBM, how does vLLM
shard the matrix multiplications across `tp_size` GPUs *while preserving exact
mathematical equivalence* to the unsharded forward pass — and why does TP=2
NOT give 2× throughput?

The chapter covers **5 movements**:

1. **The math.** GEMM `Y = X @ A` factorizes two ways:
   - Column-parallel: split `A` along its output dim → `[A_1 | A_2 | ... | A_p]`. Each rank computes `Y_i = X @ A_i`. Concat gives `Y = [Y_1 | Y_2 | ...]`. NO communication during forward unless you need the full `Y`.
   - Row-parallel: split `A` along its input dim, AND split `X` along its last dim → `Y = sum_i (X_i @ A_i)`. Requires **all-reduce** to sum partial outputs.
   - The Megatron-style trick: stack column-parallel **then** row-parallel. The intermediate result stays sharded (no all-gather between them); only ONE all-reduce per MLP/attention block.

2. **Communication primitives.** `all_reduce` (sum + broadcast, latency O(α + βS) where S is tensor size; bandwidth-bound at large S), `all_gather` (concat across ranks), `reduce_scatter` (sum + shard the result; useful for sequence-parallel intermediate). The α-β model: T = α + βS for ring algorithms, where α is per-link latency and β is inverse bandwidth. Critical insight: TP=2 doesn't give 2× because comm overhead is α-dominated for small tensors and β-dominated for large ones — neither vanishes.

3. **Attention's TP — head sharding.** QKV is column-parallel along the *head* dimension, NOT arbitrary columns. `num_heads_per_rank = total_num_heads // tp_size`. Attention itself is then local (each rank holds heads `[0..num_heads_per_rank)`, computes attention independently). The output projection is row-parallel → all-reduce after. GQA/MQA complication: if `num_kv_heads < tp_size`, KV heads must be replicated (each rank holds a copy) — `QKVParallelLinear` handles this via `num_kv_head_replicas`.

4. **MLP's TP — col-parallel + row-parallel pair.** `gate_up_proj` is column-parallel (each rank holds `[ffn_dim/tp_size]` of the weights), then SiLU activation runs locally on the sharded intermediate, then `down_proj` is row-parallel → all-reduce. Llama uses `MergedColumnParallelLinear` to fuse `gate` and `up` into one column-parallel matmul — saves a kernel launch.

5. **System impact analysis.** TP=2 splits weights → each rank holds half the params. Latency: dominated by all-reduce; small batch sizes are α-bound (worse), large batch sizes are β-bound (still not free). Throughput: TP=2 with batch size B may give ~1.5× throughput, not 2× — comm overhead steals the rest. Memory: weights cleanly halved; KV cache stays per-rank (not duplicated). When TP > num_kv_heads, KV duplicates → memory savings cap.

**OUT of scope** (do NOT re-cover):
- Pipeline parallelism (PP) and data parallelism (DP) → mentioned briefly as
  contrast. Ch11 (DCP/PCP) handles context-parallelism variants. Future
  Part-3+ chapters cover composed parallelism.
- NCCL internals or topology detection → reference, don't deep-dive.
  `device_communicators/` is the abstraction layer; we describe the API
  surface and one all-reduce path (NCCL), not implementation per backend.
- Activation/sequence parallelism (Megatron 2D variants) → mention briefly,
  defer to Ch11.
- Expert parallelism (MoE all-to-all) → Ch09.

If implementer is re-deriving NCCL ring algorithms or PP scheduling — STOP.
Those belong elsewhere.

---

## §2 — Source surface (verified at commit 98661fe)

### §2.1 — Files and exact line ranges

| File | Lines (verified) | What |
|---|---|---|
| `vllm/distributed/parallel_state.py` | total 2132 lines | TP groups, GroupCoordinator, init_distributed_environment |
| `vllm/distributed/parallel_state.py` | L290-L1136 | `class GroupCoordinator` — wraps PyTorch ProcessGroup; holds `world_size`, `rank`, `device_communicator`; methods `all_reduce`, `all_gather`, `reduce_scatter` |
| `vllm/distributed/parallel_state.py` | L502-L530 | `GroupCoordinator.all_reduce` (the user-facing entry) — bypasses if `world_size==1`; dispatches to `device_communicator.all_reduce` |
| `vllm/distributed/parallel_state.py` | L532-L580 | `GroupCoordinator.all_gather` (and `_all_gather_out_place`) |
| `vllm/distributed/parallel_state.py` | L1494-L1737 | `initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, ...)` — builds the TP/PP/DP group ranks; line 1495-1599 are the TP-relevant portion |
| `vllm/distributed/parallel_state.py` | L1229-L1235 | `get_tp_group()` — returns the TP `GroupCoordinator` |
| `vllm/distributed/parallel_state.py` | L1837-L1845 | `get_tensor_model_parallel_world_size()`, `get_tensor_model_parallel_rank()` — the two helpers every parallel layer calls |
| `vllm/distributed/parallel_state.py` | L1738-L1781 | `ensure_model_parallel_initialized` (asserts the world is set up) |
| `vllm/distributed/parallel_state.py` | L1494-L1599 | TP group construction: `all_ranks.view(-1, tp_size).unbind(0)` — the device-mesh math |
| `vllm/distributed/communication_op.py` | L12-L40 | `tensor_model_parallel_all_reduce`, `tensor_model_parallel_all_gather`, `tensor_model_parallel_reduce_scatter`, `tensor_model_parallel_gather` — thin wrappers over `get_tp_group().all_reduce(...)` etc. **THIS is the function the parallel layers call** |
| `vllm/distributed/utils.py` | L60-L66 | `divide(numerator, denominator)` — the `assert numerator % denominator == 0` divisor used everywhere |
| `vllm/distributed/utils.py` | L67-L83 | `split_tensor_along_last_dim` — used by RowParallelLinear when `input_is_parallel=False` |
| `vllm/model_executor/layers/linear.py` | total 1578 lines | The TP layer suite |
| `vllm/model_executor/layers/linear.py` | L141-L181 | `class LinearMethodBase` — quantization-method base; not TP-core but threaded throughout |
| `vllm/model_executor/layers/linear.py` | L231-L288 | `class LinearBase(PluggableLayer)` — base class, holds `quant_method` |
| `vllm/model_executor/layers/linear.py` | L289-L409 | `class ReplicatedLinear(LinearBase)` — NO TP, full weights replicated |
| `vllm/model_executor/layers/linear.py` | L410-L608 | `class ColumnParallelLinear(LinearBase)` — **§7.3 of chapter** |
| `vllm/model_executor/layers/linear.py` | L455-L460 | `__init__` sets `self.tp_rank`, `self.tp_size`, `self.output_size_per_partition = divide(output_size, tp_size)` |
| `vllm/model_executor/layers/linear.py` | L535-L560 | `weight_loader` — `loaded_weight.narrow(output_dim, tp_rank * shard_size, shard_size)` |
| `vllm/model_executor/layers/linear.py` | L579-L607 | `forward` — `quant_method.apply()` + optional `tensor_model_parallel_all_gather` if `gather_output=True` |
| `vllm/model_executor/layers/linear.py` | L609-L976 | `class MergedColumnParallelLinear(ColumnParallelLinear)` — fuses gate+up. Key for MLP §7.4 |
| `vllm/model_executor/layers/linear.py` | L977-L1393 | `class QKVParallelLinear(ColumnParallelLinear)` — **§7.5 attention TP**. L1029-L1043: `num_heads = divide(total_num_heads, tp_size)`; `num_kv_heads` and `num_kv_head_replicas` for GQA |
| `vllm/model_executor/layers/linear.py` | L1394-L1577 | `class RowParallelLinear(LinearBase)` — **§7.4 row-parallel + all-reduce** |
| `vllm/model_executor/layers/linear.py` | L1446-L1450 | `__init__`: `self.input_size_per_partition = divide(input_size, tp_size)` |
| `vllm/model_executor/layers/linear.py` | L1543-L1577 | `forward` — split input if needed, GEMM, then `tensor_model_parallel_all_reduce` if `reduce_results=True` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | total 567 lines | Vocab-parallel embedding (each rank owns a slice of the vocab) |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | L104-L191 | `VocabParallelEmbeddingShardIndices` (the slice math) |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | L192-L502 | `class VocabParallelEmbedding(PluggableLayer)` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | L503-L555 | `class ParallelLMHead(VocabParallelEmbedding)` — used at the model output |
| `vllm/distributed/device_communicators/base_device_communicator.py` | total 373 lines | Backend abstraction |
| `vllm/distributed/device_communicators/base_device_communicator.py` | L118+ | `class DeviceCommunicatorBase` — `all_reduce`, `all_gather`, `reduce_scatter` overridden per backend |
| `vllm/distributed/device_communicators/cuda_communicator.py` | total 459 lines | CUDA/NCCL backend — most-used in production |
| `vllm/model_executor/models/llama.py` | (whole file) | **Reference site** — shows TP layers in actual use: `gate_up_proj = MergedColumnParallelLinear(...)`, `down_proj = RowParallelLinear(...)`, `qkv_proj = QKVParallelLinear(...)` |

### §2.2 — Outline-vs-source mismatches to flag

**The outline uses textbook framing; the source uses class-by-class composition.**
Implementer must be aware:

- **There is NO `class TensorParallel`.** No single class implements TP. The
  TP "feature" is the *composition* of `parallel_state.py` (groups +
  collectives) + `linear.py` (sharded layers) + `vocab_parallel_embedding.py`
  + the Llama-style usage pattern. This is the OPPOSITE structure to the
  outline subsection naming, which suggests there should be a TP module.
  → **The chapter must teach via these 4-5 collaborating files, not search
  for one TP class.** Mirror Ch07's "no radix tree" reframe handling.
- The outline says "Megatron-style TP in vLLM中的实现" — the actual code is
  PyTorch DTensor / explicit shard pattern, NOT a copy of Megatron's
  framework. vLLM has its own `ColumnParallelLinear` etc. These ARE
  Megatron-style algorithmically (column then row, one all-reduce per block)
  but the implementation is clean Python on top of `torch.distributed`.
- **No standalone `tensor_parallel.py` file** — the `knowledge/INDEX.md`
  currently lists `tensor_parallel.py` for the parallelism module; that's
  outdated/incorrect. The actual files are `linear.py` +
  `parallel_state.py` + `communication_op.py`. **Update INDEX when creating
  knowledge/modules/tensor-parallelism.md.**
- The outline says "TP与Attention算子的协同" — vLLM uses
  `QKVParallelLinear` (one fused matmul producing Q, K, V already sharded
  along heads) feeding into a backend-specific attention op
  (FlashAttention/PagedAttention from Ch03), then `RowParallelLinear` for
  the output projection with all-reduce. Chapter must walk this triad
  precisely; the attention kernel itself is TP-agnostic — the heads it
  sees are already the local rank's slice.

### §2.3 — Verified absence of structures the outline could imply

- No `class TensorParallel`, no `class ParallelReplica`, no
  `class TPCoordinator`. (Confirmed via `grep -rE "class TensorParallel|class
  .*Replica|class .*Parallel" vllm/distributed/ vllm/model_executor/layers/
  linear.py` — only matches are the parallel-Linear classes listed above
  plus a `class ParallelStrategy` in
  `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/multi_process_adapter.py`
  which is unrelated to the model TP path.)
- TP groups are stored as a module-level singleton `_TP` in
  `parallel_state.py:L1494+`, accessed via `get_tp_group()` — there is no
  TP-scoped class instance the layers hold a reference to.

---

## §3 — Outline section walk-through

Outline subsections (from `book-outline.json` Ch08 entry) and how to map them
to source. Subsection text is the *topic* (the question the section answers),
not a class-name contract.

| Outline subsection | Reframed scope | Source anchor |
|---|---|---|
| 1. "Column Parallel vs Row Parallel的数学等价性证明" | Prove that sharded forward pass = unsharded forward pass mathematically. Open `linear.py:L412` ColumnParallelLinear docstring; derive `Y = X @ A = [X@A_1 | X@A_2 | ...]`. Then `linear.py:L1394` RowParallelLinear docstring; derive `Y = sum_i (X_i @ A_i)`. Compose: col→row pair = one all-reduce. **5-step rhythm: open L412 → ask "why split A's columns?" → derive math → impl `column_parallel_linear.py` → diff: vLLM's `weight_loader` does the narrow on-demand from disk shard, our impl does it on init.** | `linear.py:L410-L608`, `linear.py:L1394-L1577` |
| 2. "TP通信开销的α-β模型（延迟+带宽×数据量）" | Derive α-β from first principles for ring all-reduce: `T = 2(P-1)/P × (α + S/P × β)` for `P` ranks, payload `S`. Use `parallel_state.py:L502-L530 GroupCoordinator.all_reduce` as anchor — show that vLLM's all_reduce is one Python call, but underneath sits a multi-step ring. Bridge: this is why TP=2 doesn't give 2× — α dominates small payloads, β dominates large ones. **5-step: open L502 → ask "what does this dispatch to?" → derive ring α-β → implement a toy `ab_model.py` predicting all-reduce time → diff: production NCCL also uses tree algorithms for tiny tensors, double-binary-tree for medium, ring for large.** | `parallel_state.py:L502-L580` |
| 3. "Attention的TP切分——QKV头切分与output all-reduce" | Open `linear.py:L977 QKVParallelLinear`. Derive head-sharding: total H heads, tp_size P → each rank gets H/P heads. GQA wrinkle: KV heads can be < Q heads; if KV < P, replicate. Then `linear.py:L1394 RowParallelLinear` for `o_proj`. Show the attention block: `qkv_proj` (column-parallel) → local attention → `o_proj` (row-parallel + all-reduce). **5-step: open L977 → ask "why head dim, not arbitrary cols?" → derive (heads are independent) → implement `attention_tp.py` triad → diff: vLLM's QKVParallelLinear handles `num_kv_head_replicas` (tp_size vs total_num_kv_heads), we collapse to MHA only.** | `linear.py:L977-L1393`, `linear.py:L1394-L1577` |
| 4. "MLP的TP切分——gate/up的col-parallel + down的row-parallel" | Open `models/llama.py` LlamaMLP — show real-world layer instantiation. Derive: SwiGLU = `silu(gate(x)) * up(x)`; both gate+up are col-parallel (output dim sharded), SiLU is element-wise so works on sharded intermediate, down_proj is row-parallel → all-reduce. `MergedColumnParallelLinear` fuses gate+up into one matmul. **5-step: open llama.py LlamaMLP → ask "why fuse gate+up?" → derive (one matmul beats two for kernel-launch overhead) → impl `mlp_tp.py` with merged column → diff: vLLM uses MergedColumnParallelLinear with output_sizes list, we just split the output tensor manually.** | `linear.py:L609-L976`, `models/llama.py` LlamaMLP block |
| 5. "TP size对显存/延迟/吞吐的影响分析" | Use the demo (see §7) to plot/table: `tp_size ∈ {1, 2, 4}` × `batch ∈ {1, 8, 64}` → throughput, p99 latency, weights memory per rank, KV memory per rank. Show: weights perfectly halved by TP; KV halved when `num_kv_heads >= tp_size`, else duplicates; latency goes up due to all-reduce; throughput goes up sub-linearly. **5-step: open the analysis question → ask "why isn't TP=2 giving 2x?" → derive from α-β model + Amdahl's law → implement `tp_sweep_demo.py` → diff: vLLM in production also benefits from custom_all_reduce (`device_communicators/custom_all_reduce.py`) which uses NVLink P2P for small payloads — we simplify to torch.distributed defaults.** | `device_communicators/cuda_communicator.py`, `parallel_state.py:L1837-L1845` (the helpers all layers call), Llama config |

Use this 5-section mapping as the chapter's §8.1-§8.5 spine.
§8.6 is the source-mapping table (main + per-section mini per K15 two-tier).

---

## §4 — Knowledge dependencies

### Existing knowledge entries to read before work
- `knowledge/modules/attention.md` — Q/K/V structure (Ch01/Ch03 facts)
- `knowledge/modules/kv-cache.md` — KV cache layout per-layer-per-head (block layout interacts with TP head-sharding)
- `knowledge/modules/scheduler.md` — only relevant indirectly (TP doesn't change scheduling)

### NEW knowledge module required
**Create `knowledge/modules/tensor-parallelism.md`** — Ch08 deserves its own module:
- Forward-shared with Ch09 (EP), Ch11 (DCP/PCP), Ch15+ (model arch).
- Use **T-prefix IDs** (T01, T02, ...) like `preemption.md` uses P-prefix.
  This avoids collision with `scheduler.md`'s K-prefix.
- **WARNING (carried from Ch07 lesson)**: `learn.py` append-mode is buggy
  per `trace/cross-chapter/learn-py-append-id-bug.md`. Manual workaround:
  write the module file directly with the heading format `## T0X: <title>`,
  do NOT rely on append-mode adding it. **If after running learn.py the
  file has any `## [TKP]\d+: [TKP]\d+:` double-prefix, fix immediately.**
- Update `knowledge/INDEX.md`:
  - Add row: `| [tensor-parallelism](modules/tensor-parallelism.md) | 08, 09, 11, 15+ | linear.py, parallel_state.py, communication_op.py |`
  - REMOVE the existing inaccurate entry: `| [parallelism](modules/parallelism.md) | 08, 09 | tensor_parallel.py, expert_parallel.py |` — there is no `parallelism.md` file actually present in `modules/`, and `tensor_parallel.py`/`expert_parallel.py` files don't exist.

### Anticipated facts the implementer will discover
- T01: divide() is the universal asserting divisor — every shard size in vLLM goes through `divide(numerator, denominator)` (`utils.py:L60`); this is the contract that `tp_size | total_heads` and similar must hold.
- T02: TP group as module singleton `_TP` (`parallel_state.py:L1494+`) — accessed via `get_tp_group()`. There is no per-layer TP context.
- T03: ColumnParallelLinear's `output_partition_sizes` is a LIST when fused (e.g., MergedColumnParallelLinear) — this is what lets one matmul produce multiple sharded outputs.
- T04: GQA + TP interaction — `QKVParallelLinear` distinguishes `num_kv_head_replicas` (tp_size > total_kv_heads, KV duplicates per rank) vs normal sharding.
- T05: The all-reduce is one Python call but underneath multiple algorithm choices — `device_communicator.all_reduce` dispatches to NCCL ring/tree/double-tree based on tensor size.

---

## §5 — Wisdom hits (role priorities: implementer = debugging > architecture > testing > writing)

Read these before opening source:

- `wisdom/architecture.md` — backpressure gates, lateral comm, fix prompts not chapters.
- `wisdom/debugging.md` — F.linear weight shape `[out, in]`. **Critical for TP**: the weight_loader's `narrow` happens on the OUTPUT dim for column-parallel, INPUT dim for row-parallel. Easy to flip.
- `wisdom/testing.md` — preemption test patterns (less directly relevant for TP, but the "test the boundary, not the happy path" guidance applies — TP correctness only matters at tp_size > 1).
- `wisdom/writing.md` — formula rules (NON-NEGOTIABLE: `\mathrm{}` not `\text{}`, no `\boxed`, no `\frac` inline). Ch08 will be FORMULA-HEAVY (α-β model, equivalence proofs, head sharding math). **Plan for high formula-density mitigation per K14**: keep ≤2 inline formulas per bullet, render mid-proof variable references as plain text when the symbol isn't doing math work.

Plus the reproducible-cadence patterns from `state.json:v6_compliance.patterns_promoted_to_baseline`:
- two_tier_mapping (mandatory for Ch08 — surface is broad, ≥7 source files)
- language_trap_callouts (Ch08 has at least 5 — see §6)
- honest_demo_caveats (synthetic TP=2 throughput is sensitive to comm hardware; flag that)

---

## §6 — Candidate language traps for the writer (≥4 required, target 5)

Each candidate is a phrasing that is "easy to write and almost-but-not-quite right". The writer will pick the strongest 4-5 for explicit callouts at the relevant section + a dedicated recap section, mirroring Ch07 §7.6.4.

**Trap A — "TP=2 doubles throughput."** No. TP shards weights cleanly but
introduces an all-reduce per attention block AND per MLP block. At small
batch sizes, all-reduce is α-bound (latency-dominated); at large batch
sizes, β-bound. Real-world TP=2 throughput is typically 1.4-1.7× for
realistic prompt/decode mixes, *not* 2×. Only weights memory is exactly
halved. Forward-pointer to §8.5 system analysis.

**Trap B — "All-reduce is just `dist.all_reduce(tensor, op=SUM)`."** Hides
the algorithm choice. NCCL ring all-reduce moves `2(P-1)/P × S` bytes per
rank for payload `S` and `P` ranks; double-binary-tree halves the latency
term but doubles the bandwidth term. vLLM's `custom_all_reduce`
(`device_communicators/custom_all_reduce.py`) uses NVLink P2P for small
payloads. The "just one call" view is fine for correctness but useless for
performance reasoning.

**Trap C — "QKV is column-parallel along the feature dim."** Not arbitrary
columns — along the **head dimension**. Heads are independent in
self-attention, so head-sharding is the only safe column split. Sharding
arbitrary feature columns of the QKV weight would break attention
correctness (since each head needs a coherent contiguous slice of the
weight, not a stride-pattern). Source evidence: `linear.py:L1029-L1043`
QKVParallelLinear computes `num_heads = divide(total_num_heads, tp_size)`,
not `output_size // tp_size` directly.

**Trap D — "TP halves KV cache memory."** Only when `num_kv_heads >= tp_size`.
For models with GQA (e.g., Llama-3-70B has 8 KV heads) and tp_size=8,
each rank holds 1 KV head — clean halving. But for tp_size=16 with the
same 8 KV heads, KV gets duplicated (each rank holds the SAME 1 KV head's
worth of cache, no memory savings). The chapter must state the inequality
explicitly. Source: `linear.py:L1029-L1043` `num_kv_head_replicas` branch.

**Trap E — "MLP TP needs an all-gather between gate/up and down."** No, the
column-parallel intermediate STAYS sharded; SiLU is element-wise so works
on the local slice; down_proj is row-parallel → ONE all-reduce. The naive
"all-gather then matmul then no-comm" pattern doubles communication.
Megatron's insight: col→row pair = one all-reduce per block.

**Trap F — "RowParallelLinear's input is auto-split."** Only if
`input_is_parallel=False` is set — then `split_tensor_along_last_dim` runs.
The default is `input_is_parallel=True` (because the previous layer is
column-parallel and already produced a sharded output). Mis-setting this
flag will silently double communication or feed garbage to the layer.
Source: `linear.py:L1543-L1577`.

**Trap G — "TP communication is overlapped with compute."** Sometimes, not
always. NCCL has async semantics, but `tensor_model_parallel_all_reduce`
in vLLM is a synchronous call by default. Communication-compute overlap
needs explicit two-stream scheduling (see `device_communicators/symm_mem.py`
for the `fused_scaled_matmul_reduce_scatter` patched op at
`parallel_state.py:L178-L260` — fused matmul+reduce-scatter is one of the
few overlap mechanisms). Default path: serial.

Pick 4-5 of A/B/C/D/E (these are the most load-bearing); F/G are
secondary. Recap section §8.6.4 should explicitly enumerate them with
"claim → 错 → why → source-evidence" per Ch07 §7.6.4 template.

---

## §7 — Demo plan suggestions (numerics for verbatim narrative use)

The implementer's `demo.py` should produce numbers the writer will quote
verbatim (per the demo-numerics-verbatim hard gate, K17 lesson).

**Demo §1 — Mathematical equivalence test.** Construct a fake `nn.Linear`
with random weights. Compute `Y_ref = nn.Linear(x)`. Then split: build
`ColumnParallelLinear` simulated for `tp_size ∈ {2, 4}`; have each "rank"
compute its slice; concatenate; compare to `Y_ref` with
`torch.allclose(rtol=1e-5)`. Same for RowParallelLinear with
sum-of-partials. **Numbers to pin**: max abs diff (should be `< 1e-5` for
fp32, `< 1e-3` for fp16). This is the existence proof of math equivalence.

**Demo §2 — α-β microbench (run on real or fake all_reduce).** If GPU
available: `torch.distributed.all_reduce` on payloads `S ∈ {1KB, 1MB, 1GB}`
across 2 fake ranks (or actual 2 GPUs if WSL/CUDA permits — use a
single-process simulation otherwise). Fit `T = α + βS` via least-squares.
Pin: estimated α (μs), β (GB/s). Cite the canonical benchmarks: NVLink
P2P ~150GB/s, PCIe Gen4 ~32GB/s, NCCL ring on H100 NVLink achieves
~250GB/s effective for large payloads. **Numbers**: α and β estimates,
predicted vs measured all_reduce time at 3 payload sizes.

**Demo §3 — TP-sharding throughput sweep.** Build a tiny Llama-style block
(1 attention block + 1 MLP block, hidden=4096, heads=32) under
`tp_size ∈ {1, 2, 4}`, batch ∈ {1, 8, 64}, seq=512. Measure: forward time,
weights memory, KV memory. Use the α-β fit from Demo §2 to *predict* the
all-reduce overhead, compare to measured. **Numbers to pin**: TP=2
throughput / TP=1 throughput at each batch size (target ~1.4-1.7×, NOT 2×
— this is Trap A's evidence). Memory: weights/rank halved, KV/rank
halved if num_kv_heads=32, halved if num_kv_heads=8 (still ≥ tp_size).

**Demo §4 — GQA × TP boundary test.** Build with `num_heads=32,
num_kv_heads=8`, vary `tp_size ∈ {2, 4, 8, 16}`. Show:
- tp_size=8: clean. 1 KV head/rank. Memory savings 8×.
- tp_size=16: 1 KV head replicated 2× per "rank-pair". Memory savings only 8×, not 16× — KV memory floor.
**Numbers**: KV memory per rank at each tp_size, the saturation point.

**Demo §5 — End-to-end Llama MLP TP correctness + perf**. Use
`models/llama.py` LlamaMLP as the reference; instantiate ours with
`MergedColumnParallel` + `RowParallel`; verify outputs allclose with
unsharded; measure throughput. **Numbers**: max diff vs reference;
throughput ratio.

The above 5 demos collectively give the writer ≥10 ground-truth numbers
to quote verbatim. Test report should pin every one with `assertEqual` /
`assertLess` and explicit values — Ch07 K17 lesson: writer pre-runs
linters AND tester pins exact numbers → APPROVED in one cycle.

---

## §8 — Floor reminders (v6 hard gates, confirmed at N=4)

Implementer commit must satisfy:

- **≥5 source files in impl-notes "Source Analysis" section.** Ch08
  natural surface is 7+ files: `parallel_state.py`, `linear.py`,
  `vocab_parallel_embedding.py`, `communication_op.py`, `utils.py`,
  `device_communicators/cuda_communicator.py` (or
  `base_device_communicator.py`), `models/llama.py`. Aim for the upper
  end — the breadth IS the lesson.
- **≥60 `# REFERENCE: <path>:Lxxx` comments across impl modules.**
  Ch04: 65, Ch05: 61, Ch06: 60, Ch07: ~60. Match the floor.
- **≥10 mapping rows; aim for 25-30 main + 15-20 mini per K15 two-tier.**
  Ch07 had 27 main + 45 mini. Ch08 should adopt two-tier from §8.1
  through §8.6: main mapping at §8.6 + mini-tables in §8.3 (QKV
  variants), §8.4 (MLP merged column shard offsets), §8.5 (TP-sweep
  demo verification table).
- **Demo numerics verbatim** in tests/test-report.md, then narrative
  quotes them character-for-character.
- **Both linters PASS at the BLOCKING bar** before handoff (writer + reviewer
  re-run; mismatches trigger preemptive REVISE per K17).
- **5-step rhythm in every major section §8.1-§8.5.**
- **≥4 language traps** with explicit "claim → 错 → why" per §6 above,
  plus dedicated recap §8.6.4.
- **Forward-pointers wired**: Ch09 (EP+TP composition), Ch11 (DCP/PCP),
  Ch15+ (Llama where TP layers actually plug in). Back-pointers: Ch01
  (head structure), Ch03 (PagedAttention with TP-sharded heads).
- **Source pin verification**: implementer's first command should be
  `cd instances/vllm/source && git rev-parse HEAD` — must equal `98661fe`.
  Any line numbers that drift between brief and source → re-grep before
  citing.

### What APPROVED at cycle 1 looks like (K17)

Writer's handoff message must contain BOTH linter outputs verbatim:

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/08-tensor-parallelism/narrative/chapter.md
🟢 No blocking issues
[+ any non-blocking warnings]

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/08-tensor-parallelism/
✓ All grounding checks passed!
```

Reviewer's first action: re-run both linters, diff against writer's claim.
Match → fast review. Mismatch → preemptive REVISE without further reading.

---

## §9 — Cadence carry-forward from Ch07 (specific reminders)

Ch07 just published v6 in 1 cycle on the widest source surface yet (5 files,
72 mapping rows). Patterns that worked and Ch08 must repeat:

- **Outline subsection name = topic question, not class contract.** §8 outline
  says "Megatron-style TP在vLLM中的实现" — chapter must answer "how does vLLM
  shard linear layers for TP?", not search for a Megatron class.
- **§8.2 (radix-tree analogue for Ch08) — "TP communication primitives".**
  No NCCL ring algorithm class exists in vLLM; it's all delegated to NCCL.
  Frame this as "vLLM's all-reduce is a thin Python wrapper; the actual
  algorithm choice is in NCCL/the device communicator". Mirror Ch07 §7.2
  "vLLM doesn't have a radix tree" handling.
- **(N-1)×K formula leads numerics (K13).** For Ch08, the analogue is
  α-β leads throughput estimate. Lead §8.5 with `T_TP = T_compute/P + T_allreduce(P, S)` formula, plug demo numbers AFTER.
- **Chain-break = THE invariant for Ch07.** For Ch08, the equivalent
  load-bearing invariant is **"col→row composition needs ONE all-reduce
  per block"**. Thread this through hook + every section + closing.

---

## §10 — Operational notes for direct dispatch

- **No book-editor relay**: per current operational rule (#7 from session
  pause), team-lead direct-dispatches via SendMessage.
- **Implementer should ack receipt** with: source pin verified
  (98661fe ✓), files identified (list 5+), language-trap awareness (≥1
  paraphrased), expected impl modules (≥5).
- **Writer's brief will be derived from this** by archivist post-tester
  (Ch07 lesson: writer brief amends implementer brief with tester's
  framing tips).
- **Estimated complexity**: comparable to Ch07. Chapter MAY be longer
  (~900 lines, ~4500 words) due to formula density. Two-tier mapping
  expected to reach 60+ rows.
