# Ch08 Tensor Parallelism — Implementer Notes

- Chapter: `08-tensor-parallelism`
- Source pin: vLLM `98661fe012c5c467252d4df8411d2f46190e9268`
- Author role: implementer
- Date: 2026-05-06

The chapter teaches **how vLLM shards linear layers for tensor parallelism**,
using a class-by-class walkthrough of the source surface (which has NO
`class TensorParallel` — TP is the *composition* of `parallel_state.py` +
`linear.py` + `vocab_parallel_embedding.py` + the Llama-style usage pattern).

This implementation re-derives the TP math from scratch in numpy, then mirrors
the four production TP layers 1:1 by class name and method name. All "ranks"
are simulated in a single process so the reader can run the demo on any
machine — there is NO real `torch.distributed` / NCCL call here. Every
collective is implemented as a numpy operation; the α-β cost model
predicts what real NCCL would do.

---

## §1 — Source Analysis (HARD GATE)

### 1.1 Files implementing TP in the target repo

| File | Lines (verified at 98661fe) | Role |
|---|---|---|
| `instances/vllm/source/vllm/distributed/parallel_state.py` | total 2132; L290-L1136, L1494-L1737, L1837-L1845 most relevant | TP groups, `GroupCoordinator`, `initialize_model_parallel`, `get_tp_group`, `get_tensor_model_parallel_world_size/rank` |
| `instances/vllm/source/vllm/distributed/communication_op.py` | total 44; L12-L40 | Thin wrappers `tensor_model_parallel_all_reduce/all_gather/reduce_scatter/gather` over `get_tp_group().<op>` |
| `instances/vllm/source/vllm/distributed/utils.py` | L60-L92 | `divide` (asserting divisor used by every shard size); `split_tensor_along_last_dim` (RowParallelLinear when `input_is_parallel=False`) |
| `instances/vllm/source/vllm/model_executor/layers/linear.py` | total 1578; L410-L1577 | TP linear suite: `ColumnParallelLinear` (L410), `MergedColumnParallelLinear` (L609), `QKVParallelLinear` (L977), `RowParallelLinear` (L1394) |
| `instances/vllm/source/vllm/model_executor/layers/vocab_parallel_embedding.py` | L104-L555 | `VocabParallelEmbedding` (L192) and `ParallelLMHead` (L503) — referenced for completeness; not core to attn/MLP path |
| `instances/vllm/source/vllm/distributed/device_communicators/base_device_communicator.py` | L118+ | `DeviceCommunicatorBase` — backend abstraction, override per platform |
| `instances/vllm/source/vllm/distributed/device_communicators/cuda_communicator.py` | total 459 | NCCL-backed `all_reduce` / `all_gather` / `reduce_scatter` (CUDA path) |
| `instances/vllm/source/vllm/model_executor/models/llama.py` | L81-L121 (LlamaMLP), L124-L233 (LlamaAttention) | Real-world TP layer instantiation: `gate_up_proj = MergedColumnParallelLinear`, `down_proj = RowParallelLinear`, `qkv_proj = QKVParallelLinear`, `o_proj = RowParallelLinear` |

This is **8 source files** in the table — exceeds the v6 floor of 5 (Ch04: 4,
Ch05: 4, Ch06: 4, Ch07: 5; Ch08 broadens to 8 because the lesson is breadth).

### 1.2 Key classes and their responsibilities

| Source class | Lines | Purpose | Owns | Delegates |
|---|---|---|---|---|
| `GroupCoordinator` | parallel_state.py:L290-L1136 | One-process-per-rank wrapper of a `torch.distributed.ProcessGroup` | `world_size`, `rank`, `device_communicator` | `all_reduce`, `all_gather`, `reduce_scatter` → `device_communicator` (NCCL on CUDA) |
| `ColumnParallelLinear(LinearBase)` | linear.py:L410-L608 | `Y = X @ A` with A split column-wise | per-rank weight `[in, out/p]`, bias `[out/p]` | `weight_loader.narrow(output_dim,…)`; optional `tensor_model_parallel_all_gather` if `gather_output=True` |
| `MergedColumnParallelLinear(ColumnParallelLinear)` | linear.py:L609-L976 | Fuse N output projections into ONE matmul | `output_sizes` list; per-segment shard offsets | Each segment narrowed independently via `weight_loader_v2(...,loaded_shard_id=k)` |
| `QKVParallelLinear(ColumnParallelLinear)` | linear.py:L977-L1393 | Fused Q/K/V projection sharded along the HEAD dim | `num_heads`, `num_kv_heads`, `num_kv_head_replicas` | `weight_loader_v2(...,'q'\|'k'\|'v')` |
| `RowParallelLinear(LinearBase)` | linear.py:L1394-L1577 | `Y = X @ A` with A split row-wise; sum partials | per-rank weight `[in/p, out]`, FULL bias `[out]` | `weight_loader.narrow(input_dim,…)`; `tensor_model_parallel_all_reduce` if `reduce_results=True` |
| `LlamaMLP` | llama.py:L81-L121 | SwiGLU MLP block with TP | `gate_up_proj` (MergedColumn), `down_proj` (Row), `act_fn = SiluAndMul` | One all-reduce per block (down_proj's reduce_results=True) |
| `LlamaAttention` | llama.py:L124-L233 | Attention block with TP | `qkv_proj` (QKV), `o_proj` (Row), `attn` (FlashAttention/PagedAttention from Ch03) | One all-reduce per block (o_proj's reduce_results=True) |

### 1.3 Data flow — one Llama transformer block under TP

```
hidden_states  [B, S, hidden]
   │
   ├─ self_attn:                                                 ┐
   │    qkv_proj (QKVParallelLinear)                             │ ranks in parallel,
   │      → per-rank fused [B, S, q_local + k_local + v_local]   │ NO collective
   │    .split([q_size, kv_size, kv_size], dim=-1)               │ (column-parallel
   │    rotary_emb on (q, k)                                     │  outputs stay
   │    self.attn(q, k, v)  → [B, S, num_heads_per_rank*head]    │  sharded; attn
   │    o_proj (RowParallelLinear, reduce_results=True)          │  is local per
   │      → [B, S, hidden]   ⟵⟵⟵ ALL-REDUCE ⟵⟵⟵                 ┘  rank's heads)
   │
   ├─ residual + RMSNorm
   │
   └─ mlp:                                                       ┐
        gate_up_proj (MergedColumnParallelLinear)                │ NO collective
          → per-rank fused [B, S, 2*ffn/p]                       │ (col-parallel)
        SiluAndMul element-wise per rank → [B, S, ffn/p]         │ NO collective
        down_proj (RowParallelLinear, reduce_results=True)       │
          → [B, S, hidden]   ⟵⟵⟵ ALL-REDUCE ⟵⟵⟵                 ┘
```

**Two all-reduces per transformer block.** Their payload is `[B, S, hidden]`,
so the cost grows with sequence length. This is the load-bearing invariant:
**col→row composition needs ONE all-reduce per col-row pair.**

### 1.4 Design decisions and WHY (≥3 with trade-off analysis)

1. **Column→row pair beats column→all-gather→column→row.**
   - Decision: `linear.py:L1463` defaults `input_is_parallel=True` for
     RowParallelLinear, so the row layer consumes the column layer's sharded
     output directly without an all-gather between them.
   - Trade-off: SAVES one collective per block. Cost: SiLU runs on sharded
     intermediate, but SiLU is element-wise so this is free.
   - Source: `linear.py:L1543-L1553` (forward branch on `input_is_parallel`),
     `llama.py:L94-L121` (LlamaMLP wires col→row directly).

2. **GQA replicates KV when tp_size > total_num_kv_heads.**
   - Decision: `linear.py:L1031-L1036` — when `tp_size >= total_num_kv_heads`,
     set `num_kv_heads = 1` and `num_kv_head_replicas = divide(tp_size,
     total_num_kv_heads)`. Each rank holds the SAME 1 KV head's weights.
   - Trade-off: Memory savings cap at `total_num_kv_heads`× regardless of
     tp_size; but the math stays consistent (each rank still produces
     correct attention output since it holds the right Q-heads-to-KV-head
     pairing).
   - Source: `linear.py:L1031-L1036` (the if/else in QKVParallelLinear),
     `llama.py:L147-L155` (the `assert tp_size % self.total_num_kv_heads == 0`
     branch).

3. **Bias on rank 0 only.**
   - Decision: `linear.py:L1557-L1559` — RowParallelLinear adds bias only
     on rank 0 before all-reduce. If every rank added bias, the post-reduce
     output would have `tp_size × bias` instead of `bias`.
   - Trade-off: Tiny if/else inside the hot path, but eliminates a per-rank
     subtraction or pre-scaling. The alternative (scale bias by 1/p on every
     rank) is numerically worse (introduces fp16 quantization error).
   - Source: `linear.py:L1557-L1559`.

4. **`tensor_model_parallel_all_reduce` is a one-line wrapper.**
   - Decision: `communication_op.py:L12-L14` is literally
     `return get_tp_group().all_reduce(input_)`.
   - Trade-off: Hides algorithm choice (ring/tree/double-tree) behind NCCL
     selection logic, but exposes a clean Python API the layers can call
     without thinking about the backend. The overhead of one Python function
     call is negligible compared to the GPU-resident NCCL operation.
   - Source: `communication_op.py:L12-L14`, `parallel_state.py:L502-L530`.

5. **TP group as module singleton `_TP`.**
   - Decision: `parallel_state.py:L1494+` stores the TP `GroupCoordinator`
     in a module-global `_TP`; layers retrieve it via `get_tp_group()`.
   - Trade-off: Less ceremonial than passing a context object through every
     layer (the alternative would require hundreds of constructor arg
     plumbing changes). Cost: harder to test without proper init; easier to
     mock. The vLLM convention.
   - Source: `parallel_state.py:L1578-L1592`, `parallel_state.py:L1229-L1235`.

### 1.5 Complexity preserved (NOT simplified away)

- The **per-segment sharding** in `MergedColumnParallelLinear` (each output
  in `output_sizes` is sharded independently) — kept verbatim in
  `column_parallel.py:load_weight`. **Critical**: a naive narrow on the fused
  output would put `[gate_rank0, gate_rank1, …]` in rank 0 instead of
  `[gate_rank0_shard, up_rank0_shard]`.
- The **GQA replication branch** in `QKVParallelLinear` (`num_kv_head_replicas`)
  — kept in `qkv_parallel.py:load_qkv_weights`. Replicates KV via
  `np.repeat` on the head axis before passing to the MergedColumn-style
  loader.
- The **input_is_parallel flag** on RowParallelLinear — kept; the
  `input_is_parallel=False` branch calls `split_tensor_along_last_dim` exactly
  as in `linear.py:L1549-L1553`.
- The **bias-only-on-rank-0** rule for RowParallelLinear — kept verbatim.
- The **assert-divisibility** contract on every shard size — every shard
  computation goes through `divide(numerator, denominator)` (mirrors
  `utils.py:L60-L64`).

What we DID simplify, with comments:

- No `torch.Parameter`, no autograd, no quantization (`SIMPLIFIED:` in
  `column_parallel.py:L74` etc.).
- No multi-process: all `tp_size` ranks held in `self.rank_states[r]`. Real
  vLLM has one Python process per rank.
- No NCCL: `simulate_all_reduce` is a numpy stand-in for the ring algorithm
  that NCCL runs at large payloads. The α-β cost model predicts what NCCL
  would actually do.

---

## §2 — REFERENCE coverage (≥60 hard gate)

REFERENCE comment density across implementation modules — counted at
authoring time:

| Module | `# REFERENCE:` count |
|---|---|
| `tp_math.py` | 16 |
| `comm_primitives.py` | 5 |
| `column_parallel.py` | 13 |
| `row_parallel.py` | 12 |
| `qkv_parallel.py` | 9 |
| `mlp_block.py` | 6 |
| `demo.py` | 3 |
| **TOTAL** | **64** |

This exceeds the v6 floor of 60 (Ch04: 65, Ch05: 61, Ch06: 60, Ch07: ~60).
`scripts/lint_source_grounding.py` validates each one points to a real path
in `instances/vllm/source/`.

---

## §3 — Demo numerics (verbatim — writer quotes character-for-character)

Captured from `tests/demo-output.txt` (full run of `demo.py` on 2026-05-06
with `numpy 2.2.6 / torch 2.8.0+cu126`):

### §3.1 Mathematical equivalence (Demo §1)

| Test | tp_size | max_abs_diff |
|---|---|---|
| ColumnParallel | 2 | `0.000e+00` |
| ColumnParallel | 4 | `0.000e+00` |
| ColumnParallel | 8 | `0.000e+00` |
| RowParallel | 2 | `7.629e-06` |
| RowParallel | 4 | `9.537e-06` |
| RowParallel | 8 | `9.537e-06` |
| Col→Row block | 2 | `0.000e+00` (collectives=1) |
| Col→Row block | 4 | `2.384e-07` (collectives=1) |
| Col→Row block | 8 | `2.980e-07` (collectives=1) |

Existence proof: every TP forward reproduces the unsharded reference
within fp32 numerical tolerance. The col→row block uses **exactly one**
collective regardless of tp_size.

### §3.2 α-β fit + ring all-reduce (Demo §2)

- Synthetic ground truth: α = 5.00 μs, bandwidth = 150 GB/s
- Recovered fit: α = 4.32 μs, bandwidth = 144.6 GB/s (5% noise)
- Ring simulation correctness: max diff vs naive sum = `2.384e-07`

NVLink_HSXM4 profile (α = 2.0 μs, bandwidth = 300 GB/s) ring all-reduce
predicted times (μs):

| payload (B) | P=2 | P=4 | P=8 |
|---|---|---|---|
| 1024 | 2.00 | 3.00 | 3.50 |
| 16384 | 2.03 | 3.02 | 3.51 |
| 262144 | 2.44 | 3.33 | 3.69 |
| 4194304 | 8.99 | 8.24 | 6.56 |
| 67108864 | 113.85 | 86.89 | 52.43 |

**The signature**: small payloads are α-bound (P=8 takes 1.75× P=2); large
payloads are β-bound (P=8 is 2.17× faster than P=2). The crossover is at
~few-MB payloads. The bandwidth term scales as `(P-1)/P × β`, which is
why scaling P helps — but the latency term `α` doesn't shrink at all,
so you can never hit linear speedup.

### §3.3 TP throughput sweep (Demo §3) — Llama-7B-shaped MLP block

Hidden=4096, ffn=11008, seq=512:

| tp_size | weights/rank (MB, fp16) | predicted AR (NVLink, μs) | collectives/forward |
|---|---|---|---|
| 1 | 270.5 | 0.0 | 0 |
| 2 | 135.3 | 9.0 | 1 |
| 4 | 67.6 | 8.2 | 1 |

**HONEST CAVEAT (writer must echo verbatim, K17 honest_demo_caveats):** the
"compute_per_forward" wallclock numbers in the run log are NOT representative
of real TP performance — this is a single-process simulation that runs all
`tp_size` ranks SERIALLY in the same Python process, so wallclock grows
roughly linearly with tp_size instead of staying flat. The numbers that
matter — and that the writer should use — are: (a) **weights/rank** (cleanly
halved by tp), (b) **predicted AR overhead from the α-β model** (production
NCCL number), (c) **collectives per forward** (always 1 for tp>1). The
narrative should NOT cite the ms timings.

### §3.4 GQA × TP boundary (Demo §4) — Llama-3-70B-style

H=8192, head_size=128, total_q_heads=64, total_kv_heads=8, seq=1024.
Full KV cache per token (fp16) = `2 × 8 × 128 × 2 = 4096 bytes`.

| tp_size | kv_heads/rank | replicas | KV/rank/token (B) | save factor |
|---|---|---|---|---|
| 2 | 4 | 1 | 2048 | 2.0× |
| 4 | 2 | 1 | 1024 | 4.0× |
| 8 | 1 | 1 | 512 | 8.0× |
| 16 | 1 | 2 | 512 | **8.0× (cap)** |
| 32 | 1 | 4 | 512 | **8.0× (cap)** |

**The boundary**: at `tp_size = total_num_kv_heads = 8`, each rank holds
exactly 1 KV head — clean halving down to 1/8 the KV memory. **Above** that
threshold, KV is replicated (`num_kv_head_replicas > 1`). Memory savings
*cap* at 8× regardless of how many ranks you add. This is Trap-D in the
writer's recap.

### §3.5 LlamaMLP TP correctness + collective accounting (Demo §5)

Hidden=1024, ffn=2752, seq=16:

| tp_size | max_abs_diff vs unsharded | avg collectives/forward |
|---|---|---|
| 1 | `0.000e+00` | 0.0 |
| 2 | `6.403e-10` | 1.0 |
| 4 | `8.149e-10` | 1.0 |
| 8 | `6.912e-10` | 1.0 |

**The Megatron pair signature**: 1 all-reduce per forward, regardless of
tp_size. Always exactly 1 — never 2, never 0 (when tp>1).

---

## §4 — Language traps (≥4 required; 5 picked from brief §6 candidates A-G)

The writer must call out at least these 4-5 traps in §8.6.4 recap, and
cross-reference each at the relevant section. Each trap follows the
"claim → 错 → why → source-evidence" template (Ch07 §7.6.4 lineage).

### Trap A — "TP=2 doubles throughput." 错.
- **Claim**: Sharding weights across 2 GPUs gives 2× throughput.
- **Why wrong**: TP shards weights cleanly (memory IS halved) but introduces
  one all-reduce per attention block AND per MLP block. Small-batch traffic
  is α-bound; large-batch traffic is β-bound; neither vanishes.
- **Numbers from §3.2**: P=2 NVLink small-payload all-reduce ≈ 2.0 μs;
  P=4 ≈ 3.0 μs (α-bound regime). Even at huge payloads (64 MB), P=8 is
  only 2.17× faster than P=2 — sub-linear.
- **Source evidence**: `linear.py:L1562-L1563` — RowParallelLinear's
  `tensor_model_parallel_all_reduce` is a hard sequential dependency
  between the col-parallel computation and the next layer's input.

### Trap C — "QKV is column-parallel along the feature dim." 错.
- **Claim**: QKVParallelLinear shards arbitrary feature columns of the
  weight matrix.
- **Why wrong**: It shards along the HEAD dim. Heads are independent in
  self-attention; sharding arbitrary feature columns would break attention
  correctness because each head needs a contiguous slice of the weight.
- **Source evidence**: `linear.py:L1030` — `self.num_heads = divide(
  self.total_num_heads, tp_size)`. Not `output_size // tp_size` directly.

### Trap D — "TP halves KV cache memory." 错 (conditional).
- **Claim**: Going from tp_size=1 to tp_size=2 halves your KV cache.
- **Why wrong**: Only when `total_num_kv_heads >= tp_size`. For models
  with GQA, `total_num_kv_heads` is small (e.g., 8 for Llama-3-70B). Once
  `tp_size > total_num_kv_heads`, KV is replicated and memory savings
  cap.
- **Numbers from §3.4**: Llama-3-70B with 8 KV heads — tp_size=8 gives 8×
  savings, tp_size=16 STILL gives 8× savings (one rank, KV replicated 2×).
- **Source evidence**: `linear.py:L1031-L1036` — `num_kv_head_replicas`
  branch.

### Trap E — "MLP TP needs an all-gather between gate/up and down." 错.
- **Claim**: After column-parallel gate_up_proj, we must all-gather to
  re-form the full intermediate before down_proj.
- **Why wrong**: SiLU is element-wise — works on the sharded intermediate.
  down_proj is row-parallel, so it CONSUMES the sharded input directly.
  The col→row pair needs ONE all-reduce, NOT one all-gather + one
  all-reduce.
- **Numbers from §3.5**: every tp>1 forward observes exactly **1.0**
  collectives/forward.
- **Source evidence**: `llama.py:L94-L121` — `gate_up_proj.gather_output`
  defaults to False; `down_proj.input_is_parallel` defaults to True.

### Trap F — "RowParallelLinear's input is auto-split." 错 (conditional).
- **Claim**: Pass any tensor to RowParallelLinear and it'll split it for you.
- **Why wrong**: Only if `input_is_parallel=False`. Default is `True`,
  which assumes the caller already provides per-rank shards (the
  column→row composition case).
- **Source evidence**: `linear.py:L1547-L1553` — the `if self.input_is_parallel:`
  branch.

---

## §5 — Cross-chapter links

### Back-pointers (chapters Ch08 builds on)
- **Ch01 Self-Attention Fundamentals** — head structure (num_heads, head_size,
  Q/K/V projections). The QKV head-sharding in §8.3 / Trap-C makes sense
  only if the reader already knows heads are independent.
- **Ch03 FlashAttention/PagedAttention** — the attention kernel itself is
  TP-agnostic; the heads it sees are already the local rank's slice. Reader
  should know FlashAttention's output is `[B, S, num_heads*head_size]` so
  that the o_proj's row-parallel shape makes sense.

### Forward-pointers (chapters that build on Ch08)
- **Ch09 Expert Parallelism** — EP is the MoE analog of TP, and EP+TP
  composition is a real-world frontier-model pattern (Ch26-28).
- **Ch11 DCP/PCP** — Decode/Prefill Context Parallelism share the
  collective primitives derived in §8.2; α-β model carries forward.
- **Ch15 Llama Model Architecture** — where the TP-wrapped layers
  (MergedColumnParallelLinear, RowParallelLinear, QKVParallelLinear)
  actually plug in. `llama.py:L81-L121 LlamaMLP` and `llama.py:L124-L233
  LlamaAttention` are the canonical instantiation sites.

---

## §6 — File map

```
implementation/
├── __init__.py            # Module-map docstring; no runtime imports
├── tp_math.py             # Pure-math derivation of column/row parallel + col→row
├── comm_primitives.py     # α-β cost model, ring all-reduce simulation, fit
├── column_parallel.py     # ColumnParallelLinear + MergedColumnParallelLinear
├── row_parallel.py        # RowParallelLinear (with input_is_parallel + bias-on-rank-0)
├── qkv_parallel.py        # QKVParallelLinear (head-sharding + GQA replication)
├── mlp_block.py           # LlamaMLPTP (col→row Megatron pair) + silu_and_mul
├── demo.py                # 5 demos producing verbatim numerics for the writer
└── impl-notes.md          # this file
```

LOC: tp_math 280, comm_primitives 235, column_parallel 240, row_parallel 175,
qkv_parallel 240, mlp_block 175, demo 280 ≈ 1625 LOC across implementation modules.

---

## §7 — Honest demo caveats (K17)

When the writer quotes Demo §3 throughput numbers, they MUST include:

> "These wallclock numbers come from a single-process Python simulation that
> runs all tp_size ranks serially. Real production TP runs each rank in a
> separate process on a separate GPU; the wallclock would be roughly flat
> across tp sizes (compute bound), with the all-reduce overhead added
> on top. Use the α-β predicted overhead from §3.2 to reason about
> production cost, NOT the ms times here."

Demo §1, §2 (correctness/fit), §4 (memory math), §5 (collective count)
are all production-honest — they don't depend on simulation wallclock and
can be quoted verbatim.

---

## §8 — What's NOT in this implementation (and why)

- **No real torch.distributed call.** `simulate_all_reduce` is a numpy
  stand-in. Production code in `parallel_state.py:L502-L530` dispatches
  to NCCL via `device_communicator.all_reduce(input_)`. We model the cost,
  not the kernel.
- **No quantization / weight loading from disk.** `column_parallel.py:
  load_weight` takes a numpy `[in, out]` matrix; production
  `linear.py:L534-L569` works with `torch.Parameter`, narrowed slabs from
  HuggingFace checkpoints.
- **No CUDA Graph capture.** `parallel_state.py:L464-L500` has the
  `graph_capture` context manager for `cudaGraphCapture`-friendly TP. Out
  of scope for the educational reimpl.
- **No `custom_all_reduce`.** vLLM's
  `device_communicators/custom_all_reduce.py` uses NVLink P2P for small
  payloads — the production "fast path" for α-bound regime. We mention
  it once in Trap-B but do not reimplement.
- **No VocabParallelEmbedding implementation.** Listed in source surface
  for completeness; the math is "shard the vocab dim" which is just
  another column-parallel pattern.
- **No PP/DP composition.** Out of scope per brief §1; Ch11 will cover
  composed parallelism.
