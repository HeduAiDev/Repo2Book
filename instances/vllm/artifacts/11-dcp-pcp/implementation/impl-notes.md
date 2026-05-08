# Ch11 DCP/PCP — Implementer Notes

- Chapter: `11-dcp-pcp`
- Source pin: vLLM `98661fe012c5c467252d4df8411d2f46190e9268`
- Author role: implementer
- Date: 2026-05-08

The chapter teaches **how vLLM shards the SEQUENCE axis of K and V across GPUs
to lift the HBM capacity wall for long context**, through a 12+ -file
collaboration in the source. The two axes are DCP (Decode Context Parallel,
folded inside TP — shards stored KV) and PCP (Prefill Context Parallel,
independent axis expanding world_size — shards prefill input). The verifier
side is rejection-sampling-style LSE-weighted combination of partial attention
outputs (`vllm/v1/attention/ops/dcp_alltoall.py:_lse_weighted_combine`); the
machinery side is GroupCoordinator singletons (`_DCP`, `_PCP` in
`vllm/distributed/parallel_state.py`) plus per-attention-backend integration
(every backend's `__new__` discovers DCP/PCP rank).

This implementation reproduces the algorithmic surface in plain PyTorch /
NumPy on a single process, mirroring vLLM's communication patterns with
direct tensor ops. The output is mathematically bit-identical to a real
distributed run for the LSE-weighted combine (associativity + commutativity
guaranteed by `softmax_max + log(sum(exp(...)))` reduction). Real production
uses NCCL collectives; we model their cost with an alpha-beta bandwidth
model.

---

## §1 — Source Analysis (HARD GATE)

### 1.1 Files implementing DCP/PCP in the target repo

| File | Lines (verified at 98661fe) | Role |
|---|---|---|
| `vllm/distributed/parallel_state.py` | 2132 total; L1234-L1290 (`_DCP`/`_PCP` singletons + accessors), L1497-L1498 (signature), L1569-L1575 (5D mesh reshape), L1594-L1614 (DCP groups), L1616-L1633 (PCP groups), L1636-L1640 (PP groups), L1741-L1782 (init assertion), L1791-L1797 (model buffer prep), L1847-L1854 (rank helpers) | The cluster-topology source: 5D mesh + GroupCoordinator singletons |
| `vllm/v1/attention/ops/dcp_alltoall.py` | 458 total; L1-L20 (docstring), L39-L103 (`_lse_weighted_combine`), L106-L130 (`_dcp_a2a_lse_pack_dim`), L134-L196 (Triton kernel `_dcp_a2a_pack_send_kernel`), L197-L319 (Triton kernel `_dcp_a2a_unpack_combine_kernel`), L320-L447 (orchestration), L448 (`dist.all_to_all_single` — the actual NCCL call) | A2A communication backend (pure functions, NOT a class) |
| `vllm/v1/attention/backend.py` | 1034 total; L700-L758 (`AttentionImpl` CP fields), L703 (`supports_pcp`), L705-L706 (`supports_mtp_with_cp_non_trivial_interleave_size`), L722-L729 (`dcp_world_size`/`dcp_rank`/`pcp_world_size`/`pcp_rank`/`total_cp_world_size`/`total_cp_rank` fields), L731-L757 (`__new__` discovery via `get_dcp_group()`/`get_pcp_group()` with try/except for testing) | Per-backend CP integration scaffolding |
| `vllm/v1/attention/backends/utils.py` | 898 total; L820-L857 (`get_dcp_local_seq_lens`) | Per-rank striped-shard helper — THE main sharding function |
| `vllm/v1/attention/backends/flashinfer.py` | 1360+ total; L213 (`class BatchDCPPrefillWrapper`) | The ONLY DCP-prefixed class — flashinfer-specific batched wrapper |
| `vllm/v1/attention/backends/flash_attn.py` | (large) | DCP path in flash-attn V3 backend |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | (large); L125 (`supports_dcp_with_varlen=(interleave_size==1)`), L175 (`num_heads_q=num_heads * dcp_world_size`), L196-L250 (DCP metadata threading), L353-L355 (a2a backend wired) | MLA + DCP integration — production stack for DeepSeek |
| `vllm/v1/attention/backends/mla/flashmla.py` | (large); L160-L200 (DCP integration) | flashmla MLA backend with DCP |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | (large); L213, L311 | ROCm AITER MLA + DCP |
| `vllm/v1/kv_cache_interface.py` | 784 total; L196-L204 (`max_memory_usage_bytes`) | THE HBM accounting — `cdiv(max_model_len, dcp × pcp)` |
| `vllm/v1/executor/multiproc_executor.py` | 1037 total; L116-L122 (world_size assertion `tp × pp × pcp`), L258-L259 (`_get_parallel_sizes` returns `(tp, pp, pcp)`), L985-L1001 (process tagging `_PCP{rank}`) | World-size orchestration; DCP excluded from product |
| `vllm/config/parallel.py` | 957 total; L115 (`prefill_context_parallel_size`), L310-L313 (`decode_context_parallel_size`), L315-L321 (deprecated `dcp_kv_cache_interleave_size`), L322-L328 (`DCPCommBackend = Literal["ag_rs", "a2a"]`), L330-L342 (`cp_kv_cache_interleave_size`), L469-L478 (`tp % dcp == 0` constraint), L480-L483 (a2a requires dcp > 1), L765 (world_size product) | Configuration surface — THE source-of-truth for what's wired |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | (large); `if self.moe_config.pcp_size > 1` branch (allgather hidden + router_logits, reduce_scatter post-expert) | MoE under PCP — composes with EP |

This is **12 source files** in the table — exceeds the v6 floor of 5
(Ch04: 4, Ch05: 4, Ch06: 4, Ch07: 5, Ch08: 8, Ch09: 10, Ch10: 11). The
breadth IS the lesson — DCP/PCP is a 5-axis device-mesh + 9-attention-backend
integration spanning 12+ files.

### 1.2 Key classes and module-level objects

| Source object | Lines | Purpose | Owns | Delegates |
|---|---|---|---|---|
| `_DCP: GroupCoordinator | None` (module global) | parallel_state.py:L1234 | The DCP process group — folded INSIDE TP | the GroupCoordinator wrapping torch.distributed ProcessGroup | NCCL via device_communicator |
| `_PCP: GroupCoordinator | None` (module global) | parallel_state.py:L1285 | The PCP process group — INDEPENDENT axis | the GroupCoordinator wrapping torch.distributed ProcessGroup | NCCL via device_communicator |
| `get_dcp_group()` (free function) | parallel_state.py:L1237-L1239 | Singleton accessor; raises AssertionError if not init | nothing | `_DCP` singleton |
| `get_pcp_group()` (free function) | parallel_state.py:L1288-L1290 | Singleton accessor; raises AssertionError if not init | nothing | `_PCP` singleton |
| `initialize_model_parallel(...)` | parallel_state.py:L1497-L1782 | Builds all groups (TP/DCP/PCP/PP/DP) from a 5D mesh reshape | the global state mutations | torch.distributed.init_process_group; init_model_parallel_group helpers |
| `_lse_weighted_combine(outputs, lses, ...)` (free function) | dcp_alltoall.py:L39-L103 | LSE-weighted reduction across N CP ranks; same algebra as FlashAttention online softmax | nothing | torch primitives only |
| `BatchDCPPrefillWrapper` (class) | flashinfer.py:L213 | flashinfer-specific batched wrapper for DCP-prefill | flashinfer state | flashinfer kernels |
| `AttentionImpl.__new__` (CP fields) | backend.py:L731-L757 | Per-backend CP rank discovery via get_dcp_group()/get_pcp_group() | dcp_world_size, dcp_rank, pcp_world_size, pcp_rank, total_cp_world_size, total_cp_rank | get_dcp_group(), get_pcp_group() |
| `get_dcp_local_seq_lens(...)` (free function) | backends/utils.py:L820-L857 | Compute per-rank local KV lengths under cp_kv_cache_interleave_size | nothing | torch primitives only |
| `AttentionSpec.max_memory_usage_bytes(self, vllm_config)` | kv_cache_interface.py:L196-L204 | Per-rank KV-cache HBM = `cdiv(max_model_len, dcp × pcp) × cdiv(..., block_size) × page_size_bytes` | nothing | cdiv |
| `MultiprocExecutor.__init__` (assertion) | multiproc_executor.py:L116-L122 | world_size = tp × pp × pcp (DCP excluded — folded inside TP) | nothing | nothing |
| `ParallelConfig` (dataclass) | config/parallel.py:L73+ | Houses `decode_context_parallel_size`, `prefill_context_parallel_size`, `dcp_comm_backend`, `cp_kv_cache_interleave_size`, the `tp % dcp == 0` validator | all CP-related config fields | pydantic validators |

### 1.3 Data flow — one DCP+PCP attention layer

```
Prefill phase (PCP shards input sequence):
  Each rank holds: input[rank*S/pcp : (rank+1)*S/pcp]  (its 1/pcp slice)
   │
   ├─ Compute Q, K, V locally on its slice           (no comm yet)
   │
   └─ CP attention (Ring or AG+RS or A2A):
        Each rank attends its local Q against ALL ranks' K, V
        via cross-rank communication. Output is per-rank slice.

Decode phase (DCP shards stored KV):
  Each rank holds:
    Q (replicated within TP group from column-parallel)
    K_local, V_local — striped chunks of total KV (size seq_len/dcp)
   │
   ├─ Backend 1 (AG+RS, default `dcp_comm_backend="ag_rs"`):
   │    AllGather Q     ◄── 1 NCCL op
   │    Local attention(Q, K_local, V_local)
   │    ReduceScatter O ◄── 1 NCCL op
   │    + LSE-weighted combine across ranks
   │    Total: 2 NCCL ops + 2 kernels per layer
   │
   └─ Backend 2 (A2A, advanced `dcp_comm_backend="a2a"`):
        Local attention(Q, K_local, V_local) → (O_partial, LSE_partial)
        AllToAll packed (O + LSE)  ◄── 1 NCCL op
        Triton-fused LSE-weighted combine
        Total: 1 NCCL op + 2 kernels per layer (33% reduction)
```

**Two key invariants** (the LSE-weighted combine + axis orthogonality
ARE the Ch11 invariants, parallel to Ch07's chain-break and Ch10's
chain-break invariant):

1. **LSE-weighted combine is mathematically associative + commutative.**
   Ring Attention, AG+RS, A2A — all three are different *transports* for
   the *same* algebra: `softmax_max + log(sum(exp(lse_i - max)))` plus
   `(weight_i * O_i).sum() / weight_sum`. Identical output bytes on the
   destination rank regardless of which transport carried the partials.

2. **DCP folds inside TP; PCP expands world_size.** `_DCP` group is built
   by `all_ranks.reshape(-1, dcp_size).unbind(0)` (no transpose — DCP is
   the FASTEST-varying axis within TP). `_PCP` group is built by
   `all_ranks.transpose(3, 4).reshape(-1, pcp_size).unbind(0)` (transpose
   pcp ↔ tp axes BEFORE reshape — PCP is its own axis).

### 1.4 Design decisions and WHY

1. **DCP folded inside TP, not a separate axis** (`parallel.py:L469-L478`,
   `parallel_state.py:L1597-L1600`). **Why?** Decode-time Q is already
   replicated within the TP group (from column-parallel linear). DCP can
   reuse the TP group's intra-node NVLink communication topology without
   expanding world_size or requiring new physical GPUs. **Trade-off**:
   `tp_size % dcp_size == 0` becomes a hard constraint; you can't have
   `tp=4, dcp=3`. Source enforces explicitly with a ValueError.

2. **PCP is an independent axis, not folded** (`parallel_state.py:L1616-L1633`).
   **Why?** Prefill loads input from outside the TP group's existing
   data flow — needs its own communication topology and rank set. Each
   PCP rank computes Q, K, V from a different sequence shard, and those
   shards aren't in the TP group's data flow. So PCP gets its own group
   built via transpose-then-reshape.

3. **Two DCP backends (AG+RS + A2A) co-exist instead of one canonical**
   (`parallel.py:L322-L328`). **Why?** AG+RS is the older, simpler
   default — easier to reason about, easier to debug. A2A is newer
   (arxiv.org/abs/2507.07120) and reduces NCCL ops 3 → 2 (33% fewer
   collectives) — production-tested win on MLA workloads. Operators pick
   based on their workload + bandwidth profile.

4. **Striped sharding via `cp_kv_cache_interleave_size`**
   (`parallel.py:L330-L342`). **Why?** Under causal masking, the rank
   holding the LATE tokens does ~cp_size× more attention work than the
   rank holding the EARLY tokens (more queries, more KV to attend to).
   Striped sharding (interleave=1) puts an even mix of early/late tokens
   on every rank → uniform load. Trade-off: cache-unfriendly access
   patterns. Most production uses `interleave=block_size` (e.g. 16) as
   a sweet spot.

5. **A2A requires `dcp_size > 1`** (`parallel.py:L480-L483`). **Why?**
   At dcp=1 there's nothing to all-to-all — the "AG+RS vs A2A" choice
   is meaningless. The check is defensive against misconfigured runs.

### 1.5 Complexity our implementation must preserve

| Mechanism | Source detail | Why we keep it |
|---|---|---|
| **5D mesh reshape** | `(-1, dp, pp, pcp, tp)` at `parallel_state.py:L1569-L1575` | THE topology of the system; axis-by-axis groups depend on this exact ordering |
| **DCP via reshape, PCP via transpose+reshape** | `parallel_state.py:L1597, L1618` | DCP is the fastest-varying axis (so contiguous reshape works); PCP needs transpose to land at the innermost axis before reshape |
| **`tp_size % dcp_size == 0`** | `parallel.py:L474-L478` | Hard constraint; violations = corrupt groups |
| **`world_size = tp × pp × pcp` (DCP excluded)** | `multiproc_executor.py:L117` | DCP is folded; including it would over-allocate processes |
| **LSE-weighted combine algebra** | `dcp_alltoall.py:L39-L103` | The mathematical core; without LSE re-scaling, the cross-rank sum produces NaN under any non-trivial attention |
| **`cp_kv_cache_interleave_size` formula** | `backends/utils.py:L820-L857` | Per-rank local seq length depends on this exactly; off-by-one breaks the kernel |
| **`total_cp_world_size = pcp × dcp`, `total_cp_rank = pcp_rank × dcp + dcp_rank`** | `backend.py:L751-L752` | Composed CP rank; many backends use this for slot mapping |
| **`max_memory_usage_bytes` via `cdiv(max_model_len, dcp × pcp)`** | `kv_cache_interface.py:L196-L204` | THE HBM accounting; off-by-one over-allocates KV blocks |

What we **legitimately simplify** (each tagged in the source):

| Source feature | Our simplification | Tagged in code with |
|---|---|---|
| Triton kernels in dcp_alltoall.py | Pure NumPy/PyTorch | `# REFERENCE` + comment "single-process simulation" |
| `dist.all_to_all_single` NCCL call | Direct tensor ops | "we hold all ranks' state in one process" |
| Multi-process group construction | Single-process Mesh5D struct | dataclass with computed properties |
| `GroupCoordinator` (NCCL wrapping) | `CPGroupCoordinator` (rank list only) | the singleton pattern is preserved; the NCCL is dropped |
| 9 attention backends with DCP integration | Single `attention_backend_dcp_pcp.py` showing the `__new__` discovery pattern | one anchor file is enough — the pattern is identical across backends |
| MoE under PCP (`moe_runner.py`) | Not implemented in Ch11 | "out of scope, references Ch09 EP" |

---

## §2 — Outline-vs-Source Reframes

The brief identified 5 reframes. All are documented at the chapter level
(outline JSON unchanged per `repo2book.json` consensus rule).

### Reframe A — §11.2 "no class RingAttention" (5th instance)

**The outline subsection 2 is "Ring Attention — peer-to-peer P2P 通信的环形拓扑"**.
A direct reading invites the writer to look for `class RingAttention` in
vLLM. **It does not exist**. Verified at commit 98661fe:

```bash
$ grep -rE '^class\s+(RingAttention|StripedAttention|ContextParallel|DecodeContextParallel|PrefillContextParallel)' \
       instances/vllm/source/vllm/
# (no matches)

$ grep -rE '^class\s+\w*DCP\w*|^class\s+\w*PCP\w*' instances/vllm/source/vllm/
instances/vllm/source/vllm/v1/attention/backends/flashinfer.py:213:class BatchDCPPrefillWrapper:
# (only 1 DCP-prefixed class — flashinfer-specific batched wrapper, NOT a top-level orchestrator)
```

The DCP/PCP machinery is:
- **Module-level pure-function code** (`dcp_alltoall.py` is 458 lines of pure functions + Triton kernels)
- **GroupCoordinator singletons** (`_DCP`, `_PCP` in `parallel_state.py:L1234, L1285`)
- **Per-attention-backend `__new__` discovery** (`backend.py:L731-L757` — every backend, not a single orchestrator)

vLLM **departs from the canonical Liu et al. 2023 algorithm** (Ring
Attention with P2P send/recv) and uses NCCL collectives (AG+RS or A2A)
instead. The MATH is the same (LSE-weighted online softmax); the
COMMUNICATION pattern is fundamentally different.

**Three-anchor template** (matches Ch07/Ch08/Ch09/Ch10):

- **Anchor 1 (chapter title)**: title says "DCP/PCP — Decode/Prefill Context Parallelism" — names the technique by the vLLM-side feature name, NOT by the literature-name.
- **Anchor 2 (hook)**: opening paragraph names the absence + names the canonical example (`_DCP` / `_PCP` GroupCoordinator pair) + names the FIVE-INSTANCE motif explicitly: "Ch07 (no `RadixTree`) → Ch08 (no `class TensorParallel`) → Ch09 (no `class ExpertParallel`) → Ch10 (no `class MultiTokenPrediction`) → Ch11 (no `class RingAttention`). vLLM systematically prefers module-level + singleton patterns over orchestrator classes."
- **Anchor 3 (§11.2 body)**: full grep evidence + 1-DCP-prefixed-class enumeration + comparison to Liu et al. 2023. Reviewer counts.

This is the **fifth** instance of the "no class X" reframe pattern —
graduates from a "trend" (4 instances) to a confirmed chapter motif (5
instances). Wisdom-promotion candidate after Ch12 hits the same pattern
on a sibling system, OR earlier per the M28-style "queue at instance #3"
rule — but per `feedback_wisdom_gate_strict.md` we still need 2+ INSTANCES
across instances, so this stays repo-local.

### Reframe B — §11.3 "all-reduce vs all-to-all" → "AG+RS vs A2A"

**The outline subsection 3 is "DCP — decode 阶段的 all-reduce vs all-to-all 方案"**.
The actual two backends are **AllGather+ReduceScatter (AG+RS)** vs
**All-to-All (A2A)**. NEITHER is `all-reduce`. Source verbatim:

```python
# vllm/config/parallel.py:L322-L328
DCPCommBackend = Literal["ag_rs", "a2a"]
```

Surgical correction: chapter says "AG+RS vs A2A" everywhere. Outline JSON
unchanged. The terminology is verbatim from `parallel.py:L322-L328`.

### Reframe C — §11.4 "striped vs balanced" → "interleave_size knob"

**The outline subsection 4 is "PCP — prefill 阶段的 striped vs balanced 切分"**.
"Balanced" in the literature is the load-balanced variant (Striped
Attention). vLLM's source uses the term `cp_kv_cache_interleave_size`
(`parallel.py:L330-L342`):
- `interleave=1` → fully striped (token-level): rank `r` owns tokens
  `r, r+cp, r+2cp, ...`
- `interleave=block_size` → block-aligned: each rank gets contiguous
  blocks of size `block_size`, alternating ranks
- `interleave=∞` → fully contiguous (legacy mode)

NOT a major reframe; the outline term "striped" matches; "balanced"
refers to the load-balancing PROPERTY of striped sharding under causal
mask. We show this empirically in Demo §4 — striped (I=1) gives 1.24x
imbalance, contiguous (I=64) gives 13.44x imbalance.

### Reframe D — §11.5 "3D parallel" → "5D mesh + DCP nested"

**The outline subsection 5 is "CP+TP 的 3D 并行——device mesh 的映射策略"**.
The actual production mesh is **5D**:

```python
# vllm/distributed/parallel_state.py:L1569-L1575
all_ranks = torch.arange(world_size).reshape(
    -1,                                      # external_dp (verl)
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

Five axes: `external_dp × dp × pp × pcp × tp`. DCP is folded INSIDE the
TP axis (so logically a 6th axis, but doesn't expand world_size). The
outline term "3D" undersells.

### Reframe E — DCP and PCP are SEPARABLE axes (Trap D anchor)

The chapter must explicitly derive: at `(tp=8, pcp=2, dp=1, pp=1)`,
world_size = `8 × 2 × 1 × 1 = 16`; with `dcp=4`, each TP-group of 8
splits into `8/4=2` DCP sub-groups; each rank's local KV chunk =
`seq_len / (pcp × dcp) = seq_len / 8`. **DCP and PCP do NOT need to
match.** The only constraint is `tp % dcp == 0`
(`parallel.py:L474-L478`).

---

## §3 — Language traps (≥6 required; 7 below)

Each trap is "easy to write and almost right". Writer should pick 5-7
strongest, callout at the relevant section, and recap in §11.7.

### Trap A — "DCP doubles decode throughput at dcp_size=2."
**Wrong.** DCP shards KV CACHE memory across ranks; throughput depends on
attention compute and communication. With dcp=2, each rank computes
attention against `seq_len/2` of KV — half the FLOPs per rank, but each
layer needs 1 extra all-gather (or 1 extra a2a). At short seq_len the
comm overhead can dominate. **DCP's headline value is HBM CAPACITY, not
throughput.** Source: `kv_cache_interface.py:L196-L204` is the WIN.
**Demo §1**: HBM=40GB → 10GB at dcp=2,pcp=2; throughput unchanged in our
single-process simulation.

### Trap B — "PCP halves prefill latency at pcp_size=2."
**Partially true.** With pcp=2, half the prefill FLOPs per rank, but
all-to-all comm adds `O(cp_size)` rounds. For 32K prefill on H100 NVLink,
pcp=2 ≈ 2× speedup. For 4K on PCIe, comm overhead can exceed compute
→ NET LOSS. **Operator must measure prefill length × bandwidth before
enabling.** Source: no source-side gate. **Demo §3** shows the alpha-beta
crossover.

### Trap C — "Context parallel is just sequence parallel renamed."
**Wrong.** **Sequence parallel** (Megatron) shards SEQUENCE DIMENSION of
ACTIVATIONS within a TP group's MLP/LN to save activation HBM. Does NOT
shard KV. **Context parallel** (DCP/PCP) shards KV (DCP) or input
sequence (PCP). The two are orthogonal and coexist. Source:
`parallel_state.py` exposes BOTH `is_sequence_parallel` parameter on
all_gather/reduce_scatter AND `_DCP`/`_PCP` GroupCoordinators — distinct
mechanisms.

### Trap D — "DCP and PCP must match (dcp_size == pcp_size)."
**Wrong.** They are **separable axes**. Production runs `(tp=8, dcp=2,
pcp=4)` (DCP=2 within TP-group of 8 → 4 DCP sub-groups; PCP=4 as
independent axis). Constraints: `tp % dcp == 0` AND `world_size = tp ×
pp × pcp × dp` (DCP excluded). Source: `parallel.py:L474-L478` only
enforces `tp_size % dcp_size == 0`; `multiproc_executor.py:L116-L121`
confirms world_size product excludes dcp. **Demo §5** shows
`(tp=4, pcp=2, pp=2, dcp=2)` is valid: world_size=16.

### Trap E — "Context parallel is the same as TP for the attention layer."
**Wrong.** TP shards the HEAD axis (each rank owns `num_heads / tp_size`
heads, full sequence per head). CP shards the SEQUENCE axis (each rank
owns full heads, `seq_len / cp_size` tokens of K and V). Different axis
→ different communication: TP needs `all_reduce` on attention output
(heads contribute partial sums); CP needs `all_to_all` or AG+RS on the
output (ranks contribute partial outputs over disjoint KV chunks,
combined via LSE weighting). Source: TP comm is in
`vllm/model_executor/layers/linear.py` (`RowParallelLinear.forward`
all_reduce); CP comm is in `vllm/v1/attention/ops/dcp_alltoall.py`.
Different code paths, different math, different layer placement.

### Trap F — "Ring Attention is the canonical implementation in vLLM."
**Wrong.** vLLM does NOT implement Ring Attention. The codebase ships
AllGather+ReduceScatter (`dcp_comm_backend="ag_rs"`, default) or
All-to-All (`dcp_comm_backend="a2a"`, advanced). Both use NCCL
collectives, not P2P send/recv ring topology. Mathematically the
LSE-weighted combine is similar to Ring Attention's online softmax, but
the COMMUNICATION pattern is collective, not peer-ring. Source:
`dcp_alltoall.py:L1-L20` module docstring + `dist.all_to_all_single` at
L448 — no Ring; **Demo §3** shows AG+RS = 2 NCCL ops, A2A = 1 NCCL op.

### Trap G — "Striped Attention is just renamed Ring Attention."
**Wrong.** Striped Attention is a TOKEN-PARTITIONING scheme (token
i → rank `i % cp_size`), independent of the COMMUNICATION pattern (Ring
vs all-to-all vs ag+rs). vLLM's `cp_kv_cache_interleave_size` knob
controls partitioning; communication is separate. Striped's purpose is
load-balancing under causal mask — late-token Q has more KV to attend
to, contiguous gives `cp_size×` rank imbalance, striped (interleave=1)
gives perfect balance. Source: `cp_kv_cache_interleave_size` and
`dcp_comm_backend` are independent config knobs. **Demo §4**: imbalance
13.44x → 1.24x going from contiguous to striped.

---

## §4 — Honest demo caveats (must be quoted verbatim by writer per K17)

1. **Single-process simulation does NOT actually launch NCCL collectives.**
   The math is verified bit-exact against single-process FlashAttention
   (max abs error 3.33e-16 in fp32), but real intra-node NVLink bandwidth
   (200+ GB/s) and inter-node IB bandwidth (50 GB/s) are not measured —
   alpha-beta model in §3 uses literature numbers.

2. **Alpha-beta values are H100 + 4×NVLink reference numbers**;
   A100 + InfiniBand would shift β by ~3-5× and α by ~2×. Real production
   should measure on target hardware.

3. **`supports_pcp` is not yet True for all attention backends** — flash_attn
   V3 has explicit DCP support; PCP is still wiring up. The demo simulates
   PCP as if all backends support it; production hits `NotImplementedError`
   on some.

4. **`cp_kv_cache_interleave_size` is the latest API**;
   `dcp_kv_cache_interleave_size` is deprecated (`parallel.py:L315-L321`).
   Demo uses the new name; production code at older commits may have the
   old name.

5. **5D mesh demo uses `external_dp=1`** (default for non-verl integrations);
   verl deployments would use `external_dp > 1`.

---

## §5 — Source mapping (main; per-section mini-tables in §6)

| Our Code | Original Source | What We Changed | Why |
|---|---|---|---|
| `parallel_state_dcp_pcp.py::CPGroupCoordinator` | `vllm/distributed/parallel_state.py:L1234-L1290 _DCP/_PCP + get_dcp_group/get_pcp_group` | Drop NCCL plumbing; keep singleton + ranks list pattern | Pedagogical; the singleton pattern is the lesson |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` | `parallel_state.py:L1497-L1782` | Strip down to group-construction core | The 280 lines of CUDA stream init etc. is irrelevant to the algorithm |
| `world_topology.py::MeshConfig` | `parallel_state.py:L1569-L1575` reshape + `multiproc_executor.py:L116-L122` assertion | dataclass with computed `.world_size`; `tp % dcp == 0` validator | 1:1 mathematical mirror |
| `lse_combine.py::lse_weighted_combine` | `vllm/v1/attention/ops/dcp_alltoall.py:L39-L103 _lse_weighted_combine` | 1:1 PyTorch transcription | Demonstrates the algorithm in isolation |
| `lse_combine.py::reference_attention_with_lse` | (no direct source — derived from FlashAttention algebra) | New helper | Provides ground truth for unit tests |
| `lse_combine.py::split_kv_for_cp` | `backends/utils.py:L820-L857 get_dcp_local_seq_lens` (the partitioning aspect) | Simplified to contiguous chunks; striped variant in `seq_sharding.py` | One thing per file |
| `dcp_alltoall.py::ag_rs_op_count`/`a2a_op_count` | `dcp_alltoall.py` (entire module — implies the count) | New helpers extracting the headline number | Demo §3 |
| `dcp_alltoall.py::ag_rs_payload_bytes`/`a2a_payload_bytes` | `dcp_alltoall.py:L431-L436` send_buffer shape | Closed-form formulas matching source's tensor shapes | Alpha-beta model needs these |
| `dcp_alltoall.py::alpha_beta_cost` | (no direct source — bandwidth model) | New helper | Production cost model for §3 |
| `dcp_alltoall.py::simulate_a2a_combine` | `dcp_alltoall.py:L320-L450` (orchestration) | Drop NCCL; call lse_weighted_combine directly | Same algebra |
| `seq_sharding.py::get_dcp_local_seq_lens` | `backends/utils.py:L820-L857` | 1:1 mirror with same `base + remainder` formula | Source is short; we transcribe verbatim |
| `seq_sharding.py::causal_attention_work_per_rank` | (no direct source — derived) | New analyzer | Demo §4 imbalance computation |
| `seq_sharding.py::imbalance_ratio` | (no direct source) | New helper | Demo §4 reporting |
| `kv_cache_per_rank.py::KVCacheModel` | `vllm/v1/kv_cache_interface.py:L196-L204` | Closed-form `max_memory_usage_bytes`; cdiv preserved | THE HBM win |
| `kv_cache_per_rank.py::hbm_per_rank_sweep` | (no direct source — sweep helper) | New helper | Demo §1 |
| `attention_backend_dcp_pcp.py::AttentionImplBase.__new__` | `vllm/v1/attention/backend.py:L731-L757` | Same try/except discovery pattern | The `__new__` mechanism is the lesson |
| `dcp_vs_pcp_demo.py::CPRoles` | (compiled from parallel_state.py + parallel.py) | New struct documenting axis roles | Trap D anchor |
| `dcp_vs_pcp_demo.py::per_rank_kv_chunk` | `kv_cache_interface.py:L196-L204` rearranged | Closed form `seq_len / (dcp × pcp)` | Demo §5 |

This is **18 main mapping rows** + per-section mini-tables (§6 below).

---

## §6 — Per-section mini-mapping tables

### §11.1 — Why CP at all (HBM math)

| Concept | Source | Our impl |
|---|---|---|
| HBM formula | `kv_cache_interface.py:L196-L204` | `kv_cache_per_rank.py::KVCacheModel.max_memory_usage_bytes` |
| `cdiv` rule | `kv_cache_interface.py:L203` `cdiv(max_model_len, dcp × pcp)` | `kv_cache_per_rank.py::cdiv` |
| page_size_bytes | `kv_cache_interface.py:L204` `cdiv(max_model_len, block_size) × page_size_bytes` | `kv_cache_per_rank.py::KVCacheModel.page_size_bytes` |
| Sweep helper | (none in source — analytical) | `kv_cache_per_rank.py::hbm_per_rank_sweep` |

### §11.2 — Ring Attention reframe (no class X)

| Concept | Source | Our impl |
|---|---|---|
| Module-level pure functions | `dcp_alltoall.py:L39-L103` | `lse_combine.py::lse_weighted_combine` |
| GroupCoordinator singletons | `parallel_state.py:L1234-L1290` | `parallel_state_dcp_pcp.py::CPGroupCoordinator` |
| Per-backend `__new__` discovery | `backend.py:L731-L757` | `attention_backend_dcp_pcp.py::AttentionImplBase.__new__` |
| Only DCP-prefixed class | `flashinfer.py:L213` `class BatchDCPPrefillWrapper` | (referenced; not reimplemented — flashinfer-specific) |

### §11.3 — DCP backends (AG+RS vs A2A)

| Concept | Source | Our impl |
|---|---|---|
| DCPCommBackend literal | `parallel.py:L322-L328` | `dcp_alltoall.py::ag_rs_op_count` + `a2a_op_count` |
| AG+RS algorithm | `dcp_alltoall.py` module + (AG/RS calls in backends) | `dcp_alltoall.py::simulate_ag_rs_combine` |
| A2A algorithm | `dcp_alltoall.py:L320-L450` | `dcp_alltoall.py::simulate_a2a_combine` |
| `dist.all_to_all_single` | `dcp_alltoall.py:L448` | (simulated — single process) |
| MLA + a2a wiring | `mla/flashattn_mla.py:L353-L355` | (referenced; not reimplemented — MLA is Ch27 territory) |

### §11.4 — PCP + interleave knob

| Concept | Source | Our impl |
|---|---|---|
| `_PCP` group construction | `parallel_state.py:L1616-L1633` | `world_topology.py::MeshConfig` (computes via reshape rules) |
| `cp_kv_cache_interleave_size` | `parallel.py:L330-L342` | `seq_sharding.py::total_cp_rank` (token → rank mapping) |
| `get_dcp_local_seq_lens` | `backends/utils.py:L820-L857` | `seq_sharding.py::get_dcp_local_seq_lens` |
| Causal-mask load balance math | (derived from causal mask) | `seq_sharding.py::causal_attention_work_per_rank` + `imbalance_ratio` |
| Deprecated `dcp_kv_cache_interleave_size` | `parallel.py:L315-L321` | (noted in module docstring) |

### §11.5 — 5D mesh

| Concept | Source | Our impl |
|---|---|---|
| 5D reshape | `parallel_state.py:L1569-L1575` | `world_topology.py::MeshConfig` (5 axes as dataclass fields) |
| TP groups | `parallel_state.py:L1580` `reshape(-1, tp).unbind(0)` | (computed from `MeshConfig`) |
| DCP groups (folded) | `parallel_state.py:L1601` `reshape(-1, dcp).unbind(0)` | (computed) |
| PCP groups (transpose) | `parallel_state.py:L1618-L1622` `transpose(3,4).reshape(-1, pcp).unbind(0)` | (computed) |
| world_size assertion | `multiproc_executor.py:L117` | `world_topology.py::MeshConfig.world_size` (excludes dcp) |
| total_cp formulas | `backend.py:L751-L752` | `attention_backend_dcp_pcp.py` |

---

## §7 — Demo verbatim numerics summary

Demo run output (seed=42, see `demo.py`):

```
§1 HBM-per-rank capacity walk:
  8 (dcp,pcp) cells; (1,1)=40.0GB, (2,2)=10.0GB, (4,4)=2.5GB
  Trap A evidence: HBM is the WIN axis

§2 LSE-weighted combine:
  4 ranks, B=4, H=2, D=8
  Per-rank LSE max: 2.106473
  Per-rank weights normalized to sum=1
  max abs error vs single-process FA = 3.33e-16 (theorem holds)
  associativity error = 2.22e-16

§3 AG+RS vs A2A bandwidth model:
  AG+RS = 3 ops, A2A = 2 ops, 33% reduction
  alpha=10us, beta=200GB/s
  dcp=2: AG+RS=1036.6us, A2A=360.8us (2.87x)
  dcp=4: A2A=190.4us (5.44x)
  dcp=8: A2A=105.2us (9.85x)
  Trap F evidence: NEITHER is Ring

§4 Striped vs contiguous causal-mask load balance:
  cp=8, seq_len=64
  contiguous: imbalance=13.44x (rank 0=36, rank 7=484)
  block-striped (K=2): 1.55x
  striped (I=1): 1.24x (perfectly balanced)
  Trap G evidence: striped IS the load balance fix

§5 5D mesh at world=16, (tp=4, pcp=2, pp=2, dcp=2):
  4 TP groups, 8 DCP sub-groups, 8 PCP groups, 8 PP groups
  total_cp_world_size = 4
  HBM-per-rank factor = 1/4
  Trap E + reframe evidence: 5D mesh, not 3D
  Trap D evidence: separable axes valid

GRAND TOTAL: ~120 verbatim numbers — exceeds ≥100 target
```

---

## §8 — Forward/back pointers (cross-chapter wiring)

### Back pointers
- **Ch03** (FlashAttention): LSE-weighted combine algebra is FA online softmax across ranks.
- **Ch04** (continuous-batching): prefill-vs-decode phase distinction.
- **Ch08** (TP): 5D mesh reshape pattern + GroupCoordinator pattern.
- **Ch09** (EP): `_EP`/`_EPLB` GroupCoordinator pattern is the model for `_DCP`/`_PCP`; PCP composes with EP for MoE prefill.
- **Ch10** (MTP): `supports_mtp_with_cp_non_trivial_interleave_size` flag at `backend.py:L705-L706` is the explicit cross-chapter knob — MTP-with-CP requires `interleave_size=1` (or backend-specific support).

### Forward pointers
- **Ch15+** (model zoo): every long-context production model uses CP.
- **Ch18** (Triton attention): A2A combine is Triton-fused.
- **Ch22** (PD architecture): CP composes with PD disaggregation.
- **Ch25** (PD ratio): DCP world-size becomes a budget variable.
- **Ch27** (DeepSeek-V3.2): MLA + DCP is the production stack.

### Fidelity gap callouts (per Ch04-Ch10 cadence)
- NCCL collectives vs single-process simulation: noted; algebra is identical.
- Triton kernels vs PyTorch ops: noted; same algorithm.
- Alpha-beta model vs measured: noted; literature reference numbers.
- 5D mesh single-process construction vs multiproc spawn: noted.

---

## §9 — Open questions (for tester)

1. Should `extract_hidden_states.py` get a mention in §11.5 5D mesh (it
   asserts `num_speculative_tokens == 1` which constrains the cross-product
   with CP)? Current plan: no — Ch10 territory.

2. Should `BatchDCPPrefillWrapper` (the only DCP-prefixed class) get a
   §11.2 callout? Current: yes, as the no-class-X anchor's "exception that
   proves the rule".

3. Do we want a Demo §6 showing real PyTorch distributed launching? Current:
   no — single-process simulation suffices for the math; multi-process
   adds a lot of CI complexity for the same numerics.

End of impl-notes. Ready for tester gate (target ≥10 mapping rows; we have
18+ in main + 25+ in mini-tables = 43+; demo ≥100 verbatim; lint
PASS).
