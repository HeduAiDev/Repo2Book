# Ch09 Expert Parallelism — Implementer Notes

- Chapter: `09-expert-parallelism`
- Source pin: vLLM `98661fe012c5c467252d4df8411d2f46190e9268`
- Author role: implementer
- Date: 2026-05-07

The chapter teaches **how vLLM shards Mixture-of-Experts models across GPUs**,
through a five-file collaboration in the source: `parallel_state.py` (the
`_EP` and `_EPLB` process groups), `fused_moe/layer.py` (`FusedMoE`,
`determine_expert_map`), `fused_moe/config.py` (`FusedMoEParallelConfig`),
`device_communicators/all2all.py` (7 backend managers), and
`fused_moe/prepare_finalize/naive_dp_ep.py` (the dispatch+combine wiring).
Plus two reference sites: `models/mixtral.py:L77 MixtralMoE` (E=8, K=2,
plain softmax) and `models/deepseek_v2.py:L244 DeepseekV2MoE` (grouped
top-K, shared experts, `noaux_tc` correction bias).

This implementation re-derives the routing math and the EP+TP composition
in a single process. There are NO real `torch.distributed` collectives;
EP "ranks" are simulated by holding all rank maps in one block and running
the per-rank expert pass `ep_size` times. The α-β cost model predicts
what real NCCL / DeepEP would do, and the in-process forward gives bit-
identical outputs at `ep_size=1` and `ep_size=4` (verified — see
`tests/test_smoke.py::test_ep1_eq_ep4_forward`).

---

## §1 — Source Analysis (HARD GATE)

### 1.1 Files implementing EP in the target repo

| File | Lines (verified at 98661fe) | Role |
|---|---|---|
| `instances/vllm/source/vllm/distributed/parallel_state.py` | 2132 total; L1261-L1283 (`_EP`/`_EPLB` singletons), L1670-L1719 (mesh construction), L1797-L1801 (DeepEP buffer hooks), L1891-L1896 (teardown) | EP/EPLB process groups, GroupCoordinator |
| `instances/vllm/source/vllm/distributed/device_communicators/all2all.py` | 761 total; L40-L139 (`AgRsAll2AllManager`), L142-L325 (DeepEP HT/LL), L327-L440 (Nixl), L442-L670 (FlashInfer), L671+ (Mori) | 7 all-to-all backend managers — common interface, different kernels |
| `instances/vllm/source/vllm/model_executor/layers/fused_moe/layer.py` | 1649 total; L70-L157 (`determine_expert_map`), L160-L193 (`determine_expert_placement_strategy`), L196-L214 (`get_compressed_expert_map`), L219-L290 (`FusedMoE` decl), L290-L605 (`FusedMoE.__init__`), L1543-L1649 (forward) | The MoE composition layer — gate, experts, routing, EP/TP plumbing |
| `instances/vllm/source/vllm/model_executor/layers/fused_moe/config.py` | 1359 total; L998-L1209 (`FusedMoEParallelConfig` dataclass + `.make()`) | EP-vs-TP collapse rule: `ep_size = tp×dp×pcp; tp_size = 1` when EP=True |
| `instances/vllm/source/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | 167 total; L17-L66 (kernel dispatch), L69-L113 (`fused_topk`), L116-L167 (`FusedTopKRouter`) | Mixtral path — softmax/sigmoid → top-K → renormalize |
| `instances/vllm/source/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | 353 total; L29-L72 (fused kernel path), L81-L162 (`grouped_topk`), L165-L244 (`GroupedTopk` CustomOp), L247-L353 (`GroupedTopKRouter`) | DeepSeek path — group score → topk_group → masked top-K, optional `e_score_correction_bias` |
| `instances/vllm/source/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | 258 total; L71-L168 (`MoEPrepareAndFinalizeNaiveDPEPModular`) | Dispatch → expert exec → combine wiring; calls `get_ep_group().dispatch/.combine` |
| `instances/vllm/source/vllm/distributed/eplb/eplb_state.py` | 1166 total; L62-L210 (`EplbStats`, `EplbModelState`), L210-L920 (`EplbState`), L925-L944 (`EplbLayerState`) | Runtime expert-load rebalance state machine |
| `instances/vllm/source/vllm/model_executor/models/mixtral.py` | 601 total; L77-L154 (`MixtralMoE`) | Reference site #1: simple MoE — E=8, K=2, no shared expert |
| `instances/vllm/source/vllm/model_executor/models/deepseek_v2.py` | 1729 total; L244-L386 (`DeepseekV2MoE`) | Reference site #2: production MoE — `n_routed_experts`, grouped topk, shared experts, `noaux_tc` |

This is **10 source files** in the table — exceeds the v6 floor of 5
(Ch04: 4, Ch05: 4, Ch06: 4, Ch07: 5, Ch08: 8). The breadth IS the
lesson — EP is a 5-file collaboration plus 2 reference sites plus 1
state machine plus the all-to-all backend registry.

### 1.2 Key classes and their responsibilities

| Source class | Lines | Purpose | Owns | Delegates |
|---|---|---|---|---|
| `FusedMoE(PluggableLayer)` | layer.py:L219-L1649 | Composition: gate + experts + routing + EP/TP plumbing | `moe_parallel_config`, `_expert_map`, `quant_method`, `router`, `runner` | `RouterFactory.create_fused_moe_router(...)` for routing; `quant_method.create_weights/apply` for weights/forward |
| `FusedMoEParallelConfig` (dataclass) | config.py:L998-L1209 | EP↔TP collapse rule | `tp_size`, `pcp_size`, `dp_size`, `ep_size`, `use_ep`, `all2all_backend`, `enable_eplb` | nothing — pure data + computed properties |
| `GroupCoordinator` (the `_EP` instance) | parallel_state.py:L290-L1136 | One-process-per-rank wrapper of a torch.distributed group | `world_size`, `rank_in_group`, `device_communicator` | `dispatch()` / `combine()` → device_communicator's all2all manager |
| `AgRsAll2AllManager(All2AllManagerBase)` | all2all.py:L40-L139 | Pedagogical baseline backend | nothing (stateless) | `dist_group.all_gatherv` for dispatch; `dist_group.reduce_scatterv` for combine |
| `DeepEPHTAll2AllManager` | all2all.py:L196-L256 | High-throughput cross-node | DeepEP handle cache | DeepEP IBGDA fused kernels |
| `DeepEPLLAll2AllManager` | all2all.py:L257-L325 | Low-latency cross-node | DeepEP LL handle cache | DeepEP low-latency kernels |
| `MoEPrepareAndFinalizeNaiveDPEPModular` | naive_dp_ep.py:L71-L168 | The dispatch+combine wrapper around `quant_method.apply` | `is_sequence_parallel`, `_num_dispatchers` | `get_ep_group().dispatch(...)` / `.combine(...)` |
| `FusedTopKRouter(BaseRouter)` | fused_topk_router.py:L116-L167 | Mixtral/Switch routing | `top_k`, `renormalize`, `scoring_func` | `fused_topk(...)` Triton kernel |
| `GroupedTopKRouter(BaseRouter)` | grouped_topk_router.py:L247-L353 | DeepSeek routing | `num_expert_group`, `topk_group`, `e_score_correction_bias` | `grouped_topk(...)` (compiled native + optional `fused_grouped_topk` kernel) |
| `EplbState` | eplb_state.py:L210-L920 | Runtime load rebalance | `expert_load_window_step`, `model_states`, `policy`, `physical_to_logical_map` | `_EPLB` GroupCoordinator for cross-rank sync |
| `MixtralMoE(nn.Module)` | mixtral.py:L77-L154 | Reference site: `gate=ReplicatedLinear`, `experts=FusedMoE(num_experts=8, top_k=2)` | `gate`, `experts`, EPLB hooks | `gate(h)` → `experts(h, router_logits)` |
| `DeepseekV2MoE(nn.Module)` | deepseek_v2.py:L244-L386 | Reference site: `gate=GateLinear` (with optional `e_score_correction_bias`), `experts=FusedMoE(use_grouped_topk=True, n_shared_experts=…)` | `gate`, `shared_experts`, `experts` | `gate(h)` + `shared_experts(h)` parallel to `experts(h, router_logits)` |

### 1.3 Data flow — one MoE block under EP+TP

```
hidden_states  [M, hidden]
   │
   ├─ gate (ReplicatedLinear or GateLinear)               ──► router_logits [M, E]
   │      gate weight is REPLICATED across all EP ranks
   │
   ├─ Top-K routing  (fused_topk OR grouped_topk)         ──► (topk_weights [M,K], topk_ids [M,K])
   │      Mixtral:  softmax → topk → renormalize
   │      DeepSeek: softmax → group_max → topk_group groups → masked topk → renormalize
   │
   ├─ EP-axis all-to-all DISPATCH    ⟵⟵⟵⟵⟵
   │      AgRs: dist_group.all_gatherv([h, tw, ti], dim=0, sizes=…)
   │      DeepEP: fused IB+NVLink kernel
   │
   ├─ Local expert FFN  (per-rank, local experts only)
   │      For each token whose top-K choice is on this rank:
   │           gate|up = h @ w13.T          (MergedColumnParallelLinear)
   │           act     = SiluAndMul(gate|up)
   │           down    = act @ w2.T          (RowParallelLinear, intra-expert TP)
   │           out += weight[k] * down
   │
   └─ EP-axis all-to-all COMBINE     ⟵⟵⟵⟵⟵
          AgRs: dist_group.reduce_scatterv(summed, dim=0, sizes=…)
          DeepEP: fused combine kernel
```

**Two all-to-alls per MoE block.** The chain-break invariant for Ch09:

> **`_EP` group is orthogonal to `_TP` group.** Routing → all-to-all →
> expert FFN (TP-sharded inside) → all-to-all → DONE. EP is the
> COMPLEMENT axis of TP×DP×PCP — `ep_size = world / (tp×pcp×dp)` when
> `enable_expert_parallel=True`. (`config.py:L1162-L1165, L1192-L1208`.)

### 1.4 Design decisions and WHY (≥3 with trade-off analysis)

1. **EP=True collapses TP into ep_size; tp_size becomes 1.**
   - Decision: `config.py:L1192-L1208` — when `use_ep=True`,
     `ep_size = flatten_tp_size; tp_size = 1`. Each rank holds *whole
     experts*, not sliced experts.
   - Trade-off: SAVES one TP-internal all-reduce per expert. Cost: each
     rank's expert weights are unsharded along TP, so memory/rank only
     drops by `1/ep_size` (not `1/(tp×ep)`). Production runs combine EP+TP
     by also wrapping the expert in `MergedColumnParallelLinear` /
     `RowParallelLinear` — the chapter §5 demos that.
   - Source: `config.py:L1162-L1208` (the make() function).

2. **EPLB has its own `_EPLB` process group.**
   - Decision: `parallel_state.py:L1700-L1719` — when
     `parallel_config.enable_eplb=True`, init a SEPARATE
     `init_model_parallel_group(group_name="eplb")` with the same rank
     list as `_EP`.
   - Trade-off: Doubles the number of registered groups but eliminates a
     deadlock class. Comment in source explicitly says "to prevent
     deadlocks when using torch.distributed in execution with
     torch.distributed in EPLB" (L1700-L1701). Without separation, an
     EPLB rebalance broadcast could block on the same NCCL stream as a
     forward-pass dispatch all-to-all.
   - Source: `parallel_state.py:L1700-L1719`.

3. **`expert_map[i] = -1` is the off-rank sentinel.**
   - Decision: `layer.py:L117` — `torch.full((global_num_experts,), -1,
     dtype=torch.int32)`. The -1 marks "this rank does not own expert i".
   - Trade-off: Adds a per-token branch in the local-expert pass (skip
     if -1) but avoids materializing per-rank expert lists. The
     alternative — a list of owned global expert IDs — would need a
     hashmap lookup per token; the -1 array is one indexed-load.
   - Source: `layer.py:L117-L131` (initialization of expert_map),
     `prepare_finalize/naive_dp_ep.py:L104-L132` (uses `expert_map` to
     skip).

4. **Linear vs round-robin placement.**
   - Decision: `layer.py:L119-L131` — two strategies. Linear: rank `r`
     owns experts `[r·base, (r+1)·base)`. Round-robin: rank `r` owns
     `r, r+P, r+2P, …`.
   - Trade-off: Linear is locality-friendly when consecutive expert IDs
     correlate (e.g. the trained model's gate weights cluster the hot
     experts). Round-robin breaks that correlation, distributing hot
     experts across ranks. Round-robin is ONLY supported under DeepEP-LL
     or NIXL backends today (`layer.py:L181-L191`) — the AgRs and
     DeepEP-HT backends require linear because their kernel layouts
     assume contiguous owned blocks.
   - Source: `layer.py:L160-L193` (`determine_expert_placement_strategy`).

5. **`ep_size==1` short-circuits expert_map to `None`.**
   - Decision: `layer.py:L108-L109` returns `(global_num_experts, None,
     None)` early. The forward path tests `if expert_map is None` and
     skips the per-token branch.
   - Trade-off: Specializes the dense / single-rank path so MoE models
     can run on a single GPU without paying the EP overhead. Costs one
     extra branch but the branch is predictable.
   - Source: `layer.py:L107-L109`, `prepare_finalize/no_dp_ep.py` (the
     specialized prepare_finalize for this case).

### 1.5 Outline-vs-source mismatch handling

The outline `book/book-outline.json` lists subsection 4 as:

> "Expert Load Balancing Loss的梯度回传"

**vLLM is inference-only — there is no aux-loss / load-balance-loss /
gradient-routing code anywhere in the codebase.** Verified via grep:
no hits for `load_balance|aux.*loss|balance.*loss` in
`vllm/model_executor/`. Aux loss is a *training* technique (Switch
Transformer's `L_balance = α · Σ f_i × P_i`); the trained model's expert
distribution is what the aux loss produced. At inference, vLLM has only
EPLB.

Reframe (§4 of the chapter):

1. Open with a one-page sidebar grounding readers from training literature
   ("aux loss is what training does to balance experts").
2. Pivot to vLLM's inference response: `EplbState` + redundant experts +
   periodic logical→physical reshuffle, anchored at
   `eplb_state.py:L210 class EplbState` and the separate `_EPLB` group at
   `parallel_state.py:L1700-L1719`.

This mirrors the Ch07 "no radix tree" / Ch08 "no class TensorParallel"
reframe pattern. Reviewer will check this is documented; flagging here.

---

## §2 — Implementation Module Mapping

| Module | Source mirror | Lines | What it teaches |
|---|---|---|---|
| `routing.py` | `fused_topk_router.py:L69`, `grouped_topk_router.py:L81` | 184 | Both routing math paths (Mixtral + DeepSeek), `expert_load_counts` helper |
| `expert_map.py` | `layer.py:L70-L157, L196-L214` | 134 | `determine_expert_map` with linear + round_robin + remainder; `per_rank_token_load` for demos |
| `ep_groups.py` | `parallel_state.py:L1261-L1283, L1670-L1719`, `config.py:L998-L1209` | 244 | `FusedMoEParallelConfig` dataclass, `_EP`/`_EPLB` singletons, mesh math `transpose(1,2).reshape(-1, dp×pcp×tp)` |
| `all2all_baseline.py` | `all2all.py:L40-L139` | 162 | `AgRsAll2AllManager` (allgatherv + reduce_scatterv), α-β cost model |
| `fused_moe_block.py` | `layer.py:L219-L1649`, `prepare_finalize/naive_dp_ep.py:L71-L168` | 252 | Composition: gate → routing → dispatch → local FFN → combine; ep=1 vs ep=N invariance check |
| `eplb.py` | `eplb_state.py:L210-L944` | 184 | Toy `EplbState`: rolling window load, redundant experts, periodic reshuffle (NO aux-loss) |
| `mixtral_vs_deepseek.py` | `models/mixtral.py:L77`, `models/deepseek_v2.py:L244` | 113 | Side-by-side configs, routing fingerprint helper |
| `demo.py` | runs §1-§5 | 280 | 5 demos producing ≥30 verbatim numerics |

Total: ~1550 LOC across 8 modules + `__init__.py` + 1 smoke-test file.

---

## §3 — Demo Numerics (verbatim quotes for the writer)

All numbers below come from `tests/demo-output.txt` (regenerate with
`python implementation/demo.py`). Every number is reproducible at
`seed=42` (top-level) and per-demo seeds inside.

### §3.1 Top-K routing (Mixtral E=8 K=2; DeepSeek E=64 K=6 grouped)

```
Mixtral (E=8, K=2):
  per_expert_count = [250, 285, 277, 243, 253, 272, 247, 221]
  max=285  min=221  mean=256.00  coverage=1.000
  per-token weight sum: min=1.0000  max=1.0000  mean=1.0000

DeepSeek-V2 grouped (E=64, K=6, n_group=8, topk_group=3):
  max=131  min=78  mean=96.00  coverage=1.000
  per-token weight sum: min=1.0000  max=1.0000  mean=1.0000

Renormalize on/off (Mixtral, K=2):
  renormalize=True  → sum range [1.0000, 1.0000]  mean 1.0000
  renormalize=False → sum range [0.2730, 0.6171]  mean 0.3899
```

### §3.2 All-to-all alpha-beta cost model (P=8)

```
NVLink (alpha=5μs, beta=250 GB/s, hidden=4096 bf16):
       128 tokens →  T_AR=  16.09μs   T_A2A=   8.05μs   ratio=2.000
      1024 tokens →  T_AR=  67.47μs   T_A2A=  33.74μs   ratio=2.000
      8192 tokens →  T_AR= 478.51μs   T_A2A= 239.26μs   ratio=2.000
     65536 tokens →  T_AR=3766.85μs   T_A2A=1883.42μs   ratio=2.000

IB (alpha=8μs, beta=50 GB/s):
       128 tokens →  T_AR=    50.70μs T_A2A=    25.35μs ratio=2.000
     65536 tokens →  T_AR= 18804.48μs T_A2A=  9402.24μs ratio=2.000
```

### §3.3 Per-rank load — placement × ep_size, 60% hot 20% experts

```
E=32, K=2, tokens=4096, hot 20% of experts received 4915/8192 routed pairs (0.600)

placement      ep_size                 rank loads                  max/mean
linear           1                     [8192]                       1.000
linear           4                     [5175, 980, 1017, 1020]      2.527
linear           8                     [3329, 1846, 483, 497, 458, 559, 515, 505]   3.251
round_robin      4                     [2350, 2420, 1695, 1727]     1.182
round_robin      8                     [1199, 1195, 1186, 1205, 1151, 1225, 509, 522]   1.196
```

Take-away: under Pareto-skewed routing, linear placement at ep=8 gives
**max/mean=3.25** — one rank does 3.25× the mean rank's work — while
round-robin brings it to **1.20**, a 2.7× improvement.

### §3.4 EP×TP weight memory (E=64 DeepSeek-V2-Lite block, hidden=2048, intermediate=1408)

```
Per-expert params: 3·1408·2048 = 8,650,752
Total params:       E·3·intermediate·hidden = 553,648,128 (1056 MiB at bf16)

  ep  tp    mem/rank (MiB)    reduction
   1   1            1056.00         1.00x
   4   1             264.00         4.00x
   4   2             132.00         8.00x
   8   2              66.00        16.00x
  16   1              66.00        16.00x
   8   4              33.00        32.00x
```

Confirms `mem_per_rank ∝ 1/(ep × tp)`.

### §3.5 EPLB rebalance (100 steps, K=2, E_logical=32, redundant=4, ep=4)

```
step    placement       per-rank load                              max/mean
   0    linear          [1292, 246, 257, 253]                       2.523
  25    linear          [1295, 230, 261, 262]                       2.529
  50    round_robin     [591, 616, 423, 418]                        1.203  ← EPLB triggered
  51    round_robin     [575, 593, 463, 417]                        1.158
  75    round_robin     [594, 629, 391, 434]                        1.229
  99    round_robin     [582, 611, 434, 421]                        1.193

EplbState: num_logical=32, num_redundant=4, num_physical=36
physical_to_logical[0:8]=[0, 1, 2, 3, 4, 5, 6, 7]   (initial layout)
physical_to_logical[-4:]=[5, 2, 0, 4]               (after rearrangement: hot experts duplicated)
```

After step 50 the imbalance ratio drops from 2.53 → 1.20 (a 2.1×
improvement) because EPLB's rebalance redistributed the hot experts.

Numerics count (writer-quotable verbatim values):
- §3.1: 8 expert-count buckets + 4 aggregates + 6 weight-sum extremes = 18
- §3.2: 4 + 4 = 8 (NVLink) + 2 (IB headlines) = 10
- §3.3: 4 ratio rows + sample rank-load lists = ~15 values
- §3.4: 6 (ep,tp) cells × 2 = 12
- §3.5: 6 (step, ratio) tuples + 2 layout snapshots = ~14
- **Total: ≥65 verbatim numerics** (above the v6 floor of ≥20).

### §3.6 Honest demo caveats (writer must quote verbatim)

> Single-process all-to-all timing here is memcopy-bound, not network-
> bound. The α-β model predicts what real NCCL/DeepEP would do; the
> in-process `dispatch`/`combine` calls are pure shape demonstrations.
> Real H100+NVLink ≈ 250 GB/s; cross-node IB ≈ 50 GB/s.

> Skewed routing in §3 and §5 is synthetic (Pareto with hot 20% getting
> 60% of tokens). Real workloads have task-dependent skew that can be
> heavier or lighter; EPLB's response time depends on the
> `expert_rearrangement_step_interval` set by the operator.

> The toy `EplbState._rearrange()` uses a simple "hot experts go first
> into redundant slots" heuristic. Production EPLB lives in
> `vllm/distributed/eplb/policies.py` and uses a more sophisticated
> bin-packing solver.

---

## §4 — Language Traps (≥5 explicit recap candidates for §9.6.4)

Each trap is "easy to write and almost-but-not-quite right". Writer
should explicitly call out **at least 5** in §9.6.4 with the
"claim → 错 → why → source-evidence" structure.

**Trap A — "EP=N gives N× capacity for the same compute."**
Wrong. EP shards the *parameter store* by N (each rank holds E/N experts),
but each token still activates only K experts regardless. Compute per
token = `K × per_expert_FLOPs`, independent of E. EP scales **memory**,
not throughput-per-token. Latency *increases* by the all-to-all cost.
Source: `layer.py:L378+` distributes experts; `fused_topk_router.py:L69`
always picks K independent of E.

**Trap B — "All-to-all is symmetric so cost = all-reduce / 2."**
The α-β model gives exactly that ratio (verified in §3.2 demo: 2.000
across all payload sizes), but in practice all-to-all is
imbalance-sensitive: if rank 0 owns the popular expert and 80% of tokens
go to it, rank 0's outbound queue is huge while others idle. All-reduce
moves the same payload per rank; all-to-all is bursty by token routing.
Source: `all2all.py:L99-L102` — AgRs's dispatch uses `all_gatherv` with
per-rank `sizes` (the `_v` variant exists exactly because per-rank token
counts differ).

**Trap C — "Experts are independent so EP scaling is free."**
Two coupling effects break this:
1. Routing concentrates load on hot experts (load skew → idle ranks).
2. Shared experts (`n_shared_experts > 0` in DeepSeek) replicate per rank
   — they DON'T scale with EP.
Source: `models/deepseek_v2.py:L302-L317` — `shared_experts =
DeepseekV2MLP(...)` is constructed BEFORE and OUTSIDE `FusedMoE`, with
its own TP wiring; it runs on every rank regardless of `ep_size`.

**Trap D — "EPLB is a free runtime bolt-on."**
Three side doors:
- Requires `num_redundant_experts > 0` AND a separate `_EPLB` process group.
- Only works on quant methods that declare `supports_eplb`. Check at
  `layer.py:L548-L557`: `if self.enable_eplb and not
  self.quant_method.supports_eplb: raise NotImplementedError`.
- Round-robin placement with EPLB is forbidden — see
  `layer.py:L168-L171`: `round_robin_supported = ... and
  num_redundant_experts == 0 and not enable_eplb`.

**Trap E — "Aux loss is what makes MoE balanced in vLLM."**
vLLM is **inference-only**. Aux loss is a *training* mechanism (Switch
Transformer L_balance). The trained model's expert distribution reflects
the aux loss applied during training. At inference, vLLM has only EPLB
(runtime statistical rebalance) — no gradient, no aux loss. Verified via
grep over `vllm/model_executor/`: zero hits for `load_balance|aux.*loss|
balance.*loss` outside EPLB-related paths.

**Trap F — "FusedMoE.forward calls dispatch, then experts, then combine — that's the EP path."**
Abstractly true, but the *real* dispatch is selected at init via
`select_prepare_finalize_modular`. For `ep_size=1` and no-DP, the chosen
prepare-finalize is `MoEPrepareAndFinalizeNoDPEP`, where `dispatch` is
identity and `combine` is identity. So "all-to-all happens on every MoE
forward" is wrong; it happens only when `dp_size > 1 and use_ep`
(`config.py:L1019-L1020 use_all2all_kernels`).

**Trap G — "Top-K then softmax = softmax then Top-K."**
They do NOT commute. `softmax_first` then top_k (vLLM's path,
`fused_topk_router.py:L94-L100`): the K weights sum to ≤1 (renormalized
to 1 if `renormalize=True`). `topk_first` then softmax: weights sum to
exactly 1 over a different distribution. Demo §3.1 shows: with
renormalize=True, sums = 1.0; without, sums range
`[0.2730, 0.6171]` — those are the softmax tail probabilities.

---

## §5 — Cross-chapter Links

### Back-pointers (Ch09 reuses):
- **Ch08 (Tensor Parallelism)** — every expert's FFN is internally
  TP-sharded via `MergedColumnParallelLinear` (gate|up) +
  `RowParallelLinear` (down), with one intra-expert all-reduce. EP is
  orthogonal to TP, not a replacement.
- **Ch01 (MLP block)** — each expert is a SwiGLU MLP. Same
  `silu_and_mul` activation, same `[hidden → intermediate → hidden]`
  shape.

### Forward-pointers (Ch09 sets up):
- **Ch11 (DCP/PCP)** — shares the `parallel_state.py` group machinery.
  PCP becomes another axis the EP group is orthogonal to. The mesh math
  generalizes: `ep_size = world / (tp × dp × pcp)`.
- **Ch15+ (Llama variants)** — Llama is dense so it doesn't use
  `_EP`/`FusedMoE` directly, but the GroupCoordinator + 4D mesh
  framework is shared.
- **Part 5 model-specific chapters** — DeepSeek-V3 deep-dive
  (grouped TopK + `noaux_tc`), Mixtral deep-dive, Qwen3-MoE.

### Same-cycle hand-off:
- Tester will pin every demo numeric in §3 with `assertEqual` /
  `assertAlmostEqual`. The §3.3 placement table is the load-bearing one
  for the writer.
- Writer will use the §3.5 EPLB timeline as the §9.4 narrative spine.
- Reviewer will gate on the Trap E callout — if the chapter doesn't
  explicitly say "vLLM is inference-only, aux loss is training-only",
  REVISE.

---

## §6 — Source pin verification

```
$ cd instances/vllm/source && git rev-parse HEAD
98661fe012c5c467252d4df8411d2f46190e9268
```

Matches the brief's pin at `98661fe`. All line numbers in this
document and in `# REFERENCE:` comments throughout the implementation
modules were verified against this commit. If a future re-run hits a
drift, re-grep for the symbol (function/class name) before re-citing —
that's the lesson from Ch07/Ch08.
