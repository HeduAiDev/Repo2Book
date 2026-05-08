# Rehydration Brief — Ch09 Expert Parallelism (Implementer)

- **Chapter**: `09-expert-parallelism`
- **Title**: Expert Parallelism MoE专家并行
- **Outline level**: core (Part 2)
- **Status**: dispatch — first v6-grade pass for Ch09 (cadence baseline holds at N=5 after Ch08 single-cycle APPROVED)
- **Dependencies (per outline)**: `08-tensor-parallelism` (groups + GroupCoordinator + col/row primitives compose into the EP path); also touches Ch01 (MLP / hidden-dim structure), Ch11 (DCP/PCP as parallelism siblings)
- **Dependents downstream**: Part-3 chapters (Llama doesn't use EP, but the 3D mesh from §5 generalizes there), Part-5 model-specific (DeepSeek-V2/V3, Mixtral, Qwen3-MoE — all the canonical EP-shipping production models)
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Brief generated**: 2026-05-07 by archivist
- **Recipient**: implementer (direct dispatch by team-lead, no book-editor relay — operational rule from Ch07/Ch08)

---

## §1 — Chapter scope (5 movements — what Ch09 actually covers)

**Core question**: A Mixture-of-Experts model has `E` expert FFNs but each token only routes to `top_k` of them — why can't we just shard those experts across GPUs the same way TP shards a regular MLP, and what *new* invariants does that introduce? What does vLLM's `_EP` group + `FusedMoE` + an all-to-all dispatch do that TP's `_TP` group + `RowParallelLinear` + all-reduce does NOT?

The chapter covers **5 movements**:

1. **The routing math.** Given hidden state `h ∈ R^d` and gate weights `W_g ∈ R^{E×d}`, router logits are `g = h W_g^T ∈ R^E`. Top-K gating: pick the `top_k` largest entries of `softmax(g)`, normalize so the K weights sum to 1 (when `renormalize=True`), zero out the rest. Each token therefore activates `top_k` of `E` experts (typical: K=2 of 8 for Mixtral; K=8 of 256 for DeepSeek-V3). Critical equation: cost(MoE forward) = top_k/E × cost(dense FFN of equivalent capacity), NOT 1/E. **DeepSeek-style "grouped Top-K"**: experts are pre-clustered into `num_expert_group` groups; first pick `topk_group` groups, then take Top-K within those — reduces all-to-all cost when groups align with GPUs.

2. **Expert parallelism as a sharding axis.** TP shards a single matmul; EP shards a *set of disjoint matmuls* (each expert is its own MLP). With `ep_size=P` and `E` global experts, each rank holds `local_num_experts ≈ E/P` experts (vLLM's `determine_expert_map` distributes the remainder to the first `E mod P` ranks). The forward pass:
   - Each rank computes router logits for ALL its tokens (router weights are *replicated*).
   - Each token's `top_k` expert IDs may belong to *any* rank.
   - **Dispatch (all-to-all #1)**: tokens go to the rank that owns their experts.
   - Local experts run their FFN on the routed tokens.
   - **Combine (all-to-all #2)**: results return to the originating rank, weighted by `topk_weights` and summed.

3. **All-to-all communication.** All-to-all is *not* all-reduce — every rank sends a different chunk to every other rank. For `P` ranks and `S` total tokens dispatched, naive cost is `O(P × S/P × β + α)` per rank (roughly half the bytes of all-reduce, since each rank touches each other once not twice). vLLM ships **multiple backends**: `AgRsAll2AllManager` (allgather-then-reducescatter — the simplest), `DeepEPHTAll2AllManager` and `DeepEPLLAll2AllManager` (DeepSeek's high-throughput / low-latency kernels for cross-node), `NixlEPAll2AllManager` (CPU-staged), `FlashInferNVLinkOneSidedManager`/`TwoSided` (NVLink P2P), `MoriAll2AllManager`. The "manager" picks the kernel; `FusedMoE` calls `get_ep_group().dispatch(...)` and `.combine(...)` and never names the backend.

4. **Expert placement and load balancing.** Two placement strategies in source: `"linear"` (rank `r` owns experts `[r·E/P, (r+1)·E/P)` — contiguous block) and `"round_robin"` (rank `r` owns experts `r, r+P, r+2P, ...` — strided). Linear is cache-friendly when consecutive expert IDs co-fire; round-robin distributes "popular" experts across ranks. **Load skew is the open problem**: if 10% of experts get 50% of tokens, the GPUs holding them stall while others idle. vLLM's response is **EPLB** (Expert-Parallel Load-Balancer): a separate `_EPLB` process group + `EplbState` + redundant experts (`num_redundant_experts`) — at runtime it tracks `expert_load_view`, periodically reshuffles the logical→physical expert map, and broadcasts the new layout. This is a *runtime* rebalancer, not a training-time aux loss. (vLLM is inference-only; aux-loss-style balancing is a *training* technique briefly explained for context.)

5. **EP+TP composition (2D mesh).** Real production models compose EP and TP. `FusedMoEParallelConfig.make()` takes `tp_size_, dp_size_, pcp_size_, sp_size_` and computes the EP group as the *complement* (EP group ranks = `world_size / (tp_size × pcp_size × dp_size)` — they cover the EP axis of the device mesh). Inside each expert, the FFN can ALSO be TP-sharded (vLLM uses `MergedColumnParallelLinear` + `RowParallelLinear` inside `FusedMoE` weights, exactly like a regular MLP). So a token traversing the MoE layer goes through: replicated gate → top-K → all-to-all dispatch (EP axis) → TP-sharded FFN (TP axis, with its own intra-expert all-reduce) → all-to-all combine. The mesh construction is `parallel_state.py:L1670-L1690` — `_EP` ranks are formed by `all_ranks.transpose(1, 2).reshape(-1, dp × pcp × tp).unbind(0)`, i.e. EP is orthogonal to (TP, DP, PCP). Memory analysis: `weight_per_rank ≈ E·intermediate·hidden / ep_size / tp_size`, which is why production runs use *both*.

**OUT of scope** (do NOT re-cover):
- Pipeline parallelism (PP), data parallelism (DP) → reference, not deep-dive. Ch11 handles DCP/PCP.
- Quantized expert kernels (`fused_marlin_moe`, `triton_cutlass_moe`, FP8 paths) → mention as footnote; chapter teaches the bf16 logical surface.
- Training-side MoE concepts (auxiliary load-balance loss as in Switch Transformer, capacity factor as a training hyperparameter, gradient routing) → mention briefly under §4 to ground readers from training literature, but do not re-derive — vLLM is inference-only.
- DeepSeek-V3-specific MTP/Attention details → Part-5 chapters.
- The full DeepEP kernel internals (cross-node IBGDA tricks) → reference the manager class names, do not re-implement.

If implementer is re-deriving Switch-Transformer training math or DeepEP CUDA kernels — STOP. Those belong elsewhere.

---

## §2 — Source surface (verified at commit 98661fe)

### §2.1 — Files and exact line ranges

| File | Lines (verified) | What |
|---|---|---|
| `vllm/distributed/parallel_state.py` | 2132 lines total | EP/EPLB process groups, GroupCoordinator |
| `vllm/distributed/parallel_state.py` | L1261-L1283 | `_EP` and `_EPLB` module-level singletons; `get_ep_group()` and `get_eplb_group()` accessors. **`get_ep_group()` asserts the model is MoE** — for dense models `_EP is None` |
| `vllm/distributed/parallel_state.py` | L1670-L1719 | EP group construction. **Only created if `model_config.is_moe`**. Mesh math: `all_ranks.transpose(1,2).reshape(-1, dp × pcp × tp).unbind(0)` |
| `vllm/distributed/parallel_state.py` | L1700-L1719 | EPLB group creation (separate process group, isolates EPLB from forward-pass collectives to avoid deadlock) |
| `vllm/distributed/parallel_state.py` | L1797-L1801 | `prepare_communication_buffer_for_model` — DeepEP/DeepEPLL allocate fixed buffers per model |
| `vllm/distributed/parallel_state.py` | L1891-L1896 | EP group teardown |
| `vllm/distributed/device_communicators/all2all.py` | 761 lines total | All-to-all kernel registry — 7+ backends |
| `vllm/distributed/device_communicators/all2all.py` | L40-L140 | `AgRsAll2AllManager` — allgather-dispatch + reduce-scatter-combine. THE pedagogical baseline. `dispatch()` does `dist_group.all_gatherv(...)`; `combine()` does `dist_group.reduce_scatterv(...)` |
| `vllm/distributed/device_communicators/all2all.py` | L142-L195 | `DeepEPAll2AllManagerBase` — common base for HT/LL variants |
| `vllm/distributed/device_communicators/all2all.py` | L196-L256 | `DeepEPHTAll2AllManager` — high-throughput cross-node |
| `vllm/distributed/device_communicators/all2all.py` | L257-L325 | `DeepEPLLAll2AllManager` — low-latency cross-node |
| `vllm/distributed/device_communicators/all2all.py` | L327-L440 | `NixlEPAll2AllManager` — CPU-staged path |
| `vllm/distributed/device_communicators/all2all.py` | L442-L670 | FlashInfer NVLink one/two-sided |
| `vllm/distributed/device_communicators/all2all.py` | L671+ | `MoriAll2AllManager` |
| `vllm/model_executor/layers/fused_moe/layer.py` | 1649 lines total | The core MoE module — owns gate, experts, routing, EP/TP plumbing |
| `vllm/model_executor/layers/fused_moe/layer.py` | L70-L157 | **`determine_expert_map(ep_size, ep_rank, global_num_experts, expert_placement_strategy)`** — returns `(local_num_experts, expert_map, expert_mask)`. The `expert_map` tensor of shape `(global_num_experts,)` maps global expert IDs → local index, with `-1` for "not on this rank". Implements `"linear"` (block) and `"round_robin"` (strided) strategies. **THIS is the §3 anchor.** |
| `vllm/model_executor/layers/fused_moe/layer.py` | L219-L290 | `class FusedMoE(PluggableLayer)` declaration + docstring |
| `vllm/model_executor/layers/fused_moe/layer.py` | L290-L450 | `FusedMoE.__init__` — accepts `tp_size, ep_size, dp_size, pcp_size`; calls `FusedMoEParallelConfig.make()`; constructs `expert_map` if `use_ep`; logs the mapping. **This is the EP+TP composition site.** |
| `vllm/model_executor/layers/fused_moe/layer.py` | L548-L605 | Quant method check, EPLB compatibility assertion, weight creation through `quant_method.create_weights(layer=self, ...)` |
| `vllm/model_executor/layers/fused_moe/layer.py` | L660-L685 | `@property ep_size`, `@property use_ep`, `@property ep_rank` — read from `moe_parallel_config` |
| `vllm/model_executor/layers/fused_moe/layer.py` | L1516-L1540 | `register_eplb_state` — connects layer's `eplb_state` to module-level `expert_load_view` |
| `vllm/model_executor/layers/fused_moe/layer.py` | L1543-L1649 | `def forward` — top-level entry; calls quant_method.apply via `quant_method.moe_quant_config` machinery, which routes through `prepare_finalize` to do dispatch → expert FFN → combine |
| `vllm/model_executor/layers/fused_moe/config.py` | 1359 lines total | EP/TP config logic |
| `vllm/model_executor/layers/fused_moe/config.py` | L999-L1240 | `class FusedMoEParallelConfig` — `make(tp_size_, dp_size_, pcp_size_, sp_size_, vllm_parallel_config)`; computes `ep_size`, `tp_size`, `use_all2all_kernels` |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | 306 lines total | Wires `FusedMoE` to a `prepare_finalize` object based on the all2all manager |
| `vllm/model_executor/layers/fused_moe/all2all_utils.py` | L60-L210 | `select_prepare_finalize_modular(...)` — returns a `MoEPrepareAndFinalize{NoDPEP, NaiveDPEP, DeepEP*, ...}` instance based on `moe.moe_parallel_config.use_all2all_kernels` and the device communicator |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | 167 lines total | Standard Top-K gating |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | L69-L116 | `def fused_topk(hidden_states, gating_output, topk, renormalize, scoring_func)` — softmax/sigmoid → top-K → optional renormalize. **THE canonical routing math.** Returns `(topk_weights, topk_ids, token_expert_indices)` |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py` | L116-L167 | `class FusedTopKRouter(BaseRouter)` |
| `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | 353 lines total | DeepSeek-style grouped Top-K |
| `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | L81-L162 | `def grouped_topk(...)` — pick `topk_group` of `num_expert_group` groups first, then top-K inside. Has `e_score_correction_bias` for "noaux_tc" path (DeepSeek-V3) |
| `vllm/model_executor/layers/fused_moe/router/base_router.py` | 298 lines total | `class BaseRouter(FusedMoERouter)` |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py` | 142 lines total | `MoEPrepareAndFinalizeNoDPEP{Modular, Monolithic}` — for ep_size==1 or no-DP cases (dense path) |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | 258 lines total | `MoEPrepareAndFinalizeNaiveDPEP{Modular, Monolithic}` — uses `get_ep_group().dispatch(...)` / `.combine(...)`. **THE pedagogical reference for Ch09**. L125-L168 (Modular.apply) shows the dispatch→experts→combine triangle |
| `vllm/distributed/eplb/eplb_state.py` | 1166 lines total | EPLB state machine |
| `vllm/distributed/eplb/eplb_state.py` | L62-L210 | `EplbStats`, `EplbModelState` |
| `vllm/distributed/eplb/eplb_state.py` | L210-L920 | `class EplbState` — the rebalancer. Tracks per-expert load, decides when to reshuffle |
| `vllm/distributed/eplb/eplb_state.py` | L925-L944 | `class EplbLayerState` — per-`FusedMoE`-layer state; holds `expert_load_view` |
| `vllm/model_executor/models/mixtral.py` | 601 lines total | **Reference site #1**: simple MoE — 8 experts, K=2, plain softmax, no shared expert |
| `vllm/model_executor/models/mixtral.py` | L77-L160 | `class MixtralMoE(nn.Module)` — instantiates `gate=ReplicatedLinear(...)` + `experts=FusedMoE(num_experts=8, top_k=2, use_grouped_topk=False, ...)`. Forward: `router_logits = gate(h); out = experts(h, router_logits)` |
| `vllm/model_executor/models/deepseek_v2.py` | 1729 lines total | **Reference site #2**: production-scale MoE — `n_routed_experts` up to 256, grouped Top-K, shared experts, `noaux_tc` correction bias |
| `vllm/model_executor/models/deepseek_v2.py` | L244-L420 | `class DeepseekV2MoE(nn.Module)` — uses `GateLinear` not `ReplicatedLinear`, has `e_score_correction_bias`, has `shared_experts` (a `DeepseekV2MLP` that runs in parallel with the routed experts), passes `use_grouped_topk=True, num_expert_group=n_group, topk_group=topk_group` |

### §2.2 — Outline-vs-source mismatches to flag

**The outline subsection 4 — "Expert Load Balancing Loss的梯度回传" — is OUT OF SCOPE for vLLM source.** vLLM is an inference engine; there is no gradient backprop in any forward path verified above. There is NO aux-loss code. EPLB is a *runtime statistical rebalancer*, not a training loss. Ch09 must reframe §4 from "load balance loss" to "**runtime load balancing — EPLB and the redundant-expert mechanism**", explicitly noting that the *training-time aux loss* (Switch Transformer paper-style `L_balance`) is what produced the *trained* expert distribution, and EPLB *during inference* is the runtime response to whatever skew the trained model still has. Reader gets the math of training-time aux loss as a one-page sidebar, then we pivot to vLLM's actual code: `EplbState`, redundant experts, periodic reshuffle.

**The outline does NOT mention all-to-all backends explicitly**, but the source surface for Ch09 is dominated by `device_communicators/all2all.py` (7 manager classes). The chapter MUST cover at least `AgRsAll2AllManager` (pedagogical baseline) and *name* the DeepEP/Nixl/FlashInfer ones to anchor production reality. Subsection 2 ("AllToAll通信的延迟建模") naturally absorbs this; expand it.

**There IS a `class FusedMoE`** (good — unlike Ch07 "no radix tree" / Ch08 "no class TensorParallel"). However, FusedMoE is a *composition layer*, not the EP algorithm. The EP algorithm is **distributed across 5 files**: `parallel_state.py` (groups) + `layer.py` (`determine_expert_map` + FusedMoE init) + `config.py` (FusedMoEParallelConfig) + `all2all.py` (manager) + `prepare_finalize/naive_dp_ep.py` (dispatch/combine). Implementer must teach via this 5-file collaboration, not search for an "EP class".

**No `class ExpertParallel`, no `class MoEParallel`.** Confirmed via the verification grep — only `FusedMoE`, `FusedMoEParallelConfig`, the `*All2AllManager` family, and prepare_finalize classes.

**`knowledge/INDEX.md` currently lists `tensor-parallelism` as covering Ch09** with files `linear.py, parallel_state.py, communication_op.py`. This is WRONG for Ch09's primary surface. Knowledge update REQUIRED:
- Add row: `| [expert-parallelism](modules/expert-parallelism.md) | 09 | layer.py (fused_moe), parallel_state.py (_EP), all2all.py |`
- Update existing tensor-parallelism row to drop "09" (Ch09 reuses TP only for intra-expert sharding; the EP-specific surface lives elsewhere). New row: `| [tensor-parallelism](modules/tensor-parallelism.md) | 08, 11, 15+ | linear.py, parallel_state.py, communication_op.py |`

### §2.3 — Verified absence of structures

- No `class ExpertParallel`, `class MoEParallel`, `class ExpertSharding`, `class TopKGate` (the gate is a plain `ReplicatedLinear` or `GateLinear`).
- No training-side `aux_loss` / `load_balance_loss` function in any file under `vllm/model_executor/` (verified via grep `load_balance|aux.*loss|balance.*loss` — only EPLB-related hits).
- No "capacity factor" hyperparameter — the chapter's training-flavored content must be honest that inference time the K experts simply receive whatever tokens choose them; there is no token-dropping in vLLM (DeepEP has bucket sizing but not capacity-factor token drops).

---

## §3 — Outline section walk-through

Outline subsections (from `book-outline.json` → `parts.part2_advanced_common.chapters[09-expert-parallelism].subsections`) and how to map them to source. Subsection text is the *topic* (the question the section answers), not a class-name contract.

| Outline subsection | Reframed scope | Source anchor |
|---|---|---|
| 1. "MoE Router：Top-K gating的softmax归一化与capacity factor" | Open `router/fused_topk_router.py:L69 fused_topk(...)`. Derive: `g = h W_g^T`; `s = softmax(g)`; pick `argmax_top_k(s)`; renormalize `w_i = s_i / sum_{j∈topK} s_j`. Show with K=2 of E=8 (Mixtral) and K=8 of E=256 (DeepSeek). For *grouped* Top-K, open `router/grouped_topk_router.py:L81 grouped_topk(...)` — pre-group experts, pick top groups first. **Capacity factor**: in inference there is NO token dropping; capacity factor is a training-time concept. The chapter must say so explicitly. **5-step rhythm**: open `fused_topk_router.py:L69` → ask "why softmax then top-K not top-K then softmax?" → derive (preserves a probability distribution) → impl `routing.py` reproducing fused_topk in plain PyTorch → diff: vLLM's `fused_topk` dispatches to a Triton kernel (`dispatch_topk_softmax_func`) for batched throughput; ours uses `torch.topk`. | `router/fused_topk_router.py:L69-L167`, `router/grouped_topk_router.py:L81-L167`, `models/mixtral.py:L77-L160`, `models/deepseek_v2.py:L244-L420` |
| 2. "AllToAll通信的延迟建模（ring vs hierarchical）" | Open `device_communicators/all2all.py:L40 AgRsAll2AllManager`. Show that vLLM's "all-to-all" is an interface; the simplest backend (`AgRs`) is literally `all_gatherv` for dispatch + `reduce_scatterv` for combine — not a true symmetric all-to-all. Derive cost: `T_all2all_ring(P, S) = (P-1)/P × (α + S×β)` where S is total payload; compare to all-reduce `T_AR = 2(P-1)/P × (α + S×β)` — half the bytes. Discuss "hierarchical" as the multi-node strategy DeepEP-HT uses (intra-node NVLink + inter-node IB tiers). **5-step**: open `all2all.py:L40` → ask "why is this an allgather + reduce_scatter, not a true alltoall?" → derive (both achieve the same end state for unbalanced token counts; allgatherv handles variable per-rank counts correctly) → impl `all_to_all.py` with the AgRs strategy → diff: production DeepEP uses fused IB+NVLink kernels (`DeepEPHTAll2AllManager:L196`) for ~3-5× throughput, we use plain `dist.all_gather`/`dist.reduce_scatter`. | `device_communicators/all2all.py:L40-L325`, `all2all_utils.py:L60-L210` |
| 3. "Expert Placement——均匀分布 vs 负载感知分布" | Open `fused_moe/layer.py:L70 determine_expert_map(ep_size, ep_rank, global_num_experts, expert_placement_strategy)`. Walk both branches: `"linear"` block placement (`expert_map[start_idx : start_idx + local_num_experts] = arange(local_num_experts)`) vs `"round_robin"` strided (`expert_map[ep_rank::ep_size] = arange(local_num_experts)`). Show with E=8, P=2: linear → rank 0 owns {0,1,2,3}, rank 1 owns {4,5,6,7}; round-robin → rank 0 owns {0,2,4,6}, rank 1 owns {1,3,5,7}. Discuss: linear is locality-friendly (consecutive expert IDs often correlate); round-robin distributes hot experts. **5-step**: open `layer.py:L70` → ask "why both? what does the choice change?" → derive load distribution under correlated routing → impl `placement.py` with both strategies → diff: vLLM also supports remainder distribution (line 113, `remainder = global_num_experts % ep_size; local_num_experts = base + 1 if ep_rank < remainder else base`) — uneven last-rank case; we make E divisible by P for simplicity. | `fused_moe/layer.py:L70-L157` |
| 4. "Expert Load Balancing Loss的梯度回传" → **REFRAME to "Runtime load balancing — EPLB and redundant experts"** | The literal subsection title is OUT OF SCOPE (vLLM is inference). Reframe: open `distributed/eplb/eplb_state.py:L210 class EplbState`. Explain training-time aux loss (Switch Transformer `L_balance = α · sum_i (f_i × P_i)` where `f_i` is fraction of tokens to expert i, `P_i` is mean gate prob) as a one-page sidebar — it's how the *trained* model learned to balance. Then pivot: even with aux loss, real workloads still skew. EPLB's response: **redundant experts** (`num_redundant_experts > 0` lets the same logical expert exist on multiple physical ranks) + periodic reshuffle of the logical→physical map based on `expert_load_view`. EPLB has its OWN `_EPLB` process group (`parallel_state.py:L1700-L1719`) so reshuffle communication doesn't deadlock the forward-pass MoE collectives. **5-step**: open `eplb_state.py:L210` → ask "if training already balanced, why do we need EPLB at inference?" → derive (training distribution ≠ deployed traffic; live load skew remains) → impl `load_balance.py` with toy training-style aux loss + toy EPLB-style runtime reshuffle → diff: vLLM's EPLB has full state machine + IB-aware comm (`eplb_communicator.py`), our toy is a simple "every N steps, recompute" loop. | `distributed/eplb/eplb_state.py:L62-L944`, `parallel_state.py:L1700-L1719`, sidebar from training literature |
| 5. "EP+TP的device mesh构建（2D/3D并行拓扑）" | Open `parallel_state.py:L1670-L1719` initialize_model_parallel — show the EP group construction: `all_ranks.transpose(1,2).reshape(-1, dp × pcp × tp).unbind(0)`. Derive: world is a (DP, PP, PCP, TP) 4D tensor; EP is the *complement* axis — at fixed (DP, PP, PCP, TP) ranks combine into one EP group. Explain: with `world_size=8, tp_size=2`, EP group has `world_size / tp_size = 4` ranks (assuming dp=pp=pcp=1). Walk a token's path: gate (replicated within TP group) → top-K (per-token) → all-to-all over EP axis → expert FFN (TP-sharded internally — uses `MergedColumnParallelLinear`+`RowParallelLinear` internally with intra-expert all-reduce on TP axis) → all-to-all combine over EP axis. Memory: `weight_per_rank ∝ E × intermediate × hidden / (ep_size × tp_size)`. **5-step**: open `parallel_state.py:L1670` → ask "why is EP orthogonal to TP, not a sub-axis?" → derive (experts are FUNCTIONALLY independent; tokens routed to expert i don't need expert j's weights at all — orthogonal sharding) → impl `mesh.py` constructing (EP, TP)=(2,2) on world=4 → diff: vLLM also handles DP+PCP axes; we collapse to 2D for clarity. | `parallel_state.py:L1670-L1719`, `fused_moe/config.py:L999-L1240`, `models/deepseek_v2.py` (uses both ep_size and tp_size simultaneously) |

Use this 5-section mapping as the chapter's §9.1-§9.5 spine. §9.6 is the source-mapping table (main + per-section mini per K15 two-tier). §9.6.4 is the language-trap recap.

---

## §4 — Knowledge dependencies

### Existing knowledge entries to read before work
- `knowledge/modules/tensor-parallelism.md` — T01 `divide()`, T02 `_TP` singleton, T03 `MergedColumnParallelLinear` output_partition_sizes list, T04 GQA × TP head replication, T05 all-reduce algorithm dispatch. **Critical**: Ch09 reuses TP machinery *inside each expert's FFN*, so all 5 are load-bearing.
- `knowledge/modules/attention.md` — only tangentially relevant; Ch09 doesn't touch attention.

### NEW knowledge module REQUIRED
**Create `knowledge/modules/expert-parallelism.md`** — Ch09 owns its own module:
- Use **E-prefix IDs** (E01, E02, ...) — distinct from Ch07's K-prefix, Ch06's P-prefix, Ch08's T-prefix. **MUST avoid collision** per `feedback_double_prefix_headings.md` user feedback.
- Forward-shared with: future MoE-architecture chapters (DeepSeek deep-dive in Part 5), Ch11 (DCP/PCP shares the parallel_state.py group machinery), any chapter touching `_EP` group.
- **WARNING (carried from Ch07/Ch08 lessons)**: `learn.py` append-mode bugs were fixed (P1-1 task #36 completed), but if doubled `## E0X: E0X:` headers show up after extraction, fix immediately.
- Update `knowledge/INDEX.md`:
  - Add row: `| [expert-parallelism](modules/expert-parallelism.md) | 09 | fused_moe/layer.py, parallel_state.py (_EP), all2all.py |`
  - Update tensor-parallelism row chapters from `08, 09, 11, 15+` to `08, 11, 15+` (Ch09 has its own primary module now; reference only).

### Anticipated facts the implementer will discover (E-prefix candidates)
- E01: `_EP` group is created **only if `model_config.is_moe`** (`parallel_state.py:L1672`); `get_ep_group()` raises a helpful AssertionError on dense models. This is asymmetric vs `_TP` which is always created.
- E02: `EPLB` uses a **separate process group** (`_EPLB`, `parallel_state.py:L1700-L1719`) intentionally to isolate rebalance comm from MoE forward-pass collectives — comment in source explicitly says "to prevent deadlocks". Cross-ref to wisdom `architecture.md` (gates and isolation patterns).
- E03: `determine_expert_map` returns `(local_num_experts, expert_map_tensor, expert_mask_tensor)` where `expert_map[i] = local_idx_or_-1`. This `-1` sentinel is the standard "not-on-this-rank" marker. (`layer.py:L116` → `torch.full((global_num_experts,), -1, dtype=torch.int32)`.)
- E04: vLLM's "all-to-all" interface (`get_ep_group().dispatch/.combine`) is backend-pluggable. The simplest backend `AgRsAll2AllManager` is **NOT a true `dist.all_to_all`** — it's `all_gatherv` (dispatch) + `reduce_scatterv` (combine). Same end state, simpler dependency surface.
- E05: `FusedMoE` does NOT take a `routing_method` argument; it takes `use_grouped_topk: bool` + `custom_routing_function: Callable`. The router is selected internally via `RouterFactory` based on these flags. (`layer.py` __init__ around L260, router selection around L470.)
- E06: Mixtral and DeepSeek use **different gate types** — Mixtral uses `ReplicatedLinear` (every rank holds a copy of the gate weight), DeepSeek uses a custom `GateLinear` (`router/gate_linear.py`) with optional `e_score_correction_bias` for noaux_tc. Reader who only knows Mixtral will be surprised by DeepSeek's gate.
- E07: **Shared experts pattern** (DeepSeek's "always-on expert" that processes every token). Fused into `FusedMoE` via `n_shared_experts`/`shared_experts=` constructor arg (`layer.py:L290+`, `models/deepseek_v2.py:L302-L316`). For Mixtral, `shared_experts=None`. This is a model-architecture choice, not a parallelism choice — but it interacts with EP because the shared expert runs on every rank (no all-to-all for it).
- E08: `FusedMoEParallelConfig.make` (`config.py:L999-L1212`) computes `ep_size = world / (tp × pcp × dp)` — **EP is the COMPLEMENT axis**, not a free hyperparameter. Operators choose tp/pcp/dp; ep is determined.

---

## §5 — Wisdom hits (role priorities: implementer = debugging > architecture > testing > writing)

Read these before opening source:

- `wisdom/debugging.md` — `F.linear` weight shape `[out, in]`; for MoE the per-expert weight has shape `[E, out, in]` (or `[E, intermediate, hidden]` for w13_weight) — easy to forget the leading expert axis. Sharding with `expert_map` indexes the `E` dim, not the `out` dim. Cross-check this when the implementer writes the toy expert FFN.
- `wisdom/architecture.md` — **backpressure gates and lateral comm patterns**. EPLB's separate process group (E02 above) is a textbook isolation gate; the chapter should call this out as architecture wisdom. Also: "**fix prompts not chapters**" — if reviewer gates on outline-vs-source mismatch (esp. §4 reframe), implementer must NOT silently rewrite the outline subsection name; flag explicitly in impl-notes "outline mismatch handling" subsection.
- `wisdom/testing.md` — preemption test design generalizes to "test the boundary of the parallelism axis": `ep_size=1` (no-op path), `ep_size=E` (one expert per rank — degenerate), `ep_size > E` (impossible, must assert), uneven case where `E % ep_size != 0`. Tester will need all four.
- `wisdom/writing.md` — formula rules (NON-NEGOTIABLE: `\mathrm{}` not `\text{}`, no `\boxed`, no `\frac` inline). **Ch09 will be FORMULA-HEAVY** (softmax + top-K + renormalize, alpha-beta cost models, aux loss, EPLB rebalance criterion). Plan for high formula-density mitigation per the writer-side K-series patterns: ≤2 inline formulas per bullet, render mid-proof variable references as plain text when the symbol isn't doing math work.

Plus the reproducible-cadence patterns from `state.json:v6_compliance.patterns_promoted_to_baseline` (after Ch08):
- `two_tier_mapping` — mandatory; Ch09 surface is BROAD (10+ files). Aim for 30 main + 25 mini.
- `language_trap_callouts` — Ch09 has plenty (see §6); plan ≥5 explicit recap items.
- `honest_demo_caveats` — synthetic single-process EP simulation does NOT give true all-to-all timings (intra-process is just memcopy); flag this loudly. K17 lesson: writer must quote the caveat verbatim from impl-notes.
- `single_cycle_approval` — Ch08 hit this. Ch09 must replicate.

---

## §6 — Candidate language traps for the writer (target 5-7)

Each candidate is a phrasing that is "easy to write and almost-but-not-quite right". Writer picks the strongest 5-6 for explicit callouts at the relevant section + a dedicated recap section, mirroring Ch07 §7.6.4 and Ch08 §8.6.4.

**Trap A — "EP=N gives N× capacity for the same compute."** No. EP shards the *parameter store* by N (each rank holds E/N experts), but each token only activates K experts regardless. The compute *per token* is `top_k × per_expert_FLOPs`, which doesn't depend on E. EP scales **memory**, not throughput-per-token. Throughput goes UP only because you can fit a bigger model in aggregate HBM; latency *increases* by the all-to-all cost. Source evidence: `fused_moe/layer.py:L378+` distributes experts; routing in `fused_topk_router.py:L69` selects K independent of E.

**Trap B — "All-to-all is symmetric so cost = all-reduce / 2."** Algorithmically yes (each rank touches each other rank once not twice), but in practice all-to-all is *imbalance-sensitive*: if rank 0 owns the popular expert and 80% of tokens go to it, rank 0's outbound queue is huge while others idle. All-reduce always moves the same payload per rank; all-to-all is bursty by token routing. Source evidence: AgRs's `dispatch` uses `all_gatherv` with per-rank `sizes` (`all2all.py:L78` `sizes = dp_metadata.get_chunk_sizes_across_dp_rank()`) — the `_v` variant exists exactly because per-rank token counts differ.

**Trap C — "Experts are independent so EP scaling is free."** No, two coupling effects break this:
1. Routing concentrates load on hot experts (load skew → idle ranks).
2. Shared experts (`n_shared_experts > 0` in DeepSeek) replicate per rank — they DON'T scale with EP. Memory savings are only on `routed` experts.
   Source evidence: `models/deepseek_v2.py:L295-L317` shows `shared_experts = DeepseekV2MLP(...)` constructed *before* and outside `FusedMoE`, with its own TP wiring.

**Trap D — "EPLB is just a runtime load balancer."** It is, but the side door matters: EPLB requires `num_redundant_experts > 0` AND has a separate `_EPLB` process group AND can ONLY work on quant methods that declare `supports_eplb`. The check is at `fused_moe/layer.py:L548-L557` — `if self.enable_eplb and not self.quant_method.supports_eplb: raise NotImplementedError`. Writer should not present EPLB as a free bolt-on; there's a quant-method compatibility surface.

**Trap E — "Aux loss is what makes MoE balanced in vLLM."** vLLM is **inference-only**. Aux loss is a *training* mechanism. The trained model's *expert distribution* reflects the aux loss that was applied during training. At inference time, vLLM has only EPLB (runtime statistical rebalance) — no gradient, no aux loss. The chapter must be honest about this when reframing outline §4.

**Trap F — "FusedMoE.forward calls dispatch, then experts, then combine — that's the EP path."** True at the abstraction layer, but the *real* dispatch is selected at init time via `prepare_finalize_modular` (`all2all_utils.py:L60-L210`). For `ep_size=1` and no-DP, the chosen prepare-finalize is `MoEPrepareAndFinalizeNoDPEP`, which does NO all-to-all — `dispatch` is identity, `combine` is identity. So "all-to-all happens on every MoE forward" is wrong; it happens only when ep_size > 1 (and there's more than one DP replica for some backends).

**Trap G — "Top-K then softmax = softmax then Top-K."** They do NOT commute. `softmax_first` then top_k (vLLM's path, `fused_topk_router.py:L93-L96`): the K weights sum to ≤1 (renormalized to 1 if `renormalize=True`). `topk_first` then softmax: weights sum to exactly 1. Different output distributions. vLLM uses softmax-first for compatibility with the training procedures of supported models. Source evidence: `fused_topk_router.py:L93-L100` — softmax dispatched, then `topk_func` selects.

Pick 5-6 of A/B/C/D/E for primary callouts; F/G are secondary. Recap section §9.6.4 should explicitly enumerate them with "claim → 错 → why → source-evidence" per Ch07/Ch08 template.

---

## §7 — Demo plan suggestions (numerics for verbatim narrative use)

The implementer's `demo.py` should produce numbers the writer will quote verbatim (per the demo-numerics-verbatim hard gate, K17 / N=5 baseline).

**Demo §1 — Top-K routing distributions (Mixtral and DeepSeek scales).** Build a fake gate: `W_g ∈ R^{E×d}` random init. Generate `n_tokens=1024` random hidden states. Run `fused_topk` reproduction. Pin numerics:
- For E=8, K=2: per-expert token counts (mean ~256, but show actual skew σ).
- For E=256, K=8 grouped: pick `num_expert_group=8, topk_group=4`. Pin: per-group counts, in-group skew.
- Renormalize on/off comparison: the `renormalize=True` path makes `sum(weights)=1`; without it, sum is in `[K/E, 1]`. Pin one example.
**Numbers**: 6+ pinned values across the two scales.

**Demo §2 — All-to-all alpha-beta microbench.** Single-process simulation (no real GPUs needed; use `torch.distributed` over `gloo` on localhost with `world_size=4`, or fully simulate). For payloads `S ∈ {128, 1024, 8192, 65536} tokens × hidden=4096 × bf16`, time `dispatch + combine` via the AgRs pattern (allgatherv + reduce_scatterv). Fit `T_dispatch = α + β × S_per_rank`. Pin α (μs) and β (GB/s). Compare to all-reduce of same total payload — show all-to-all is roughly half the bytes. **Numbers**: α/β fits, predicted vs measured at 4 sizes, all-reduce/all-to-all ratio (target ≈2).

**Demo §3 — Per-expert load distribution under skewed routing.** Build a synthetic distribution where 20% of experts receive 60% of tokens (Pareto-like). With E=32, K=2, `ep_size ∈ {1, 4, 8}`, `placement_strategy ∈ {"linear", "round_robin"}`, measure per-rank token count. Show:
- linear placement: rank holding hot block stalls (max/mean ratio ~3×).
- round_robin: hot experts are spread; max/mean ratio ~1.4×.
**Numbers**: max/mean/min per rank under each (placement × ep_size) cell. ≥6 cells × 3 stats = 18 numbers.

**Demo §4 — EP+TP composition (2D mesh memory).** Build a toy MoE block with `E=64, hidden=4096, intermediate=14336` (DeepSeek-V2-style). Compute weights memory under:
- (ep=1, tp=1): all weights per rank.
- (ep=4, tp=1): E/4 experts per rank.
- (ep=4, tp=2): E/4 experts per rank, each expert TP-sharded.
- (ep=8, tp=4): E/8 experts per rank, each TP-sharded by 4.
**Numbers**: weights memory per rank in MiB for each cell — confirms `mem ∝ 1/(ep × tp)`. 4-8 cells.

**Demo §5 — EPLB rebalance toy.** Simulate 100 forward passes with skewed routing (the §3 distribution). At step 0, "linear" placement means rank-0 is hot. After step 50, swap: redistribute experts so the formerly-hot ones are spread. Measure max/mean rank load at steps {0, 25, 50, 51, 75, 100} — show the rebalance kicks in at step 50. **Numbers**: 6 timestamps × 3 stats = 18 numbers showing the load curve.

These 5 demos collectively give the writer **≥20 ground-truth numbers** to quote verbatim. Test report should pin every one with `assertEqual` / `assertLess` / `assertAlmostEqual` and explicit values — Ch07 K17 lesson: writer pre-runs linters AND tester pins exact numbers → APPROVED in one cycle. **Honest demo caveats** the impl-notes must state (then writer quotes verbatim):
- Single-process all-to-all timing is memcopy-bound, not network-bound; β estimates are upper bounds for shared-memory and lower bounds for cross-node IB. Real H100+NVLink ≈ 250 GB/s; cross-node IB ≈ 50 GB/s.
- Skewed routing is synthetic; real workloads have task-dependent skew.

---

## §8 — Floor reminders (v6 hard gates, confirmed at N=5 after Ch08)

Implementer commit must satisfy:

- **≥5 source files in impl-notes "Source Analysis" section.** Ch09 natural surface is **8+ files**: `parallel_state.py`, `fused_moe/layer.py`, `fused_moe/config.py`, `fused_moe/all2all_utils.py`, `device_communicators/all2all.py`, `fused_moe/router/fused_topk_router.py`, `fused_moe/router/grouped_topk_router.py`, `fused_moe/prepare_finalize/naive_dp_ep.py`, `eplb/eplb_state.py`, `models/mixtral.py`, `models/deepseek_v2.py`. **Aim for 8-10**; the breadth IS the lesson (per Ch07/Ch08 cadence).
- **≥60 `# REFERENCE: <path>:Lxxx` comments across impl modules.** Match the floor. Ch04: 65, Ch05: 61, Ch06: 60, Ch07: ~60, Ch08: ~60. Target ≥65 for Ch09 given the broader surface.
- **≥10 mapping rows; aim for 30 main + 25 mini per K15 two-tier.** Ch07: 27+45. Ch08: ~30+~25. Ch09 should adopt two-tier from §9.1 through §9.6: main mapping at §9.6 + mini-tables in §9.1 (router variants — Mixtral vs DeepSeek), §9.2 (all-to-all backends — 7 managers), §9.3 (placement strategies — 2 strategies), §9.5 (mesh axis math).
- **Demo numerics verbatim** in tests/test-report.md, then narrative quotes them character-for-character.
- **Both linters PASS at the BLOCKING bar** before handoff (writer + reviewer re-run; mismatches trigger preemptive REVISE per K17).
- **5-step rhythm in every major section §9.1-§9.5.**
- **≥4 language traps with explicit "claim → 错 → why" per §6 above**, plus dedicated recap §9.6.4.
- **Forward/back-pointers wired**: back to Ch08 (TP groups + col/row primitives reused inside experts), Ch01 (MLP structure). Forward to Ch11 (DCP/PCP shares the parallel_state.py group machinery), Part-5 chapters (DeepSeek-V3 deep-dive, Mixtral deep-dive).
- **Source pin verification**: implementer's first command should be `cd instances/vllm/source && git rev-parse HEAD` — must equal `98661fe`. Any line numbers that drift between brief and source → re-grep before citing.
- **Outline §4 reframe documented in impl-notes**: explicit "outline-vs-source mismatch handling" subsection that names the issue (training-time aux loss is out of scope) and the resolution (sidebar + pivot to EPLB). Reviewer will check this — Ch07 lesson.

### What APPROVED at cycle 1 looks like (K17 / N=5 baseline)

Writer's handoff message must contain BOTH linter outputs verbatim:

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/09-expert-parallelism/narrative/chapter.md
🟢 No blocking issues
[+ any non-blocking warnings]

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/09-expert-parallelism/
✓ All grounding checks passed!
```

Reviewer's first action: re-run both linters, diff against writer's claim. Match → fast review. Mismatch → preemptive REVISE without further reading.

---

## §9 — Cadence carry-forward from Ch08 (specific reminders)

Ch08 just published v6 in 1 cycle on 7+ source files, ~30 mapping rows, ~60 REFERENCE comments. Patterns that worked and Ch09 must repeat:

- **Outline subsection name = topic question, not class contract.** §4 outline says "Expert Load Balancing Loss的梯度回传" — chapter must answer "how does vLLM keep experts balanced at inference time?", not search for an aux-loss training implementation that doesn't exist in inference code. Mirror Ch07 "no radix tree" / Ch08 "no class TensorParallel" reframe handling.
- **Source-vs-outline reframe must be IN-CHAPTER, not buried.** Ch07's §7.2 explicitly opens with "vLLM doesn't have a radix tree". Ch08's §8 opens with "no class TensorParallel". Ch09 §9.4 should open with "vLLM is inference-only — there is no aux-loss code in this codebase. Here's what production deployments use instead: EPLB."
- **Tester's 5 framing tips from Ch07/Ch08** were instrumental — implementer-tester pair work uncovered framing issues before writer touched the chapter. Ch09 should plan for this (e.g., tester might point out that AgRs is not a "true" all-to-all; absorb that framing).
- **K17 honest demo caveat**: the synthetic single-process all-to-all timing is NOT representative of real network all-to-all. impl-notes must say so; writer must quote it verbatim; reviewer will check.
- **(N-1)×K formula leads numerics (K13 lesson).** For Ch09, the analogue is **`mem_per_rank ∝ 1/(ep × tp)`** — lead §9.5 with this formula, plug demo numbers AFTER. Same for §9.2: `T_alltoall ≈ T_allreduce / 2` lead, demo evidence AFTER.
- **Chain-break = THE invariant for Ch07. For Ch08 it was "col→row pair = ONE all-reduce".** For Ch09 the load-bearing invariant is **"`_EP` group is orthogonal to `_TP` group; routing → all-to-all → expert FFN(TP-sharded) → all-to-all → DONE"**. Thread this through hook + every section + closing.

---

## §10 — Operational notes for direct dispatch

- **No book-editor relay**: per current operational rule, team-lead direct-dispatches via SendMessage. Implementer should NOT wait for book-editor.
- **Implementer should ack receipt** with: source pin verified (98661fe ✓), files identified (list 5+ from §2.1), language-trap awareness (≥1 paraphrased from §6), expected impl modules (≥5), outline-§4-reframe acknowledged.
- **Writer's brief will be derived from this** by archivist post-tester (Ch07/Ch08 lesson: writer brief amends implementer brief with tester's framing tips and impl-notes specifics).
- **Estimated complexity**: comparable to Ch08, possibly slightly larger (~950 lines, ~4800 words) due to (a) breadth — 8+ source files, (b) the training-vs-inference reframe in §9.4 needs careful explanation, (c) two distinct router math derivations (`fused_topk` and `grouped_topk`).
- **Knowledge module is NEW** (`knowledge/modules/expert-parallelism.md` with E-prefix). Implementer should write this AFTER WORK via `learn.py extract 09-expert-parallelism implementer`. If `learn.py` interactive mode fails, write the module directly with E-prefix headers (no double-prefix).
- **EPLB is a sidebar, not a separate chapter.** Resist the urge to do a full EPLB deep-dive — `EplbState` is 1166 lines and could be its own chapter. Treat as: name the class, show the public interface, walk one happy-path scenario, defer internals.
