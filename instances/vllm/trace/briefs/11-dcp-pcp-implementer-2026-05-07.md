# Rehydration Brief — Ch11 DCP/PCP (Decode/Prefill Context Parallelism) — Implementer

- **Chapter**: `11-dcp-pcp`
- **Title**: DCP/PCP：解码/预填充上下文并行
- **Outline level**: advanced (Part 2)
- **Status**: dispatch — first v6-grade pass for Ch11 (cadence baseline holds at N=7 after Ch10 single-cycle APPROVED with broadest source surface yet — 11 vLLM modules + 5 proposer-family classes; 4th "no class X" reframe graduated to chapter motif; 2nd training→inference reframe stable)
- **Dependencies (per outline)**: `08-tensor-parallelism` (axes orthogonality, group construction in `parallel_state.py`), `04-continuous-batching` (decode-vs-prefill phase distinction)
- **Dependents downstream**: Ch15+ model-zoo chapters (every long-context model deployed at 128K+ uses CP); Ch18 (Triton attention) — DCP all-to-all kernel is a Triton kernel; Ch22 (PD architecture) — CP composes with PD disaggregation; Ch25 (PD ratio) — DCP world-size becomes a budget variable; Ch27 (DeepSeek-V3.2) — MLA + DCP is the production stack
- **Source pin**: vLLM commit `98661fe012c5c467252d4df8411d2f46190e9268` at `instances/vllm/source/` (verified by archivist 2026-05-07)
- **Brief generated**: 2026-05-07 by archivist
- **Recipient**: implementer (direct dispatch by team-lead, no book-editor relay — operational rule from Ch07/Ch08/Ch09/Ch10)

---

## §1 — Chapter scope (5 movements — what Ch11 actually covers)

**Core question**: Tensor parallelism shards the *model* (heads, expert weights). Pipeline parallelism shards the *stages* (early layers vs late layers). Data parallelism shards the *batch* (independent requests in parallel). All three are great until your *single sequence* exceeds 128K tokens — then no per-rank GPU has enough HBM for the K+V tensors of one request, and TP can't help because every TP rank still stores KV for the WHOLE sequence. **Context Parallelism (CP)** shards the *sequence dimension* of K and V across GPUs, so each rank only stores `seq_len / cp_size` tokens of KV. vLLM splits CP into two semantically distinct axes: **DCP** (Decode Context Parallel — shards KV cache across decoding ranks; reuses TP group's GPUs and splits each TP group into `tp_size/dcp_size` sub-groups) and **PCP** (Prefill Context Parallel — shards the prefill input sequence across ranks; orthogonal axis in the device mesh, multiplies world_size). What does the math look like (Ring Attention's K,V circulation; Striped Attention's load-balanced sharding), what communication patterns does vLLM actually emit (AG+RS vs all-to-all backend choice), and how do DCP+PCP+TP compose in the 5-axis device mesh?

The chapter covers **5 movements**:

1. **Why CP at all — long-context capacity wall + KV memory math.** Open `vllm/v1/kv_cache_interface.py:L195-L205 max_memory_usage_bytes` — the actual HBM accounting code. KV cache size scales as `seq_len × num_layers × num_kv_heads × head_dim × 2 (K and V) × dtype_bytes`. For a 128K-token request on a 70B-params model with 80 layers × 8 KV heads × 128 head_dim × bf16, this is `128K × 80 × 8 × 128 × 2 × 2 ≈ 33.5 GB` — **per request**. TP=8 doesn't help because every TP rank stores KV for ALL 128K tokens (TP shards heads, not sequence). PCP=4 shards the prefill sequence among 4 ranks → each rank only owns 32K of prefill tokens. DCP=4 shards the decode KV across 4 ranks → each rank only owns 32K stored KV. The chapter must derive: `max_model_len_per_rank = max_model_len / (dcp_size × pcp_size)` (`kv_cache_interface.py:L199-L203`). Why split into TWO axes (DCP and PCP)? Because prefill and decode have different access patterns — prefill computes attention over the WHOLE prompt sequence at once (every token attends to every other token; needs cross-rank K,V exchange via Ring Attention or all-to-all); decode reads ONE query against the WHOLE KV cache (needs cross-rank K,V gather via AllGather or all-to-all). Different communication topologies for different phases → different axes.

2. **Ring Attention / Striped Attention — the theoretical communication pattern.** The CP attention algorithm is well-known in the literature (Ring Attention: Liu et al. 2023, https://arxiv.org/abs/2310.01889; Striped Attention: load-balancing variant). At PCP=4 with sequence sharded as `[Q0, Q1, Q2, Q3]` and `[K0, K1, K2, K3]` and `[V0, V1, V2, V3]` (each chunk on rank 0, 1, 2, 3), Ring Attention runs **4 rounds** of P2P send/receive: each rank holds its local Q, computes partial attention against (K_i, V_i) it has, then RECEIVES (K_{i+1}, V_{i+1}) from the next rank, accumulates online softmax (LSE-based combine like FlashAttention §2.3), then forwards to the next rank, until each rank has seen all KV chunks. Total comm: `cp_size` rounds × `seq_len/cp_size × num_kv_heads × head_dim × dtype` bytes per round. **Striped Attention** load-balances by chunking each rank's tokens not as a contiguous range but as a striped pattern (token i → rank `i % cp_size`), which equalises causal-masked work per rank (under causal masking, late-token Q has more KV to attend to → rank-imbalance under contiguous partitioning). Walk the math: with contiguous partitioning, rank with the latest Q does `O(L × L)` work; rank with the earliest Q does `O(L × L/cp_size)` work — `cp_size×` imbalance. With striped partitioning: every rank has `~50%` the average causal load. **5-step rhythm**: open `vllm/v1/attention/ops/dcp_alltoall.py:L1-L40` (the all-to-all backend, more vLLM-native than Ring) → ask "why doesn't vLLM ship Ring Attention?" → derive the LSE-weighted combine math → impl `cp_attention.py` reproducing both Ring (P2P) and a2a (LSE-combine) variants in plain PyTorch → diff: vLLM's a2a uses Triton kernels for the combine; ours uses pure PyTorch.

3. **DCP — Decode Context Parallel; the all-gather-or-all-to-all backend choice.** Open `vllm/distributed/parallel_state.py:L1234-L1260` — the `_DCP` global GroupCoordinator + `get_dcp_group()`. Open `parallel_state.py:L1593-L1614` — `initialize_model_parallel` builds DCP group by reshaping `all_ranks` along the dcp axis: `group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)`. **DCP world size DOES NOT increase total ranks** — it splits each TP group into `tp_size//dcp_size` sub-groups. Walk the constraint at `vllm/config/parallel.py:L474-L478`: `tp_size % dcp_size == 0` is a hard assertion (the comment at L466-L472 says "dcp_size must not exceed tp_size, because the world size does not change by DCP, it simply reuses the GPUs of TP group, and split one TP group into tp_size//dcp_size DCP groups"). At decode time, attention computes Q (already replicated within TP group from `column-parallel`) against K,V (now sharded across DCP). Each rank has `seq_len/dcp_size` tokens of KV. Two communication backends: **AG+RS (default `dcp_comm_backend="ag_rs"`)** — AllGather Q to full size, compute attention against local K,V partition, ReduceScatter the output back; 3 NCCL ops per layer. **A2A (`dcp_comm_backend="a2a"`)** — exchange partial output + LSE values via all-to-all, combine with Triton kernel; **2 NCCL ops per layer (33% reduction)**. Reference: arxiv.org/abs/2507.07120 (cited in `dcp_alltoall.py:L17`). The two backends are mathematically equivalent (LSE-weighted combine is associative); a2a is the production choice for MLA. Open `vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355` — the a2a path is wired into MLA backend with `cp_world_size`, `cp_rank`, `cp_tot_seqused_k` parameters.

4. **PCP — Prefill Context Parallel; orthogonal axis with full-mesh sharding.** Open `vllm/distributed/parallel_state.py:L1285-L1290` — `_PCP` global + `get_pcp_group()`. Open `parallel_state.py:L1616-L1633` — `_PCP` group built differently from DCP: `group_ranks = all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size).unbind(0)` — **PCP IS an independent axis in the world_size product** (`tp × dp × pp × pcp` in the 4-axis layout; with DCP folded inside TP group, the canonical mesh is 5-axis: `[external_dp, dp, pp, pcp, tp]`). At prefill time, the input sequence is sharded across PCP ranks: `seq_chunk = seq[rank * S/pcp_size : (rank+1) * S/pcp_size]`. Each rank computes its local Q,K,V, then runs CP attention to produce per-rank output partitions. Open `vllm/v1/attention/backend.py:L703` — `supports_pcp: bool = False` is the per-backend flag; only some attention impls support PCP. Open `vllm/v1/attention/backend.py:L725-L753` — `pcp_world_size` + `pcp_rank` + `total_cp_world_size = pcp × dcp` + `total_cp_rank = pcp_rank × dcp_world_size + dcp_rank` is a **2D CP composition** wired into every attention backend. Open `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` — MoE under PCP requires `all_gather`/`reduce_scatter` on `hidden_states` and `router_logits` across the PCP group (search the `if self.moe_config.pcp_size > 1` branch). PCP composes with EP: each PCP rank may host different experts, so prefill MoE has a `pcp × ep` 2D communication pattern.

5. **CP+TP+EP+PCP+DCP 5D mesh — device mesh layout and the cp_kv_cache_interleave_size knob.** Open `vllm/distributed/parallel_state.py:L1573-L1582` — the `all_ranks` reshape: `(-1, data_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)`. The 5D mesh ordering is `external_dp × dp × pp × pcp × tp`. DCP is folded INSIDE the TP axis (sub-group of size `dcp_size` within each TP group of size `tp_size`). Forward-pointer: Ch15+ Llama at `tp=8, pp=2, pcp=2, dp=1` — total ranks 32, plus internally each TP-group of 8 GPUs split into `8/dcp_size` DCP sub-groups. Open `vllm/v1/attention/backends/utils.py:L820-L857` — `get_dcp_local_seq_lens` is the helper that computes per-DCP-rank local KV lengths under the `cp_kv_cache_interleave_size` knob (config field at `parallel.py:L330-L342` says `cp_kv_cache_interleave_size` controls the per-rank KV shard granularity: "store interleave_size tokens on total_cp_rank i, then store next interleave_size tokens on total_cp_rank i+1" — striped sharding, not contiguous). Block_size must be ≥ and divisible by interleave_size. Note that `dcp_kv_cache_interleave_size` (deprecated alias at `parallel.py:L315-L324`) was replaced by `cp_kv_cache_interleave_size` — implementer must surface this rename. Walk the system impact: at PCP=2 × DCP=2, `total_cp_world_size=4`, max_model_len_per_rank = `max_model_len/4`; HBM per rank scales as `1/4` of single-rank; comm overhead scales as `O(cp_size)` in Ring Attention or `O(log cp_size)` in tree-reduce variant. Open `vllm/v1/executor/multiproc_executor.py:L985-L997` — process names are tagged `_PCP{rank}` to make per-process CP rank visible. Open `vllm/v1/executor/multiproc_executor.py:L116-L121` — world_size assertion: `world_size == tp_size × pp_size × pcp_size`. (NOT × dcp_size — DCP is folded.) **5-step rhythm**: open `multiproc_executor.py:L985` (the rank-naming) → ask "where does the 5D mesh actually get realized?" → derive (above + `parallel_state.py:L1573-L1582`) → impl `cp_mesh.py` reproducing the 5D reshape + group-builder logic → diff: real production mesh wires NCCL groups; ours uses tuple of (dim_name, rank) pairs.

**OUT of scope** (do NOT re-cover):
- TP group construction / linear sharding internals → handled in Ch08 (TP). Reference, do NOT redrive `column-parallel`/`row-parallel` math.
- EP routing kernels (top-k, fused_moe) → handled in Ch09 (EP). Reference for the PCP-EP composition only.
- KV cache block manager internals (`block_pool.py`, `kv_cache_manager.py`) → those are Ch05's territory. Use `kv_cache_interface.py:L195-L205` `max_memory_usage_bytes` as the entry point and stop there.
- Pipeline parallel scheduling / micro-batching → not Ch11. Mention as the orthogonal axis in §11.5.
- Continuous batching scheduler decisions → Ch04. Reference only the prefill-vs-decode phase distinction.
- MLA (Multi-head Latent Attention) deep-dive → Ch27 (DeepSeek-V3.2). Mention that DCP+MLA is the most production-deployed CP combo.
- The full FlashAttention algorithm (online softmax, tile schedule) → Ch03. Reference the LSE-weighted combine; the algorithm itself is Ch03.

If implementer is re-deriving Megatron column-parallel `[h, h/tp]` shapes or re-implementing FlashAttention's online softmax — STOP. Those belong elsewhere.

---

## §2 — Source surface (verified at commit 98661fe)

### §2.1 — Files and exact line ranges

| File | Lines (verified) | What |
|---|---|---|
| `vllm/distributed/parallel_state.py` | L1234-L1243 | `_DCP: GroupCoordinator | None` + `get_dcp_group()` + backward-compat alias `get_context_model_parallel_group = get_dcp_group` |
| `vllm/distributed/parallel_state.py` | L1285-L1290 | `_PCP: GroupCoordinator | None` + `get_pcp_group()` |
| `vllm/distributed/parallel_state.py` | L1497-L1498 | `initialize_model_parallel` signature: `prefill_context_model_parallel_size: int = 1, decode_context_model_parallel_size: int | None = 1` |
| `vllm/distributed/parallel_state.py` | L1573-L1582 | 5D mesh reshape: `all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)` |
| `vllm/distributed/parallel_state.py` | L1593-L1614 | DCP group construction — `group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)` + `init_model_parallel_group(group_name="dcp")` |
| `vllm/distributed/parallel_state.py` | L1616-L1633 | PCP group construction — `group_ranks = all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size).unbind(0)` + `init_model_parallel_group(group_name="pcp")` |
| `vllm/distributed/parallel_state.py` | L1741-L1782 | `ensure_model_parallel_initialized` with PCP world-size assertion |
| `vllm/distributed/parallel_state.py` | L1791-L1797 | `prepare_communication_buffer_for_model` — `_PCP.prepare_communication_buffer_for_model(model)` if PCP enabled |
| `vllm/distributed/parallel_state.py` | L1847-L1854 | `get_decode_context_model_parallel_world_size` / `_rank` helpers |
| `vllm/distributed/parallel_state.py` | L1871+ | global `_DCP` reset in cleanup paths |
| `vllm/v1/attention/ops/dcp_alltoall.py` | 458 lines total | DCP A2A communication backend (NOT a class — pure functions). Reference: arxiv.org/abs/2507.07120 |
| `vllm/v1/attention/ops/dcp_alltoall.py` | L1-L40 | Module docstring + imports — explicit "Provides All-to-All (A2A) communication as an alternative to AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP)" |
| `vllm/v1/attention/ops/dcp_alltoall.py` | L40-L100 | `_lse_weighted_combine` — CPU reference for LSE-weighted combination of partial outputs from CP ranks |
| `vllm/v1/attention/ops/dcp_alltoall.py` | L448 | `dist.all_to_all_single` — the actual NCCL call |
| `vllm/v1/attention/backend.py` | L700-L755 | `AttentionImpl` base class CP fields: `supports_pcp: bool = False`, `dcp_world_size`, `dcp_rank`, `pcp_world_size`, `pcp_rank`, `total_cp_world_size = pcp × dcp`, `total_cp_rank = pcp_rank × dcp_world_size + dcp_rank`. **`__new__` initializes from `get_dcp_group()` / `get_pcp_group()` with try/except for testing.** |
| `vllm/v1/attention/backend.py` | L703-L706 | `supports_pcp` flag + `supports_mtp_with_cp_non_trivial_interleave_size` flag (cross-link to Ch10 MTP — interaction matters under interleave_size>1) |
| `vllm/v1/attention/backend.py` | L725-L753 | DCP/PCP fields with `__new__`-time discovery — pattern for subclasses |
| `vllm/v1/attention/backends/utils.py` | L820-L857 | `get_dcp_local_seq_lens(seq_lens, dcp_size, dcp_rank, cp_kv_cache_interleave_size)` — per-rank local KV length under striped sharding. The MAIN sequence-sharding helper |
| `vllm/v1/attention/backends/utils.py` | L824 | parameter `cp_kv_cache_interleave_size: int = 1` |
| `vllm/v1/attention/backends/flash_attn.py` | L? (gate at `decode_context_parallel_size > 1`) | DCP path in flash-attn V3 backend |
| `vllm/v1/attention/backends/flashinfer.py` | L213 | `class BatchDCPPrefillWrapper` — flashinfer-specific DCP-prefill batched wrapper. **Only DCP-prefixed class in the entire codebase** (not a top-level CP orchestrator) |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | L125 | `supports_dcp_with_varlen=(interleave_size == 1)` per-backend capability flag |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | L175 | `num_heads_q=self.num_heads * self.dcp_world_size` — Q replicated across DCP ranks |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | L196-L250 | `dcp_tot_seq_lens_device`, `dcp_context_kv_lens` metadata threaded through MLA forward |
| `vllm/v1/attention/backends/mla/flashattn_mla.py` | L353-L355 | a2a backend wired: `cp_world_size=self.dcp_world_size, cp_rank=self.dcp_rank, cp_tot_seqused_k=attn_metadata.decode.dcp_tot_seq_lens` |
| `vllm/v1/attention/backends/mla/flashmla.py` | L160-L200 | `dcp_tot_seq_lens_device` parameter in forward (DCP integration in MLA's flashmla impl) |
| `vllm/v1/attention/backends/mla/rocm_aiter_mla.py` | L213, L311 | DCP integration in ROCm AITER MLA backend |
| `vllm/v1/attention/backends/utils.py` | L820 (`get_dcp_local_seq_lens`) | Comment says "Only consider dcp now, we can extend the case of cp based on this" — PCP version still TBD in source |
| `vllm/v1/kv_cache_interface.py` | L195-L205 | `max_memory_usage_bytes` — the HBM accounting code. `max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)` if `dcp × pcp > 1` |
| `vllm/v1/kv_cache_interface.py` | L195-L205 | The actual `cdiv(max_model_len, dcp_world_size × pcp_world_size)` formula — chapter MUST quote this verbatim |
| `vllm/v1/executor/multiproc_executor.py` | L116-L121 | `world_size == tp_size × pp_size × pcp_size` assertion (NOT × dcp; DCP is folded inside TP group) |
| `vllm/v1/executor/multiproc_executor.py` | L258-L259 | `_get_parallel_sizes` → `(tp_size, pp_size, pcp_size)` |
| `vllm/v1/executor/multiproc_executor.py` | L985-L997 | Process naming: `_PCP{rank}` if `pcp_size > 1` |
| `vllm/v1/executor/multiproc_executor.py` | L493 | `* self.parallel_config.prefill_context_parallel_size` — world-size product for spawn |
| `vllm/v1/executor/ray_executor_v2.py` | L263-L268 | Ray executor: same world_size = tp×pp×pcp assertion |
| `vllm/v1/executor/ray_utils.py` | L338-L346 | `pcp_size: int` parameter in Ray bundle scheduling |
| `vllm/model_executor/layers/fused_moe/config.py` | L? (multiple) | `pcp_size`, `pcp_rank`, `flatten_tp_across_dp_and_pcp(tp, dp, pcp)` helper — PCP composes with EP for MoE prefill |
| `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | L? (multiple) | `if self.moe_config.pcp_size > 1: hidden_states = get_pcp_group().all_gather(...)` then `reduce_scatter` after expert compute — MoE under PCP |
| `vllm/config/parallel.py` | L115 | `prefill_context_parallel_size: int = 1` (member of `ParallelConfig`) |
| `vllm/config/parallel.py` | L310 | `decode_context_parallel_size: int = 1` (member of `ParallelConfig`) |
| `vllm/config/parallel.py` | L315-L324 | `dcp_kv_cache_interleave_size: int = 1` — DEPRECATED alias for `cp_kv_cache_interleave_size`; will be removed when PCP is fully supported |
| `vllm/config/parallel.py` | L325-L329 | `dcp_comm_backend: DCPCommBackend = "ag_rs"` — `"ag_rs"` (default) vs `"a2a"` |
| `vllm/config/parallel.py` | L330-L342 | `cp_kv_cache_interleave_size: int = 1` — striped sharding granularity. "store interleave_size tokens on total_cp_rank i, then store next interleave_size tokens on total_cp_rank i+1". **Block_size must be ≥ and divisible by interleave_size** |
| `vllm/config/parallel.py` | L465-L484 | DCP validation: `tp_size % dcp_size == 0` hard assertion + `dcp_comm_backend="a2a"` requires `dcp_size > 1` |
| `vllm/config/parallel.py` | L765 | `* self.prefill_context_parallel_size` in `world_size` product property |

This is **at least 12 source files** — comparable to Ch10's 11.
Aim for the v6 floor of ≥10 in impl-notes.

### §2.2 — Outline-vs-source mismatches to flag (CRITICAL)

**Five potential reframes — verify each:**

**Reframe A — Outline §1 "为什么需要CP——超长序列（128K+）的单GPU放不下"**: this is EXACTLY what `kv_cache_interface.py:L195-L205 max_memory_usage_bytes` derives. NOT a reframe — straightforward; just open with HBM math from this function. This validates Ch10's pattern of "outline subsection title is the question, source-grounded answer is the chapter".

**Reframe B — Outline §2 "Ring Attention——peer-to-peer P2P通信的环形拓扑"**: **vLLM does NOT ship `class RingAttention` — verified via `grep -rE '^class\s+(RingAttention|StripedAttention)'` returns zero matches.** Instead vLLM ships:
- `vllm/v1/attention/ops/dcp_alltoall.py` (458 lines) — pure-function module providing ALL-TO-ALL (NOT P2P ring) communication for DCP
- LSE-weighted combine math (`_lse_weighted_combine` at L40-L100) is the same MATHEMATICALLY as Ring Attention's online softmax accumulation; the COMMUNICATION pattern is different (a2a vs ring)
- AG+RS backend (`dcp_comm_backend="ag_rs"`) is the alternative, also NOT ring

**This is a 5th instance candidate for the "no class X" reframe.** §11.2 should:
1. Title-anchor: "第11章：DCP/PCP —— 没有 `class RingAttention` 的上下文并行"
2. Hook-anchor: "Ch07 'no radix tree' → Ch08 'no class TensorParallel' → Ch09 'no class ExpertParallel' → Ch10 'no class MultiTokenPrediction' → Ch11 是这条系列的 **第五件**"
3. Body-anchor: explicit grep evidence + `dcp_alltoall.py` module-name (not class-name) reveal + reference to the Liu et al. 2023 paper as the literature anchor that vLLM departs from
4. Recap: §11.7 list with 5-instance enumeration

If the brief is right about this, Ch11 establishes **N=5** for the chapter motif and validates the pattern further. **HIGH-PROBABILITY 5th instance — implementer should verify on first pass and claim this if confirmed.**

**Reframe C — Outline §3 "DCP——decode阶段的all-reduce vs all-to-all方案"**: source uses **AllGather+ReduceScatter (AG+RS)** as the default backend, NOT all-reduce. The two options are AG+RS vs A2A, NOT all-reduce vs all-to-all. **Surgical correction**: chapter must say "AG+RS vs A2A" (verbatim from `parallel.py:L325-L329` which lists `DCPCommBackend = Literal["ag_rs", "a2a"]`). Outline subsection title is wrong on this; outline JSON unchanged, chapter corrects.

**Reframe D — Outline §4 "PCP——prefill阶段的striped vs balanced切分"**: source has `cp_kv_cache_interleave_size` knob (`parallel.py:L330-L342`) which IS the striped sharding granularity. "Balanced" partitioning (Striped Attention from the literature) is what `cp_kv_cache_interleave_size > 1` enables. NOT a reframe — outline term "striped" matches; "balanced" refers to the load-balanced variant. Implementer should verify: with `interleave_size=1`, sharding is fully striped (token i → rank `i % cp_size`); with `interleave_size=K`, K-token blocks are striped (block i → rank `i % cp_size`). Math walk: under causal masking, contiguous (`interleave_size=∞`) gives `cp_size×` rank imbalance; striped (`interleave_size=1`) gives perfectly balanced; intermediate K trades off cache-line-friendliness vs load balance.

**Reframe E — Outline §5 "CP+TP的3D并行——device mesh的映射策略"**: actual device mesh is **5-DIMENSIONAL** at `parallel_state.py:L1573-L1582`, with axes `(external_dp × dp × pp × pcp × tp)` and DCP folded INSIDE the TP axis. So "CP+TP的3D" undersells it: CP itself is 2 axes (DCP + PCP), and the production mesh has 5 axes. Implementer reframes §11.5 as "5-axis mesh: external_dp × dp × pp × pcp × tp, with DCP nested inside TP" — outline JSON unchanged, chapter narrows to source reality.

**The chapter has TWO sub-distinctions that must be precise:**
1. **DCP world-size DOES NOT change total ranks** (folded inside TP). PCP DOES (independent axis).
2. **DCP and PCP are SEPARABLE axes** — not "DCP and PCP must match". The chapter must derive: at `(tp=8, pcp=2, dp=1, pp=1)`, world_size = `8 × 2 × 1 × 1 = 16`; with `dcp=4`, each TP-group of 8 GPUs splits into `8/4=2` DCP sub-groups; each rank's local KV chunk = `seq_len / (pcp × dcp) = seq_len / 8`.

Implementer brief should hit ALL FIVE reframes; reviewer will count.

**`knowledge/INDEX.md` must add a new module**:
- Add row: `| [dcp-pcp](modules/dcp-pcp.md) | 11 | parallel_state.py, attention/backend.py, dcp_alltoall.py |`
- This is a NEW module; use **D-prefix IDs** (D01, D02, ...) — distinct from K (prefix-cache), P (preemption), T (tensor-parallelism), E (expert-parallelism), M (multi-token-prediction). **MUST avoid collision** per `feedback_double_prefix_headings.md`.

### §2.3 — Verified absence of structures

- No top-level `class RingAttention` anywhere in `vllm/`. Verified.
- No top-level `class StripedAttention` anywhere in `vllm/`. Verified.
- No top-level `class DecodeContextParallel` / `class PrefillContextParallel` / `class ContextParallel`. Verified.
- The ONLY DCP-prefixed class in the codebase is `class BatchDCPPrefillWrapper` at `vllm/v1/attention/backends/flashinfer.py:L213` — a flashinfer-specific batched wrapper for DCP-prefill. NOT a top-level orchestrator.
- The DCP/PCP machinery is **module-level pure-function code** (`dcp_alltoall.py`) + **GroupCoordinator singletons** (`_DCP`, `_PCP` in `parallel_state.py`) + **per-attention-backend integration** (each backend in `vllm/v1/attention/backends/` has its own DCP wiring).
- This is the **5th potential "no class X" instance** — confirms the chapter motif from Ch07/Ch08/Ch09/Ch10 extends.

### §2.4 — vLLM commit pin verification

```
$ cd instances/vllm/source && git rev-parse HEAD
98661fe012c5c467252d4df8411d2f46190e9268
```

Matches Ch10 brief's pin at `98661fe`. All file:line refs in §2.1 verified against this commit. If implementer hits drift on a second-pass check, re-grep for the symbol (function/class name) — DCP/PCP code paths are still active development territory in vLLM and may shift in the next release.

---

## §3 — Outline section walk-through

Outline subsections (from `book-outline.json` →
`parts.part2_advanced_common.chapters[11-dcp-pcp].subsections`)
and how to map them to source. Subsection text is the *topic* (the question the
section answers), not a class-name contract.

| Outline subsection | Reframed scope | Source anchor |
|---|---|---|
| 1. "为什么需要CP——超长序列（128K+）的单GPU放不下" | Open `vllm/v1/kv_cache_interface.py:L195-L205 max_memory_usage_bytes`. Walk the HBM math: KV size = `seq_len × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes`. Compute concrete: 128K × 80 × 8 × 128 × 2 × 2 = 33.5 GB per request. Show that `max_model_len_per_rank = max_model_len / (dcp_size × pcp_size)` from `cdiv(max_model_len, dcp_world_size * pcp_world_size)` at L203. **5-step rhythm**: open → ask "why TP doesn't shard this?" → derive (TP shards heads, KV is per-head, so each TP rank still stores all tokens) → impl `cp_capacity.py` reproducing the formula → diff: real production has TP=8, PCP=4, DCP=4 → 32× HBM reduction. | `vllm/v1/kv_cache_interface.py:L195-L205`, `vllm/config/parallel.py:L115, L310` |
| 2. "Ring Attention——peer-to-peer P2P通信的环形拓扑" | **REFRAME to "no class RingAttention; vLLM ships AllToAll + AllGather+ReduceScatter, not P2P ring"**. Open `dcp_alltoall.py:L1-L40` module docstring. Sidebar: Liu et al. 2023 Ring Attention algorithm (P2P ring, online-softmax LSE accumulation, `cp_size` rounds × `seq_len/cp_size` chunk per round). Pivot: vLLM's choice — `dcp_comm_backend="ag_rs"` (AllGather Q + ReduceScatter output, 3 NCCL ops) OR `dcp_comm_backend="a2a"` (single all-to-all of partial output + LSE, then Triton-kernel combine, 2 NCCL ops). Derive LSE-weighted combine math (same as FlashAttention §2.3). Striped Attention as the load-balancing variant under causal masking. **5-step rhythm**: open `dcp_alltoall.py:L40` `_lse_weighted_combine` → ask "why no Ring?" (answer: NCCL collectives are highly tuned; bandwidth on a2a/ag_rs is comparable to or better than P2P ring on intra-node NVLink, especially with FP8 LSE precision; ring is more bandwidth-optimal at very large cp_size but vLLM targets cp_size 2-8 typically) → derive LSE combine → impl `cp_combine.py` reproducing both Ring (P2P sequential) and a2a (LSE combine) variants → diff: vLLM's a2a is Triton-fused, ours is pure PyTorch. | `vllm/v1/attention/ops/dcp_alltoall.py:L1-L100`, `vllm/v1/attention/ops/dcp_alltoall.py:L448` |
| 3. "DCP——decode阶段的all-reduce vs all-to-all方案" | **CORRECT to "AG+RS vs A2A"**. Open `vllm/distributed/parallel_state.py:L1234-L1260`. Walk `_DCP` GroupCoordinator + `get_dcp_group()`. Walk init at L1593-L1614: `group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)` — DCP folded inside TP group. Walk constraint `tp_size % dcp_size == 0` at `parallel.py:L474-L478`. Open `vllm/v1/attention/backends/utils.py:L820-L857 get_dcp_local_seq_lens` — per-rank local KV length under striped sharding. Two backends side-by-side: AG+RS (3 NCCL ops, default) vs A2A (2 NCCL ops, advanced). Open `vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355` for a2a wiring with MLA. **5-step rhythm**: open `parallel_state.py:L1593-L1614` → ask "why DCP folded inside TP, not independent axis?" (answer: DCP shards KV across decoding ranks but Q is already replicated within TP group from column-parallel; so DCP can reuse the TP intra-node communication topology without expanding world_size) → derive AG+RS vs A2A NCCL op count → impl `dcp_decode.py` reproducing both backends → diff: production a2a fuses LSE combine into Triton kernel. | `vllm/distributed/parallel_state.py:L1234-L1260, L1593-L1614`, `vllm/v1/attention/backends/utils.py:L820-L857`, `vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355` |
| 4. "PCP——prefill阶段的striped vs balanced切分" | Open `vllm/distributed/parallel_state.py:L1285-L1290 _PCP / get_pcp_group`. Walk init at L1616-L1633: PCP is INDEPENDENT axis (transpose 3,4 for the right layout reshape). Walk `vllm/v1/attention/backend.py:L725-L753` — `pcp_world_size`, `total_cp_world_size = pcp × dcp`. Walk `cp_kv_cache_interleave_size` at `parallel.py:L330-L342`: striped sharding (interleave=1) vs blocked striped (interleave=K) vs near-contiguous (interleave very large). Derive load-balance math: under causal masking, contiguous gives `cp_size×` rank imbalance; striped gives balanced; intermediate is cache-line-friendly. Walk MoE under PCP at `fused_moe/runner/moe_runner.py` — all_gather hidden_states + router_logits, then reduce_scatter post-expert. **5-step rhythm**: open `parallel_state.py:L1616-L1633` → ask "why PCP is independent but DCP folded?" (answer: prefill loads input from outside the TP group's existing data flow; needs its own communication topology; while decode reuses Q which already moves through TP) → derive striped-vs-blocked load balance → impl `pcp_prefill.py` reproducing the prefill sequence shard + Ring-style attention → diff: production has Triton-fused causal-masked Ring kernel; ours is naive matmul. | `vllm/distributed/parallel_state.py:L1285-L1290, L1616-L1633`, `vllm/v1/attention/backend.py:L703-L755`, `vllm/config/parallel.py:L330-L342`, `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` |
| 5. "CP+TP的3D并行——device mesh的映射策略" | **REFRAME to "5D mesh: external_dp × dp × pp × pcp × tp, DCP folded inside TP"**. Open `vllm/distributed/parallel_state.py:L1573-L1582`: 5D `all_ranks` reshape. Walk axis-by-axis: external_dp (verl integration), dp (in-model DP, generates simultaneously), pp (pipeline stages), pcp (prefill seq shard), tp (head/MoE shard). DCP nested inside TP: each TP-group of `tp_size` GPUs splits into `tp_size/dcp_size` DCP sub-groups. Open `multiproc_executor.py:L116-L121` world_size assertion. Forward-pointer: at production scale `(tp=8, pp=2, pcp=2, dcp=2, dp=1)`, total ranks = `8×2×2×1×1 = 32` (DCP folded), HBM-per-rank = `1/(pcp × dcp) = 1/4` of original. **5-step rhythm**: open `parallel_state.py:L1573-L1582` → ask "why 5D not 4D?" (answer: external_dp + pcp are both "outside TP+EP+DCP unit" axes; pcp expands world_size, external_dp doesn't) → derive composition math (per-rank HBM, per-layer NCCL ops) → impl `mesh_5d.py` reproducing the reshape + axis-by-axis group construction → diff: real production wires NCCL groups; ours uses tuples + asserts. | `vllm/distributed/parallel_state.py:L1573-L1582`, `vllm/v1/executor/multiproc_executor.py:L116-L121, L985-L997`, `vllm/v1/kv_cache_interface.py:L195-L205` |

Use this 5-section mapping as the chapter's §11.1-§11.5 spine. §11.6+ for source-mapping table (main + per-section mini per K15 two-tier; Ch10 hit 80+51 mini=206; Ch11 should aim for the same density). §11.7 for language-trap recap (6 candidates in §6 below). §11.8 for verification with demo numerics. §11.9 for forward-pointers (Ch15+/Ch18 attention/Ch22 PD/Ch27 DeepSeek-V3.2).

---

## §4 — Knowledge dependencies

### Existing knowledge entries to read before work

- `knowledge/modules/tensor-parallelism.md` — T-prefix, T01-T19. **Critical**: T01 group construction (`parallel_state.py:L1573-L1582` 5D reshape); T05 `tp_size % dcp_size == 0` constraint origin; T03 column-parallel + Q replication that DCP relies on. The TP group is the carrier for DCP folding.
- `knowledge/modules/expert-parallelism.md` — E-prefix, E01-E24. **Critical**: E04 `_EP` GroupCoordinator pattern (mirrored by `_DCP`/`_PCP`); E15 axis orthogonality logic from §9.5; E07 `pcp × ep × tp` flatten in `fused_moe/config.py:flatten_tp_across_dp_and_pcp`.
- `knowledge/modules/multi-token-prediction.md` — M-prefix, M01-M30. **Cross-link**: M-? (none direct yet) but `vllm/v1/attention/backend.py:L705 supports_mtp_with_cp_non_trivial_interleave_size` IS the explicit cross-link. Implementer should call out that MTP-with-CP requires `interleave_size=1` (or backend-specific support).
- `knowledge/modules/attention.md` — only tangentially relevant; Ch11 doesn't rederive attention but DOES use the LSE-weighted combine which IS FlashAttention's online softmax. Reference Ch03.
- `knowledge/modules/kv-cache.md` — relevant for the HBM-per-rank math; reference for `max_memory_usage_bytes`.
- `knowledge/modules/scheduler.md` — P-prefix. Not directly relevant to Ch11, but the prefill-vs-decode phase distinction came from Ch04; reference only.

### NEW knowledge module REQUIRED

**Create `knowledge/modules/dcp-pcp.md`** — Ch11 owns its own module:

- Use **D-prefix IDs** (D01, D02, ...) — distinct from K (prefix-cache), P (preemption), T (tensor-parallelism), E (expert-parallelism), M (multi-token-prediction). **MUST avoid collision** per `feedback_double_prefix_headings.md`.
- Forward-shared with: Ch15+ Llama deep-dive (every long-context model), Ch18 (Triton attention — DCP a2a is Triton-fused), Ch22 (PD architecture — CP composes with PD), Ch25 (PD ratio — DCP world-size becomes a budget knob), Ch27 (DeepSeek-V3.2 — MLA + DCP production stack).
- **WARNING (carried from Ch07-Ch10 lessons)**: `learn.py` append-mode bugs were fixed (P1-1 task #36 completed), but if doubled `## D0X: D0X:` headers show up after extraction, fix immediately. Also: multi-token-prediction.md hit 30 facts (Ch10), expert-parallelism.md 24 (Ch09), tensor-parallelism.md 19 (Ch08), prefix-cache.md 17 (Ch07) — **`learn.py compact()` is broken** (`_parse_module_file` returns []). Manual workaround if Ch11 module exceeds 15 facts. Ch11 will likely also exceed (5-axis mesh + 2 backends + load-balance math has rich fact surface).
- Update `knowledge/INDEX.md`:
  - Add row: `| [dcp-pcp](modules/dcp-pcp.md) | 11 | parallel_state.py, attention/backend.py, dcp_alltoall.py |`

### Anticipated facts the implementer will discover (D-prefix candidates)

- D01: `_DCP` is folded INSIDE the TP group; `_PCP` is a separate world-size-expanding axis. (`parallel_state.py:L1593-L1633`)
- D02: DCP communication backends: `"ag_rs"` (default, 3 NCCL ops) vs `"a2a"` (2 NCCL ops, advanced; ref arxiv.org/abs/2507.07120). (`parallel.py:L325-L329`, `dcp_alltoall.py`)
- D03: `cp_kv_cache_interleave_size` controls striped-sharding granularity. block_size must be divisible by interleave_size. (`parallel.py:L330-L342`)
- D04: `dcp_kv_cache_interleave_size` is DEPRECATED; use `cp_kv_cache_interleave_size`. Will be removed when PCP fully supported. (`parallel.py:L315-L324`)
- D05: 5D mesh = `(external_dp × dp × pp × pcp × tp)` at `parallel_state.py:L1573-L1582`. NOT 3D and NOT 4D; the outline term "3D parallel" undersells.
- D06: `world_size = tp × pp × pcp × dp` (NOT × dcp; DCP is folded). (`multiproc_executor.py:L116-L121`)
- D07: `total_cp_world_size = pcp × dcp` (composed CP world size); `total_cp_rank = pcp_rank × dcp_world_size + dcp_rank`. (`vllm/v1/attention/backend.py:L725-L753`)
- D08: `tp_size % dcp_size == 0` is HARD CONSTRAINT; violating raises ValueError. (`parallel.py:L474-L478`)
- D09: `BatchDCPPrefillWrapper` is the ONLY DCP-prefixed class — flashinfer-specific. NOT a top-level orchestrator. (`flashinfer.py:L213`)
- D10: Per-attention-backend `supports_pcp: bool = False` flag — many backends do NOT yet support PCP. (`vllm/v1/attention/backend.py:L703`)
- D11: `supports_mtp_with_cp_non_trivial_interleave_size` — explicit cross-chapter knob between Ch10 MTP and Ch11 CP. (`vllm/v1/attention/backend.py:L705`)
- D12: `max_memory_usage_bytes` formula: `cdiv(max_model_len, dcp × pcp) × cdiv(max_model_len, block_size) × page_size_bytes`. (`vllm/v1/kv_cache_interface.py:L195-L205`)
- D13: `get_dcp_local_seq_lens` is the per-rank striped-shard helper. (`vllm/v1/attention/backends/utils.py:L820-L857`)
- D14: `_lse_weighted_combine` is the LSE-weighted merge for partial CP outputs — same algebra as FlashAttention §2.3 but across ranks not tiles. (`vllm/v1/attention/ops/dcp_alltoall.py:L40-L100`)
- D15: PCP composes with EP: `flatten_tp_across_dp_and_pcp(tp, dp, pcp)` flattens 3 axes into a single per-rank EP scope. (`fused_moe/config.py`)

---

## §5 — Wisdom hits (role priorities: implementer = debugging > architecture > testing > writing)

Read these before opening source:

- `wisdom/debugging.md` — `F.linear` shape rules don't directly apply (Ch11 is comm-side, not weight-side), but the mesh-reshape pattern is similar: `all_ranks.reshape(-1, dim).unbind(0)` produces per-axis groups; mismatched shape gives wrong group composition silently. Write small mesh tests (e.g., `world_size=8, tp=2, pcp=2, pp=2, dp=1` should produce TP-groups `[[0,1],[2,3],[4,5],[6,7]]` and PCP-groups `[[0,2],[1,3],[4,6],[5,7]]` after the transpose).
- `wisdom/architecture.md` — **backpressure gates and lateral comm patterns**. CP ranks must synchronise per-layer (every CP rank waits for the all-gather/reduce-scatter or the all-to-all to complete before next layer). Different from EP (where dispatch + combine are 2 collectives per layer). DCP a2a backend reduces from 3 to 2 ops/layer — the PROOF this matters is the paper arxiv.org/abs/2507.07120 reporting end-to-end throughput improvements. **fix prompts not chapters** still applies: if reviewer flags "PCP claim is wrong on world-size folding", fix the impl-notes axis-orthogonality discussion, not the chapter directly.
- `wisdom/testing.md` — preemption test design generalizes to "test the boundary": `cp_size=1` (degenerate, equivalent to no CP), `cp_size=2` (smallest non-trivial — verify single round of all-to-all preserves attention output), `cp_size=4` (composition with TP=2), `dcp=2 + pcp=2` (axis composition), `interleave_size=1` (fully striped) vs `interleave_size=block_size` (contiguous). Tester needs all of these. **Plus**: M25 lesson — parametrise grid cells. `cp_size ∈ {1,2,4,8}` × `interleave_size ∈ {1,2,4,8}` × `(ag_rs, a2a)` backends gives 64 parametrised test cells from 2 fact-tests.
- `wisdom/writing.md` — formula rules (NON-NEGOTIABLE: `\mathrm{}` not `\text{}`, no `\boxed`, no `\frac` inline). **Ch11 will be FORMULA-HEAVY** (Ring Attention LSE-weighted combine, HBM math, axis composition). Plan for high formula-density mitigation per K-series writer patterns: ≤2 inline formulas per bullet, render mid-proof variable references as plain text. **M30 from Ch10 wisdom** says inline-density warnings up to ~15 acceptable IF every flagged paragraph passes single-symbol check; Ch11 may go higher.

Plus the reproducible-cadence patterns from `state.json:v6_compliance` (after Ch10):

- `two_tier_mapping` — mandatory; Ch11 surface is BROAD (12+ files). Aim for 80+ main + 50+ mini per Ch10 cadence.
- `language_trap_callouts` — Ch11 has plenty (see §6); plan ≥6-7 explicit recap items, matching Ch09/Ch10's 7.
- `honest_demo_caveats` — synthetic CP simulation on single-process is NOT real distributed comm; the chapter must caveat. K17 lesson: writer must quote caveat verbatim from impl-notes.
- `single_cycle_approval` — Ch04-Ch10 all hit this. Ch11 must replicate; cadence holds at N=7.
- `framing_tip_three_anchor_verification` (E22 from Ch09; M29 from Ch10) — every framing tip must show up at hook + body + recap. Reviewer will count.
- `no_class_X_three_anchor_pattern` — the "no class RingAttention" reframe needs title + hook + §11.2 body anchors with grep evidence. Pattern is 5th instance now (Ch07 → Ch08 → Ch09 → Ch10 → Ch11).
- `training_to_inference_reframe_template` — M28 explicitly says "queue for wisdom promotion at instance #3". Ch11 does NOT have a training→inference reframe (CP is purely inference machinery), so this template doesn't apply this chapter; Ch16 quantisation may.

---

## §6 — Candidate language traps for the writer (target 5-7)

Each candidate is a phrasing that is "easy to write and almost-but-not-quite right". Writer picks the strongest 5-7 for explicit callouts at the relevant section + a dedicated recap section, mirroring Ch07 §7.6.4 / Ch08 §8.6.4 / Ch09 §9.7 / Ch10 §10.7.

**Trap A — "DCP doubles decode throughput at dcp_size=2."** No. DCP shards KV CACHE memory across ranks; throughput depends on attention compute and communication. With dcp=2, each rank computes attention against `seq_len/2` of KV — half the FLOPs per rank, but each layer needs 1 extra all-gather (or 1 extra a2a). At short seq_len, the comm overhead dominates and DCP can be slightly slower than no-CP. At long seq_len (128K+), the FLOPs reduction wins. **DCP's headline value is HBM CAPACITY (enabling long context at all), NOT throughput.** Source evidence: `kv_cache_interface.py:L195-L205 max_memory_usage_bytes` is the WIN axis; throughput is workload-dependent.

**Trap B — "PCP halves prefill latency at pcp_size=2."** Partially true. With pcp=2, each rank computes Q,K,V on `seq_len/2` tokens — half the prefill compute per rank. BUT the all-to-all (or ring) communication for cross-rank attention adds `O(cp_size)` rounds of comm. For a 32K prefill on H100 with NVLink (200+ GB/s), pcp=2 communication overhead is `~5%` of compute time → near-2× speedup. For a 4K prefill on PCIe (50 GB/s), comm overhead can exceed compute → PCP is NET LOSS. **Operator must measure prefill length × inter-GPU bandwidth before enabling PCP.** Source: no source-side gate; operator chooses `prefill_context_parallel_size` in `ParallelConfig`.

**Trap C — "Context parallel is just sequence parallel renamed."** No. **Sequence parallel** in Megatron usage shards the SEQUENCE DIMENSION of activations within a TP group's MLP/LN to save activation HBM (see `is_sequence_parallel: bool` in `vllm/distributed/parallel_state.py:L? ALL operations`). It does NOT shard KV cache. **Context parallel** (DCP/PCP) shards KV cache (DCP) or input sequence (PCP). The two are orthogonal and coexist: `is_sequence_parallel=True` is for activation HBM; DCP/PCP is for KV HBM. Source evidence: `parallel_state.py` exposes BOTH `is_sequence_parallel` parameter on `all_gather`/`reduce_scatter` AND `_DCP`/`_PCP` GroupCoordinators — distinct mechanisms.

**Trap D — "DCP and PCP must match (dcp_size == pcp_size)."** No. They are **separable axes** in the device mesh. Production configs run `(tp=8, dcp=2, pcp=4, pp=2, dp=1)` (DCP=2 within each TP-group of 8 → 4 DCP sub-groups; PCP=4 as independent axis). The constraints are: `tp % dcp == 0` (DCP folds inside TP) and `world_size = tp × pp × pcp × dp` (PCP enters world_size, DCP doesn't). Source evidence: `parallel.py:L474-L478` only enforces `tp_size % dcp_size == 0`; `multiproc_executor.py:L116-L121` confirms world_size product excludes dcp.

**Trap E — "Context parallel is the same as tensor parallel for the attention layer."** No. TP shards the HEAD axis (each rank owns `num_heads / tp_size` heads, full sequence per head). CP shards the SEQUENCE axis (each rank owns full heads, `seq_len / cp_size` tokens of K and V). Different axis → different communication: TP needs `all_reduce` on attention output (heads contribute partial sums); CP needs `all_to_all` or AG+RS on the output (ranks contribute partial outputs over disjoint KV chunks, combined via LSE weighting). Source evidence: TP comm is in `vllm/model_executor/layers/linear.py` (the `RowParallelLinear.forward` `all_reduce`); CP comm is in `vllm/v1/attention/ops/dcp_alltoall.py` (the all-to-all + LSE combine). **Different code paths, different math, different layer placement.**

**Trap F — "Ring Attention is the canonical implementation in vLLM."** No, **vLLM does not implement Ring Attention**. The codebase ships AllGather+ReduceScatter (`dcp_comm_backend="ag_rs"`, default) or All-to-All (`dcp_comm_backend="a2a"`, advanced). Both use NCCL collectives, not P2P send/recv ring topology. Mathematically the LSE-weighted combine is similar to Ring Attention's online softmax, but the COMMUNICATION pattern is collective, not peer-ring. Source evidence: `dcp_alltoall.py:L1-L20` module docstring explicitly compares "All-to-All (A2A) communication as an alternative to AllGather + ReduceScatter (AG+RS)" — no mention of Ring; `dist.all_to_all_single` at L448 is the actual NCCL call.

**Trap G — "Striped Attention is just renamed Ring Attention."** No. Striped Attention is a TOKEN-PARTITIONING scheme (token i → rank `i % cp_size`), independent of the COMMUNICATION pattern (Ring vs all-to-all vs ag+rs). vLLM's `cp_kv_cache_interleave_size` knob (`parallel.py:L330-L342`) controls the partitioning granularity; communication is separate. Striped sharding's purpose is load-balancing under causal masking — late-token Q has more KV to attend to, and contiguous partitioning gives `cp_size×` rank imbalance. Striped (interleave=1) gives perfect balance at the cost of cache-unfriendly access; intermediate K trades off. Source evidence: `cp_kv_cache_interleave_size` and `dcp_comm_backend` are independent config knobs.

Pick 5-7 of A/B/C/D/E/F/G for primary callouts; reviewer expects ≥5 (Ch09 hit 7, Ch10 hit 7, exceeds floor). Recap section §11.7 should explicitly enumerate them with "claim → 错 → 为什么 → 源码证据 → Demo/测试" per Ch07/Ch08/Ch09/Ch10 template.

---

## §7 — Demo plan (numerics for verbatim narrative use, target ≥20 verbatim)

The implementer's `demo.py` should produce numbers the writer will quote
verbatim (per the demo-numerics-verbatim hard gate, K17 / N=7 baseline; Ch10
hit ≥85 verbatim values).

**Demo §1 — HBM-per-rank capacity walk under (DCP, PCP) sweep.** Compute `max_memory_usage_bytes` from the actual formula: `seq_len = 128K` (max_model_len), `num_layers = 80`, `num_kv_heads = 8`, `head_dim = 128`, `dtype_bytes = 2 (bf16)`. Sweep `(dcp, pcp) ∈ {(1,1), (1,2), (2,1), (2,2), (1,4), (4,1), (2,4), (4,4)}`. Pin: HBM bytes per rank for each of 8 cells (formula `cdiv(seq_len, dcp × pcp) × num_layers × num_kv_heads × head_dim × 2 × dtype_bytes`). 8 cells × 1 value = **8 verbatim numbers**. Show: `(1,1) = 33.5 GB`; `(2,2) = 8.4 GB`; `(4,4) = 2.1 GB`. Trap-A evidence: HBM is the WIN axis, not throughput.

**Demo §2 — LSE-weighted combine math (Ring Attention algebra).** Build a toy: 4 CP ranks each own `[seq_len/4, num_heads, head_dim]` of partial attention output `O_i` and corresponding `lse_i`. Implement `_lse_weighted_combine` (mirroring `dcp_alltoall.py:L40-L100`): compute `lse_max = max_i(lse_i)`, then `weight_i = exp(lse_i - lse_max)`, then `O_global = (Σ_i weight_i × O_i) / Σ_i weight_i`. Verify against ground-truth single-process FlashAttention. Pin: per-rank `lse_i` values, per-rank `weight_i`, final `O_global` against ground-truth (max abs error). 4 ranks × 3 vectors per rank + final error = ~13 values verbatim. **Demo also covers Ring Attention's mathematical equivalence to all-to-all when LSE weights are correct.**

**Demo §3 — AG+RS vs A2A NCCL op count and bandwidth model.** For dcp_size ∈ {2, 4, 8}, count NCCL ops per layer: AG+RS = `1 (AllGather) + 1 (Attention) + 1 (ReduceScatter)` = 3 NCCL ops + 1 GEMM kernel; A2A = `1 (Attention partial) + 1 (AllToAll partial+LSE) + 1 (Triton combine)` = 2 NCCL ops + 2 kernels. Use α-β bandwidth model: `T_AG_RS(N) = 2 × (α + β × seq_len/dcp × heads × dim × dtype) + α + β × seq_len × heads × dim × dtype`; `T_A2A(N) = α + β × (output_partial + LSE) × seq_len/dcp × heads × dim × dtype`. Pin α=10μs, β=200 GB/s. For seq_len=32K, num_heads=8, head_dim=128, dtype=2: 4 verbatim cells. Trap-F evidence: A2A is faster but neither is Ring.

**Demo §4 — Striped vs contiguous KV partitioning under causal mask.** Build a toy: 8 CP ranks, seq_len=64. Causal mask gives token-i `i+1` KV positions to attend. Compute per-rank work under (a) contiguous partition (rank 0 owns tokens 0..7, rank 7 owns tokens 56..63 — rank 7 work ≈ `8 × 60 = 480` KV-attends, rank 0 work ≈ `8 × 4 = 32` → 15× imbalance) and (b) striped (interleave=1, rank r owns tokens r, r+8, ..., r+56 — every rank has same average attention depth ≈ `8 × 32 = 256` KV-attends → balanced). Pin: per-rank work counts under both schemes (8 cells × 2 schemes = 16 values), imbalance ratio. Trap-G evidence: Striped IS the load-balancing scheme.

**Demo §5 — 5D mesh group construction.** Build a toy: world_size=16 with `(tp=4, pcp=2, pp=2, dp=1, external_dp=1)`. Reproduce `parallel_state.py:L1573-L1582` reshape and group-builder logic. Pin per-axis groups: TP groups `[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]`; PP groups via transpose 2,4 and reshape; PCP groups via transpose 3,4 and reshape. With `dcp_size=2` folded inside TP: DCP sub-groups `[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]`. **Verify all groups partition world_size correctly** — 16 ranks total. Pin verbatim: 4 TP-groups × 4 ranks each = 16 IDs; 2 PP-groups × 8 ranks each = 16 IDs; 4 PCP-groups × 4 ranks each = 16 IDs; 8 DCP sub-groups × 2 ranks each = 16 IDs. 4 group lists × 16 IDs = ~64 verbatim integers (visible as bracketed lists in narrative).

These 5 demos collectively give the writer **≥100 ground-truth numbers** to
quote verbatim (target ≥20). Test report should pin every one with
`assertEqual` / `assertLess` / `assertAlmostEqual` and explicit values.

**Honest demo caveats** (impl-notes states; writer quotes verbatim):

- Single-process simulation does NOT actually launch NCCL collectives. The math is verified bit-exact against single-process FlashAttention, but real intra-node NVLink bandwidth (200+ GB/s) and inter-node IB bandwidth (50 GB/s) are not measured — α-β model in §3 uses literature numbers.
- α-β bandwidth values are H100 + 4×NVLink reference numbers; A100 + InfiniBand would shift β by ~3-5× and α by ~2×. Real production should measure on target hardware.
- "supports_pcp" is not yet True for all attention backends — flash_attn V3 has explicit DCP support; PCP is still wiring up. The demo simulates PCP as if all backends support it; production will hit `NotImplementedError` for some.
- `cp_kv_cache_interleave_size` is the latest API; `dcp_kv_cache_interleave_size` is deprecated. Demo uses the new name; production code at older commits may have the old name.
- 5D mesh demo uses `external_dp=1` (default for non-verl integrations); verl deployments would use `external_dp > 1`.

---

## §8 — Floor reminders (v6 hard gates, confirmed at N=7 after Ch10)

Implementer commit must satisfy:

- **≥5 source files in impl-notes "Source Analysis" section.** Ch11 natural surface is **12+ files**: `parallel_state.py`, `dcp_alltoall.py`, `vllm/v1/attention/backend.py`, `vllm/v1/attention/backends/utils.py`, `flash_attn.py`, `flashinfer.py`, `mla/flashattn_mla.py`, `mla/flashmla.py`, `kv_cache_interface.py`, `multiproc_executor.py`, `parallel.py` (config), `fused_moe/runner/moe_runner.py` (PCP-EP integration). **Aim for 12**; the breadth IS the lesson (per Ch07-Ch10 cadence; Ch10 hit 11).
- **≥60 `# REFERENCE: <path>:Lxxx` comments across impl modules.** Match the floor. Ch04: 65, Ch05: 61, Ch06: 60, Ch07: ~60, Ch08: 64, Ch09: 66, Ch10: 151 (proposer family inflation). Target ≥70 for Ch11 (without proposer-family-style boilerplate inflation).
- **≥10 mapping rows; aim for 80 main + 50 mini per K15 two-tier.** Ch07: 27+45 = 72. Ch08: 122. Ch09: 49+39+helper = 151. Ch10: 80+51+helper = 206. Ch11 should match Ch10 cadence (broad 5-axis surface).
- **Demo numerics verbatim** in tests/test-report.md, then narrative quotes them character-for-character.
- **Both linters PASS at the BLOCKING bar** before handoff (writer + reviewer re-run; mismatches trigger preemptive REVISE per K17). Non-blocking inline-density warnings up to ~15 acceptable IFF every inline token is single symbol (M30 from Ch10). The bar is "0 blocking AND every inline token single-symbol", NOT "0/0".
- **5-step rhythm in every major section §11.1-§11.5.**
- **5-7 language traps with explicit "claim → 错 → 为什么 → 源码证据 → Demo/测试" per §6 above**, plus dedicated recap §11.7. Ch10 hit 7 traps matching Ch09's 7.
- **Forward/back-pointers wired**:
  - Back to Ch03 (FlashAttention — LSE-weighted combine algebra), Ch04 (continuous-batching — prefill-vs-decode phase), Ch08 (TP — group construction in `parallel_state.py`, axis orthogonality), Ch09 (EP — `_EP` GroupCoordinator pattern, axis composition), Ch10 (MTP — `supports_mtp_with_cp_non_trivial_interleave_size` cross-link).
  - Forward to Ch15+ (model zoo; every long-context model uses CP), Ch18 (Triton attention — DCP a2a is Triton-fused), Ch22 (PD architecture — CP composes with PD disaggregation), Ch25 (PD ratio — DCP becomes a budget knob), Ch27 (DeepSeek-V3.2 — MLA + DCP production stack).
- **Source pin verification**: implementer's first command is `cd instances/vllm/source && git rev-parse HEAD` — must equal `98661fe`. Any line numbers that drift between brief and source → re-grep before citing.
- **§11.2 "no class RingAttention" reframe documented in impl-notes**: title + hook + §11.2 body anchors with grep evidence. **5th instance** after Ch07/Ch08/Ch09/Ch10 — pattern is now well-rehearsed, reviewer will count anchors. Hook should enumerate 5 prior cases including Ch11 self.
- **§11.3 + §11.4 + §11.5 surgical corrections** to outline language: AG+RS vs A2A (not all-reduce vs all-to-all); 5D mesh (not 3D); separable DCP/PCP axes (not "must match"). Document as outline-vs-source corrections in impl-notes.

### What APPROVED at cycle 1 looks like (K17 / N=7 baseline)

Writer's handoff message must contain BOTH linter outputs verbatim:

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/11-dcp-pcp/narrative/chapter.md
[expected: 🟢 No blocking issues]
[acceptable: ≤15 non-blocking inline-density warnings IFF every inline token is single symbol per M30]

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/11-dcp-pcp/
[expected: ✓ All grounding checks passed!]
```

Plus the hard gates: ≥10 mapping rows, all source files in impl-notes
referenced in narrative, 5-step rhythm in every §11.X, demo numerics verbatim,
≥5 trap callouts (target 6-7), forward-pointers wired, no `class
RingAttention` reframe applied at three anchors (title + hook + §11.2),
DCP/PCP/AG+RS/A2A/5D-mesh corrections surfaced in §11.3-§11.5.

### Cadence projection from Ch04-Ch10

| Metric | Ch04 | Ch05 | Ch06 | Ch07 | Ch08 | Ch09 | Ch10 | Ch11 (target) |
|---|---|---|---|---|---|---|---|---|
| Lines | 712 | 757 | 655 | 859 | 1051 | 1204 | 1345 | ≥1300 |
| Words | 3064 | 3849 | 3351 | 4440 | 6058 | 7792 | 8888 | ≥8000 |
| Mapping rows | 13 | 21 | 40 | 72 | 122 | 151 | 206 | ≥180 |
| Tests | 48 | 74 | 97 | 83 | 144 | 204 | 311 | ≥250 |
| Source files | 5 | 7 | 6 | 5 | 8 | 10 | 11 | ≥10 |
| Cycles to APPROVED | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 (target) |
| Lang trap callouts | 0 | 0 | 4 | 4 | 5 | 7 | 7 | ≥6 |

Ch11 should be in the Ch10 ballpark on lines/words (broad surface). Test count
should match Ch10 (parametrising `cp_size × interleave_size × backend` is the
3-axis lever, just like Ch10's α-K-c grid).

---

## §9 — Cadence carry-forward from Ch10

**Ch10 hit single-cycle APPROVED with broadest source surface yet (11 vLLM
modules) and quality bar holding. Ch11 must replicate.** Specific
carry-forwards:

1. **"No class X" reframe pattern is now a chapter motif (N=4, going N=5).** Ch07 (radix tree) → Ch08 (TensorParallel) → Ch09 (ExpertParallel) → Ch10 (MultiTokenPrediction). Ch11 is the FIFTH instance candidate (no `class RingAttention` / `class StripedAttention` / `class ContextParallel`). Pattern is now well-rehearsed; reviewer expects three anchors with grep evidence + opener "this chapter, like Ch07/Ch08/Ch09/Ch10, opens with what's NOT in source, BUT MORE — vLLM departs from the canonical Liu et al. 2023 algorithm structure entirely, choosing all-to-all over P2P ring".

2. **Ch10's training→inference reframe template (Ch09 §9.4 pattern, Ch10 §10.3 instance #2) does NOT apply to Ch11.** CP is purely inference machinery; no training-time concept to reframe. M28 says queue for wisdom promotion at instance #3; Ch11 doesn't contribute, but Ch16 quantisation is the next likely candidate.

3. **Three-anchor framing-tip verification (Ch09 E22 + Ch10 M29 reviewer wisdom).** Every framing tip from tester must appear at hook + body + recap. Reviewer counts. Ch10 had 5 tips × 3 anchors = 15 verifications. Ch11 should target the same.

4. **Tester framing-guidance loop (Ch06-Ch10, N=5 in a row).** Tester is expected to produce 5+ surgical narrative-shaping tips from test code. Implementer's brief should HINT at what tester will discover — the demo plan in §7 above is structured to surface those tips:
   - Demo §1 HBM-per-rank → Tip "DCP/PCP wins HBM, not throughput; throughput is workload-dependent"
   - Demo §2 LSE combine → Tip "Ring Attention's algebra IS LSE-weighted combine; vLLM uses the same math, different transport"
   - Demo §3 AG+RS vs A2A → Tip "A2A reduces 3 NCCL ops → 2; production-tested 33% reduction"
   - Demo §4 striped vs contiguous → Tip "Striped sharding solves causal-mask imbalance, NOT communication"
   - Demo §5 5D mesh → Tip "Production mesh is 5D with DCP folded inside TP; outline says 3D undersells"

5. **Honest-demo caveat OR-skip discipline (Ch09 K17/E11; Ch10 M17 OR-skip strict).** §7 lists 5 caveats; impl-notes must state them, writer quotes verbatim, reviewer cross-checks.

6. **Knowledge module D-prefix discipline.** Avoid double-prefix headings (`## D0X: D0X:`). The `learn.py compact()` is broken — manual workaround in use across Ch07-Ch10. **P2-2 (system-improvements) flagged "must-fix-before-Ch12"** in Ch09 delivery, then **escalated in Ch10 delivery** ("blocking 5 consecutive chapters; rate accelerating"). Ch11 will be the **6th consecutive chapter** to add 15+ new facts. If P2-2 isn't fixed before Ch11 lands, manual compact will be required for D-prefix module.

7. **# REFERENCE comment count is workload-dependent.** Ch04-Ch09 hit 60-66; Ch10 jumped to 151 due to 7-file proposer family boilerplate. Ch11's source surface is comm-side (not algorithm family), so expected count is 70-80 (not 150+). Don't artificially inflate; quality > count.

---

## §10 — Direct-dispatch operational notes

**Per `feedback_direct_dispatch.md` rule**: book-editor's idle-summary handoffs
were unreliable in Ch07; team-lead direct-SendMessages each agent. Ch08-Ch10
followed this; Ch11 will too.

**Handoff sequence for Ch11**:

1. **Team-lead → implementer**: SendMessage with this brief's path
   (`/home/zjq/Repo2Book/instances/vllm/trace/briefs/11-dcp-pcp-implementer-2026-05-07.md`)
   plus the §1 chapter scope summary as inline context. Mention the 5
   reframes (no class RingAttention; AG+RS vs A2A correction; striped
   sharding under interleave; 5D mesh not 3D; separable axes not "must
   match") as upfront framing.
2. **Implementer → tester**: SendMessage when implementation + impl-notes
   complete; include linter passes in handoff message.
3. **Tester → writer**: SendMessage with framing tips and demo verbatim
   numerics; tester is expected to produce 5+ tips per Ch06-Ch10 cadence.
4. **Writer → reviewer**: SendMessage with both linters' outputs verbatim
   (K17 protocol).
5. **Reviewer → archivist**: SendMessage on APPROVED with verdict report path.
6. **Archivist → team-lead**: SendMessage with delivery summary + state.json
   diff + Ch12 (KV offload) brief immediately per brief-on-approval.

**Brief-on-approval discipline (`feedback_brief_on_approval.md`)**: when Ch11
is APPROVED, archivist immediately writes the Ch12 (KV offload) brief without
waiting for explicit user prompt.

**Source-grounding-verify-before-dispatch (`feedback_outline_topic_not_contract.md`,
rule #6)**: this brief was written with archivist running source-verification
queries first (DCP/PCP machinery in `parallel_state.py`, init logic, attention
backend integration, `cp_kv_cache_interleave_size`, `dcp_alltoall.py` module
walk, no `class RingAttention` grep). All file:line refs in §2.1 verified at
commit 98661fe. **5 outline-vs-source mismatches identified** (per §2.2).
If implementer hits drift on a specific symbol, re-grep — DCP/PCP code is
still active in vLLM main.

**Loop-escalation rule**: if reviewer/writer ping-pong > 3 cycles, escalate to
team-lead per `repo2book.json:pipeline.topology.decision_protocol.escalate`.
Ch04-Ch10 each completed in 1 cycle; Ch11 should too.

---

**END OF BRIEF**

Brief author: archivist (2026-05-07).
Source pin: 98661fe012c5c467252d4df8411d2f46190e9268.
Outline source: `instances/vllm/book/book-outline.json` →
`parts.part2_advanced_common.chapters[11-dcp-pcp]`.
Reframes flagged:
- §11.2 "no class RingAttention" (5th instance — N=5 of the chapter motif)
- §11.3 "AG+RS vs A2A" (correction to outline "all-reduce vs all-to-all")
- §11.4 "striped via cp_kv_cache_interleave_size knob" (matches outline)
- §11.5 "5D mesh not 3D" (correction to outline "3D并行")
- §11.5 "separable DCP/PCP axes" (correction to "must match" misconception in trap D)
