# DCP/PCP Knowledge — vLLM

Repo-specific facts about the Decode/Prefill Context Parallelism surface in vLLM.
Source pin: `98661fe`. D-prefix IDs (D01..) — distinct from the K
(prefix-cache), P (preemption), T (tensor-parallelism), E (expert-parallelism),
M (multi-token-prediction) families to avoid double-prefix collisions.

Roles: **I**=implementer, **T**=tester, **W**=writer, **R**=reviewer.

---

## D01: `_DCP` is folded INSIDE the TP group; `_PCP` is a separate world-size-expanding axis

- File: `vllm/distributed/parallel_state.py:L1593-L1633`
- Audience: I, R, W
- Fact: `_DCP` group is built by `all_ranks.reshape(-1, dcp_size).unbind(0)` —
  no transpose, because DCP is the FASTEST-varying axis within TP. Each
  TP-group of size `tp_size` contains `tp_size//dcp_size` DCP sub-groups.
  `_PCP` group is built by `all_ranks.transpose(3, 4).reshape(-1, pcp_size).unbind(0)`
  — transpose pcp ↔ tp axes BEFORE reshape, because PCP is its own axis.
- Implication: any reimpl must mirror this exactly. Off-by-axis transpose
  produces wrong group composition silently (no NCCL error until runtime).

## D02: DCP communication backends — AG+RS (default, 3 ops) vs A2A (2 ops, advanced)

- File: `vllm/config/parallel.py:L322-L328`, `vllm/v1/attention/ops/dcp_alltoall.py`
- Audience: I, W, R
- Fact: `DCPCommBackend = Literal["ag_rs", "a2a"]`. Default is `"ag_rs"`
  (AllGather Q + local attention + ReduceScatter O = 3 NCCL ops).
  Alternative is `"a2a"` (local attention + packed AllToAll of partial
  output + LSE = 2 NCCL ops; reference: arxiv.org/abs/2507.07120).
  A2A reduces NCCL latency by ~33%.
- Implication: reviewer must check chapter says "AG+RS vs A2A", NOT
  "all-reduce vs all-to-all" (outline subsection 3 has wrong terminology).

## D03: `cp_kv_cache_interleave_size` controls striped-sharding granularity

- File: `vllm/config/parallel.py:L330-L342`
- Audience: I, T, W
- Fact: Token `i` is stored on `total_cp_rank = (i // I) % total_cp_world_size`
  where `I = cp_kv_cache_interleave_size`. `I=1` → fully striped (token-level);
  `I=block_size` → block-aligned; `I=∞` → fully contiguous. Block_size must
  be ≥ AND divisible by interleave_size.
- Implication: tests must cover `I ∈ {1, 4, block_size}`. Demo §4 shows
  imbalance ratio is 13.44x at contiguous, 1.24x at striped (cp=8, seq=64).

## D04: `dcp_kv_cache_interleave_size` is DEPRECATED

- File: `vllm/config/parallel.py:L315-L321`
- Audience: I, R
- Fact: `dcp_kv_cache_interleave_size` is the OLD name; replaced by
  `cp_kv_cache_interleave_size`. Will be removed when PCP is fully supported.
- Implication: chapter uses the NEW name; production code at older commits
  may still use the deprecated alias.

## D05: 5D mesh = (external_dp × dp × pp × pcp × tp) at parallel_state.py:L1569-L1575

- File: `vllm/distributed/parallel_state.py:L1569-L1575`
- Audience: I, R, W
- Fact: `all_ranks = torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)`.
  Five axes. The `-1` absorbs `external_dp` (verl integration; defaults to 1).
  DCP is folded INSIDE TP; logically a 6th axis but doesn't expand world_size.
  Outline subsection 5 says "3D parallel" — undersells.
- Implication: chapter MUST say "5D mesh + DCP nested", not "3D".

## D06: `world_size = tp × pp × pcp × dp` (NOT × dcp)

- File: `vllm/v1/executor/multiproc_executor.py:L116-L121`
- Audience: I, R
- Fact: Source verbatim assertion: `world_size == tp_size * pp_size * pcp_size`.
  DCP is excluded — folded inside TP. (DP also excluded from this single-engine
  scope; DP is a separate engine.)
- Implication: any reimpl computing world_size from DCP × something is wrong.
  Tests should verify `Mesh5D.world_size` excludes dcp.

## D07: `total_cp_world_size = pcp × dcp`; `total_cp_rank = pcp_rank × dcp_world + dcp_rank`

- File: `vllm/v1/attention/backend.py:L725-L753`
- Audience: I, T, W
- Fact: Composed CP world size for cross-rank attention. `total_cp_rank` is
  the index into the SHARED kv-cache stripe (every (pcp, dcp) pair has its
  own slice of the global sequence). The formula is `pcp_rank × dcp_size + dcp_rank`
  — DCP-major ordering.
- Implication: tests for the composition should verify both axis orders give
  consistent results. Demo §5 uses `(pcp=2, dcp=2) → 4 total_cp_ranks`.

## D08: `tp_size % dcp_size == 0` is a HARD CONSTRAINT

- File: `vllm/config/parallel.py:L469-L478`
- Audience: I, T
- Fact: Source verbatim ValueError: `f"tp_size={...} must be divisible by dcp_size={...}"`.
  This is the ONLY hard CP constraint. PCP has no equivalent (PCP is independent
  axis; tp doesn't constrain pcp).
- Implication: tests must verify ValueError on `tp=4, dcp=3`. Tests for valid
  configs should cover `tp=8, dcp ∈ {1, 2, 4, 8}`.

## D09: `BatchDCPPrefillWrapper` is the ONLY DCP-prefixed class — flashinfer-specific

- File: `vllm/v1/attention/backends/flashinfer.py:L213`
- Audience: I, R, W
- Fact: Verified by `grep -rE '^class\s+\w*DCP\w*'` returning ONE match. The
  rest of the DCP machinery is module-level pure functions
  (`vllm/v1/attention/ops/dcp_alltoall.py`) + GroupCoordinator singletons
  (`_DCP`, `_PCP`).
- Implication: chapter §11.2 hook should call out this 1-class-only fact as
  the "exception that proves the rule" for the no-class-X reframe.

## D10: Per-attention-backend `supports_pcp: bool = False` flag

- File: `vllm/v1/attention/backend.py:L703`
- Audience: I, T
- Fact: Each attention backend declares `supports_pcp: bool = False` by default;
  only some backends override to True. flash_attn V3 supports DCP; PCP is still
  wiring up across backends.
- Implication: production runs hit `NotImplementedError` for some backends.
  Demo simulates as if all backends support it.

## D11: `supports_mtp_with_cp_non_trivial_interleave_size` — the Ch10 ↔ Ch11 cross-link

- File: `vllm/v1/attention/backend.py:L705-L706`
- Audience: I, W
- Fact: Explicit per-backend flag for whether MTP works with CP at
  `interleave_size > 1`. This is the explicit Ch10 ↔ Ch11 cross-link.
- Implication: chapter §11.4 should mention this when discussing
  interleave_size; Ch10 narrative should also have a forward pointer.

## D12: `max_memory_usage_bytes` formula = `cdiv(max_model_len, dcp × pcp) × cdiv(..., block_size) × page_size_bytes`

- File: `vllm/v1/kv_cache_interface.py:L196-L204`
- Audience: I, T, W
- Fact: Source verbatim:
  ```python
  if dcp_world_size * pcp_world_size > 1:
      max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
  return cdiv(max_model_len, self.block_size) * self.page_size_bytes
  ```
- Implication: chapter §11.1 must derive THIS formula verbatim. Demo §1 shows
  HBM=40GB → 2.5GB at dcp=4, pcp=4 for 70B/128K/bf16 model.

## D13: `get_dcp_local_seq_lens` is the per-rank striped-shard helper

- File: `vllm/v1/attention/backends/utils.py:L820-L857`
- Audience: I, T
- Fact: Formula: `base = (seq // I // dcp) × I; remainder = seq - base × dcp;
  remainder = clip(remainder - rank_offsets × I, 0, I); local_seq_lens = base + remainder`.
  Source comment: "Only consider dcp now, we can extend the case of cp based on this"
  — PCP version still TBD.
- Implication: this is the CORE per-rank seq-length math. Tests must verify
  `sum_across_ranks(local_seq_lens[r]) == global_seq_len` for any (cp_size, I).

## D14: `_lse_weighted_combine` is the FlashAttention online softmax across ranks

- File: `vllm/v1/attention/ops/dcp_alltoall.py:L39-L103`
- Audience: I, W
- Fact: `lse_max = max_i(LSE_i); weights = exp(LSE - lse_max); O_global = (sum_i
  weight_i × O_i) / sum_i(weight_i)`. Same algebra as FlashAttention §2.3
  online softmax, applied across N CP ranks instead of N attention tiles.
  Verified bit-exact (3.33e-16 max abs error in fp32) against single-process
  FA in Demo §2.
- Implication: tests must verify associativity (split 4-way then 2+2-way →
  same result). Theorem: AG+RS, A2A, Ring all produce IDENTICAL output via
  this combine.

## D15: PCP composes with EP via `flatten_tp_across_dp_and_pcp`

- File: `vllm/model_executor/layers/fused_moe/config.py` (helper),
  `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` (use site)
- Audience: I, R
- Fact: Helper flattens 3 axes (tp, dp, pcp) into a single per-rank EP scope.
  Under PCP, MoE prefill needs `all_gather` on hidden_states + router_logits
  across the PCP group, then `reduce_scatter` after expert compute.
- Implication: PCP-MoE is a 2D communication pattern (pcp × ep). Out of
  scope for this chapter (Ch09 territory); chapter §11.4 should mention as
  forward-pointer.

## D16: Tester-discovered — total_cp_rank composition needs witnesses for both pcp_rank and dcp_rank

- File: `vllm/v1/attention/backend.py:L751-L752`
- Audience: T, W
- Fact: ``total_cp_rank = pcp_rank * dcp_world_size + dcp_rank``. A test
  that just checks total_cp_world_size = pcp * dcp passes for the wrong
  reason if either factor is 1. The composition test must use
  ``(dcp_world_size > 1, pcp_world_size > 1)`` BOTH at once — otherwise
  one or both fields hide their value.
- Implication: tests for D07 must use a (dcp=2, pcp=2) world_size=8 mesh
  to witness independent contributions, not (dcp=1, pcp=N) or vice versa.

## D17: Tester-discovered — naive HBM total = 40.0 GB (NOT 33.5 GB from brief)

- File: `vllm/v1/kv_cache_interface.py:L195-L205` (formula source);
  `implementation/kv_cache_per_rank.py::hbm_naive_total` (verified)
- Audience: T, W
- Fact: For Llama-70B at 128K (80 layers, 8 KV heads, head_size=128, bf16):
  ``128K * 80 * 2 (K+V) * 8 * 128 * 2 = 42,949,672,960 bytes = 40.0 GB``.
  The impl-notes brief/summary mentioned 33.5 GB — that was a prior rev with
  different head config. Demo §1 reports the correct 40.0 GB.
- Implication: chapter narrative MUST quote 40.0 GB (the demo output) as
  the headline number. 33.5 GB elsewhere should be corrected. Knowledge
  D17 prevents this rot recurring.

## D18: Tester-discovered — `combine` function takes pre-summed partials in source

- File: `vllm/v1/attention/ops/dcp_alltoall.py:L320-L450`
- Audience: T
- Fact: The pedagogical ``simulate_a2a_combine`` and ``simulate_ag_rs_combine``
  produce identical outputs because both delegate to ``lse_weighted_combine``.
  In production, the difference is in pre-combine packing — A2A uses
  ``dist.all_to_all_single`` on a pre-packed (output, lse) buffer, AG+RS
  uses two collectives (allgather + reduce_scatter) with separate buffers.
  The math is the same; only buffer-packing differs.
- Implication: the integration test
  ``test_section_2_a2a_equals_ag_rs_combine`` is Trap-F evidence at the
  ALGEBRAIC level. To exercise transport differences we'd need real NCCL.

## D19: Tester-discovered — striped (interleave=1) imbalance is 1.24x, NOT 1.0x

- File: `implementation/seq_sharding.py::causal_attention_work_per_rank`
- Audience: T, W
- Fact: Under causal mask, even striped (interleave=1) is NOT perfectly
  balanced — Demo §4 reports 1.24x imbalance. The reason: token 0 has 1
  KV-attend, token 1 has 2, ..., token 63 has 64. Round-robin over 8 ranks
  gives rank 0 tokens [0, 8, 16, ..., 56] = sum(1..7 + 1) = 232, rank 7
  tokens [7, 15, ..., 63] = 288. Ratio 288/232 = 1.241. That's "perfectly
  balanced" *relative to* contiguous (13.44x) but NOT absolutely 1.0.
- Implication: chapter wording matters — say "near-balanced (1.24x vs
  contiguous's 13.44x)" not "perfectly balanced". Demo §4 uses the phrase
  "perfectly balanced" loosely; the writer should refine.

## D20: Tester-discovered — A2A payload shape divides by dcp_size, not by num_heads

- File: `vllm/v1/attention/ops/dcp_alltoall.py:L431-L436`
- Audience: T, W
- Fact: A2A packed payload shape is ``[num_ranks, num_tokens, num_heads/dcp_size,
  head_dim + lse_pack_dim]``. The bytes scale as ``num_heads / dcp_size``
  (not just num_heads). At dcp=2 → bytes = num_tokens * 4 * (128+2) * 2 =
  34,078,720 (for 32K tokens, 8 heads). At dcp=8 → 4× smaller.
- Implication: writer's α-β framing in §11.3 should highlight that A2A's
  payload SHRINKS with dcp_size while AG+RS payload stays constant. That's
  the second axis of A2A's win, beyond the 33% NCCL-op reduction.

## D21: Tester-discovered — Naive sum vs LSE combine differ by 12+ orders of magnitude

- File: `vllm/v1/attention/ops/dcp_alltoall.py:L39-L103`
- Audience: T, W, R
- Fact: Naive (1/N)·sum(O_i) gives error ~1e-3 to ~1e-1 vs single-process
  attention; LSE-weighted combine gives error <1e-15 (fp64). The gap is
  12+ orders of magnitude. This is the "Trap C is wrong" anchor — proves
  the combine math is NOT a numerical approximation but an algebraic
  identity. `test_lse_combine_strictly_better_than_naive` seals the gap.
- Implication: writer should derive the identity BEFORE quoting the
  3.33e-16 number. Frame as "the math composes; numerics confirm" (mirrors
  Ch03 FlashAttention framing). Avoid leading with the empirical bound.

## D22: Tester-discovered — DCP groups are CONTIGUOUS chunks of TP groups (no transpose)

- File: `vllm/distributed/parallel_state.py:L1594-L1614`
- Audience: I, T, W
- Fact: DCP sub-groups are built by chunking each TP group into `tp/dcp`
  contiguous slices. e.g., tp=4, dcp=2 gives DCP sub-groups [0,1] and
  [2,3] — NOT [0,2] and [1,3]. PCP groups, by contrast, are NON-contiguous
  (built via `transpose(3, 4).reshape(-1, pcp).unbind(0)`). The
  contiguity vs non-contiguity is the load-bearing reshape detail. Tested
  via `test_dcp_sub_groups_chunk_tp_groups_contiguously` and
  `test_pcp_groups_via_transpose_3_4`.
- Implication: chapter §11.5 must show BOTH reshape lines verbatim and
  explain why DCP doesn't need transpose (fastest-varying axis within TP)
  while PCP does (own outer axis).

## D23: Tester-discovered — `__new__` discovery snapshots singleton state at call time

- File: `vllm/v1/attention/backend.py:L731-L757`
- Audience: I, T
- Fact: AttentionImpl.__new__ uses try/except AssertionError to read
  singletons. The discovery happens at instance creation time — if
  groups are uninit, the instance gets size-1/rank-0 fallback values
  permanently. Re-initialising groups after instance creation does NOT
  update existing instances. Tested via
  `test_discovery_reflects_state_at_new_call_time`.
- Implication: production code must initialise groups BEFORE creating
  attention backends. Re-init mid-run causes silent stale rank values.
  Writer's §11.2 should mention this as a "subtle gotcha" callout.

## D24: Tester-discovered — total_cp_rank uses pcp-major formula (NOT dcp-major)

- File: `vllm/v1/attention/backend.py:L752`
- Audience: I, T, W
- Fact: Source verbatim: `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`.
  PCP-MAJOR ordering — pcp_rank is the slow-varying index, dcp_rank is the
  fast-varying. Composed CP rank addresses the global KV stripe in
  pcp-major order. Reversing (dcp-major) would mis-address kv-cache
  segments. Tested at `test_total_cp_rank_formula_pcp_major`.
- Implication: writer must show the formula verbatim with the dimension
  order; getting the order wrong silently corrupts cross-rank attention.

## D25: Tester-discovered — `world_size NEVER includes DCP` is invariant across 11 grid cells

- File: `vllm/v1/executor/multiproc_executor.py:L116-L121`
- Audience: I, T, R
- Fact: For ANY valid (tp, pcp, dcp) combination, `mesh.world_size = tp × pp × pcp × dp`
  — DCP is excluded. Tested at `test_world_size_excludes_dcp_grid` across
  11 cells covering tp=8 × pcp ∈ {1,2,4} × dcp ∈ {1,2,4,8}. The world_size
  value is a function of (external_dp, dp, pp, pcp, tp); DCP's value never
  appears. This is the KEY mental-model fix for Trap D.
- Implication: chapter must repeat this invariant in §11.4 (Trap D anchor)
  AND §11.5 (5D mesh). Writer should derive world_size for the production
  config (tp=8, dcp=2, pcp=4) and explicitly say "the answer is 32 = tp×pcp,
  NOT 64 = tp×pcp×dcp".

## D26: Writer-discovered — demo-text vs narrative-text discipline

- File: `instances/vllm/artifacts/11-dcp-pcp/implementation/demo.py:L246`
  (demo prints "perfectly balanced") vs narrative §11.5.2 (must say
  "near-balanced (1.24×)" with absolute number).
- Audience: W, R
- Fact: Demo's loose phrase "perfectly balanced" is correct **relative
  to** contiguous's 13.44×, but the absolute number is 1.24×, not 1.0×
  — narrative must surface the absolute number and refine the phrase.
  This is a verbatim-vs-claim mismatch: writer quotes demo numbers
  verbatim BUT refines demo prose where it overshoots.
- Implication: when writer extracts demo text into narrative, identify
  any phrasing the demo used loosely (e.g. "perfectly balanced",
  "fully balanced", "infinitely fast") and refine it with the actual
  absolute number, while still quoting the digits verbatim.

## D27: Writer-discovered — formula linter `\text{}` blocking discipline + sed mass-replace

- File: `scripts/lint_formulas.py:L148-L160`
- Audience: W, R
- Fact: `lint_formulas.py` flags `\text{}` as auto-REJECT BLOCKING
  but treats "Too Many Inline" and "Complex Inline" as advisory
  warnings (exit 0). Mass replacement
  `sed -i 's/\\text{/\\mathrm{/g' chapter.md` is safe because the
  pattern only collides with literal "\text{" which is always wrong
  in math contexts; outside math it doesn't appear.
- Implication: writer can run sed on a clean draft to clear all
  `\text{}` issues in one shot, then verify with `lint_formulas.py`
  exit code (0 = ok regardless of warning count).

## D28: Reviewer-discovered — three-anchor template scales cleanly to N=5

- File: `instances/vllm/artifacts/11-dcp-pcp/narrative/chapter.md:L1` (title)
  + `:L15` (hook) + `:L218-L262` (§11.2 body)
- Audience: W, R, BE
- Fact: The "no class X" reframe pattern (Ch07 radix tree → Ch08 TP →
  Ch09 EP → Ch10 MTP → Ch11 RingAttention) survives the jump from
  N=4 ("trend") to N=5 ("motif") with the same three-anchor template:
  title-names-the-absence + hook-enumerates-prior-instances +
  body-grep-evidence-at-pin. The hook MUST list ALL prior instances
  by chapter number — if it just says "the same as Ch10" the reader
  loses the cumulative weight.
- Implication: future "no class X" reframes (likely Ch12+ on KV-offload,
  Ch15 on model-zoo, Ch22 on PD-architecture) should follow the same
  template and explicitly count "this is the Nth instance".

## D29: Reviewer-discovered — non-blocking lint warnings are acceptable when each inline is a derivation step

- File: `instances/vllm/artifacts/11-dcp-pcp/narrative/chapter.md:L102, L126,
  L412, L554, L571, L739, L911-L919, L1316`
- Audience: W, R
- Fact: `lint_formulas.py` warns on "Complex Inline" and "Too Many
  Inline" but these are ADVISORY (exit 0). When inline expressions
  are part of a narrative derivation chain (e.g. "代入 spec 参数
  $N_\mathrm{layers}=80, H_\mathrm{kv}=8, ..." or "代入
  $Z_i \cdot O_i = \sum_j ..."), promoting them to block formulas
  fragments the prose flow. Per Ch10 precedent (11 non-blocking
  warnings, APPROVED) and Ch11 (15 non-blocking, APPROVED), these
  are acceptable.
- Implication: reviewer should INSPECT each warning and clear
  derivation-chain inline formulas; only flag when the inline is a
  standalone expression that interrupts the prose without context.

## D30: Archivist-discovered — wisdom-promotion gate is "2+ INSTANCES" not "N within one"; 5th 'no class X' graduates to chapter motif but stays repo-local

- File: `instances/vllm/trace/deliveries/2026-05-08_ch11-dcp-pcp-ch11-v6-published.md`
  + `CLAUDE.md` (Wisdom section) + `feedback_wisdom_gate_strict.md`
- Audience: BE, A
- Fact: After Ch11 hits the 5th "no class X" instance (Ch07/Ch08/Ch09/Ch10/Ch11),
  the temptation is to promote the motif to `wisdom/`. The strict gate is
  "2+ INSTANCES" meaning 2+ DIFFERENT REPO BOOKS. All 5 vllm instances
  are within one repo book. Promotion is BLOCKED until a second
  repo2book instance hits the same outline-vs-source pattern. Document
  in chapter-internal `kv-offload.md` / `dcp-pcp.md` as a chapter motif;
  do NOT touch `wisdom/architecture.md`. The strict gate protects wisdom
  from instance-specific noise — without a 2nd repo, the motif could be
  a vllm-architectural quirk rather than a universal pattern.
- Implication: archivist must explicitly remind on every "Nth instance"
  delivery that wisdom promotion is GATE-LOCKED at 2+ INSTANCES; future
  archivists should resist promotion-pressure even when N reaches 6, 7,
  or higher within one instance.

## D31: Archivist-discovered — implementer brief format that yields zero-patch handoff has 4 invariants

- File: `instances/vllm/trace/briefs/11-dcp-pcp-implementer-2026-05-07.md`
  + `instances/vllm/trace/briefs/12-kv-offload-implementer-2026-05-08.md`
- Audience: BE, A, IMP
- Fact: Ch11 was the cleanest implementer→tester handoff in book (zero
  patches required during testing). The brief that produced this had 4
  invariants: (1) §2 verified source surface table with EXACT `:Lxxx`
  line ranges at the source pin commit, (2) §2.X outline-vs-source
  mismatches surfaced explicitly with reframe direction, (3) §6
  candidate language traps with claim → 错 → 为什么 substructure
  pre-listed (5-7), (4) §7 demo plan with target verbatim numerics
  count. When all 4 invariants are met, implementer artifacts pass
  testing without modification. Ch12 brief reproduces all 4 invariants.
- Implication: future implementer briefs MUST include all 4 invariants;
  archivist verifies before dispatch. The "4-invariant-brief" pattern
  is a candidate cross-instance pattern but per D30 wisdom-promotion
  gate, document at instance level until a 2nd repo book validates.
