# Ch11 DCP/PCP v6 PUBLISHED — eighth v6 chapter, cadence holds at N=8; cleanest implementer handoff in book (zero patches required during testing); broadest comm-side surface yet (12 vLLM modules); 5th "no class X" instance graduates to chapter motif within vllm instance (wisdom-promotion gate of 2+ INSTANCES still NOT met)

- **Type**: delivery
- **Chapter**: 11-dcp-pcp
- **Date**: 2026-05-08
- **Timestamp**: 2026-05-08T08:00:00Z
- **Agents involved**: implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: v6, dcp-pcp, decode-context-parallel, prefill-context-parallel, no-class-reframe-5th-graduated-motif, ag-rs-vs-a2a-outline-correction, 5d-mesh-outline-correction, lse-allreduce, ring-attention, hbm-40gb-correction, framing-tips, language-trap, cleanest-handoff

## What happened

Reviewer-3 APPROVED in 1 cycle (no REVISE iterations). All **11 hard gates pass**.
Both linters PASS at the BLOCKING bar: formula linter **0 blocking + 15
non-blocking inline-density warnings** (4 "Too Many Inline Formulas" at
L126/L554/L739/L911-L919, 11 "Complex Inline Formulas" — each warning sits
inside a derivation chain where the inline expression is part of the prose
flow, e.g. L102 spec parameters list, L412 LSE substitution, L571 α/β H100
reference numbers, L1316 mapping-table cell). Per Ch10 M30 calibration
("warnings up to ~15 acceptable IF every flagged paragraph passes the
single-symbol-token check") and the Ch10 precedent of 11 non-blocking,
Ch11's 15 non-blocking sits at the edge of the calibrated band but each
warning was inspected individually by reviewer-3 and cleared.
Source-grounding linter all PASS.

**474/474 tests pass at vLLM source commit `98661fe`**. Test count is **52%
above Ch10's 311** (Ch04: 48; Ch05: 74; Ch06: 97; Ch07: 83; Ch08: 144;
Ch09: 204; Ch10: 311; Ch11: **474**). Critically, **ZERO implementer
patches were required during testing** — this is the **cleanest
implementer→tester handoff in the entire book to date**. Every other v6
chapter required at least one tester-side fix during the run (Ch10 had M16
shape-mismatch + M23 missing dataclass + M24 dict-key divergence; Ch09 had
mem-formula-leads adjustments; Ch08 had MergedColumn bug). Ch11 ran
end-to-end on the implementer artifact unchanged.

**Stats**: **1394 lines, 8124 words, 149 mapping rows** (81 main §11.9 +
16 cross-chapter §11.9.1/§11.9.2 + per-section mini-tables in §11.1.6 /
§11.2.5 / §11.3.5 / §11.4.7 / §11.5.7 / §11.6.8 / §11.7.1 / §11.8 → 149
total `|`-prefix lines). Lines **exceed Ch10's 1345** (+3.6%); words
**slightly under Ch10's 8888** (-8.6%, but well above the 5K v6 floor —
prose density per topic is intentionally tighter because Ch11 leans on
formula-derivation chains for AG+RS/A2A and α-β cost models rather than
narrative breadth); mapping rows **149 vs Ch10's 206** (-27.7%, but well
above the 60-row floor; broader Ch10 was driven by 5-proposer family
boilerplate, not a quality jump). All three metrics strictly exceed v6
floors. impl-notes "Source Analysis" lists **12 distinct vLLM source
files** (parallel_state.py, dcp_alltoall.py, v1/attention/backend.py,
v1/attention/backends/utils.py, v1/attention/backends/mla/flashattn_mla.py,
v1/kv_cache_interface.py, v1/executor/multiproc_executor.py,
config/parallel.py, flashinfer.py, mla/rocm_aiter_mla.py, moe_runner.py,
flash_attn.py) — **broadest comm-side surface in the book to date**,
exceeds Ch10's 11 by 1 file but spans an entirely new territory
(distributed comm + KV-cache layout + multiproc executor + MLA backend
plumbing). **78 # REFERENCE: comments** across impl modules — count is
LOWER than Ch10's 151 because Ch10's count was inflated by the proposer
family's 7-file boilerplate (per Ch10 delivery note); Ch11's 78 is
above Ch09's 66 baseline and represents real density across 12 source
files (≈6.5 REFERENCEs/file vs Ch10's 13.7/file but Ch10 had 7 boilerplate
proposer files). Reviewer flagged all 12 source files with explicit
`:Lxxx` line refs in narrative — 41 mentions for `dcp_alltoall.py`
(stripe transport + AG+RS + A2A), 28 for `parallel_state.py` (5D mesh
construction), 20 for `v1/attention/backend.py` (DCP-aware attention
plumbing), 13 for `v1/kv_cache_interface.py` (per_rank_len + striped
layout), 12 each for `multiproc_executor.py` and `config/parallel.py`,
10 each for `attention/backends/utils.py` and `mla/flashattn_mla.py`, 4
for `flashinfer.py` (the `class BatchDCPPrefillWrapper` reframe anchor),
1 each for `rocm_aiter_mla.py`, `moe_runner.py`, `flash_attn.py`
(mapping-table only). Coverage is complete; nothing referenced indirectly.

5-step rhythm verified §11.1-§11.6 by reviewer (gate 4): each section opens
with source location + `:Lxxx` → bridge → theory derive → our impl → diff
table. Two-tier mapping (K15 from Ch07) reused: 81 main + 16
cross-chapter forward/back + 7 per-section mini-tables.

**§11.2 reframe applied** as designed in the implementer brief — outline
subsection "Ring Attention 数据结构详解" reframed at chapter level as
**"vLLM 没有 `class RingAttention`，也没有 `class DCPRingAttention` —
只有 `class BatchDCPPrefillWrapper` 在 `flashinfer.py:L213` 一个文件里"**.
Established at THREE structural anchors:
1. §11.2.1 grep evidence: explicit `grep -rE "^class\s+(RingAttention|
   DCPRingAttention|RingAttn|DCPRing)\b"` at commit `98661fe` returning
   "(zero matches)"
2. §11.2 body explicit `class BatchDCPPrefillWrapper` enumeration at
   `flashinfer.py:L213` as the SOLE wrapper class (1 file vs the
   3-transport flat structure expected from outline)
3. §11.10 recap: explicit "**第 5 件** 'no class X'" lineage list (Ch07
   radix → Ch08 TensorParallel → Ch09 ExpertParallel → Ch10
   MultiTokenPrediction → Ch11 RingAttention)

This is the **5th instance of the "no class X" framing pattern** —
graduates from "recognized chapter motif" (Ch10) to firmly established
chapter motif within the vllm instance. **CRITICAL CLARIFICATION**:
all 5 instances are within ONE INSTANCE (vllm); the wisdom-promotion gate
specified by feedback_wisdom_gate_strict and CLAUDE.md is "2+ INSTANCES"
(2+ different repo books). N=5 within vllm does NOT meet that bar. The
motif is documented in `knowledge/modules/dcp-pcp.md` as a chapter-internal
motif and **is NOT promoted to `wisdom/`** until a second repo book hits
the same pattern.

**§11.4 surgical outline correction applied** — outline §11.3 said
"all-reduce vs all-to-all" which is wrong as a description of the
DCP-comm-side trade. The actual choice in the source is **AG+RS
(all-gather + reduce-scatter) vs A2A (all-to-all)** — 2 NCCL ops vs 3,
which yields the 33% NCCL-op reduction headline number (verified in
demos and `dcp_alltoall.py:L66-L70`). The chapter pivots without
modifying outline JSON: §11.4.3 derives the AG+RS / A2A α-β cost models
side-by-side at H100 reference numbers (α=1.0µs, β=2.5e-5µs/byte) and
walks the 3-cell speedup (2.87× / 5.44× / 9.85× at dcp ∈ {2,4,8}) with
verbatim payload bytes (67,108,864 / 34,078,720 / 17,039,360 / 8,519,680)
and verbatim latencies (1036.6 / 360.8 / 190.4 / 105.2 µs). Outline JSON
unchanged; chapter honestly represents source reality. **Reviewer praised
this as "the cleanest outline-vs-source surgical correction in the book
to date" — surfaces the discrepancy explicitly rather than silently
absorbing it.**

**§11.6 surgical outline correction applied** — outline §11.5 said
"3D 并行" which is wrong. The actual mesh in `parallel_state.py:L1569-L1575`
is **5D**: `external_dp × dp × pp × pcp × tp` with **DCP folded inside
TP**. (DCP shares the TP communicator rather than being its own
mesh-dim — a fact that took implementer significant verification to
establish at commit 98661fe.) Reframe at §11.6.1 / §11.6.3 / §11.6.5
walks the 5D enumeration with `world_size=16` demo numerics and the
specific `(tp=8, dcp=2, pcp=4) → world_size=32` trap (NOT 64, because
DCP is folded into TP). Outline JSON unchanged.

**5 framing tips from tester applied surgically with three-anchor
verification** (gate 8 — hook + body + recap, per Ch10 M29 reviewer rule
codified as the **D28 rule** in dcp-pcp.md):
1. **Tip 1: HBM correction "40.0 GB exact, NOT 33.5 GB"** — Hook L41,
   L63 (formula `128K × 80 × 8 × 128 × 2 × 2 = 40.0 GB exact`) + 8-cell
   table at L188-L195; Body §11.1.4-§11.1.5 explicit derivation; Recap
   §11.7 Trap A + §11.10 L1384. Earlier impl-notes revisions had stale
   33.5 GB; reviewer grep-confirmed `33.5 GB` does NOT appear anywhere.
2. **Tip 2: AG+RS vs A2A 33% NCCL-ops reduction** — Hook L65 + Body
   §11.4.3 / §11.4.6 / demo §4 + Recap §11.7 Trap B + §11.10 L1388. With
   verbatim α-β cost-model numbers and 3-cell speedup grid.
3. **Tip 3: Striped 13.44× / 1.55× / 1.24×** — Hook L66 + Body §11.5
   striped-vs-contiguous derivation with cell-by-cell verbatim
   reproduction (L725, L735, L788, L801-L803, L811) + Recap §11.7
   Trap C + §11.10 L1390.
4. **Tip 4: 5D mesh `world=16`** — Hook implicit + Body §11.6.1-§11.6.5
   walk with verbatim §5 numerics (L989, L991, L1031, L1222) + Recap
   §11.10 explicit "DCP folded inside TP" enumeration. Plus Trap D
   "(tp=8, dcp=2, pcp=4) → world_size=32 NOT 64" at L1099.
5. **Tip 5: 5th 'no class X' graduated motif** — Hook L17 + Body §11.2.1
   grep + §11.2 body 1-file enumeration + Recap §11.10 5-instance
   lineage list. Three anchors; dcp-pcp.md D26 explicitly documents this
   as **chapter-internal motif** (5 within vllm; wisdom-promotion gate
   2+ INSTANCES NOT met).

**7 language-trap callouts** in §11.7 (gate 9) following the
Ch06/Ch07/Ch08/Ch09/Ch10 lineage style "claim → 错 → 为什么 → 源码证据
→ Demo/测试":
- Trap A: HBM 40.0 GB → 2.5 GB at (dcp=4, pcp=4) — NOT 33.5 GB stale
- Trap B: AG+RS vs A2A is 2 NCCL ops vs 3, 33% reduction (NOT
  all-reduce vs all-to-all per outline)
- Trap C: Striped vs contiguous KV-cache layout — striped is 13.44× at
  per_rank_len=8192 NOT a uniform constant ratio
- Trap D: `(tp=8, dcp=2, pcp=4)` → `world_size=32` NOT 64 — DCP folded
  inside TP, NOT a separate mesh dimension
- Trap E: LSE allreduce associativity error 2.22e-16 vs sequential
  3.33e-16 — BOTH within float64 ULP; LSE is associative under the
  log-sum-exp identity (NOT non-associative as some readers assume)
- Trap F: Ring Attention is NOT a class in vLLM — `flashinfer.py:L213`
  has `class BatchDCPPrefillWrapper` as the SOLE wrapper (5th
  "no class X")
- Trap G: DCP "decode-context" vs PCP "prefill-context" — name
  asymmetry signals different sharding semantics (verified at
  `kv_cache_interface.py:L195-L205` per_rank_len + striped-layout
  invariants)

7 traps **matches Ch09 and Ch10's** 7 (target floor was 5-7 per Ch10
delivery); each follows the same 5-substructure template per E19/M29
codified as **D28 rule**.

**Forward-pointers** to Ch12+/Ch13/Ch22-Ch25 wired (gate 10):
- Ch12 (kv-offload): per_rank_len + striped-layout invariant carries
  forward to multi-tier KV-cache (HBM → CPU → disk)
- Ch13 (prefix-cache-pooling): 5D mesh world_size invariant carries
  forward to pool-size sizing across DCP/PCP-replicated ranks
- Ch22-Ch25 (PD-disagg): AG+RS vs A2A trade reappears in PD comm-side
  decisions (cross-stage transfer protocols)

**Knowledge appended**: D01-D29 in `knowledge/modules/dcp-pcp.md`:
- D01-D15: implementer-supplied (5D mesh + DCP folding into TP +
  per_rank_len + striped-layout + LSE allreduce algebra + AG+RS vs A2A
  cost model)
- D16-D20: tester-added (HBM 40.0 GB derivation + zero-patch handoff
  reproducibility + striped-cell-grid invariants + 5D enumeration
  edge cases)
- D21-D25: tester-added more (3-cell AG+RS speedup payloads + 4-cell
  striped grid + max-abs-error 3.33e-16 vs 2.22e-16 distinction +
  LSE associativity claim + 7-trap fidelity verification)
- D26-D27: writer-added (5th-no-class-X-as-chapter-motif documentation +
  reframe-template stability)
- D28-D29: reviewer-added (three-anchor mechanical grep verification
  rule formalised as D28; HBM 33.5-vs-40.0 disambiguation discipline as
  D29)

At **29 facts**, dcp-pcp.md exceeds the 15-fact compaction trigger by 14
facts. **Sixth chapter** to demonstrate compact() brokenness in succession
(Ch07: 17 → Ch08: 19 → Ch09: 24 → Ch10: 30 → Ch11: 29). **CRITICAL
DIFFERENCE FROM PRIOR CHAPTERS**: per system-improvements P1-1 fix,
`learn.py compact` now WORKS — Ch11 is the **first chapter** where the
compact path is operational. Recommendation: run
`python3 scripts/learn.py compact dcp-pcp` to LLM-summarise oldest 5
facts (D01-D05) into one and bring file under threshold.

**Minor non-blocking observations** (do NOT trigger REVISE; reviewer noted
for writer's future awareness):
1. 15 non-blocking inline-density warnings — at the edge of M30
   calibrated band (Ch08:0, Ch09:4, Ch10:11, Ch11:15). Reviewer-3
   inspected each and cleared per single-symbol-token rule. If next
   chapter trends higher, M30 may need re-calibration to ≤20.
2. §11.4 AG+RS vs A2A correction was the cleanest outline-vs-source
   surgical correction in the book to date — reviewer flagged as
   reference example for future outline-correction patterns.
3. §11.6 5D mesh "DCP folded inside TP" took implementer significant
   verification effort at commit 98661fe — reviewer praised explicit
   surface as "no reader can mistake DCP for a separate mesh dim".

Source pinned at vLLM commit `98661fe`. Snapshot location TBD (matching
prior pattern `trace/snapshots/{N}-{slug}/v6-2026-05-08/`).

## Why it matters

**EIGHTH chapter under v6 standards** — cadence holds at N=8. Critically,
Ch11 establishes **the cleanest implementer→tester handoff in the entire
book to date** (zero patches required during testing) and **two surgical
outline-vs-source corrections** (AG+RS vs A2A; 5D not 3D mesh) without
modifying outline JSON. Three structural ways v6 is now strictly better
at N=8 than at N=7:

1. **Implementer-handoff quality reproduces and improves.** Ch10 needed
   M16/M23/M24 patches. Ch11 needed ZERO. The implementer brief
   (`12-kv-offload-implementer-2026-05-08.md` precedent set by Ch11
   brief) format with verified source surface table + outline-vs-source
   mismatches + candidate language traps + demo plan is now operationally
   reproducing zero-patch handoffs. Pattern: **briefs that pre-verify
   outline-vs-source mismatches at the source commit before dispatch
   produce zero-patch implementer artifacts**. Document in dcp-pcp.md
   D27 as a candidate cross-chapter pattern.

2. **Comm-side source surface scales without losing focus.** Ch08 (8 TP
   files) → Ch09 (10 EP files) → Ch10 (11 spec-decode files) → Ch11
   (12 DCP/PCP comm-side files). The Ch11 surface is **structurally
   different** — it spans distributed comm primitives
   (`dcp_alltoall.py`), KV-cache layout invariants
   (`kv_cache_interface.py`), MLA backend plumbing (`flashattn_mla.py`,
   `rocm_aiter_mla.py`), multiproc executor scaffolding
   (`multiproc_executor.py`), and 5D mesh construction
   (`parallel_state.py`). The 5-step rhythm and two-tier mapping survive
   the breadth — every section opens with a specific file:line, every
   file gets a mini-mapping table, the 81-row master + 16-row
   cross-chapter index the chapter.

3. **Outline-as-topic-not-class-contract pattern proves at scale.**
   Ch07 (radix-tree topic) → Ch08 (TensorParallel-class) → Ch09
   (ExpertParallel-class + ExpertLoadBalancingLoss-training) → Ch10
   (MultiTokenPrediction-class + MTP-CE-loss-training) → Ch11
   (RingAttention-class + AG+RS-vs-A2A-correction + 5D-not-3D-mesh-correction).
   Outline JSON unchanged in all five cases; chapters honestly represent
   source reality. **Ch11 has TWO surgical corrections in one chapter**
   (AG+RS vs A2A; 5D mesh) — the topic-not-contract rule scales to
   multiple corrections per chapter.

Beyond metrics, four patterns reproduce that lock in v6 robustness:

4. **The "no class X" framing pattern reaches 5 instances within vllm
   (graduated motif) — but wisdom-promotion gate of 2+ INSTANCES is
   STILL NOT met.** Ch07 "no radix tree" → Ch08 "no class TensorParallel"
   → Ch09 "no class ExpertParallel" → Ch10 "no class MultiTokenPrediction"
   → Ch11 "no class RingAttention". All 5 within ONE instance (vllm).
   Per `feedback_wisdom_gate_strict` and CLAUDE.md, the gate is "2+
   INSTANCES" (different repo books), not "N within one". **The motif is
   documented in `knowledge/modules/dcp-pcp.md` as chapter-internal
   motif (D26) and is NOT promoted to `wisdom/`.** The strict gate
   protects wisdom from instance-specific noise; without a 2nd repo book
   that hits the same outline-vs-source pattern, the motif could be a
   vllm-architectural quirk rather than a universal pattern. Future
   instances will resolve this question.

5. **§11.4 reproduces Ch10 §10.3's pattern of in-chapter outline
   correction.** Ch10 §10.3 corrected MTP-CE-loss-training as
   inference-only-not-training. Ch11 §11.4 corrects AG+RS-vs-A2A and
   §11.6 corrects 5D-not-3D-mesh. Pattern: **outline subsection names
   describe questions/topics, not class names or implementation details
   — verify at source-commit before dispatch**. This is exactly what
   `feedback_outline_topic_not_contract` codifies; Ch11 is now the third
   strong instance of in-chapter surgical correction.

6. **Tester framing-guidance loop reproduces (N=6).** Ch06 → Ch07 → Ch08
   → Ch09 → Ch10 → Ch11. Each tester's surgical tips applied by writer
   and verified by reviewer with explicit three-anchor (hook + body +
   recap) check, now formalised as the **D28 rule** in dcp-pcp.md.
   Six chapters in a row — testers consistently produce load-bearing
   narrative-shaping guidance, and writers/reviewers consistently apply
   it with three-anchor discipline.

7. **APPROVED-at-cycle-1 discipline holds across all 8 v6 chapters.**
   Ch04 through Ch11 all 1-cycle. Pipeline cost is predictably low.
   Reviewer-3 introduced the **D28 three-anchor mechanical grep
   verification rule** which formalises the M29 / E22 lineage as a
   reviewer-checkable gate.

8. **Knowledge file growth matches Ch10 (29 vs 30 facts) but compact()
   is NOW operational.** Ch11 is the first chapter where P1-1 (compact()
   brokenness) is fixed — recommend immediate compaction of dcp-pcp.md to
   bring under 15-fact threshold. This unblocks automated knowledge
   hygiene that has been blocked across 5 prior chapters.

cadence_holds_at_n8 unlocks the inference: v6 is robust enough to walk
**the broadest comm-side surface yet** (12 vLLM modules across 5 distinct
distributed-systems territories) in a single cycle WITH zero implementer
patches. Pipeline can plan Ch12 (kv-offload) → Ch13 (prefix-cache-pooling)
on this basis. The "no class X" pattern is firmly established as a
chapter motif within vllm but the strict 2+ INSTANCES wisdom-promotion
bar holds. The §11.4/§11.6 dual-outline-correction pattern proves the
outline-as-topic-not-contract rule scales to multi-correction chapters.

## What to remember

Reviewer-3 APPROVED in 1 cycle. Both linters PASS — formula 0 blocking +
15 non-blocking inline-density warnings (M30-acceptable, edge-of-band);
source-grounding all PASS. **474/474 tests pass** at commit 98661fe (52%
test count increase over Ch10's 311). **1394 lines, 8124 words, 149
mapping rows** (81 main + 16 cross-chapter + 7 per-section mini —
two-tier per K15 with broadest comm-side surface yet). **12 vLLM source
files cited** (parallel_state, dcp_alltoall, v1/attention/backend,
v1/attention/backends/utils, v1/attention/backends/mla/flashattn_mla,
v1/kv_cache_interface, v1/executor/multiproc_executor, config/parallel,
flashinfer, mla/rocm_aiter_mla, moe_runner, flash_attn) — 78 REFERENCE
comments, lower than Ch10's 151 only because Ch10 was inflated by
proposer-family boilerplate. **§11.2 "no class RingAttention" reframe**
is the **5th instance** of "no class X" within vllm — graduates to
chapter motif but **wisdom-promotion gate (2+ INSTANCES) NOT yet met**;
documented as chapter-internal motif in dcp-pcp.md D26, NOT promoted to
wisdom/. **§11.4 AG+RS vs A2A** surgical correction of outline §11.3
("all-reduce vs all-to-all" was wrong) and **§11.6 5D mesh** surgical
correction of outline §11.5 ("3D 并行" was wrong; actual is
`external_dp × dp × pp × pcp × tp` with DCP folded inside TP) — both
without modifying outline JSON. **HBM correction**: 40.0 GB exact (NOT
33.5 GB stale in earlier impl-notes rev) — formula
`128K × 80 × 8 × 128 × 2 × 2 = 40.0 GB exact`. 5 framing tips from tester
applied with three-anchor verification (D28 rule). 7 language-trap
callouts (A-G) matching Ch09/Ch10. Forward-pointers to Ch12+/Ch13/Ch22-Ch25
wired. Knowledge D01-D29 in dcp-pcp.md (29 facts, 14 over 15-trigger;
**P1-1 compact() now operational — recommend
`python3 scripts/learn.py compact dcp-pcp`**). **CLEANEST
IMPLEMENTER→TESTER HANDOFF IN BOOK** — zero patches required during
testing. **cadence_holds_at_n8** with broadest comm-side surface yet —
v6 robust at N=8; not just holding but actively scaling source breadth
(12 files vs Ch10's 11) and test breadth (474 vs 311) without dilution.
**Brief-on-approval discipline triggers Ch12 (kv-offload) brief
immediately** — likely 6th "no class X" candidate pending source
verification at commit 98661fe (KV-offload semantics + connector taxonomy
+ v0/v1 divergence at `vllm/distributed/kv_transfer/`).
