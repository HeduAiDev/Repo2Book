# Ch07 Prefix Cache 与 APCAwareAllocation v6 PUBLISHED — fourth v6 chapter, cadence holds at N=4 on widest source surface yet

- **Type**: delivery
- **Chapter**: 07-prefix-cache
- **Date**: 2026-05-06
- **Timestamp**: 2026-05-06T12:47:15Z
- **Agents involved**: implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: v6, prefix-cache, radix-tree-reframe, language-trap, two-tier-mapping

## What happened

Reviewer-3 APPROVED in 1 cycle (no REVISE iterations). Both linters PASS at the
BLOCKING bar: formula linter 0 blocking issues with 5 non-blocking inline-density
warnings (L32, L86, L88, L628, L841 — K14-aligned, writer applied mitigation but
proof paragraphs in §7.1.2 / §7.5.4 still pile symbols; tolerable for math-heavy
proofs); source-grounding linter all PASS.

83/83 tests pass at vLLM source commit `98661fe`. Demo numerics reproduce verbatim
in narrative (every ground-truth number from `tests/test-report.md` greps clean):
9.3% / 49.5% / 88.2% hit-rate sweep; 373348/s hash table vs 80291/s Trie vs 77232/s
RadixTree microbench; 4.65× speedup explicitly framed as Python interpreter
overhead (NOT asymptotic — K12 anti-pattern enforced); naive=2000 vs prefix-aware=432
KV blocks; saved=1568=49×32; ratio=0.784 (78%).

**Stats**: 859 lines, 4440 words, 72 mapping rows total = 27 main (§7.6.3) + 45
mini-rows (§7.1.5 hash primitives 6, §7.4.5 cache_blocks lifecycle 6, §7.3.3
4-variant find_longest 4, §7.5.4 sweep verify 3, demo §1 hit-rate 6, §7.4.3
three-state lifecycle 3, plus header rows). impl-notes "Source Analysis" lists
5 distinct vLLM source files (kv_cache_utils.py, block_pool.py,
single_type_kv_cache_manager.py, kv_cache_manager.py, request.py) all cited with
`:Lxxx` references multiple times in narrative. ~60 # REFERENCE: comments across
6 impl modules (block_hash, prefix_cache_index, radix_tree, prefix_cache_manager,
paged_integration, demo).

5-step rhythm verified §7.1-§7.5 by reviewer (gate 4): each section opens with
source location → bridge → theory → impl → diff. Two-tier mapping (K15) reused
from Ch06 baseline and extended: main 27-row + per-section mini-tables anchored
to the section's source walk.

**§7.2 reframe applied** as designed in the implementer brief: outline subsection
"Radix Tree 数据结构详解" was reframed at chapter level as "为什么没用 radix tree:
链式 hash 替代方案的设计权衡". The hook (L24) opens with the radix-tree absence
fact — "整个 vLLM v1 源码树里搜不到一个 `class.*Radix|Trie|PrefixTree`
（commit 98661fe 实测 0 个匹配）" — and §7.2.1 cites `block_pool.py:L34-L127`
to show the dict-not-tree implementation. The pedagogical `radix_tree.py` module
is framed as a "教学对照 foil" that quantifies what vLLM avoided rather than
what it uses. This validates outline-as-topic-not-class-contract for chapters
where the source surface differs from the textbook expectation.

**4 language-trap callouts** explicitly enumerated (Trap 1: vLLM 用 radix tree;
Trap 2: cache hit on partial block; Trap 3: cache hit makes request free;
Trap 4: need exhaustive search) at hook + per-section + dedicated recap §7.6.4.
Pattern matches Ch06 (4 traps) and exceeds the v6 floor (≥1).

**Chain-break framed as THE invariant** (gate 8): hook L39 sets it up; §7.3.5
demo trace L383-L408 walks chain-break step-by-step; L410 explicit framing
"链断裂不是一个 bug，它就是 prefix cache 的核心不变量 I3"; §7.6.1 invariants I1/I2/I3
trio; closing summary L843-L847. Tester's framing tip surgically applied.

**(N-1)×K formula leads §7.5** (gate 6, K13 anti-pattern): §7.5.4 L613 prose
intro "先讲公式，再讲数字"; L616-L628 derives general formula `saved=(N-1)×K`;
L630 plugs demo numbers AFTER the general formula; L651 explicit K13 callout.

**Knowledge appended**: K14 (lint_formulas non-blocking density warnings flag
real readability risk — writer-discovered), K15 (two-tier mapping table layout
as v6 standard for broad source surface — writer-discovered, lifted from Ch06).
Total `prefix-cache.md` now has K01-K17 (17 facts) — exceeds the 15-fact
compaction trigger; oldest-5 LLM-compaction may be due before next session.
Reviewer also added K16 (hard-gate review checklist) and K17 (APPROVED-at-cycle-1
discipline) during this review.

**Minor non-blocking observations** (do NOT trigger REVISE, captured by reviewer
for writer's future awareness): L725 backtick/dollar typo (cosmetic, lint-clean);
§7.6.4 trap headers mix Chinese label + English content (consistent with
impl-notes language traps section); 5 inline-density warnings already cited.

Source pinned at vLLM commit `98661fe`. Snapshot location TBD (matching Ch06
pattern `trace/snapshots/06-scheduling/v6-2026-05-06/` if archivist creates one).

## Why it matters

**FOURTH chapter under v6 standards** — cadence holds at N=4. Critically, Ch07
is the **widest source surface** v6 has tackled (5 distinct vLLM files, 5 cited
modules in impl-notes vs Ch04's 5 / Ch05's 7 / Ch06's 6) AND the most
misconception-loaded topic ("vLLM uses radix tree" is a literature-driven false
expectation). Single-cycle APPROVE on this combo confirms:

1. **v6 is robust against high-misconception chapters**, not just topics where the
   source maps cleanly to textbook structures. The implementer brief's
   "verify source surface FIRST, reframe outline as needed" pattern (Ch07 §7.2
   radix-tree-reframe) is now validated as a v6 procedural standard.

2. **Two-tier mapping (K15) is reproducible at high density**. Ch06 introduced
   the pattern at 40 rows (29 main + 11 mini); Ch07 extended to 72 rows
   (27 main + 45 mini-rows in 6 mini-tables). The pattern scales — readers can
   anchor mini-tables to section source walks while keeping the main table as
   chapter-level index.

3. **Language-trap callouts are now a v6 invariant for misconception-heavy
   chapters**. Ch06: 4 traps; Ch07: 4 traps (plus dedicated §7.6.4 recap).
   Pattern locks in: hook + per-section + recap = three-tier defense against
   reader misconceptions.

4. **Tester framing-guidance loop reproduces** (Ch06 → Ch07 second instance):
   tester's K11 (chain-break test pattern), K12 (Python overhead, not asymptotic),
   K13 ((N-1)×K formula leads numerics) all surgically applied by writer and
   verified by reviewer. The Knowledge-as-narrative-shaping-guidance pattern
   is now N=2 within instance — STILL not eligible for wisdom promotion per
   strict 2+ repos rule (this is intra-instance reproducibility).

5. **APPROVED-at-cycle-1 discipline (K17)** holds when writer pre-runs both
   linters and reports outputs in handoff. Ch04, Ch05, Ch06, Ch07 all 1-cycle.
   Pipeline cost is now predictably low when writer follows the lint-first
   protocol.

6. **Outline-as-topic-not-class-contract** (rule #6 from session pause) is now
   operationally proven on Ch07's radix-tree subsection. The outline JSON was
   NOT updated mid-run; the chapter framed the absent topic as "why isn't this
   here?". This protects outline JSON stability while letting chapters honestly
   represent source reality.

Cadence_holds_at_n4 unlocks the inference: v6 is reliable enough that future
chapters can be expected to deliver in 1 cycle when writer + tester +
implementer pre-flight cleanly. Pipeline can plan accordingly.

**No framework bugs surfaced this chapter** (the learn.py append-mode and
_parse_module_file bugs were already known; they remain open with manual
workarounds in use).

## What to remember

Reviewer-3 APPROVED in 1 cycle. Both linters PASS (formula 0 blocking + 5
non-blocking density warnings; source-grounding all PASS). 83/83 tests pass at
commit 98661fe. 859 lines, 4440 words, 72 mapping rows (27 main + 45 mini in
6 mini-tables — two-tier per K15). 5 vLLM source files cited; ~60 REFERENCE
comments across 6 impl modules. §7.2 radix-tree reframe applied successfully —
outline JSON unchanged, chapter honestly frames absent class as "why not?".
4 language-trap callouts + dedicated §7.6.4 recap. Chain-break = THE invariant
threaded through hook + demo + closing. (N-1)×K formula leads §7.5 (K13).
Knowledge K14 (lint density), K15 (two-tier mapping), K16 (review checklist),
K17 (approve-at-cycle-1 discipline) all appended. Cadence_holds_at_n4 — v6
robust on widest+most-misconception-loaded chapter to date. Pipeline pattern
of pre-run-linters-in-handoff confirmed across all 4 v6 chapters. Forward to
Ch08 (Tensor Parallelism) — verify source surface BEFORE implementer dispatch
per Ch07 lesson; outline subsection names are topic guides, not class contracts.
