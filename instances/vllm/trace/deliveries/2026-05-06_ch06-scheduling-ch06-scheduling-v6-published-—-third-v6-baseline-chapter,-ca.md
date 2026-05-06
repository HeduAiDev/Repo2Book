# Ch06 scheduling v6 PUBLISHED — third v6-baseline chapter, cadence holds at N=3

- **Type**: delivery
- **Chapter**: 06-scheduling
- **Date**: 2026-05-06
- **Timestamp**: 2026-05-06T06:11:51Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: 

## What happened

Reviewer-2 APPROVED in 1 cycle. Both linters PASS (formula 0 blocking, 1 non-blocking inline-density on line 111 — W06-allowed). 97/97 tests pass in 0.11s. Demo numerics reproduce verbatim: 8.00x head-of-line factor, recompute/swap/abort latencies at 8K (164/62/5000ms), [B,D,C,A] priority order, 16x p95 TTFT (800→50ms). §6.6 main mapping table 29 data rows + §6.3.5 mini-table (6 rows) + §6.5.5 mini-table (5 rows) = 40 total source crossrefs. impl-notes lists 8 source-file references (6 distinct files). 60 # REFERENCE: comments across 6 impl modules. 5-step rhythm verified §6.1-§6.6. Scope discipline maintained: §6.1 references Ch04 §4.3.2 by section number, does NOT re-derive mechanics. Tester framing guidance applied with surgical precision: §6.3.2 'swap-faster, recompute-simpler' (no 'recompute is faster'); §6.4.1 K18 invariant first; §6.5.3 P05 sweep-pair framing for 16x. Strong language-trap callouts at §6.3.2/§6.3.4/§6.4.5/§6.5.4 prevent reader misconceptions. §6.5.4 honest framing '1 Pareto point is model artifact, not vLLM truth'. Diagrams correctly omitted — §6.3 7-axis ASCII trade-off matrix and §6.5 Pareto sweep table convey data better than figures. 655 lines, ~3351 words. Source pinned at vLLM commit 98661fe. Snapshot at trace/snapshots/06-scheduling/v6-2026-05-06/.

## Why it matters

THIRD chapter under v6 standards — cadence_holds_at_n3. Ch06 strictly exceeds Ch04+Ch05 baselines on mapping table density (29 main + 11 mini = 40 vs Ch04: 13, Ch05: 21). Confirms: (a) v6 is robust at N=3, not just bouncing between two outliers; (b) tester framing-guidance pattern (P05 sweep-pair, K18 invariant-first) is reproducible — sets precedent for future chapter testers to apply Knowledge facts as direct narrative-shaping guidance for the writer; (c) two-tier mapping (main + per-section mini-tables) is a new pattern worth promoting to subsequent chapters where source surface is broad. handoff-protocol candidate now N=3 within instance (Ch04, Ch05, Ch06) — STILL not eligible for wisdom promotion per CLAUDE.md strict 2+ repos rule. Ch06 also surfaced two framework-level bugs in scripts/learn.py (append-mode produces malformed double-prefix headings AND is module-prefix non-aware) — captured in trace/cross-chapter/learn-py-append-id-bug.md.

## What to remember

Reviewer-2 APPROVED in 1 cycle. Both linters PASS (formula 0 blocking, 1 non-blocking inline-density on line 111 — W06-allowed). 97/97 tests pass in 0.11s. Demo numerics reproduce verbatim: 8.00x head...
