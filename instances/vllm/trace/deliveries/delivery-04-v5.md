# Delivery: Chapter 04 — Continuous Batching v5 Rewrite

- **Chapter**: 04-continuous-batching
- **Date**: 2026-05-04
- **Status**: published
- **Gate results**: implementation_exists=true, tests_pass=true, narrative_complete=true, review_approved=true

## What was delivered

Complete v5 rewrite of Chapter 4 (Continuous Batching) to the latest writing standard.

### Implementation (Implementer)
- File: `artifacts/04-continuous-batching/implementation/scheduler.py` (748 lines, 22 REFERENCE comments)
- Key bug fixed: `_preempt_request` was calling `self.running.remove(req)` but caller already popped
- All REFERENCE comments verified against vLLM source at `vllm/v1/core/sched/scheduler.py`

### Tests (Tester)
- File: `artifacts/04-continuous-batching/tests/test_scheduler.py` (14 tests)
- Fixed `test_preempt_lowest_priority_when_oom` — both requests now fit initially before preemption triggers
- Result: 14/14 passed in 0.14s

### Narrative (Writer)
- File: `artifacts/04-continuous-batching/narrative/chapter.md`
- Bubble analysis with formal mathematical proof
- Line-by-line code walkthrough with verified line numbers
- Numerical trace with concrete 3-request example
- Diagram: `diagram_bubble.png` (static vs continuous batching comparison)
- Formula lint: 0 blocking issues, 21 non-blocking warnings

### Review (Reviewer)
- File: `artifacts/04-continuous-batching/reviews/review-report.json`
- 9/10 dimensions pass, 1 BLOCKING formula issue (`\text{ 步}`) → fixed by Writer
- Verdict: APPROVED after fix

## Key decisions made during this chapter

1. **Bubble proof placement**: Decided to include formal proof in Cell 4 rather than Cell 7 (numerical example). Proof needs to come before implementation walkthrough.
2. **Token budget in formulas**: Use `B` for token budget consistently. `B=2048` in vLLM, `B=512` in our simplified implementation.
3. **Preemption test design**: Both requests must fit initially, THEN trigger preemption. This is a universal testing pattern → recorded in `wisdom/testing.md`.

## Context snapshot for future agents

When working on related chapters (06-scheduling, 22-pd-architecture), read `knowledge/modules/scheduler.md` for repo-specific facts. The wisdom entries W02 (preemption tests) and W04 (backpressure gates) apply.

## User interactions during this chapter

- User requested v5 rewrite with code walkthrough + bubble proof
- User caught `\text{}` formula issue → led to Reviewer formula dimension
- User requested lateral communication (Reviewer→Writer without Lead)
