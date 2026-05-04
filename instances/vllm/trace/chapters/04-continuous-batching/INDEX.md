# Chapter 04 — Continuous Batching: Trace Index

## Delivery

- **Date**: 2026-05-04
- **Status**: published (v5 rewrite)
- **Gate results**: implementation_exists=true, tests_pass=true, narrative_complete=true, review_approved=true
- **Tests**: 14/14 passed
- **Formula lint**: 0 blocking issues

## Decisions

| Date | Decision | Made by |
|------|----------|---------|
| 2026-05-04 | Bubble proof placement in Cell 4 (before walkthrough) | writer |
| 2026-05-04 | Token budget `B=512` in simplified implementation | implementer |
| 2026-05-04 | `_preempt_request` must NOT call `self.running.remove()` — caller pops | implementer (bug fix) |

## User Interactions

| Date | Interaction | Response |
|------|-------------|----------|
| 2026-05-04 | User requested v5 rewrite: code walkthrough + bubble proof + source trail | Full rewrite executed |

## Session Backups

| Date | Agent | Transcript |
|------|-------|-----------|
| 2026-05-04 | implementer | [sessions/2026-05-04_implementer.json](sessions/2026-05-04_implementer.json) |
| 2026-05-04 | writer | [sessions/2026-05-04_writer.json](sessions/2026-05-04_writer.json) |
| 2026-05-04 | reviewer | [sessions/2026-05-04_reviewer.json](sessions/2026-05-04_reviewer.json) |

## Key Artifacts

- Implementation: `artifacts/04-continuous-batching/implementation/scheduler.py` (748 lines, 22 REFERENCE comments)
- Tests: `artifacts/04-continuous-batching/tests/test_scheduler.py` (14 tests)
- Narrative: `artifacts/04-continuous-batching/narrative/chapter.md`
- Review: `artifacts/04-continuous-batching/reviews/review-report.json`

## Knowledge Generated

See `instances/vllm/knowledge/modules/scheduler.md` for repo-specific facts discovered.
See `wisdom/testing.md` W02 for the preemption test pattern (promoted to wisdom).
