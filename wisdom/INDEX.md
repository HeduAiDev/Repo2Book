# Wisdom Index — Universal repo2book Patterns

These patterns apply across ALL book projects. They are gated: must appear in 2+ repos before promotion.

| # | Category | Pattern | Roles | Confirmed In |
|---|----------|---------|-------|-------------|
| W01 | debugging | F.linear weight shape is [out, in] | impl, tester | vllm, pytorch |
| W02 | testing | Preemption tests: both requests must fit initially | tester | vllm |
| W03 | writing | Numerical trace: 2+ iterations, ALL intermediates | writer, reviewer | vllm |
| W04 | architecture | Backpressure gates prevent cascading errors | all | vllm |
| W05 | debugging | Docker vs local: CUDA version mismatch causes silent failures | tester, impl | vllm |
| W06 | writing | Formula lint: \text → \mathrm (blocking) | writer, reviewer | vllm |
| W07 | writing | Code walkthrough line numbers: read implementation FIRST | writer | vllm |
| W08 | architecture | Lateral communication: Reviewer↔Writer bypasses Lead | reviewer, writer | vllm |
| W09 | writing | 大白话 not 书面语 per cell formality spectrum | writer | vllm |
| W10 | testing | ref_cnt = -1 means "not ready" in cache pools | tester, impl | vllm |
| W11 | debugging | SVG text clipping: text-anchor="end" at x < 30 | writer | vllm |
| W12 | architecture | Chapter pipeline: emit skeleton + 4 gates → publish | all | vllm |

## Category Quick Reference

- **debugging**: Patterns that cause silent failures, wrong results, or confusing errors
- **testing**: Test design patterns, edge case catalogs, Docker/container quirks
- **writing**: Narrative structure, formula rules, diagram guidelines, style rules
- **architecture**: Pipeline design, agent communication, gate design, orchestration

## Role Quick Reference

| Role | Query Priority |
|------|---------------|
| implementer | debugging > architecture > testing > writing |
| tester | testing > debugging > architecture > writing |
| writer | writing > architecture > debugging > testing |
| reviewer | writing > architecture > testing > debugging |
| book-editor | architecture > all others equally |
