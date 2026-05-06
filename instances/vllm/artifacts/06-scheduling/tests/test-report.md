# Test Report — Ch06 Request Scheduling Policy

**Tester**: tester@book-factory
**Date**: 2026-05-06
**Source commit**: `98661fe`
**Verdict**: APPROVED → handoff to Writer (Task #13)

## Summary

```
97 passed, 0 failed in 0.16s
```

| Module | Tests | Status |
|---|---|---|
| `test_request_queue.py` | 22 | PASS |
| `test_policy.py` | 16 | PASS |
| `test_preemption_strategy.py` | 17 | PASS |
| `test_starvation_analysis.py` | 14 | PASS |
| `test_pareto.py` | 16 | PASS |
| `test_integration.py` | 12 | PASS |

Run from chapter dir:

```bash
cd instances/vllm/artifacts/06-scheduling
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

## Demo numerics — every headline number reproduced exactly

| Quantity | Demo | Test |
|---|---|---|
| FCFS victim (§1) | C | C |
| PRIORITY victim (§1) | B | B |
| Pause budget table (§1) | 2048/2048/0 | 2048/2048/0 |
| Skipped beats waiting (§1) | yes (FCFS) | yes |
| Head-of-line factor (§2) | 8.00x | 8.00x |
| has_starvation FCFS (§2) | True | True |
| has_starvation PRIORITY (§2) | False | False |
| Recompute 8K (§3) | 163.8 ms | 163.84 ms |
| Swap 8K (§3) | 62.5 ms | 62.5 ms |
| Abort (§3) | 5000 ms | 5000 ms |
| Crossover length-independent (§3) | True (P02) | verified for L=128, L=131072 |
| E[recompute] @ p=0.05/100 steps (§3) | 983 ms | 983.04 ms |
| E[swap] (§3) | 476 ms | 476.34 ms |
| E[abort] (§3) | 25164 ms | 25163.84 ms |
| Priority order (§4) | [B,D,C,A] | [B,D,C,A] |
| FCFS arrival order (§4) | [A,B,C,D] | [A,B,C,D] |
| aged_priority(A, t=10) (§4) | -7 | -7 |
| Pareto front size (§5) | 1 point | 1 point |
| P95 TTFT improvement factor (§5) | 16x (800ms → 50ms) | 16.0x exact |

## Coverage by behavior class

1. **Queue family** (vLLM `request_queue.py:L1-L208`):
   - `PolicyRequest.__lt__` 4-tier ordering — priority, arrival, request_id, id() — pinned in 4 unit tests.
   - `FCFSRequestQueue`: FIFO add/pop, peek-empty raises IndexError, prepend appendleft puts request at head, iter doesn't consume.
   - `PriorityRequestQueue`: heap pop returns smallest priority first, arrival breaks ties, **K18 invariant** (`prepend_request == add_request`), iter pops a copy (heap stays intact), determinism across reseeds (3 distinct insertion orders → identical pop order).
   - `create_request_queue` factory: dispatches by enum, raises on unknown.

2. **Policy primitives** (`scheduler.py:L478-L504, L1567-L1577, L372-L374`):
   - **Victim selection**: FCFS pops `running[-1]` (last admitted); **K19** PRIORITY uses `max(priority, arrival)` — picks WORST priority value (not best). Tie-break: latest arrival.
   - **Waiting-queue selection**: **K20** FCFS unconditionally prefers `skipped` over `waiting`; PRIORITY peeks both heads and picks the better one. Returns `None` when both empty.
   - **Pause-state gating**: `PAUSED_ALL` zeros budget, `PAUSED_NEW`/`UNPAUSED` unchanged.

3. **Preemption strategies** (analytical):
   - KV-bytes formula: `2 * NL * NH * D * dt * L`; 8K/32/8/128/fp16 = 1 GiB exact.
   - Demo §3 table reproduces all 4 prompt sizes × 3 strategies.
   - **P02 length-independence**: `crossover_prompt_length(L=128)` = `crossover_prompt_length(L=131072)` for same model shape — confirmed by direct test, plus the threshold-TP arithmetic (`32 GiB / 256 KiB = 131072 tok/s`) verified.
   - `expected_latency_under_oom_rate` formula matches demo's 983/476/25164 ms.

4. **Starvation analysis**:
   - `WorkloadProfile.head_of_line_blocking_factor` = `prefill_steps / own_steps` matches demo's 8.00x.
   - `has_starvation` predicate: True under FCFS when factor > 1, False under PRIORITY (assumes well-assigned).
   - `priority_ordering` reproduces demo §4 order [B,D,C,A]; doesn't mutate caller's list.
   - `aged_priority` formula: `priority - rate * max(0, now - arrival)`. Pedagogical only — vLLM does NOT auto-age.

5. **Pareto frontier**:
   - `estimate_p95_ttft`: with `long_prefill_token_threshold` set, short request fits in `B - threshold` leftover → 50ms. Without, scales as `ceil(L / B) * 50ms`.
   - `pareto_front` dominance: A dominates B iff A has higher tput AND lower TTFT.
   - **The 16x claim**: 800ms (max_seqs=8/B=512) → 50ms (max_seqs=128/B=8192) verified exactly.
   - The demo's threshold-row design (max_seqs=32, B=2048, threshold=512) compresses to 50ms because leftover=1536 fits short(64) in 1 step.

6. **Integration**:
   - All 5 demo sections replay end-to-end.
   - **Cross-chapter regression**: Ch04 `implementation.scheduler` and Ch05 `implementation.block_pool` both importable alongside Ch06 in the same Python session (handles the `implementation` package-name collision gracefully).
   - **Invariant summary suite**: dedicated tests for K18, K19, K20, P01, P02 — explicit pin-points for future regression checks.

## Knowledge applied

- **scheduler.md K18-K20** and **preemption.md P01-P03** — implementer-supplied. Tests pin every quantitative claim.
- **K15 (pytest legacy exclusion)**: `tests/pytest.ini` + `--ignore=tests/_legacy` flag. Belt-and-suspenders pattern from Ch04 reused.

## Wisdom applied

- **W02 (preemption-test design)** doesn't apply directly (Ch06 has no live preemption flow — the flow lives in Ch04). The broader principle "don't let the test pass for the wrong reason" is honored: every demo number is asserted exactly. The `test_length_independent` check is the cleanest application — instead of asserting "swap is faster", it asserts "the dispatch is the same at L=128 and L=131072", which would fail loudly if anyone refactored the formula away from L-cancelation.

## Reference count observation

Implementer reported 60 `# REFERENCE:` comments at the baseline floor (Ch04=65, Ch05=61). Per-module check:
- `request_queue.py` — heavy (every method tagged).
- `policy.py` — heavy (each primitive tagged).
- `preemption_strategy.py` — moderate; covers vLLM L952-L972 + kv_offload + finish_requests.
- `starvation_analysis.py` — moderate; analytical model derived from L387-L556 + L296-L307.
- `pareto.py` — light (~5 references) — JUSTIFIED, this module is analytical synthesis, not a 1:1 port. Annotated as such in module docstring.

**Not flagged as a fidelity concern.** All non-port modules are clearly marked. Tests anchor each demo number to a vLLM line via REFERENCE comment in the test docstring or assertion comment.

## Fidelity findings

**None.** The module's biggest source-grounded surface (request_queue.py + policy.py = 8 functions/classes) tests 1:1 against vLLM source line ranges. Analytical surfaces (preemption_strategy, starvation_analysis, pareto) are clearly marked as such in implementer's docstrings; tests verify the formulas reproduce the demo's numbers, which is the right level of fidelity for a policy/strategy chapter.

## Backpressure gate

OPEN. Writer (Task #13) is clear to start.
