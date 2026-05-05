# Knowledge: Scheduler (vLLM)

Repo-specific facts about `vllm/v1/core/sched/scheduler.py` and surrounding code.

---

## K01: Main schedule() entry point

**Module**: scheduler
**Chapter**: 04-continuous-batching, 06-scheduling
**Discovered by**: implementer
**TTL**: permanent (core fact)
**Access count**: 5

`scheduler.py:L352` is the main `schedule()` entry. The method's own comment says:
"There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just
has num_computed_tokens and num_tokens_with_spec."

**Implication**: Continuous batching is NOT a separate mode — it naturally falls out
of treating all tokens uniformly. Each request requests N tokens per step, and the
scheduler grants up to the token budget.

---

## K02: _preempt_request — caller pops from running

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Discovered by**: implementer (bug fix)
**TTL**: permanent
**Access count**: 4

`scheduler.py:L952-L972`: `_preempt_request()` does NOT remove the request from
`self.running`. The docstring says: "NOTE: The request should be popped from the
running queue outside of this method."

The caller (`schedule()` at L503-L504) does `self.running.pop()` BEFORE calling
`_preempt_request()`. Adding `self.running.remove(req)` inside `_preempt_request`
causes `ValueError: list.remove(x): x not in list`.

**For testers**: Preemption tests must ensure both requests fit initially, THEN trigger
preemption when one needs more resources.

---

## K03: Token budget B = max_num_scheduled_tokens

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Discovered by**: implementer
**TTL**: permanent
**Access count**: 3

`scheduler.py:L264`: `token_budget = self.max_num_scheduled_tokens`. Default is 2048
(or 512 in our simplified implementation). This budget caps how many tokens can be
scheduled per step across ALL requests.

**For writers**: Use `B` for the token budget in formulas. The reader should understand
that B << total_tokens for realistic workloads.

---

## K04: Phase 1 (running requests) before Phase 2 (waiting)

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Discovered by**: implementer
**TTL**: permanent
**Access count**: 3

`scheduler.py:L388-L556` (Phase 1, running) runs before `L568-L846` (Phase 2, waiting).
This is the "running-first" principle: already-admitted requests get priority over
new requests. This prevents starvation and minimizes preemption churn.

**For writers**: This is a key design insight worth explaining in Cell 4 (theory).

---

## K05: Chunked prefill logic

**Module**: scheduler
**Chapter**: 04-continuous-batching (chunked-prefill)
**Discovered by**: implementer
**TTL**: 60 days
**Access count**: 2

When `enable_chunked_prefill=True`, long prompts are split across multiple steps.
The scheduler caps `num_new_tokens` at `token_budget` for waiting requests.
Without chunked prefill, a request that can't fully fit in the budget is deferred.

**Key code**: `scheduler.py:L682-L692` (chunked prefill decision) and `L684-L690`
(when chunked prefill is disabled, skip requests that don't fit).

---

## K06: FCFS scheduling order

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Discovered by**: implementer
**TTL**: 60 days
**Access count**: 2

vLLM uses First-Come-First-Served (FCFS) by default. The running list order = arrival
order. The LAST request in `self.running` is the lowest priority for preemption.
This is why `self.running.pop()` preempts the lowest-priority request.

**For testers**: When testing preemption, the request added LAST should be preempted first.

---

## K07: Config file path — no /v1/ directory

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Discovered by**: reviewer (source grounding spot-check)
**TTL**: 60 days
**Access count**: 1

The scheduler config is at `vllm/config/scheduler.py`, NOT `vllm/v1/config/scheduler.py`.
The `/v1/` prefix applies to `core/sched/` (scheduler.py, output.py) and `request.py`, but
NOT to the `config/` directory. The config tree lives directly under `vllm/config/`.

**For writers/reviewers**: Always verify source paths against the actual repo. The source
grounding linter doesn't verify individual file:line references in narrative prose — it
only checks implementation REFERENCE comments and source mapping rows.

**Key defaults in config/scheduler.py**:
- L56: `max_num_scheduled_tokens: int | None = None` (computed dynamically at runtime)
- L63: `max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)`
- L84: `enable_chunked_prefill: bool = True`
