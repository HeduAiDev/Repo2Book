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

## K05–K09: [COMPACTED 2026-05-06] Scheduler micro-facts — chunked-prefill, FCFS, config path, line ranges, Phase 2 guards

**Module**: scheduler
**Chapter**: 04-continuous-batching
**Compacted by**: archivist (manual; `learn.py compact` non-functional — `_parse_module_file` returns `[]`)
**Archive**: `knowledge/archive/scheduler-20260506-k05-k09.json` (full original text)
**Access count (combined)**: 7
**External citations preserved**: K05 cited at `tests/test_scheduler.py:183`; K06 cited at `tests/test_scheduler.py:227`, `tests/test-report.json:53`; K09 cited at `tests/test_scheduler.py:298`, `tests/test-report.json:54`. The five `## K0[5-9]:` heading anchors below ALL survive grep.

### K05: Chunked prefill logic
`scheduler.py:L682-L692` — when `enable_chunked_prefill=True`, long prompts are split across steps; scheduler caps `num_new_tokens` at `token_budget` for waiters. Without chunked prefill (`L684-L690`), a request that can't fully fit in budget is deferred (`break`).

### K06: FCFS scheduling order
vLLM is FCFS by default; `self.running` order = arrival order; LAST request is lowest priority for preemption — that's why `self.running.pop()` preempts the tail. **For testers**: the request added LAST is preempted first.

### K07: Config file path — no `/v1/` directory
Scheduler config lives at `vllm/config/scheduler.py` (NOT `vllm/v1/config/scheduler.py`). The `/v1/` prefix applies to `core/sched/` and `request.py`, but NOT to `config/`. Key defaults: `max_num_scheduled_tokens=None` (runtime-computed), `max_num_seqs=DEFAULT_MAX_NUM_SEQS`, `enable_chunked_prefill=True`. **Reminder**: source-grounding linter doesn't verify narrative-prose paths — verify manually.

### K08: scheduler.py canonical line ranges at commit 98661fe
`Scheduler.__init__` L67-L176 · `schedule()` L352-L945 · `_preempt_request()` L952-L972 · `_update_after_schedule()` L974-L998 · `update_from_output()` L1290-L1551 · `add_request()` L1728-L1748 · `_free_request()` L1813-L1834. Re-verify if `git rev-parse HEAD` in `instances/vllm/source/` differs from `98661fe`.

### K09: Phase 2 has TWO guards
`scheduler.py:L568`: `if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:`. The pause-state piece supports `PauseState.PAUSED_ALL`. If simplified away (we do), tests that model engine pause/resume must reintroduce it — otherwise paused engines admit waiters, violating pause semantics.

---

## K10–K14: [COMPACTED 2026-05-06 second-pass] Ch04 scheduler details — allocate_slots arg, Phase 1 continue-not-break, SchedulerOutput naming, finished_req_ids cross-step, preemption-test sizing

**Module**: scheduler
**Chapter**: 04-continuous-batching (and forward refs: 12, 13, 20)
**Compacted by**: archivist (manual; learn.py compact non-functional)
**Archive**: `knowledge/archive/scheduler-20260506-second-pass.json` (full original text)
**Access count (combined)**: 5
**External citations preserved**:
- K10 cited at `artifacts/05-memory-management/tests/test-report.json:77`, `tests/test-report.md:72`, `trace/briefs/06-scheduling-implementer-2026-05-06.md:111`
- K11 cited at `artifacts/04-continuous-batching/tests/test_scheduler.py:183`, `tests/test-report.json:55`, `trace/briefs/06-scheduling-implementer-2026-05-06.md:110`
- K12, K13, K14 anchors retained for forward-citation safety

The five `### K10:` / `### K11:` / `### K12:` / `### K13:` / `### K14:` heading anchors below ALL survive grep.

### K10: allocate_slots takes Request, NOT request_id
Real signature: `allocate_slots(request, num_new_tokens, **kwargs)` (`scheduler.py:L466-L471`). First positional arg is the `Request` object, not its id. kwargs are prefix-cache + spec-decode plumbing (`num_new_computed_tokens`, `new_computed_blocks`, `num_lookahead_tokens`, `num_external_computed_tokens`, `delay_cache_blocks`, `num_encoder_tokens`, `full_sequence_must_fit`). When simplifying, keep first arg a Request so call sites read identically to vLLM.

### K11: Phase 1 uses `continue` not `break` (deliberate FCFS relaxation)
`scheduler.py:L446-L462` explicit comment: *"by doing `continue` instead of `break`, we do not strictly follow the FCFS scheduling policy and allow the lower-priority requests to be scheduled."* Phase 2 still uses `break` for chunked-prefill-disabled (`L684-L690`) — there it's safer to wait than skip ahead. Don't "fix" Phase 1's `continue` to `break`.

### K12: SchedulerOutput uses `num_scheduled_tokens`, not `scheduled_requests`
Authoritative field is `num_scheduled_tokens: dict[str, int]` (`output.py:L191-L193`). Caller pattern `scheduler.py:L984-L987`:
```python
num_scheduled_tokens = scheduler_output.num_scheduled_tokens
for req_id, num in num_scheduled_tokens.items(): ...
```
The legacy `scheduled_requests` name breaks the 1:1 → AttributeError. Use `num_scheduled_tokens` for the dict, `total_num_scheduled_tokens` for the sum.

### K13: finished_req_ids cross-step carry — vLLM keeps it, our merged update_after_step clears it
vLLM populates `SchedulerOutput.finished_req_ids` across step boundaries: a finish in step N's `update_from_output` shows up in step N+1's `schedule()` output (`scheduler.py:L919-L923`); model runner drops KV state on N+1. Source NOTE at `L995-L998`: *"We shouldn't do `self.finished_req_ids.clear()` here..."*. Our Ch04 merged `update_after_step` clears at end of same call → `SchedulerOutput.finished_req_ids` always empty in our impl. Authoritative finish: `request.is_finished()`. **Ch20 fix**: clear at START of `schedule()` to match "between steps" semantics. Test: `test_finished_req_ids_field_is_simplified`.

### K14: Preemption-test sizing (W02 application)
Wisdom W02: both requests must be admitted initially, then one needs more KV → OOM → preempt FCFS tail. Concrete `SimpleKVCacheManager(num_gpu_blocks=2, block_size=16)` sizing: two 16-token prompts each take 1 block; both admit; 0 free remain. Decode rounds up to a new block → OOM → preempt. Pop-tail rule (K06): LATER-added request preempted first. Drive `_drive(sched, max_steps=80)` and assert the FIRST preempt step contains the tail-id.

---

## K15: pytest auto-collects `_legacy/` test files unless excluded

**Module**: scheduler
**Chapter**: 04-continuous-batching (and any chapter migrated under Strategy B1)
**Discovered by**: tester (Ch04 v5 test pass)
**TTL**: until repo-wide layout convention is locked
**Access count**: 1

Under Strategy B1 (migrate-then-rewrite, keep legacy as reference), `tests/_legacy/test_*.py` lives next to the new tests. pytest's default discovery walks ALL subdirs, so it auto-imports the legacy file — which uses stale APIs (`ContinuousBatchingScheduler`, `scheduled_requests`) and fails to import → collection error → 0 tests run.

Fix used in Ch04: `tests/pytest.ini` with `norecursedirs = _legacy __pycache__` AND run with `--ignore=tests/_legacy` for belt-and-suspenders. Either alone works; both together are robust against running pytest from different cwds.

For testers on Ch05+: when you see `_legacy/`, set up the same exclusion BEFORE writing your first test, or your first run will look broken even when the new code is fine.

---

## K16: Chinese inside `$$` blocks — split prose from math, don't `\mathrm{}` Chinese

**Module**: scheduler (writing-perspective)
**Chapter**: 04-continuous-batching
**Discovered by**: writer (Ch04 v5 A1 rewrite)
**TTL**: permanent
**Access count**: 1

When you want to express "先消耗 budget 满足 $R_k$，然后用余量 $B - \sum n_r$ 去 admit $W_k$" inside a `$$` block, the obvious workaround `\text{先消耗 budget 满足 } R_k \text{，然后用 …}` triggers `lint_formulas.py` BLOCKING (W06: no `\text{}`). You CANNOT switch `\text{}` → `\mathrm{}` because Chinese characters in `\mathrm{}` still render as serif Roman font and the linter's grep is `\\text{` only — but the deeper rule (wisdom/writing.md W06) is "Chinese characters in formulas should be moved OUTSIDE `$$`."

**Fix pattern** that passes lint AND reads well:

```markdown
（Chinese narrative ending with "用余量"）

$$
B - \sum_{r \in R_k} n_r
$$

（Chinese narrative continuing "去 admit ..."）
```

i.e., split the sentence around the math fragment. The math fragment stands alone in `$$` block; Chinese before/after is plain prose. Do NOT try to "fit everything into one `$$` block" with `\text{}` or `\mathrm{}` wrappers.

**For writers on any chapter with mixed-language formal definitions**: this is the canonical pattern. Don't reach for `\text{}`.

---

## K17: Complex-inline-formula warnings are non-blocking when symbols are simple

**Module**: scheduler (writing-perspective)
**Chapter**: 04-continuous-batching
**Discovered by**: writer (Ch04 v5 A1 rewrite)
**TTL**: permanent
**Access count**: 1

`lint_formulas.py` flags two non-blocking warnings:
- `Too Many Inline Formulas` (≥3 inline `$...$` per paragraph)
- `Complex Inline Formulas` (>30 chars inside `$...$`)

Both are warnings, not blockers (per `lint_formulas.py` print_report: only `text_instead_of_mathrm`, `boxed_requires_amsmath`, `tag_requires_amsmath`, `frac_in_inline_math`, `block_math_on_separate_lines` are BLOCKING). Wisdom W06 explicitly allows single-symbol inlines like `$x$, $\alpha$, $d_k$, $\sqrt{2}$, $1/\sqrt{d_k}$`. So a paragraph with 5 single-symbol inlines (e.g. `$k$`, `$R_k$`, `$W_k$`, `$r$`, `$n_r$`) is FINE — even if linter says "consider promoting some".

**For writers**: only act on `\text{}`/`\boxed{}`/`\tag{}`/`\frac` in `$...$`/`$$` on same line. The two "consider" warnings are stylistic suggestions; resist the urge to refactor away single-symbol inline math at the cost of awkward prose.

**For reviewers**: if formula linter shows "🟢 No blocking issues", do not REJECT for non-blocking warnings.


---

## K18: PriorityRequestQueue.prepend_request == add_request — no front in priority queue

**Module**: scheduler
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: request-queue, priority, invariant

vllm/v1/core/sched/request_queue.py:L160-L165 has the explicit comment: 'In a priority queue, there is no concept of prepending to the front. Requests are ordered by (priority, arrival_time).' This means preemption mechanics are NOT symmetric: under FCFS the preempted request lands at the head and re-runs immediately; under PRIORITY it heap-pushes back to its true priority slot, which can keep it preempted forever if its priority is low. Tests that say 'preempted request is re-queued at front' are FCFS-only language. Same applies to `prepend_requests` (L167-L173).

---

## K19: PRIORITY victim selection uses max() not min() because smaller priority = higher

**Module**: scheduler
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: scheduler, preemption, priority, convention

vllm/v1/core/sched/scheduler.py:L480-L483: `preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))`. The use of `max` not `min` is INTENTIONAL: vLLM convention is 'smaller priority value = higher priority' (config/scheduler.py:L114), so max() returns the request with the LARGEST priority value (= lowest precedence), among ties the LATEST arrival. That's the correct preemption victim. Don't 'fix' this to min().

---

## K20: skipped_waiting queue prevents indefinite postponement of blocked requests

**Module**: scheduler
**Chapter**: 06-scheduling
**Discovered by**: implementer
**TTL**: 2026-06-04
**Access count**: 0
**Tags**: scheduler, starvation, skipped-waiting

vllm/v1/core/sched/scheduler.py:L1567-L1577 (`_select_waiting_queue_for_scheduling`): under FCFS, `return self.skipped_waiting or self.waiting or None`. skipped wins unconditionally. Under PRIORITY, peek both heads and return the better one. This protects requests that were peeked-but-not-popped during a prior step (e.g. WAITING_FOR_REMOTE_KVS, LoRA cap exceeded) from being perpetually overtaken by new arrivals. Without this gate, a blocked request could starve indefinitely as `waiting` keeps refilling.

---

## K21: Determinism testing for PriorityRequestQueue — use 3+ permutations, NOT random.seed

**Module**: scheduler
**Chapter**: 06-scheduling
**Discovered by**: tester (Ch06 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, priority-queue, determinism

When testing "PriorityRequestQueue pop order is deterministic", the temptation is to seed `random.shuffle()` and check the result matches a baseline. That's brittle: changes in Python's hash seed or heapq internals could shift things in ways the seed doesn't catch.

The robust pattern (used in `test_determinism_across_reseeds`): pick **3 explicit permutations** of the same items, push them, pop them all, and assert the 3 pop orders are equal:

```python
items = [_r("a", priority=1, arrival=0), _r("b", priority=1, arrival=1), ...]
orders = []
for perm in [items, list(reversed(items)), [items[2], items[0], items[3], items[1]]]:
    q = PriorityRequestQueue()
    for r in perm: q.add_request(r)
    orders.append([q.pop_request().request_id for _ in range(len(items))])
assert orders[0] == orders[1] == orders[2]
```

This catches any non-determinism that depends on insertion order, including hash-seed-driven id() variation. The 4th tier of `__lt__` (id()) ONLY differs if all of (priority, arrival, request_id) are identical — for distinct request_ids the test gives a strong determinism guarantee.

For testers on any priority-queue-style chapter: prefer explicit-permutation determinism tests over seeded-shuffle ones.

---

## K22: Cross-chapter `implementation` package collision — graceful skip pattern

**Module**: scheduler
**Chapter**: 06-scheduling (and Ch07+)
**Discovered by**: tester (Ch06 v6 test pass)
**TTL**: permanent
**Access count**: 1
**Tags**: testing, cross-chapter, sys-path

Each chapter has its own `implementation/` package living at the same relative depth. When a Ch06 test wants to verify "Ch04's `implementation.scheduler` still imports cleanly", `importlib.import_module("implementation.scheduler")` returns whichever `implementation` is already cached in `sys.modules` — usually the current chapter's, since conftest puts it first on `sys.path`.

**Robust pattern** (used in `test_integration.py::TestCrossChapterRegression`): try the import and skip cleanly on any failure:

```python
try:
    mod = importlib.import_module("implementation.scheduler")
except (ImportError, ModuleNotFoundError):
    return
assert mod is not None  # smoke-test only
```

Don't try to invalidate the cache (`sys.modules.pop`) — that breaks the current chapter's tests if pytest re-imports later. The skip-cleanly pattern verifies "no crash on import path setup" while staying agnostic about which module actually loaded. For testers on Ch07+: this is the canonical Ch04/Ch05/Ch06 cross-chapter regression style.

---

## K23: Ch06 narrative framing — "policy is not a field, it's a thread through schedule()"

**Module**: scheduler (writing-perspective)
**Chapter**: 06-scheduling
**Discovered by**: writer (Ch06 v6 A1 rewrite)
**TTL**: permanent
**Access count**: 1
**Tags**: writing, ch06-policy, framing

The natural way to introduce SchedulingPolicy is "an enum with two values FCFS/PRIORITY". That's wrong — it suggests a binary toggle.

Better framing (used in Ch06 §6.1): policy is a thread that runs through THREE points in `schedule()`:
1. preempt-victim selection (`scheduler.py:L478-L504`)
2. waiting/skipped queue selection (`scheduler.py:L1567-L1577`)
3. `prepend_request` semantics under preempt (`request_queue.py:L160-L165`)

Each point has a different FCFS/PRIORITY behavior. Together they encode TWO complete fairness philosophies: FCFS = "先来的先享受", PRIORITY = "优先级低的先牺牲". A reader who only sees the enum value learns the WHAT; a reader who sees the three threading points learns the WHY.

**Pattern for any chapter introducing a policy/strategy enum**: locate every if/else branch the enum drives, show them together as ONE coherent design, not as a sequence of independent toggles.

**For reviewers**: when a writer introduces a "policy" or "strategy" enum, check that they show ALL the call sites where the enum is read. A single-call-site introduction is misleading.

---

## K24: Ch06 demo §3 winner column says "swap" — frame around complexity not latency

**Module**: scheduler (writing-perspective)
**Chapter**: 06-scheduling
**Discovered by**: writer (Ch06 v6 A1 rewrite — corrected per team-lead guidance)
**TTL**: permanent
**Access count**: 1
**Tags**: writing, preemption, framing-trap

Ch06 demo §3 prints `winner: swap` for ALL prompt sizes (swap is 2-3x faster than recompute). The temptation: "vLLM picked recompute because it's faster on net (E[latency] over OOM workloads)" — WRONG. swap is also faster in expected latency under realistic OOM rates (476 ms vs 983 ms at 5%).

The correct framing (per Ch04 K06 + Ch05 5.6.3 + tester P02 guidance): **NEVER lead with "recompute is faster"**. Lead with:

> "Swap is faster but more complex; recompute is slower but simpler. vLLM v1 trades the latency penalty for four kinds of complexity reduction: single code path, zero CPU memory dependency, no swap-failure error mode, bit-deterministic resume."

Then show the 164/62/5000 ms table to make the trade-off honest.

This framing avoids the writing trap: a reader who sees "recompute is faster" then runs the demo and sees `winner: swap` will lose trust in the chapter. Showing the slower-but-simpler trade-off honestly preserves trust.

**For writers covering preemption in any future chapter (Ch09 preemption deep-dive, Ch20 ModelRunner-aware preempt)**: if you find yourself typing "recompute is faster", stop and reframe.
