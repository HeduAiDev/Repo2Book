# Candidate: learn.py append-mode produces malformed double-prefix headings under stale next_id

- **Recorded**: 2026-05-06
- **Recorder**: archivist (with team-lead concurrence)
- **Type**: framework / tooling defect candidate
- **Source**: discovered during 2026-05-06 second-pass compaction of `instances/vllm/knowledge/modules/scheduler.md`

## Status

```yaml
status: candidate
gate_status: awaiting_second_instance
candidate_only: true
repos_confirmed: ["vllm-from-scratch"]
occurrences_within_instance:
  - module: scheduler.md
    chapter: 06-scheduling
    bad_headings: ["## K13: K18:", "## K14: K19:", "## K15: K20:"]
    discovered_when: 2026-05-06 second compaction pass
  - module: preemption.md
    chapter: 06-scheduling
    bad_headings: ["## K01: P01:", "## K02: P02:", "## K03: P03:"]
    discovered_when: 2026-05-06 (concurrent with above)
operational_status: "Archivist now fixes ## K??: P??: / ## K??: K??: double-prefix patterns immediately on encounter, no permission needed (per team-lead 2026-05-06)."
```

## What happened

When Ch06 implementer + tester ran `learn.py` in append-mode to record new facts, the resulting markdown had double-prefix headings:

In `scheduler.md`:
- `## K13: K18: PriorityRequestQueue.prepend_request == add_request — no front in priority queue`
- `## K14: K19: PRIORITY victim selection uses max() not min() because smaller priority = higher`
- `## K15: K20: skipped_waiting queue prevents indefinite postponement of blocked requests`

In `preemption.md` (the new module Ch06 created):
- `## K01: P01: vLLM v1 has only ONE preemption strategy — recompute.`
- `## K02: P02: Recompute-vs-swap crossover is prompt-length INDEPENDENT...`
- `## K03: P03: long_prefill_token_threshold gives 4x p95 TTFT at constant throughput`

Note the patterns:
- `## K13: K18:` — same module (scheduler), so the *first* prefix was the **expected next ID per learn.py's count_facts regex**, while the second was the agent's intended new ID. They diverged because the agent had its own next_id derivation that disagreed with learn.py.
- `## K01: P01:` — different modules. The agent populated a freshly-created `preemption.md` and got `K01` as the auto-assigned ID for an empty file (the bug isn't module-aware), while the agent's intended ID was `P01` since the module is preemption-themed.

## Why it matters

Each malformed heading creates a **silent duplicate**. Top-level `^## K\d+:` regex (used by `_count_facts` in `scripts/learn.py:189-195`) matches only the FIRST K-id per heading line, so:
- Two `## K13:` headings exist (the original Ch04-era one at line 149 and the malformed Ch06 one at line 250).
- `_count_facts` returns the count *as if* there are 17 facts, when really there are 17 facts with 3 silent ID collisions.
- A subsequent `compact()` call would see the duplicates as one fact (LLM-collapse keyed on first match), silently destroying one of the two K13 / K14 / K15 entries on archive.

This bug would cause **silent data loss on the next compaction event** — exactly what the manual-compaction discipline is designed to prevent.

## Suspected root cause (in `scripts/learn.py`)

The append-mode flow appears to:
1. Call `_count_facts(file_path)` → returns N (top-level `^## K\d+:` count).
2. Set `next_id = f"K{N+1:02d}"` (caller-supplied or derived inside append).
3. The agent (Ch06 implementer/tester) computed its OWN next_id from a different basis (e.g., manual reading of file, or stale local state), got a DIFFERENT answer, and supplied that.
4. learn.py's append-mode prepended the auto-derived next_id WITHOUT checking — and now you have two prefixes.

A robust fix would be: append-mode derives next_id by **scanning existing IDs** (top-level + subheadings) and picking `max(existing) + 1`. It should NEVER trust a caller-provided next_id; it should ignore stale callers and recompute. Belt-and-suspenders: assert no `## K\d+: K\d+:` or `## K\d+: P\d+:` pattern appears in the rendered append before writing.

## What was done as a temporary fix

- 2026-05-06 second compaction pass: archivist manually renamed all 3 malformed scheduler.md headings (`## K13: K18:` → `## K18:`, etc.) and all 3 malformed preemption.md headings (`## K01: P01:` → `## P01:`, etc.) before completing compaction.
- team-lead authorized archivist to fix the `## K??: K??:` and `## K??: P??:` patterns immediately on future encounter, no permission needed.

## Recommended actions (open)

1. Fix `scripts/learn.py` append-mode to derive next_id from file scan, not caller-supplied. Add an assertion that no `^## [KP]\d+: [KPM]\d+:` heading exists in the rendered output.
2. Fix `scripts/learn.py` append-mode to be **module-aware** (scheduler.md uses K, preemption.md uses P, memory.md uses M, etc.) — currently it appears to default to K-prefix regardless of module name.
3. After step 1+2, the manual-compaction-discipline can relax: `_parse_module_file` returning `[]` is still a problem, but the append-mode bug is what produced silent duplicates.

## Promotion gate

Per CLAUDE.md, wisdom rules require pattern recurrence in 2+ **repos**, not chapters. Two occurrences within the vllm instance (scheduler.md and preemption.md, both during Ch06) is N=1 instance. Hold for second-instance confirmation OR for an explicit framework-level fix to `scripts/learn.py` (which would obviate the need to escalate to wisdom — fixing the script removes the failure mode entirely).

## Cross-references

- `instances/vllm/knowledge/archive/scheduler-20260506-second-pass.json` — archive of K10-K14 originals (compaction pass that discovered the bug).
- `scripts/learn.py:L189-L195` — `_count_facts` regex.
- `scripts/learn.py:L383-L390` — `_parse_module_file` (returns `[]`, the related compaction-mode bug already known).
- `instances/vllm/trace/cross-chapter/handoff-protocol-2026-05-06.md` — companion candidate ("compaction must preserve externally-cited fact IDs"). Both candidates relate to learn.py limitations.
