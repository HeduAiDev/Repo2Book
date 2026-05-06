# Cross-chapter process improvements observed during Ch04 v6 pipeline

- **Date**: 2026-05-06
- **Recorder**: archivist
- **Source**: team-lead post-Ch04 review
- **Type**: process / handoff protocol
- **Status**: candidate for promotion to `wisdom/architecture.md` (universal pattern, framework-level)

## Observations from the first full-pipeline v6 chapter

### O1. Hooks alone do NOT wake the next-stage agent
**What**: The Task system status-transition hook (Ch04 implementer→tester→writer→reviewer) auto-advances task statuses, but does NOT wake the assigned agent. Each stage needs an explicit `SendMessage` from the prior agent or team-lead. Ch04 stalled ~26min on tester for this reason, and again at writer.

**Implication**: The pipeline contract must be: "completing a task includes (a) `TaskUpdate` to completed AND (b) `SendMessage` to the next assignee with a stage-handoff blurb." Status alone is not a wake signal.

### O2. book-editor "idle-summary" preview is not a SendMessage
**What**: book-editor frequently emitted text like `[to implementer] start now` in its idle/summary output. This rendered to the user but did NOT actually invoke the SendMessage tool. Team-lead has been backstopping these as out-of-band relays.

**Implication**: Train book-editor (and any orchestrator agent) to ALWAYS use the `SendMessage` tool for inter-agent communication. Plain text is invisible to teammates. Preview text in summaries is for the user, not a delivery mechanism.

### O3. learn.py instance-scoping bug
**What**: Surface during Ch04 mid-chapter; team-lead patched. Specifics in commit history of `scripts/learn.py`. Root cause: knowledge-extraction CLI defaulted to repo-root `knowledge/` instead of `instances/vllm/knowledge/`.

**Implication**: All scripts that read/write per-instance state must read `repo2book.json` for the active instance dir, not assume a fixed path.

### O4. Premature task completion on bootstrap conflation
**What**: Archivist closed Ch04 Task #5 during bootstrap (after recording the migration trace entry), then the team-lead re-opened the task to track the actual pipeline-final archival. Pattern is: bootstrap-tier tasks and pipeline-tier tasks share IDs in some flows.

**Implication**: Either (a) bootstrap actions get separate task IDs from pipeline actions, or (b) agents must distinguish "I did one piece of my role's work" from "the pipeline-final work for this chapter is done." The conservative rule: don't close a stage task until the chapter status in `state.json` reaches `published_*` for THAT chapter.

## Proposed wisdom rule (for `wisdom/architecture.md`)

**Title**: "Agent-team handoff requires explicit message + task update, not hook alone"

**Body**:
> When a pipeline stage completes, the agent responsible MUST do BOTH:
> 1. `TaskUpdate` the stage task to `completed` (this fires status-transition hooks).
> 2. `SendMessage` the next-stage assignee with a one-paragraph handoff (artifact paths, gate checks passed, anything they need to know).
>
> The hook alone does NOT wake the next agent. Hooks are bookkeeping; messages are wake signals.
>
> **Why**: observed during vLLM Ch04 v6 — pipeline stalled twice because the next-stage agent received no explicit message, only a hook-fired status change that nothing was watching.
>
> **How to apply**: every stage agent's role .md should include a "Handoff" section listing (a) what to TaskUpdate, (b) who to SendMessage, (c) what the message should contain.
>
> **Corollary**: orchestrator agents (book-editor, team-lead) must use the SendMessage tool for inter-agent comms. Plain-text "[to X] do Y" in summary output is invisible to other agents — it only renders to the user.

**Promotion gate**: per CLAUDE.md, wisdom rules require the pattern to recur in 2+ repos. Ch04 is N=1; promote when the next instance (or Ch05 of vLLM) confirms the same pattern. Until then, this lives in `trace/cross-chapter/` as a candidate.

## Status: candidate (HOLD per team-lead 2026-05-06)

```yaml
status: candidate
gate_status: awaiting_second_instance
repos_confirmed: ["vllm-from-scratch"]
chapters_confirmed_within_instance: ["Ch04", "Ch05"]
promotion_gate: "Per CLAUDE.md, must appear in 2+ REPOS before promotion. N=2-within-one-instance does NOT meet the bar — that would defeat the cross-instance gating purpose. Hold until a second book instance independently confirms."
operational_status: "team-lead is keeping this rule operationally in *this* book regardless of promotion. Team-lead has assumed the handoff role from book-editor — direct SendMessage between stages is the working norm."
```

### Initial promotion attempt + revert (2026-05-06)

Archivist briefly added W13 to `wisdom/architecture.md` and `wisdom/INDEX.md` after misinterpreting an earlier team-lead message as authorization. team-lead clarified: the strict CLAUDE.md gate is "2+ repos", not "2+ chapters within one repo". W13 reverted from both `wisdom/architecture.md` and `wisdom/INDEX.md`. This file remains the canonical home of the candidate until a second instance confirms.

### Ch05 evidence (within-instance N=2, NOT yet wisdom-grade)

Ch05 pipeline (memory-management) ran end-to-end and confirmed the same pattern as Ch04:
- Implementer→tester, tester→writer, writer→reviewer, reviewer→archivist all required explicit `SendMessage` to wake the next stage. Hooks alone never woke an agent.
- book-editor's `[to X] do Y` summary lines continued to render to the user but never reached teammates — team-lead has now formally taken over the handoff role.
- Ch05 archival itself was the cleanest test of the rule: reviewer-2 → archivist-2 explicit SendMessage = no stall; clean archival sequence.

## Recommended actions

1. Update each stage agent's role .md (`.claude/agents/{implementer,tester,writer,reviewer}.md`) with a "Handoff" section. **(open — local to this instance, lower priority since team-lead is backstop)**
2. Add a `SendMessage`-required step to the Chapter Pipeline section in CLAUDE.md (or to `repo2book.json` pipeline config). **(open — local to this instance)**
3. ~~Log Ch05's pipeline carefully...~~ **(done — Ch05 confirmed pattern within this instance)**
4. **NEW**: when starting a second repo2book instance, watch for the same handoff-stall pattern in its first 2 chapters. If it recurs there, the cross-instance N=2 gate is met → promote to `wisdom/architecture.md` as the next available W##.

## Companion candidate (added 2026-05-06)

Discovered during scheduler.md compaction: **"Knowledge-fact compaction must preserve externally-cited fact IDs"** — `learn.py compact()` would have auto-renumbered K-IDs via `_write_module_file`, breaking external test citations from `tests/test_scheduler.py:183/227/298`. Manual compaction kept all five `K05`/`K06`/`K07`/`K08`/`K09` heading anchors as `### K0X:` subheadings within a single consolidated block.

This is a candidate for `wisdom/debugging.md` or a new `wisdom/knowledge-management.md` category. Currently N=1 (single occurrence in scheduler.md). Hold for next compaction event to confirm.
