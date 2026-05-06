# Multi-Agent System — Improvement Log

Living document maintained by Team Lead at every session start. Records observed pain points in the book-factory pipeline, with proposed fixes ranked by leverage. Reviewed/updated each session before dispatching agents.

**Last reviewed**: 2026-05-07 (session 2 — Ch08 published; user-triggered pause to fix P0/P1; resolved P0-1, P0-2, P0-3, P1-1)
**Owner**: team-lead (main session)

---

## Open issues (priority order)

### P0-1 — In-process agents trap in idle-and-wait loop
**Symptom**: Boot prompts saying "go idle and wait for SendMessage" cause agents to ACK forever without producing artifacts. Confirmed for writer-2 on Ch07 (30+ min, multiple SendMessage dispatches, zero file output).

**Root cause**: `backendType: in-process` agents are ephemeral — each turn boots fresh, no persistent loop polling the inbox. The "idle and wait" instruction completes their boot turn; subsequent SendMessages may or may not trigger a follow-up turn, and when they do, the agent treats it as another boot-style readiness check.

**Workaround in use**: Bake the FULL dispatch into the spawn prompt with explicit "EXECUTE in this same turn — do not idle-and-wait". Confirmed to work for writer-3, reviewer-3, archivist-3, implementer-3, tester-3, writer-4.

**Proposed permanent fix**:
1. Update `.claude/agents/<role>.md` files to remove "go idle and wait" language and replace with "execute the dispatch you receive in the same turn".
2. Update `reference_agent_respawn.md` boot prompt template accordingly.
3. Or: switch team config `backendType` from `in-process` to `tmux` (requires `tmux` server running) for genuine persistent agents — but that adds infra burden.

---

### P0-2 — TaskCreate auto-broadcast misroutes to all agents
**Symptom**: Creating tasks fires task-list broadcasts to ALL registered agents in the team regardless of `owner` or `blockedBy`. Stale `<role>-N-1` instances react with "misroute" complaints (correctly defensive). Stuck tasks get auto-flipped to `in_progress` when broadcast acknowledgments come in. Observed twice this session (Tasks #2-#4 and #5-#8).

**Root cause** (verified 2026-05-07 via investigation task): The broadcast is **harness-internal**, not a project hook. The two hooks in `.claude/settings.local.json` are narrowly scoped and never fan out:
- `TeammateIdle → hook_idle.py` — single-target inbox writes (writer→reviewer etc.)
- `PostToolUse: TaskUpdate → hook_pipeline.py` — single-target keyword routing; fires on TaskUpdate, NOT TaskCreate

The actual fan-out comes from the experimental Agent Teams machinery (gated by `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`). When TaskCreate runs, the harness pushes the task-list event to every member in `~/.claude/teams/book-factory/config.json` — regardless of `owner` or `blockedBy`. The metadata fields are populated correctly in task JSON (e.g., `~/.claude/tasks/book-factory/1.json` has `owner:"implementer"`, `blockedBy:[]`) but the broadcaster ignores them.

**Why our blast radius is so large**: that registry currently holds **21 entries, 0 active** (1 tmux + 14 stale in-process supersessions). Inbox listing confirms `team-lead`, `book-editor[-2]`, `implementer[-2]`, `tester[-2]`, `writer[-2]`, `reviewer[-2]`, `archivist[-2,-3]` have all received messages. P0-2 and P0-3 are the same problem with two leverage points.

**Filterable from our side?** Not at the broadcast moment — there is no `PreToolUse: TaskCreate` hook in the current `settings.local.json` schema (worth probing `claude --help` for the full hook event list). We can only filter post-hoc.

**Workaround in use**: After spawning supersession `-N+1`, send stale `-N-1` instance ONE explicit "you are SUPERSEDED, ignore broadcasts" message; ignore subsequent noise.

**Proposed permanent fixes** (in leverage order):
1. **Shrink the broadcast set.** Wire P0-3's `cleanup_stale_agents.py` so the registry only contains active members. With 14→0 stale entries, the harness still fans out but to a small clean set. Highest leverage; fixes P0-2 and P0-3 together.
2. **Stop using `TaskCreate` for in-flight pipeline stages.** Drive entirely via SendMessage + state.json/file-on-disk verification. Reserve TaskCreate for genuine multi-step user-facing tracking. Currently the workaround.
3. **Receive-side filter in role prompts.** Add to each `.claude/agents/<role>.md`: "ignore TaskCreate notifications unless `owner == your name`". Defense in depth even if registry stays bloated.
4. **Investigate `PreToolUse: TaskCreate` hook event** existence. If exposed, we can write a project-level filter that drops the event before it reaches the broadcast machinery.
5. Document: never create unowned/un-blocked tasks.

---

### P0-3 — Stale `-N-1` agents persist in team registry
**Symptom**: After spawning `<role>-N+1` supersession, `<role>-N-1` and `<role>-N` remain registered in `~/.claude/teams/book-factory/config.json` (currently 6 stale + 4 active). They keep receiving broadcasts and emitting idle pings.

**Root cause**: No "kill" tool for in-process agents. Only the team config persists their identity.

**Workaround**: Standdown SendMessage + ignore. Visual noise but not pipeline-blocking.

**Proposed permanent fix**:
1. Add a script `scripts/cleanup_stale_agents.py` that prunes `in-process` entries with `isActive: None/False` from team config.
2. Or: convention to always reuse `<role>` (no suffix) by deleting prior entry before spawn.

---

### P1-1 — `learn.py` compact() and append-mode bugs
**Symptom**: Knowledge file `compact()` returns `[]` due to `_parse_module_file` regex bug → script-driven compaction non-functional; manual compaction required when modules exceed 15 facts. Append-mode produces malformed `## K??: K??:` / `## K??: P??:` double-prefix headings.

**Status**: Known from prior session (session-pause notes). `prefix-cache.md` now has 18 facts (over the 15 trigger), `tensor-parallelism.md` has 16 — both due for compaction soon.

**Workarounds**: Manual compaction; standing rule to fix double-prefix on encounter.

**Proposed permanent fix**:
1. Fix `_parse_module_file` regex in `scripts/learn.py` to actually extract `## K\d+:` entries.
2. Fix append-mode to detect existing module prefix and not re-add it.
3. Add a CI-style lint that scans all knowledge module files for double-prefix headings.

---

### P1-2 — No reliable agent-progress heartbeat
**Symptom**: Boot prompts request periodic heartbeats; agents send idle pings without content. Hard to distinguish "still working" from "dead". Implementer-3 took ~25 min to ship Ch08; was hard to know mid-flight if it was alive.

**Workaround**: File-on-disk inspection by team-lead (Bash `ls`/`wc -l`) every few minutes during long stages. Two-strike rule: one diagnostic ping, then supersede.

**Proposed permanent fix**:
1. Add `scripts/agent_heartbeat.sh <role> <interval>` that backgrounds and emits heartbeats to a status file (this script exists but not wired into agent dispatch).
2. Or: each agent writes a status file (`/tmp/book-factory/<chapter>/<role>-status.json`) on every major step transition; team-lead polls instead of pinging.
3. Implementer-3 already wrote one (`/tmp/book-factory/08-tensor-parallelism/implementer-status.json`) — promote this to a convention in agent role definitions.

---

### P1-3 — `book-editor` agent role unclear / unused
**Symptom**: book-editor agent exists, is spawned, but team-lead does direct-dispatch per `feedback_direct_dispatch.md`. book-editor's "idle-summary handoffs" don't deliver. Currently book-editor sits idle.

**Workaround**: Direct dispatch from team-lead.

**Proposed permanent fix**:
1. Either: re-architect book-editor to do something genuinely useful (e.g., topology decisions, cross-chapter consistency checks, batch outline updates) — currently CLAUDE.md describes "内容调度" but the role hasn't proved load-bearing.
2. Or: remove book-editor from the team and update CLAUDE.md.

---

### P2-1 — No project-level test runner / regression harness
**Symptom**: Each chapter's tests run independently. No "run all chapters" command. Hard to detect cross-chapter regressions when an upstream change (e.g., Ch05 BlockPool API) breaks downstream (e.g., Ch07 prefix-cache integration tests).

**Proposed**: Add `scripts/run_all_tests.py` that walks `instances/vllm/artifacts/*/tests/` and reports per-chapter pass/fail. Run before each chapter publish.

---

### P2-2 — Knowledge modules over the 15-fact compaction trigger ✅ RESOLVED 2026-05-07
**Symptom**: `prefix-cache.md` has 18 facts; `tensor-parallelism.md` has **19 facts after Ch08 publish (T01-T19)** — 4 over the trigger. Both over CLAUDE.md's 15-fact limit. learn.py compact() doesn't work.

**Severity escalation 2026-05-06 (Ch08 publish)**: tensor-parallelism.md is now the third chapter file to demonstrate compact() brokenness (after scheduler.md and prefix-cache.md). With the same root cause confirmed across 3 modules and growing-without-bound, compact() failure has moved from "annoying" to "blocking automated knowledge hygiene". Manual workaround in use; if not fixed before Ch09-13, every new chapter adds 4-7 facts and the index will keep drifting.

**Resolution 2026-05-07** (book-editor, manual compaction):
- `prefix-cache.md`: 18 → 14 (compacted K01-K05 into one parent block; archived to `knowledge/archive/prefix-cache-20260507-k01-k05.json`).
- `tensor-parallelism.md`: 19 → 15 (compacted T01-T05 into one parent block; archived to `knowledge/archive/tensor-parallelism-20260507-t01-t05.json`).
- All 10 ID anchors (`### K01:` … `### K05:`, `### T01:` … `### T05:`) preserved as subheadings — heavy external citations in narrative/tests/reviews/briefs remain grep-resolvable.
- `knowledge/INDEX.md` updated with rows for `prefix-cache`, `preemption`, `memory` (previously missing).

**Still open** (P1-1 root cause): learn.py compact() remains non-functional; manual compaction will be needed again when Ch09-Ch13 push these or other modules back over the cap. Until P1-1 fixes, this task recurs each chapter publish.

---

### P2-3 — Outline-vs-source mismatch detection is ad-hoc
**Symptom**: Ch07 (radix tree absence) and Ch08 (TensorParallel class absence) both required manual reframe at chapter level. Discovered by archivist's source verification per rule #6.

**Working pattern**: Pre-implementer-dispatch source verification by archivist now standardized per session-pause rules.

**Proposed permanent fix**: Add `scripts/check_outline_classes.py` that reads `book/book-outline.json`, extracts class names mentioned in subsections, greps the source repo, and emits a mismatch report. Run this once across all 28 chapters; flag every ghost class for advance reframe planning.

---

## Resolved (kept for context)

### R-1 — `scripts/learn.py` instance-scoping (resolved 2026-05-05)
KNOWLEDGE_DIR resolved from framework `repo2book.json.source.source_dir.parent` — now correctly instance-scoped to `instances/vllm/knowledge/`.

### R-2 — Hook paths after WSL→Linux migration (resolved 2026-05-05)
TeammateIdle and PostToolUse hook paths rebased from `/mnt/e/...` to `/home/zjq/Repo2Book/...`. 7 stale `cwd` paths in team config also rebased.

### R-3 — P0-1 in-process idle-loop trap (resolved 2026-05-07)
Updated 5 agent role files (writer.md, implementer.md, reviewer.md, tester.md, book-editor.md): replaced misleading "Continuous Session — You Are a Persistent Agent" sections (which described a tmux-persistent runtime that doesn't exist for in-process agents) with accurate "Lifecycle — Ephemeral Spawn-Per-Task" sections that instruct "EXECUTE the dispatch in your first turn — do not idle-and-wait". Also updated `memory/reference_agent_respawn.md` boot prompt template to remove idle-and-wait language and require full dispatch in spawn prompt.

### R-4 — P0-3 stale agent cleanup script (resolved 2026-05-07)
Wrote `scripts/cleanup_stale_agents.py`. Prunes `in-process` entries from team config; preserves tmux base entries. Supports `--dry-run` and `--keep <name>` flags. Ran once: pruned 14 stale `-2/-3/-4` instances, surviving 7 tmux skeletons. Backup at `config.json.bak`.

### R-5 — P0-2 hook_idle.py path bug (resolved 2026-05-07)
`hook_idle.py` had `ARTIFACTS = Path("/mnt/e/Laboratory/...")` — broken WSL-mount path since 2026-05-05 migration. Hook silently no-op'd because `find_active_chapter()` couldn't locate any chapters. Rebased to `/home/zjq/Repo2Book/...`. (Note: the task-list auto-broadcast itself is built into the team task-list machinery, not in this hook; the standing workaround is to assign owner/blockedBy at TaskCreate and standdown stale agents explicitly.)

### R-6 — P1-1 learn.py compact() and count parser bugs (resolved 2026-05-07)
Root cause: `_parse_module_file`, `_count_facts`, and `_detect_module_prefix` all used `^## [A-Z]\d+:` regex which missed `### K01:` sub-entries inside compacted `## K01–K05:` parent blocks. Compacted entries are still valid distinct facts (5 of them per parent block), so they MUST count toward the 15-fact compaction trigger. Fix: extended regex to `^##+ [A-Z]\d+:` (matches both `## K01:` top-level AND `### K01:` sub-entries). Verified: prefix-cache.md now reports 18 (was 13), tensor-parallelism.md 19 (was 14), scheduler.md 24 (was 14). All three over the 15 trigger; compact() can now be used.

---

## Maintenance protocol

At every new session:
1. Read this file before dispatching any agents.
2. Check whether any P0/P1 items can be addressed before resuming pipeline.
3. After session, append new observations to relevant items or add new ones.
4. Mark resolved items with date and move to "Resolved" section.

User-issued pause (e.g., "完成第八章的时候暂停一下，改进一下当前多智能体协作系统的问题") is the trigger to actually fix the highest-leverage P0 items, not just observe them.
