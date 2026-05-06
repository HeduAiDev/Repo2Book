# Session Summary — 2026-05-06 Session 1 (Pause)

- **Date**: 2026-05-06
- **Recorded by**: archivist-2
- **Reason for snapshot**: user pausing session for restart; this trace + auto-memory must enable clean rehydration
- **Project**: vllm-from-scratch (28-chapter book on the vLLM inference engine)
- **Working dir**: `/home/zjq/Repo2Book`
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`

---

## 1. Pipeline state at pause

### Published (6 / 28)

| ID | Title | Version | Lines | Tests | Notes |
|---|---|---|---|---|---|
| 01-self-attention-fundamentals | Self-Attention 算子深度解析 | published_v5 | — | — | pre-v6 era; will not rewrite |
| 02-kv-cache | KV Cache 内存模型与实现 | published_v5 | — | — | pre-v6 era |
| 03-flashattention-pagedattention | FlashAttention & PagedAttention | published_v5 | — | — | pre-v6 era |
| **04-continuous-batching** | Continuous Batching 动态调度 | **published_v6** | 712 / 3064w | 48/48 | first v6, 1-cycle APPROVE |
| **05-memory-management** | GPU 显存管理系统 | **published_v6** | 757 / 3849w | 74/74 | second v6, 1-cycle APPROVE |
| **06-scheduling** | 请求调度系统 | **published_v6** | 655 / 3351w | 97/97 | third v6, 1-cycle APPROVE, 40 mapping crossrefs |

### In progress (Ch07)

`07-prefix-cache` — Prefix Cache 与 APCAwareAllocation:
- **Implementer**: ✓ DONE
- **Tester**: ✓ DONE — 83/83 tests pass; all 4 fidelity checks verified; appended K11-K13 to `prefix-cache.md` (now 13 facts there)
- **Writer**: dispatched by team-lead but **no SendMessage actually delivered before pause** — this is the immediate next step on resume
- **Reviewer**: pending
- **Archivist**: pending (Task #20)

Tasks at pause: #16-#19 completed/in_progress, #20 pending.

### Remaining (21 / 28): Ch08-Ch28

- needs_rewrite: Ch08-Ch13 (had legacy content snapshotted to `_legacy/`; awaiting v6 rewrite)
- not_started: Ch14-Ch28 (no legacy)

---

## 2. Active agents at pause (all `-2` suffixed, idle)

book-editor-2, implementer-2, tester-2, writer-2, reviewer-2, archivist-2.

**WARNING**: agents likely terminate when main session ends. Next session must re-spawn all 6 with `Agent` tool, `team_name=book-factory`, matching `name`. Reference team-lead's prior Ch04-Ch07 dispatch messages for the prompt template.

---

## 3. Framework patches landed this session

### `scripts/learn.py`
- `KNOWLEDGE_DIR` resolved from framework `repo2book.json.source.source_dir.parent` — now correctly instance-scoped to `instances/vllm/knowledge/` (was a repo-root fallback before).
- `save_knowledge` switched from rewrite-from-scratch to append-only mode (preserves existing K## entries).
- `_count_facts` regex helper for module size checks (regex: `^## K\d+:` top-level only).
- Accepts both `{fact}` (legacy) and `{title, body, source, tags}` (rich) input formats.

**Known remaining bugs** (candidate-only, not promoted to wisdom):
- `_parse_module_file` returns `[]` → `compact()` is non-functional via script, must compact manually.
- Append-mode produces malformed `## K??: K??:` / `## K??: P??:` double-prefix headings — not module-prefix-aware. Six observed instances during Ch06 work (3 in scheduler.md, 3 in preemption.md) — all manually cleaned.

### `.claude/settings.local.json`
- TeammateIdle and PostToolUse hook paths rebased from `/mnt/e/Laboratory/...` → `/home/zjq/Repo2Book/...`

### `.claude/teams/book-factory.json`
- 7 stale `cwd` paths rebased from the legacy WSL-Windows-mount path to `/home/zjq/Repo2Book`

### Knowledge base ops (this session)
- `scheduler.md` compacted twice (manual; learn.py compact() non-functional): K05-K09 (first pass) → K10-K14 (second pass after Ch06 work pushed it back over cap). Final state: 12 top-level facts; 10 IDs preserved as `### KXX:` subheadings under two parent compacted blocks.
- `preemption.md` created during Ch06; double-prefix labels cleaned post-Ch06 (P01-P03 were `## K01: P01:` form).
- `prefix-cache.md` created during Ch07 (NEW module); 13 facts; all clean single-prefix headings.

---

## 4. Operational rules enforced in this book (NOT yet wisdom — gate awaiting 2nd repo)

These are operationally enforced by team-lead in *this* book; they remain candidates per CLAUDE.md strict 2+-repos promotion gate:

1. **Explicit SendMessage required at every handoff**. Task hook alone doesn't wake the next agent. Confirmed N=3-within-vllm-instance (Ch04, Ch05, Ch06). Candidate at `trace/cross-chapter/handoff-protocol-2026-05-06.md`.
2. **Wisdom 2+ repos is literal across instances**, never N-within-one-instance promote. Lesson learned hard this session — twice over-eagerly promoted W13, twice reverted. See `memory/feedback_wisdom_gate_strict.md`.
3. **`## K??: P??:` / `## K??: K??:` double-prefix headings are always wrong** — fix on encounter, no permission needed. Standing rule per team-lead 2026-05-06. Candidate at `trace/cross-chapter/learn-py-append-id-bug.md`.
4. **Archivist's standing protocol**: produce next-chapter brief immediately on every reviewer-APPROVE — drift-corrected this session (skipped Ch06 + Ch07 briefs in succession before correction). See `memory/feedback_brief_on_approval.md`.
5. **Outline subsection names describe TOPICS, not contracts on which classes must exist** — Ch07 radix tree near-miss. Decision: leave outline JSON alone, reframe at chapter level. See `state.json:outline_notes[0]`.
6. **Source-grounding verification before implementer dispatch when outline mentions specific data structures** — archivist must verify presence/absence of named classes at the pinned commit before writing the brief.

---

## 5. v6 cadence baseline (locked at N=3)

Floor thresholds confirmed at N=3 across Ch04, Ch05, Ch06:

| Threshold | Floor | Ch04 | Ch05 | Ch06 | Ch07 (in-progress) |
|---|---|---|---|---|---|
| Source files in `impl-notes.md` | ≥ 5 | 5 | 7 | 6 | TBD |
| `# REFERENCE:` comments | ≥ 60 | 65 | 61 | 60 | 60 |
| Source mapping table rows | ≥ 10 | 13 | 21 | 29 (+11 mini = 40) | TBD |
| Tests pass rate | 100% | 48/48 | 74/74 | 97/97 | 83/83 |
| Lint formula blocking | 0 | 0 | 0 | 0 | TBD |
| Lint formula non-blocking | ideally 0 | 4 | 0 | 1 | TBD |
| Lint source-grounding | all green | PASS | PASS | PASS | TBD |
| Demo numerics in narrative | verbatim | ✓ | ✓ | ✓ | TBD |
| 5-step rhythm | every major § | ✓ | ✓ | ✓ | TBD |
| Review cycles to APPROVE | 1 (typical) | 1 | 1 | 1 | TBD |
| Forward-pointer to dependents | when fidelity gap | ✓ (Ch20) | ✓ (Ch20+Ch06) | ✓ (Ch07) | (will need Ch13 + Ch23) |

Patterns Ch06 introduced (`state.json:v6_compliance.patterns_promoted_to_baseline`):
- two_tier_mapping (main + per-section mini-tables when source surface is broad)
- tester_framing_guidance (testers shape narrative through Knowledge facts) — proven again in Ch07 with 3 framing tips from tester
- language_trap_callouts (explicit "don't say X" — Ch06 had 4, Ch07 will have at least 1 around radix-tree-myth)
- honest_demo_caveats (flag what's model artifact vs vLLM truth)

These are "reach for these when triggers apply", NOT new floors.

---

## 6. Open framework bugs (candidate, not promoted)

1. **`scripts/learn.py` `_parse_module_file` returns `[]`** → `compact()` is non-functional via script. Workaround: compact manually with full external-citation-preservation discipline (preserve `### KXX:` subheadings under a single consolidated parent block).
2. **`scripts/learn.py` append-mode produces malformed `## K??: K??:` / `## K??: P??:` double-prefix headings** — not module-prefix-aware. Workaround: standing rule to fix on encounter without permission.

Both at:
- `instances/vllm/trace/cross-chapter/learn-py-append-id-bug.md`
- `instances/vllm/trace/cross-chapter/handoff-protocol-2026-05-06.md` (companion candidate, same root cause area)

A framework-level fix to `scripts/learn.py` would obviate the need to escalate either to wisdom — fixing the script removes the failure mode entirely.

---

## 7. Open work — where to resume

### Immediate next step: Ch07 writer dispatch

Pattern: see team-lead's prior Ch06 writer dispatch message — same framing-tips-from-tester structure, plus the radix-tree language-trap reframe for §7.2.

**Tester gave 3 framing tips for the writer** (apply surgically):
1. **Don't say "asymptotically faster"** — the 4.6x lookup speedup is Python overhead artifact, not asymptotics. Hash-chain is asymptotically EQUIVALENT to radix-tree (both are O(L)); the win is constant-factor + simplicity.
2. **Lead with the (N-1)*K savings formula**, not the 78% number. The formula generalizes; the percent is workload-specific.
3. **Chain-break is THE invariant** for prefix cache — frame the chapter around it (eviction breaks the chain → subsequent blocks become unreachable via prefix lookup but their KV stays in memory until individually evicted; this is W10 "ref_cnt = -1 means not ready" wisdom in action).

Plus the Ch07 brief headline (already in `trace/briefs/07-prefix-cache-implementer-2026-05-06.md` §2):
- **vLLM v1 has NO radix tree** — verified zero `class.*Radix|Trie|PrefixTree` matches at commit 98661fe. Suggested writer framing: "vLLM v1 没有 radix tree——它用链式 hash + 平面 hash 表替代" (in Ch06's "不要说 recompute 更快" style).
- §7.2 reframed at chapter level as "为什么没用 radix tree:链式 hash 替代方案的设计权衡". Outline JSON unchanged — see `state.json:outline_notes[0]` for full context.

### Then: Ch07 reviewer → archive (Task #20) → Ch08

Ch08 = `08-tensor-parallelism` — scope shift to TP / communication primitives. New source surface (`vllm/distributed/parallel_state.py`, `vllm/model_executor/layers/linear.py:Linear*Replica`, etc.). Brief candidate areas:
- TP partitioning math (column-parallel vs row-parallel)
- All-Reduce / All-Gather collective primitives — `dist.all_reduce` placements
- Forward + backward sync points
- Why TP=2 doesn't always = 2x throughput (communication overhead)

When archivist resumes for Ch08 brief: verify `vllm/distributed/` source structure at commit 98661fe before writing the brief, per rule #6.

---

## 8. Knowledge base state at pause

| Module | Facts | State |
|---|---|---|
| `scheduler.md` | 12 (top-level) | compacted twice this session; K05-K09 + K10-K14 in two compacted parent blocks; K01-K04 + K15-K22 active |
| `attention.md` | 4 | untouched this session |
| `kv-cache.md` | 12 | post-compaction, healthy |
| `memory.md` | 3 (M01-M03) | created during Ch05, populated by Ch05 implementer extract |
| `preemption.md` | 5 (P01-P05) | created during Ch06, all clean labels post-relabel |
| `prefix-cache.md` | 13 (K01-K13) | created during Ch07, all clean single-prefix headings |

Note: `prefix-cache.md` uses K-prefix per the actual-on-disk state (which I have not verified at pause — team-lead reported "K01-K13"). If next session needs to verify, `grep -nE "^## " prefix-cache.md` is the check. P-prefix would have matched the Ch07 brief recommendation, but K-prefix was what landed; either is acceptable as long as it's consistent within the file and learn.py append-mode doesn't reintroduce double-prefixes.

`knowledge/INDEX.md` may or may not have been updated for the new `prefix-cache.md` row — verify on resume.

---

## 9. Trace files written this session (relative to `instances/vllm/`)

### Decisions
- `trace/decisions/2026-05-05_ch05-ch28-directory-remap-to-outline-ids-+-legacy-snapshot.md` (early-session migration)

### Briefs
- `trace/briefs/05-memory-management-implementer-2026-05-06.md`
- `trace/briefs/06-scheduling-implementer-2026-05-06.md`
- `trace/briefs/07-prefix-cache-implementer-2026-05-06.md` (180 lines, includes radix-tree language-trap §2)

### Cross-chapter
- `trace/cross-chapter/handoff-protocol-2026-05-06.md` (candidate, gate_status: awaiting_second_instance, repos_confirmed: ["vllm-from-scratch"], chapters within instance: [Ch04, Ch05, Ch06])
- `trace/cross-chapter/learn-py-append-id-bug.md` (candidate, gate_status: awaiting_second_instance)

### Deliveries (canonical + auto-record per chapter)
- `trace/deliveries/04-continuous-batching.md` (canonical, with v6-baseline posterity note)
- `trace/deliveries/05-memory-management.md` (canonical, with Ch04↔Ch05 comparison)
- `trace/deliveries/06-scheduling.md` (canonical, with Ch04↔Ch05↔Ch06 floor table)
- Plus 3 timestamped auto-records from `archivist.py record`

### Snapshots
- `trace/snapshots/04-continuous-batching/v6-2026-05-06/` (24 files, 560K)
- `trace/snapshots/05-memory-management/v6-2026-05-06/` (22 files, 256K)
- `trace/snapshots/06-scheduling/v6-2026-05-06/` (20 files, 240K)

### Sessions
- `trace/sessions/2026-05-06_session-1-pause.md` (this file)

### Knowledge archives
- `knowledge/archive/scheduler-20260506-k05-k09.json` (first compaction)
- `knowledge/archive/scheduler-20260506-second-pass.json` (second compaction)

### state.json blocks added
- `outline_notes` (with the §07.2 radix-tree-reframe entry)
- `v6_compliance.metrics_per_chapter` (per-chapter metrics dict for Ch04/05/06)
- `v6_compliance.cadence_holds_at_n3` flag
- `v6_compliance.patterns_promoted_to_baseline` (4 Ch06-introduced patterns)

---

## 10. Auto-memory entries (cross-session persistence)

`/home/zjq/.claude/projects/-home-zjq-Repo2Book/memory/`:
- `MEMORY.md` — index
- `feedback_brief_on_approval.md` — produce next-chapter brief immediately on every approval; standing protocol
- `feedback_wisdom_gate_strict.md` — CLAUDE.md "2+ repos" is literal across instances
- `feedback_double_prefix_headings.md` — fix `## [KP]\d+: [KP]\d+:` immediately on encounter
- `project_vllm_book.md` — project state snapshot

These will load automatically on next session start.

---

## 11. Rehydration checklist for next session

When resuming:
1. Read `state.json` — confirm `last_session.context_hash = session-2026-05-06-pause-1`.
2. Read this file (`trace/sessions/2026-05-06_session-1-pause.md`).
3. Read `trace/briefs/07-prefix-cache-implementer-2026-05-06.md` — Ch07 context including radix-tree headline.
4. Read auto-memory in `/home/zjq/.claude/projects/-home-zjq-Repo2Book/memory/` (loads automatically).
5. Re-spawn 6 agents (book-editor-2, implementer-2, tester-2, writer-2, reviewer-2, archivist-2) via `Agent` tool with `team_name=book-factory`.
6. Team-lead dispatches Ch07 writer with the 3 framing tips + radix-tree reframe.
7. Pipeline resumes from Task #18 (Ch07 writer).

Verify before resuming Ch07 writer:
- `prefix-cache.md` actually exists with 13 K-prefix facts (sanity check; team-lead reported K01-K13).
- `knowledge/INDEX.md` has a row for `prefix-cache.md` (add if missing).
- No `## [KP]\d+: [KP]\d+:` double-prefixes in any module file (`grep -rE "^## [KPM]\d+: [KPM]\d+:" instances/vllm/knowledge/modules/`).
