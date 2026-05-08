# Session Summary — 2026-05-07 → 2026-05-09 Session 2 (Pause)

- **Dates**: 2026-05-07 (resume from session-1-pause) → 2026-05-09 (this pause)
- **Recorded by**: team-lead (main session)
- **Reason for snapshot**: user pausing session for restart
- **Project**: vllm-from-scratch (28-chapter book on the vLLM inference engine)
- **Working dir**: `/home/zjq/Repo2Book`
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/`
- **Branch**: `session-2026-05-07-ch07-08-system-fixes` (push to origin; main has 2 stale Ch05 commits we elected to leave alone)

---

## 1. Pipeline state at pause

### Published (9 v6 / 12 total of 28)

| ID | Title | Version | Lines / Words | Tests | Mapping | Cycles | Notes |
|---|---|---|---|---|---|---|---|
| 01 | Self-Attention | published_v5 | — | — | — | — | pre-v6 |
| 02 | KV Cache | published_v5 | — | — | — | — | pre-v6 |
| 03 | FlashAttention/Paged | published_v5 | — | — | — | — | pre-v6 |
| 04 | Continuous Batching | published_v6 | 712/3064 | 48/48 | 13 | 1 | first v6 |
| 05 | Memory Management | published_v6 | 757/3849 | 74/74 | 21 | 1 | second v6 |
| 06 | Scheduling | published_v6 | 655/3351 | 97/97 | 40 | 1 | third v6 |
| 07 | Prefix Cache | published_v6 | 859/4440 | 83/83 | 72 | 1 | session 2 begins; "no class X" #1 (radix tree) |
| 08 | Tensor Parallelism | published_v6 | 1051/6058 | 144/144 | 122 | 1 | "no class X" #2 (TensorParallel) |
| 09 | Expert Parallelism | published_v6 | 1204/7792 | 204/204 | 151 | 1 | "no class X" #3 (ExpertParallel); training→inference reframe #1 (§9.4 EPLB) |
| 10 | Multi-Token Prediction | published_v6 | 1345/8888 | 311/311 | 206 | 1 | "no class X" #4 (MTP); training→inference reframe #2 (§10.3) |
| 11 | DCP/PCP | published_v6 | 1394/8124 | 474/474 | 149 | 1 | "no class X" #5 (RingAttention) — GRADUATED MOTIF; cleanest implementer handoff (zero patches) |
| **12** | **KV Offload** | **published_v6** | **1583/10178** | **314/314** | **285** | **1** | **NOT a 6th "no class X" — series retired at N=5; new motif: 4 TOPIC-level outline reframes (NVMe/LFU/attention-score/predictive); two-reviewer convergence** |

### In progress (Ch13)

`13-prefix-cache-pooling` — **the LAST `needs_rewrite` chapter**:
- **archivist-2 mid-flight** writing Ch12 publish + Ch13 brief at the moment of pause
- Ch13 brief target: `instances/vllm/trace/briefs/13-prefix-cache-pooling-implementer-2026-05-08.md`
- Open question: is Ch13 a 6th "no class X" candidate, a 5th TOPIC-reframe instance, or hybrid? Source verification in progress

### Remaining (15 / 28): Ch14-Ch28

- not_started: Ch14-Ch28 (15 chapters; Part 3+ Triton-from-operators, Part 4 PD architecture, Part 5 model-specific)
- Ch14 = `triton-primer` (Part 3 opener)

---

## 2. Multi-agent system improvements (this session)

User-triggered pause after Ch08 → P0/P1 work batch. All 4 P0/P1 items resolved this session:

### Resolved (recorded in `instances/vllm/trace/system-improvements.md`)
- **R-3 P0-1**: 5 agent .md files (writer/implementer/reviewer/tester/book-editor) replaced misleading "Continuous Session — Persistent Agent" sections with accurate "Lifecycle — Ephemeral Spawn-Per-Task" lifecycle. Eliminates the in-process idle-and-wait trap.
- **R-4 P0-3**: New `scripts/cleanup_stale_agents.py` prunes dead in-process agents from team config (--dry-run / --keep flags). Multiple cleanup runs throughout session.
- **R-5 P0-2**: Fixed `hook_idle.py` stale WSL-mount path `/mnt/e/...` → `/home/zjq/Repo2Book/...`. The task-list auto-broadcast itself is harness-internal (CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1) and not filterable from our side.
- **R-6 P1-1**: `learn.py` `_parse_module_file` / `_count_facts` / `_detect_module_prefix` regexes now use `^##+` to count compaction sub-entries. Verified consistent counts across all knowledge modules.

### Still open (P2-2)
- Knowledge modules over 15-fact compaction trigger: kv-offload.md (28), dcp-pcp.md (29), multi-token-prediction.md (30), expert-parallelism.md (24), tensor-parallelism.md (19), prefix-cache.md (18). With P1-1 fixed, `learn.py compact()` should now work — manual compaction pass would help future hygiene but doesn't block Ch13.

---

## 3. New patterns recognized this session

### "no class X" reframe series (Ch07 → Ch11; RETIRED at Ch12)
- Ch07 §7.2: no `class RadixTree` — flat dict at `block_pool.py:L34-L127`
- Ch08 §8.2: no `class TensorParallel` — 5-file composition (parallel_state + linear + comm wrappers + vocab_parallel_embedding + Llama)
- Ch09 §9.2: no `class ExpertParallel/MoEParallel/TopKGate` — 5-file composition (parallel_state _EP + fused_moe/{layer,config} + all2all + naive_dp_ep)
- Ch10 §10.2: no `class MultiTokenPrediction` — 5 proposer family classes via SpeculativeMethod enum (Eagle/Medusa/DraftModel/Ngram/DeepSeek-MTP)
- Ch11 §11.2: no `class RingAttention/StripedAttention/ContextParallel` — 8-file composition (parallel_state _DCP/_PCP + dcp_alltoall + attention/backend + utils + flashattn_mla + kv_cache_interface + multiproc_executor + config/parallel)
- **Ch12: series RETIRED at N=5** — KV offload has honestly-named classes (`OffloadingManager`, `CPUOffloadingManager`, 18 connectors). Don't force 6th. Wisdom-promotion gate (2+ instances) NOT met since all 5 are within vLLM only.

### Training→inference reframe series
- Ch09 §9.4: aux-loss CE training → EPLB inference-time runtime balancer (1st)
- Ch10 §10.3: multi-step CE training → `_rewrite_spec_layer_name` inference weight loading (2nd)
- Pattern: training-literature sidebar → pivot to inference-time response. M28 says queue for wisdom promotion at instance #3.

### TOPIC-level outline reframe pattern (Ch12 introduced)
4 surgical corrections to outline TOPICS that don't exist in vLLM at 98661fe:
1. NVMe SSD third tier → vLLM is 2-tier (HBM ↔ CPU pinned only)
2. LFU eviction → only LRU + ARC (Megiddo-Modha 2003)
3. Attention-score-based eviction → block-hash semantics
4. Predictive ML prefetch → REACTIVE block-hash matching

### Honest demo caveat (K17 lineage)
- Ch06: 1 Pareto point is model artifact, not vLLM truth
- Ch10: K17 OR-skip discipline (ms times paired with caveat OR omitted)
- Ch12: ARC LOSES to LRU on phase_shift (LRU 2.60% vs ARC 14.15% miss) — preserved verbatim across 8 anchors with Megiddo-Modha 2003 Table 4 reference

### Three-anchor framing-tip verification (D28 / M29)
Reviewer checks each tester framing tip is applied at 3 places: hook + body + recap. Used in Ch10/Ch11/Ch12 reviews.

### Brief-write 4-invariant pattern (D31)
Implementer brief should contain: (1) verified-source-surface table at commit pin, (2) outline-vs-source mismatches enumeration, (3) candidate-language-traps list, (4) demo-plan with numerics-count target. Yields zero-patch implementer handoff (Ch11 + Ch12 both demonstrated).

---

## 4. Active agents at pause

After last cleanup: only `archivist-2` (in-process, mid-task on Ch12 publish + Ch13 brief).
All other in-process spawns from this session have been pruned.

7 tmux skeleton entries (book-editor, implementer, tester, writer, reviewer, archivist, researcher) — never activated; placeholders only.

---

## 5. Open work — where to resume

### Immediate next step: wait for archivist-2 to deliver Ch12 publish + Ch13 brief

Expected artifacts:
- `instances/vllm/trace/deliveries/2026-05-08_ch12-kv-offload-ch12-v6-published.md`
- state.json patched: total_published_v6=9; cadence_holds_at_n9 NEW key
- `instances/vllm/trace/briefs/13-prefix-cache-pooling-implementer-2026-05-08.md`

If archivist-2 didn't finish before pause:
- Verify state.json `chapters."12-kv-offload"` is `published_v6` (if not, redo task A manually OR respawn archivist)
- Check Ch13 brief exists; if missing, respawn archivist with brief-only task

### Then: Ch13 implementer dispatch

Ch13 = `13-prefix-cache-pooling` — **the LAST v5→v6 rewrite chapter**. After Ch13 ships, the book transitions from "rewrite Part 1-2" mode to "write Part 3+ from scratch" mode (Ch14-Ch28 are `not_started`).

### Then: Part 3 begins at Ch14 (triton-primer)

Big mode shift — Part 3 (Ch14-Ch21) is "build Llama-3.2-1B from Triton operators". Each chapter is from-scratch, not a vLLM-rewrite. The brief-on-approval discipline still applies but source verification will look different (Triton kernels vs vLLM Python).

---

## 6. Important framework state

### Branch
- Local + origin branch: `session-2026-05-07-ch07-08-system-fixes`
- Origin/main has 2 stale Ch05 commits (1b207a9, d6eb93a) from a different machine; we did NOT merge them. Decision was to keep our v6 work canonical and let user reconcile main later if needed.
- All Ch07-Ch12 work + system improvements are on the session-2026-05-07 branch.

### Keep-alive
- Background bash process (PID 299329, Claude background ID `btahjhvsa`) writes to `/tmp/book-factory/keep-alive.log` every 30s
- Started 2026-05-06 to prevent WSL2 VM idle suspension
- Stop on session resume if not needed: `kill 299329` or stop background task

### Memory
Auto-memory updated this session:
- `feedback_team_lead_as_controller.md` — control-theory framing of team-lead role
- `feedback_verify_dispatch_action.md` — supersession + heartbeat + two-strike rule
- `feedback_system_improvements_doc.md` — maintain `instances/vllm/trace/system-improvements.md` at every session start
- `reference_agent_respawn.md` — updated boot prompt template (no idle-and-wait language; full dispatch baked into spawn prompt)

### Knowledge modules
| Module | Facts | Prefix |
|---|---|---|
| prefix-cache.md | 18 | K |
| tensor-parallelism.md | 19 | T |
| expert-parallelism.md | ~28 | E (with overlap from impl/tester/writer/reviewer phases) |
| multi-token-prediction.md | ~30 | M |
| dcp-pcp.md | 29 | D |
| kv-offload.md | 28 | O |

P1-1 fix means `learn.py compact()` should now work; manual compaction pass deferred to a future session pause.

---

## 7. Cadence trajectory (v6 chapters)

| Ch | Lines | Words | Mapping | Tests | Source files | # REFERENCE | Cycles |
|---|---|---|---|---|---|---|---|
| 04 | 712 | 3064 | 13 | 48 | 5 | 65 | 1 |
| 05 | 757 | 3849 | 21 | 74 | 7 | 61 | 1 |
| 06 | 655 | 3351 | 40 | 97 | 6 | 60 | 1 |
| 07 | 859 | 4440 | 72 | 83 | 5 | 60 | 1 |
| 08 | 1051 | 6058 | 122 | 144 | 8 | 64 | 1 |
| 09 | 1204 | 7792 | 151 | 204 | 10 | 66 | 1 |
| 10 | 1345 | 8888 | 206 | 311 | 11 | 151 | 1 |
| 11 | 1394 | 8124 | 149 | 474 | 12 | 78 | 1 |
| 12 | 1583 | 10178 | 285 | 314 | 22 | 81 | 1 |

**9 chapters in a row, single-cycle APPROVED.** v6 cadence is robust at N=9.

---

## 8. Misc operational notes

- TaskCreate auto-broadcasts to all team members regardless of owner — ongoing P0-2 nuisance. Best practice: don't TaskCreate for in-flight pipeline; always assign owner+blockedBy at creation.
- Stale agents respond defensively to misroutes — they're correct to do so. Standdown SendMessage once, then ignore further idle pings.
- In-process agents need full dispatch in boot prompt; "go idle and wait" is the failure mode (P0-1 fixed).
- Two-reviewer convergence (reviewer-2 + reviewer-3 both APPROVING Ch12 independently) is a useful pattern — cheap insurance against single-reviewer bias when registry has duplicates.

---

## 9. To-do at next session start

1. Read this file. Read `instances/vllm/trace/state.json`. Read `instances/vllm/trace/system-improvements.md`.
2. Verify Ch12 published_v6 (state.json + delivery file). If missing, complete archivist-2's work.
3. Verify Ch13 brief on disk. If missing, dispatch archivist for brief.
4. If both are done: dispatch Ch13 implementer (M-prefix? L-prefix? per archivist's brief recommendation).
5. Resume pipeline through Ch28.
6. (Optional) Run `python3 scripts/learn.py compact <module>` on the 6 over-trigger modules — P1-1 fix should make it work now.
