# repo2book — Multi-Agent Book Factory

Turn any code repository into a source-grounded technical book. This is the **framework** — the current instance is `vllm-from-scratch` (a 28-chapter book on the vLLM inference engine).

See `repo2book.json` for instance-specific config (source repo, outline path, reader profile).

## Architecture

```
repo2book/                              ← Framework (this repo)
├── repo2book.json                      ← Master config
├── CLAUDE.md                           ← You are here
├── .gitmodules                         ← Submodule tracking
│
├── .claude/                            ← Project-scoped config
│   ├── settings.local.json
│   ├── agents/                         # (book-editor, implementer, tester, writer, reviewer, archivist)
│   ├── teams/book-factory.json
│   └── skills/svg-diagram/
│
├── schemas/
│
├── scripts/                            ← Framework CLI tools
│   ├── archivist.py, learn.py, decide.py
│   ├── hook_pipeline.py, team_orchestrator.py
│   ├── lint_formulas.py, lint_source_grounding.py
│   └── guard_narrative.py
│
├── wisdom/                             ← Shared across ALL instances
│
└── instances/
    └── vllm/                           ← vLLM book instance (complete)
        ├── repo2book.json              ← Instance config
        ├── artifacts/                  ← Chapter content (all chapters)
        ├── book/book-outline.json       ← 28-chapter outline
        ├── source/                     ← Git submodule: vllm-project/vllm
        │   └── (vllm source at tracked commit)
        ├── knowledge/                  ← Repo-specific facts
        ├── trace/                      ← Project long-term memory
        │   ├── INDEX.md, state.json
        │   ├── chapters/               # Per-chapter trace entries
        │   ├── cross-chapter/          # Framework-wide decisions
        │   └── sessions/               # Session summaries
        └── .claude/agents/archivist.md ← Per-instance archivist
```

## ⛔ HARD RULE: Narrative Guardian

**You (the main orchestrator) CANNOT directly write or edit any
`artifacts/*/narrative/chapter.md` file.** These files are OWNED by the Writer agent.

Every narrative edit MUST go through the Writer agent or the Chapter Pipeline.
If you find yourself wanting to `Edit` or `Write` a narrative file — STOP.
Run `python3 scripts/guard_narrative.py check {chapter_id}` first.

**Only modify:** .claude/agents/, schemas/, scripts/, CLAUDE.md, repo2book.json.
**FORBIDDEN to modify directly:** artifacts/*/narrative/chapter.md.

If the chapter quality is wrong, fix the PROMPTS, not the chapter content.

## Agent Teams Pipeline

### 架构

**子 agent 没有 Agent 工具，无法嵌套 spawn。** Team Lead 角色由主 session（我，即 Claude Code 主进程）扮演。

```
用户 ⇄ 主 session (Team Lead)  ← 唯一有 Agent 工具，可 spawn/kill/restart
         │
         ├── book-editor   ← 内容调度（TaskCreate、SendMessage、拓扑决策）
         │
         └── Pipeline: implementer → tester → writer → reviewer → archivist
              │              │           │          │            │
              源码实现      反压闸门    教育叙事    最终闸门    终端备份

    Lateral: Reviewer ↔ Writer (REVISE, 直接 SendMessage)
             Writer ↔ Implementer (修改请求, 直接 SendMessage)
    Loop >3 → 升级到 Team Lead (主 session)
```

### Startup（主 session 执行）

```
1. python3 scripts/setup.py                         # 一次性初始化
2. 主 session 用 Agent 工具 spawn 全部 6 个 agent    # 后台运行
   - book-editor (内容调度)
   - implementer, tester, writer, reviewer, archivist (流水线)
3. 报告用户: "团队就绪"
```

### 每章流程

```
用户提需求 → 主 session 判断意图：
  - "写第N章" → 告诉 book-editor 启动 pipeline
  - "全书进度" → 读 state.json，汇总
  - "第N章太枯燥" → 调度 writer 重写 → reviewer 重审
  - agent 挂了 → 主 session 重新 spawn
```

### Per-Chapter Pipeline (book-editor 调度)

```
book-editor 创建 5 个 tasks → SendMessage 给 implementer
  → implementer 完成 → hook → tester
  → tester 完成 → hook → writer
  → writer 完成 → hook → reviewer
  → reviewer APPROVED → hook → archivist 备份 → 章节发布
```

### Lateral Communication
- **Reviewer → Writer**: REVISE，直接 SendMessage
- **Writer → Implementer**: 修改请求，直接 SendMessage
- **Loop >3**: 升级到主 session (Team Lead)

### Organizational Topology

The pipeline topology is NOT fixed. Each chapter can override the default `linear` mode.

| Mode | Agents/Stage | When to Use |
|------|-------------|-------------|
| `linear` (default) | 1 per stage | Standard chapters, well-understood concepts |
| `pair` | 2 on implementer | Complex algorithms, unfamiliar code patterns |
| `panel` | 2 reviewers | First-of-part chapters (set pattern for all subsequent) |
| `writer_editor` | 2 on writer (writer+editor) | Style-critical chapters, many formulas |
| `swarm` | 2+ on all stages | Very complex, brand new concepts, Part-openers |

**Changing topology for a chapter:**
```bash
python3 scripts/decide.py propose-topology {chapter_id} {mode} "Reason for change"
```

**Decision protocols** (see `repo2book.json` → `pipeline.topology.decision_protocol`):
- **Vote**: Proposal → agents vote (APPROVE/REJECT/ABSTAIN) → tally → decision
  - majority (>50%): standard decisions
  - supermajority (>66%): structural decisions (topology change, chapter redesign)
  - consensus (100%): fundamental decisions (outline changes, prompt rewrites)
- **Discuss**: Proposal → round 1 arguments → round 2 counter → synthesis → vote_or_escalate
- **Escalate**: Any agent can escalate to Lead when stuck (>3 review cycles, pair deadlock, pipeline stalled)

## Agent Memory System — Knowledge & Wisdom

Agents **self-improve through experience**. This is a two-tier system:

### Knowledge (知识) — Repo-Specific Facts

**Location**: `knowledge/` (per-instance)
**What**: File locations, API patterns, gotchas, conventions of THIS repo
**Growth**: Linear (one fact per encounter)
**Lifespan**: TTL with decay — unused facts archived after 30 days
**Anti-bloat**: Max 15 facts per module file; oldest 5 get LLM-compacted into summary

```
knowledge/
├── INDEX.md                    ← Module map: which chapters use which modules
├── modules/
│   ├── scheduler.md            ← Facts about scheduler.py
│   ├── attention.md            ← Facts about attention backends
│   └── kv-cache.md             ← Facts about KV cache
└── archive/                    ← Stale facts (unused > 30 days)
```

### Wisdom (智慧) — Universal Patterns

**Location**: `wisdom/` (framework-level, shared across all instances)
**What**: High-level patterns that apply to ANY repo2book project
**Growth**: Logarithmic (pattern must re-occur in 2+ repos)
**Lifespan**: Permanent, refined over time
**Gating**: Must appear in 2+ repos → book-editor promotes

```
wisdom/
├── INDEX.md                    ← Pattern catalog with role relevance
├── debugging.md                ← F.linear shapes, CUDA mismatches, SVG clipping
├── testing.md                  ← Preemption test design, OOM paths, Docker cmd
├── writing.md                  ← Formula rules, code walkthrough, 大白话 spectrum
└── architecture.md             ← Backpressure gates, lateral comm, fix prompts not chapters
```

### The Self-Learning Loop

```
Agent works on chapter → encounters issue → resolves it
  ↓
Was this a repo-specific fact or a universal pattern?
  ↓                    ↓
Knowledge DB          Wisdom DB
(tagged, TTL'd)       (gated, categorized)
  ↓                    ↓
Future agents         Future agents
query before work     always have access

Cross-role enrichment:
  Implementer discovers source gotcha → tester benefits when writing tests
  Tester discovers edge case → writer explains it in narrative
  Writer discovers formula pitfall → reviewer adds to auto-REJECT list
```

### Learning Protocol (per agent, per task)

**Before work**: Query `knowledge/INDEX.md` for the relevant module. Read the module file.
Filter by your role. Also read `wisdom/` category files ranked by your role's priority.

**After work**: Run `python3 scripts/learn.py extract {chapter_id} {role}`. This prompts
you to:
1. Extract: What facts did you learn about THIS repo?
2. Classify: Repo-specific → `knowledge/modules/`. Universal pattern → propose to `wisdom/`.
3. Compact: If module file exceeds 15 facts, oldest 5 are LLM-summarized into one.

**Role-specific query priorities** (which wisdom categories to read first):
- implementer: debugging > architecture > testing > writing
- tester: testing > debugging > architecture > writing
- writer: writing > architecture > debugging > testing
- reviewer: writing > architecture > testing > debugging
- book-editor: architecture > all equally

## Writing Standard

### Every chapter section MUST have BOTH:
1. **Source Trail:** Open a specific source file, show the code, explain where it lives
   - Format: `{source_dir}/path/to/file.py:L123`
2. **Theory Deep Dive:** Derive the math, prove correctness, explain the WHY

### The 5-step rhythm per major section:
```
1. Open {source_dir}/xxx.py:L123 → ClassName.method()     ← Source Trail
2. What does this method do? Why?                          ← Bridge
3. Let's derive the principle from scratch...              ← Theory Deep Dive
4. Our simplified implementation: [code]                   ← Implementation
5. Original adds X and Y because...                        ← Source Diff
```

### Anti-patterns (both extremes are WRONG):
- ALL source references, zero theory → "source dump" — reader learns nothing
- ALL theory, zero source refs → "textbook copy" — not about the source
- EVERY section: source entry point + theory derivation

## Formula Rules (NON-NEGOTIABLE)

### Blocking (auto-REJECT):
- `\text{}` → use `\mathrm{}`
- `\boxed{}` → use markdown bold header above formula
- `\tag*{}` → put annotation outside `$$` block
- `\frac` inside inline `$...$` → promote to `$$...$$` block
- `$$` on same line as formula content → separate lines

### Allowed inline ($...$):
- Single symbols: `$x$`, `$\alpha$`
- Simple expressions: `$d_k$`, `$\sqrt{2}$`, `$1/\sqrt{d_k}$`

Run `python3 scripts/lint_formulas.py artifacts/{chapter_id}/narrative/chapter.md` before marking complete.

## Source Grounding Rules (NON-NEGOTIABLE)

### Requirements:
- Every Cell (2-7) must have 1+ source file reference
- Every implementation function must have `# REFERENCE: ...` comment
- Source Mapping Table must have 5+ rows
- impl-notes.md must list 3+ source files

### Reference format:
- Full path: `{source_dir}/path/to/file.py:L123`
- Short path: `file.py:L123` (after full path established)
- Class.method: `flash_attn.py → FlashAttentionImpl.forward()`

Run `python3 scripts/lint_source_grounding.py artifacts/{chapter_id}/` before marking complete.

## By-Repo Instance Structure

Each repo lives in `instances/{repo_name}/` with its own team, knowledge, trace, and artifacts.

```
repo2book/
├── repo2book.json              ← Framework config (shared)
├── wisdom/                     ← Shared wisdom (all instances)
├── scripts/                    ← Framework CLI tools
├── .claude/agents/             ← Agent definitions & prompts
│
└── instances/
    └── vllm/                   ← vLLM book instance
        ├── repo2book.json      ← Instance config
        ├── knowledge/          ← Repo-specific facts (TTL'd)
        ├── trace/              ← ★ Project long-term memory
        │   ├── INDEX.md         # Master index of everything
        │   ├── state.json       # Single source of truth
        │   ├── decisions/       # Every design decision + rationale
        │   ├── deliveries/      # Per-chapter delivery records
        │   ├── user_interactions/ # User feedback, Q&A
        │   └── context_summaries/ # Session summaries for rehydration
        └── .claude/
            └── agents/
                └── archivist.md  # ★ Per-instance archivist agent
```

**artifacts/ and book/ stay at root** for the current instance (referenced by instance config).

## The Archivist — Long-Term Memory System

**Why this exists**: Claude Code sessions compress context after ~50 turns. After weeks of work, agents forget why decisions were made. The user has to re-explain preferences. Bugs are re-discovered. Chapter quality degrades.

**The archivist agent is the project's long-term memory.** One archivist per instance.

### Archivist Duties
1. **Record** every significant event (decisions, deliveries, user feedback, bugs)
2. **Rehydrate** agents with context briefs before they start work
3. **Summarize** sessions for continuity across session boundaries
4. **Answer** queries about past work from agents or the user
5. **Alert** when context loss risk is high (>50 turns, >24h sessions)
6. **Cross-reference** related entries (bug→fix→test→chapter section)

### Context Rehydration Protocol

Before ANY agent works on a chapter, the archivist provides:
```
1. Current project state (from state.json)
2. Relevant past decisions (from trace/decisions/)
3. Previous deliveries for this chapter (from trace/deliveries/)
4. User feedback about this chapter (from trace/user_interactions/)
5. Most recent session summary (from trace/context_summaries/)
6. Wisdom entries relevant to the agent's role
7. Knowledge entries for the relevant module
```

### Trace Entry Format
```markdown
# [Title]
- Type: decision | delivery | user_interaction | bug | design_change
- Chapter: (if applicable)
- Date: YYYY-MM-DD
- Agents involved: [agent1, agent2]
- User present: true/false
- Tags: [tag1, tag2]

## What happened
(factual account)

## Why it matters
(impact on future work)

## What to remember
(for context rehydration — under 200 words)
```

### Archivist CLI
```bash
python3 scripts/archivist.py record --type decision --chapter 04 --title "..." --what "..." --why "..."
python3 scripts/archivist.py brief --chapter 14 --role implementer
python3 scripts/archivist.py summary --date 2026-05-04 --accomplishments "..."
python3 scripts/archivist.py alert --session-turns 45
python3 scripts/archivist.py state  # View current state
python3 scripts/archivist.py query --chapter 04 --type delivery
```

## Creating a New Book Instance

1. Clone repo2book as template
2. Clone the target repo into `{source_dir}/`
3. Initialize instance: `python3 scripts/archivist.py init --instance {repo_name}`
4. Configure `instances/{repo_name}/repo2book.json`
5. Write `book/book-outline.json` for the target repo
6. Create the team via `TeamCreate` with name `book-factory-{repo_name}`
7. Spawn the archivist first (records everything from the start)
8. Start the pipeline:
```bash
python3 scripts/team_orchestrator.py pipeline {chapter_id}
```

## Current Instance: vLLM

- **Repo**: https://github.com/vllm-project/vllm
- **Source dir**: `instances/vllm/source/`
- **Outline**: 28 chapters, 5 parts (see `book/book-outline.json`)
- **Progress**: 13 chapters published (Parts 1-2 complete), 15 remaining (Parts 3-5)

## Common Pitfalls

1. **Don't write generic textbook chapters.** Every concept must trace to a source file:line.
2. **Don't over-correct to source-only.** Theory and source references are NOT mutually exclusive.
3. **Run formula linter before claiming complete.** `\text{}` in formulas is the #1 issue.
4. **Run source grounding linter before claiming complete.**
5. **`F.linear` weight shape is `[out, in]`.** `nn.Linear(in, out)` stores weight as `[out, in]` and `F.linear(x, weight)` does `x @ weight^T`.
6. **Chapter IDs must be unique across old and new outlines.**
