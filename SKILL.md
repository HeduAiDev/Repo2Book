---
name: agent-teams
description: Use when creating multi-agent teams with split-pane mode, configuring team member models, or setting up collaborative agent workflows in Claude Code
---

# Agent Teams with Split-Pane Mode

## Overview

Claude Code supports research-preview **agent teams**: multiple agents collaborating with lateral communication, task dependency chains, and optional visual split-pane display. On Linux/WSL the backend is **tmux**; on macOS it uses **iTerm2 native split-panes**.

**Core principle:** One team lead orchestrates, team members execute independent tasks in isolated panes, communicating via SendMessage.

## Prerequisites

### Required Settings

In `/root/.claude/settings.json` or project `.claude/settings.local.json`:

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### Platform Requirements

| Platform | Backend | Requirements |
|----------|---------|-------------|
| Linux/WSL | tmux | `tmux` installed (`apt install tmux`). Must run Claude Code INSIDE a tmux session. |
| macOS | iTerm2 | iTerm2 terminal. Automatic detection. |
| Other | In-process only | No visual split-pane. Agents run inline (toggle with Shift+Down). |

**Critical for tmux:** Claude Code detects `TMUX` env var to decide whether to create tmux panes. Always start Claude Code inside a tmux session.

## Project-Level vs Global Configuration

Agent teams span multiple configuration layers. Only some are project-scopable:

| Layer | Global Path | Project Path | Project-Scopable? |
|-------|------------|-------------|-------------------|
| Feature flag | `~/.claude/settings.json` | `.claude/settings.local.json` | **Yes** |
| Custom agent defs | `~/.claude/agents/*.md` | `.claude/agents/*.md` | **Yes** |
| Team config + inboxes | `~/.claude/teams/{name}/` | _(not supported)_ | **No** — always global |
| Tasks | `~/.claude/tasks/{name}/` | _(not supported)_ | **No** — always global |

### Project-Scoped Setup (Recommended)

In your project root `.claude/settings.local.json`:
```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

Define project-specific agents in `.claude/agents/`:
```markdown
---
name: code-reviewer
color: purple
tools: Read, Grep, Glob, Bash
---

You are a code reviewer for this project...
```

**`model` 字段无需写**——Agent 工具在 team spawn 时不读取 `.md` 的 `model` 字段（v2.1.126 实测）。所有 agent 默认 opus，由 `ANTHROPIC_DEFAULT_OPUS_MODEL` env var 决定最终模型。

### Global-Only Limitation

Team configs (`TeamCreate`) always write to `~/.claude/teams/` regardless of which project you're in. If you need project isolation for team data, use unique team names prefixed with the project name (e.g., `vllm-book-ch4-rewrite` instead of `ch4-rewrite`).

## Team Configuration Reference

Team configs live at `/root/.claude/teams/{team-name}/config.json`. Created automatically by `TeamCreate`, or built manually.

### Member Schema

```json
{
  "agentId": "name@team-name",
  "name": "display-name",
  "agentType": "general-purpose | Explore | Plan",
  "model": "opus | sonnet | haiku  (Agent 工具写入；不传默认 opus。.md 的 model 不生效)",
  "prompt": "custom system prompt for this agent",
  "color": "blue | green | yellow | red | purple",
  "backendType": "tmux | in-process",
  "tmuxPaneId": "",
  "cwd": "/absolute/path/to/working/dir",
  "subscriptions": [],
  "planModeRequired": false,
  "isActive": true
}
```

### Model Configuration — Agent 团队的模型控制

#### 核心发现（v2.1.126，tmux 模式）

Agent 工具的 team spawn 路径**硬编码** model 解析。`opus` 始终解析为 `claude-opus-4-7`，然后以 `--model claude-opus-4-7` 传给 agent。

**所有客户端方法均无效（已逐项验证）：**

| # | 尝试的方法 | 结果 |
|---|-----------|------|
| 1 | settings.json `env.ANTHROPIC_DEFAULT_OPUS_MODEL` | Agent 启动命令中 env 白名单不包含此变量 |
| 2 | `.bashrc` export | Shell 有变量但 `--model` CLI 参数覆盖 |
| 3 | tmux `setenv` | 同上，`--model` 优先级更高 |
| 4 | `.claude/agents/*.md` `model:` 字段 | Team spawn 时不读取 |
| 5 | settings.json `agents.xxx.model` | Team spawn 时也不读取 |

**Agent 启动命令（captured from tmux pane）：**
```
env CLAUDECODE=1 CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 ANTHROPIC_BASE_URL=...
  /path/to/claude --model claude-opus-4-7 ...
```
注意：`ANTHROPIC_BASE_URL` 被传入，但 `ANTHROPIC_DEFAULT_OPUS_MODEL` 没有。`--model` 是已解析的**具体模型 ID**，绕过了别名系统。

#### in-process vs tmux 模式差异

| | in-process（不在 tmux） | tmux（在 tmux 内） |
|---|---|---|
| backendType | `in-process` | `tmux` |
| 窗格 | 无，内联运行 | `tmux split-window` 创建真实窗格 |
| 环境继承 | 完整（共享父进程） | 部分（env 白名单过滤） |
| `--model` | `claude-opus-4-7` | `claude-opus-4-7`（都一样） |
| 实际模型 | 由父进程 env var 决定 | 由 `--model` CLI 决定 |

**在 in-process 模式下，agent 共享父进程环境，`ANTHROPIC_DEFAULT_OPUS_MODEL` 可能生效。** 但在 tmux 模式下完全无效。

#### 唯一解法：API 端映射

在 API 端点（`ANTHROPIC_BASE_URL`）上配置模型路由：

```
claude-opus-4-7    → deepseek-v4-pro[1m]
claude-sonnet-4-6  → deepseek-v4-pro[1m]
claude-haiku-4-5   → deepseek-v4-flash
```

这样无论 Claude Code 传什么 `--model`，API 层都会路由到正确后端。改一次永久生效，和 Claude Code 版本无关。

#### 配置清单

| 位置 | 配置 | 作用 |
|------|------|------|
| API 端点 | `claude-opus-4-7` → 目标模型 | **唯一生效的方案** |
| `settings.json` | `ANTHROPIC_DEFAULT_OPUS_MODEL` | 仅影响主 session，不影响 agent spawn |
| `settings.json` | `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` | 启用 agent teams 功能 |
| `Agent()` 调用 | 不传 `model` 参数 | 避免报错（只接受 opus\|sonnet\|haiku） |
| `.claude/agents/*.md` | 写 `prompt`、`tools`、`color` | model 字段不生效，但其他字段正常 |

#### 补充：如果 API 端无法改

在 agent 窗格中，`/model` 列表已包含目标模型。可以手动切换：
1. Agent spawn 后，SendMessage 告诉 agent 运行 `/model <目标模型名>`
2. Agent 切换模型后，再分配实际任务
3. 这需要 agent 支持 slash command 执行

### backendType Field — Controlling Split-Pane

This is THE key field for split-pane. It determines how the agent appears:

| `backendType` | Visual | When to Use |
|---------------|--------|-------------|
| `"tmux"` | Separate tmux pane (visible split) | Linux/WSL: want side-by-side view |
| `"in-process"` | Inline, no separate pane | Research agents, sub-tasks, no visual feedback needed |

**Team lead** always has `"backendType": ""` and `"tmuxPaneId": ""` — they own the primary terminal.

**Pane lifecycle:**
1. Team member spawned → `tmux split-window` executed automatically
2. New pane gets `tmuxPaneId: "%N"` (tmux pane ID)
3. API env vars auto-propagated to the pane process
4. Pane auto-closes when lead exits or teammate is shut down

**No pane sizing/layout controls** — panes are auto-managed.

## Communication Patterns

### SendMessage — Lateral Communication

```json
SendMessage({
  to: "agent-name",
  summary: "5-10 word preview",
  message: "message content (plain text or structured JSON)"
})
```

Messages are stored in `/root/.claude/teams/{team}/inboxes/{agent-name}.json` with read/unread tracking. Team lead receives idle notifications automatically.

**No broadcast:** Communication is point-to-point only. `to: "*"` is NOT supported. To reach multiple teammates, send separate messages to each. Use `to: "agent-name"` or `to: "agent@team-name"`.

**Peer-to-peer supported:** Any member can SendMessage to any other member — not just lead→member. Implementer can ask Writer for clarification, Reviewer can query Tester, etc. Members discover each other by reading the team config at `~/.claude/teams/{team-name}/config.json` (lists all `members[].name`). No architectural restriction limits communication to hub-and-spoke.

### Structured Message Types

1. **Task assignment:** `{"type": "task_assignment", "taskId": "1", ...}`
2. **Shutdown request:** `{"type": "shutdown_request", "reason": "..."}`
3. **Idle notification:** Auto-sent by system when agent becomes available

### Task Coordination

Tasks support dependency chains via `blockedBy`/`blocks`:

```json
{
  "id": "4",
  "subject": "Synthesize findings",
  "description": "...",
  "status": "pending",
  "blockedBy": ["1", "2", "3"],
  "blocks": []
}
```

**Pattern:** Research tasks → Synthesis task (blocked by all research).

**Critical: task completion does NOT auto-trigger the next agent.** Dependencies only gate TaskList — an agent checking TaskList won't see an unblocked task unless they explicitly poll. You must actively hand off.

### Handoff Patterns (Passing Work to the Next Agent)

**Option 1: Peer-to-peer (simplest)**

Agent A finishes → SendMessage directly to Agent B:
```json
SendMessage({
  to: "reviewer",
  summary: "Implementation ready for review",
  message: "Task #1 complete. scheduler.py at artifacts/04-continuous-batching/implementation/. Your review task #2 is now unblocked."
})
```
Agent B receives this as a teammate-message and starts working. No lead involvement needed.

**Option 2: Lead relay (centralized control)**

Agent A → reports to lead → lead assigns to Agent B:
```
Agent A: SendMessage({to: "team-lead", summary: "Done", message: "Task #1 complete"})
Lead:    TaskUpdate({taskId: "1", status: "completed"})  // unblocks #2
Lead:    SendMessage({to: "agent-b", summary: "Your turn", message: "Task #2 unblocked"})
```

**Option 3: TaskCompleted hook (automated, requires setup)**

Configure in `settings.json`:
```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "TaskUpdate",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/skills/agent-teams/handoff.sh"
      }]
    }]
  }
}
```
The handoff script reads task status and sends SendMessage to the agent assigned to the now-unblocked task. Most powerful but requires scripting.

**Recommendation:** Use Option 1 for simple chains (2-3 agents). Use Option 2 for complex workflows or when the lead needs visibility into every transition. Only invest in Option 3 for recurring pipeline-style teams (like the vLLM book's Implementer→Tester→Writer→Reviewer chain).

### Hook Events

| Event | Fires When |
|-------|-----------|
| `TeammateIdle` | Team member becomes idle/available |
| `TaskCompleted` | Team member completes a task |

Register hooks in `settings.json` to automate reactions (e.g., auto-assign next task).

## Reliable Pipelines (A→B→C with Gates)

**The limitation:** Built-in `blockedBy`/`blocks` are passive constraints — they prevent premature execution but don't actively drive the pipeline forward. For a production pipeline you need: automatic stage transitions, failure handling, quality gates, and state persistence.

### The Lead-Orchestrated Pattern (Recommended)

The team lead acts as the **pipeline executor**. This is the pattern used by the vLLM book factory's `pipelines/chapter_pipeline.py` — each stage has a quality gate, and the lead only advances when the gate passes:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Agent A  │────→│  GATE A  │────→│ Agent B  │
│ (Implement)   │(tests pass?)   │ (Write)   │
└──────────┘     └──────────┘     └──────────┘
                       │                │
                       ↓ fail           ↓
                  Back to A        ┌──────────┐
                                   │  GATE B  │
                                   │(review ok?)│
                                   └──────────┘
                                       │
                                       ↓ pass
                                       DONE
```

**Lead orchestration protocol:**

```json
// Step 1: Lead creates pipeline tasks with dependencies
TaskCreate({ subject: "Implement X", description: "..." })      → task #1
TaskCreate({ subject: "Test X", description: "...",              → task #2
  metadata: { gate: "tests_pass", gateCheck: "pytest returns 0" }})
TaskCreate({ subject: "Write narrative", description: "..." })   → task #3
TaskUpdate({ taskId: "2", addBlockedBy: ["1"] })
TaskUpdate({ taskId: "3", addBlockedBy: ["2"] })

// Step 2: Lead spawns Agent A, assigns task #1
Agent({ name: "implementer", ... })
SendMessage({ to: "implementer", summary: "Task #1: Implement", ... })

// Step 3: Lead monitors Agent A's inbox for completion
// When Agent A finishes:
//   (a) Agent A: TaskUpdate({ taskId: "1", status: "completed" })
//   (b) Agent A: SendMessage({ to: "team-lead", summary: "Done", ... })

// Step 4: Lead runs GATE CHECK before advancing
// If tests pass → spawn Agent B, assign task #2
// If tests fail → SendMessage back to Agent A with fix instructions

// Step 5: Repeat for each stage
```

**Key principle:** The lead, not the system, owns pipeline progression. Each gate is explicitly checked before the next agent is spawned or assigned. This gives you:
- **Failure handling:** Gate fails → lead sends fix request back to previous agent
- **Retry control:** Lead decides whether to retry or abort
- **State visibility:** Lead always knows exactly where the pipeline is
- **Persistence:** Task state survives session restarts

### Automation Spectrum

| Level | Mechanism | Reliability | Effort |
|-------|----------|-------------|--------|
| **Manual** | Lead polls TaskList, manually dispatches each stage | High (human in loop) | Low |
| **Peer-driven** | Each agent SendMessages the next when done | Medium (agents must know successors) | Low |
| **Hook-assisted** | TaskCompleted hook fires script that checks unblocked tasks | Medium (script is single point of failure) | Medium |
| **Orchestrator script** | A Python script (like `chapter_pipeline.py`) reads task state, spawns agents, checks gates, advances pipeline | High (script logic + human oversight) | High |

### The vLLM Book Factory Pipeline (Real Example)

This project uses a **custom orchestrator** (`agents/book_editor.py` + `pipelines/chapter_pipeline.py`) rather than the built-in agent teams. Its pipeline:

```
Implementer ──[tests_pass]──→
  Tester ──[report_valid]──→
    Writer ──[lint_pass]──→
      Reviewer ──[review_approved]──→ Published
```

Each gate is a **Backpressure gate**: if it fails, the pipeline stops and the previous agent is asked to fix. No downstream agent ever sees broken work. This is implemented via file-system state tracking (`context.json` per chapter) rather than the built-in task system.

### Bottom Line

The built-in agent teams provide **task dependency primitives** (`blockedBy`/`blocks`, `TaskCompleted` hook). They do NOT provide a pipeline execution engine. For reliable A→B→C pipelines with gates, you need one of:

1. **Lead orchestration** — lead checks gates, spawns next agent only on pass (good for most cases)
2. **Custom orchestrator** — a script/agent that implements the pipeline state machine (good for recurring pipelines)
3. **Hook-based automation** — TaskCompleted hooks that check gates and auto-dispatch (good for simple linear chains)

## Complete Workflow — 启动 Agent Team 的完整步骤

### Step 0: 一次性环境配置

**API 端（推荐）：** 在 `ANTHROPIC_BASE_URL` 上配置模型路由——`claude-opus-4-7` → 目标模型。Agent 的 `--model` 无法通过客户端 env var 改变，必须服务端映射。

**settings.json 最小配置：**
```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
    "ANTHROPIC_BASE_URL": "http://your-api:port"
  }
}
```

`ANTHROPIC_DEFAULT_OPUS_MODEL` 仅影响主 session——agent team spawn 时不读取此变量。

### Step 1: 进入 tmux（Linux/WSL）

```bash
tmux new-session -s book-factory
# 在 tmux 内启动 Claude Code
```

Claude Code 检测 `$TMUX` 环境变量，自动为团队成员创建 tmux 分屏。

### Step 2: 创建 team

```json
TeamCreate({
  team_name: "book-factory",
  description: "Multi-agent book writing system"
})
```

### Step 3: 定义 agent（.claude/agents/*.md）

```yaml
---
name: implementer
color: blue
tools: Read, Write, Bash, Grep, Glob, Agent
---

# Implementer Agent
你是一个代码实现 agent……
```

**注意：** `model` 字段可以不写——写了也不会被 Agent 工具读取。真正起作用的字段是 `prompt`、`tools`、`color`。

### Step 4: Spawn agent 成员

```json
// ✅ 正确写法：不传 model 参数
Agent({
  subagent_type: "implementer",   // 引用 .claude/agents/implementer.md
  team_name:     "book-factory",
  name:          "implementer",
  description:   "Start implementer agent",
  prompt:        "可选——覆盖 .md 里的 prompt 的特定部分"
})

// ❌ 错误写法：传 model: "inherit" → Agent 工具报错
// Agent 工具只接受 model: "opus" | "sonnet" | "haiku" 三个值
```

**Agent 工具自动：**
- 读取 `.claude/agents/implementer.md` 的 prompt + tools + color
- 默认 opus 模型 → 写入 `model=claude-opus-4-7` 到 team config
- 在 tmux 中创建分屏（如果检测到 tmux）
- API 调用时 `ANTHROPIC_DEFAULT_OPUS_MODEL` 映射为 deepseek-v4

### Step 5: 批量 spawn（4 个角色一次性启动）

```json
// 4 个独立调用，可以一起发出
Agent({ subagent_type: "implementer", team_name: "book-factory", name: "implementer", description: "Start implementer" })
Agent({ subagent_type: "tester",       team_name: "book-factory", name: "tester",       description: "Start tester" })
Agent({ subagent_type: "writer",       team_name: "book-factory", name: "writer",       description: "Start writer" })
Agent({ subagent_type: "reviewer",     team_name: "book-factory", name: "reviewer",     description: "Start reviewer" })
```

**注意：** 如果 team config 里已有同名成员（即使 `active=False`），Agent 工具会创建 `name-2`。启动前先清理 config 里的旧条目。

### Step 6: 分配任务

```json
// 先创建任务
TaskCreate({ subject: "Implement Chapter 5", description: "..." })

// 发送给 agent
SendMessage({
  to: "implementer",
  summary: "Task: Implement Chapter 5",
  message: "请完成 Chapter 5 的代码实现。完成后 TaskUpdate 并通知我。"
})
```

### Step 7: 监控和握手

```
Agent 完成任务 → TaskUpdate({ status: "completed" })
              → SendMessage({ to: "team-lead", summary: "Done" })

Lead 收到通知 → 检查门禁 → 分配下一个任务给下一个 agent
```

### `/resume` 后恢复 team

```
1. /resume → 恢复 lead session
2. team config 还在，但成员全是 active=False
3. 重新执行 Step 4，spawn 所有成员
4. 成员名字会变 name-2（因为旧的还在 config 里）
   → 可以先清理 config 里的旧条目，或者接受 -2 后缀
```

## Common Mistakes

- **Not in tmux, expecting split-pane** — Linux/WSL must be inside a tmux session for panes
- **Passing `model: "inherit"` to Agent tool** — throws validation error. Agent tool only accepts `"opus"|"sonnet"|"haiku"`. Simply omit the `model` parameter entirely
- **Spawning all agents with `backendType: "tmux"`** — use tmux for agents you want to watch; in-process for background tasks
- **Not setting task dependencies** — agents may start work before prerequisites are done
- **Forgetting to SendMessage after TaskCreate** — creating a task doesn't notify agents; you must explicitly send task assignments
- **Waiting for idle notifications instead of checking tasks** — idle notifications are informational; check TaskList for actual progress

## Task Assignment Pattern (Quick Reference)

```json
// 1. Create
TaskCreate({ subject: "Do X", description: "Details..." })
// 2. Block
TaskUpdate({ taskId: "2", addBlockedBy: ["1"] })
// 3. Assign
SendMessage({ to: "worker", summary: "Task #2 assigned", message: "..." })
// 4. Track
TaskList()  // check statuses
// 5. Complete
TaskUpdate({ taskId: "2", status: "completed" })
```

## Team Config File Locations

```
/root/.claude/teams/{team-name}/config.json     ← Team definition + members
/root/.claude/teams/{team-name}/inboxes/        ← One JSON file per member
/root/.claude/tasks/{team-name}/{id}.json       ← One JSON file per task
/root/.claude/teams/{team-name}/.lock           ← Concurrency guard
/root/.claude/teams/{team-name}/.highwatermark  ← Last-read task ID
```

## Version Requirements

| Version | Capability |
|---------|-----------|
| v2.1.32+ | Agent teams (research preview) |
| v2.1.33+ | Tmux messaging, TeammateIdle/TaskCompleted hooks |
| v2.1.77+ | Pane cleanup, SendMessage auto-resume, iTerm2 detection |
| v2.1.126 | **已验证版本**——`.md` model 字段不生效；tmux 模式下 `--model` 硬编码；唯一解法 API 端映射 |

## Known Limitations (v2.1.126)

| 问题 | 影响 | 状态 |
|------|------|------|
| `--model` 硬编码 `claude-opus-4-7` | tmux agent 无法用 env var 切换模型 | API 端映射可解 |
| `.claude/agents/*.md` `model:` 被忽略 | 所有 agent 统一用默认模型 | 不影响运行，model 由 API 决定 |
| `/resume` 后成员 `active=False` | 需手动重新 spawn | 设计限制 |
| 手动写 config 无法激活进程 | 必须用 Agent() 工具 spawn | 设计限制 |
