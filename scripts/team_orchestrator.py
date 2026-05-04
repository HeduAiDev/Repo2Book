#!/usr/bin/env python3
"""
repo2book Team Orchestrator — CLI for persistent Agent Teams pipeline.

Two modes:
  Plan → plan all chapters, review structure, then execute one at a time
  Batch → auto-write all remaining chapters without intervention

Key principle: Agents spawn ONCE and persist across ALL chapters via tmux panes.
  start-team:     Print one-time spawn instructions for ALL agents
  pipeline:       Create chapter tasks, print SendMessage instructions (agents already alive)
  batch/plan:     Works with persistent agents — no respawning needed

Usage:
  python3 scripts/team_orchestrator.py start-team
  python3 scripts/team_orchestrator.py pipeline <chapter_id>
  python3 scripts/team_orchestrator.py plan
  python3 scripts/team_orchestrator.py batch

Architecture:
  The orchestrator creates 4 tasks with Ralph Backpressure dependencies
  and spawns the first agent (Implementer). Downstream agents are triggered
  when upstream tasks complete (via TaskCompleted hooks + Lead dispatch).

  Pipeline: Implementer → Tester → Writer → Reviewer

  Lateral communication is DIRECT:
  - Reviewer → Writer (REVISE instructions, bypassing Lead)
  - Writer → Implementer (change requests, bypassing Lead)

  The Lead only intervenes for:
  - New chapter dispatch
  - Loop detection (Reviewer↔Writer > 3 rounds → escalate to Lead)
  - Status queries from user
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

# ── Load repo2book config ─────────────────────────────────────────
def load_config() -> dict:
    config_file = ROOT / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
INSTANCE = CONFIG.get("instances", {}).get("current", {})
SOURCE_DIR = INSTANCE.get("source_dir", CONFIG.get("source", {}).get("source_dir", "src"))
ARTIFACTS_DIR = ROOT / INSTANCE.get("artifacts_dir", "artifacts")
TEAM_NAME = CONFIG.get("team", {}).get("name", "book-factory")
PIPELINE_STAGES = CONFIG.get("pipeline", {}).get("stages", ["implementer", "tester", "writer", "reviewer"])
PIPELINE_ORDER = PIPELINE_STAGES
TEAM_DIR = Path.home() / ".claude" / "teams" / TEAM_NAME
TASKS_DIR = Path.home() / ".claude" / "tasks" / TEAM_NAME


def team_exists() -> bool:
    return (TEAM_DIR / "config.json").exists()


def create_chapter_tasks(chapter_id: str) -> dict:
    """Create the 4 pipeline tasks for a chapter with dependencies."""
    chapter_dir = ARTIFACTS_DIR / chapter_id

    # Ensure skeleton exists
    for subdir in ["implementation", "tests", "narrative", "reviews"]:
        (chapter_dir / subdir).mkdir(parents=True, exist_ok=True)
    init_file = chapter_dir / "implementation" / "__init__.py"
    if not init_file.exists():
        init_file.touch()

    tasks = {}
    task_ids = {}

    # Task template: (stage, subject, description, blocked_by)
    stages = [
        (
            "implementer",
            f"Implement {chapter_id}",
            f"Read vLLM source → write reimplementation code for {chapter_id}. "
            f"Output to artifacts/{chapter_id}/implementation/. "
            f"Every function must have # REFERENCE: vllm/path/file.py:L123. "
            f"Must produce impl-notes.md with Source Mapping Table (5+ rows).",
            [],
        ),
        (
            "tester",
            f"Test {chapter_id}",
            f"Write and run pytest tests for {chapter_id}. "
            f"Unit tests + integration tests + teaching example tests. "
            f"ALL tests must pass before marking complete. "
            f"Output: test_{{module}}.py + test-report.json.",
            ["implementer"],
        ),
        (
            "writer",
            f"Write {chapter_id}",
            f"Write v5-standard narrative for {chapter_id}. "
            f"Code walkthrough + mathematical proof + numerical trace + source trail. "
            f"Every Cell 2-7 must have vLLM file:line references. "
            f"Use svg-diagram skill for diagrams. "
            f"Read implementation FIRST for correct line numbers.",
            ["tester"],
        ),
        (
            "reviewer",
            f"Review {chapter_id}",
            f"Review {chapter_id} narrative across 9 dimensions. "
            f"Run formula lint + source grounding lint. "
            f"AUTO-REJECT: missing proof, missing code walkthrough, \\text{{}} in formulas. "
            f"If REVISE: SendMessage directly to writer with fix instructions. "
            f"If APPROVED: mark complete, trigger archivist.",
            ["writer"],
        ),
        (
            "archivist",
            f"Archive {chapter_id}",
            f"Terminal stage for {chapter_id}. "
            f"Backup ALL agent session transcripts to trace/chapters/{chapter_id}/sessions/. "
            f"Create delivery record. Update state.json. "
            f"Chapter is PUBLISHED after this task completes.",
            ["reviewer"],
        ),
    ]

    for stage, subject, description, blocked_by in stages:
        task_id = _generate_task_id(stage, chapter_id)
        task_ids[stage] = task_id

        tasks[task_id] = {
            "subject": subject,
            "description": description,
            "stage": stage,
            "chapter_id": chapter_id,
            "blocked_by": [task_ids[b] for b in blocked_by] if blocked_by else [],
            "status": "pending",
        }

    print(f"  Chapter: {chapter_id}")
    print(f"  Team: {TEAM_NAME}")
    print(f"  Tasks:")
    for stage in PIPELINE_ORDER:
        tid = task_ids[stage]
        blocked = tasks[tid]["blocked_by"]
        block_str = f" [blocked by: {', '.join(blocked)}]" if blocked else ""
        print(f"    {tid}: {tasks[tid]['subject']}{block_str}")

    return tasks


def get_chapter_status(chapter_id: str) -> dict:
    """Read chapter context.json for gate status."""
    context_file = ARTIFACTS_DIR / chapter_id / "context.json"
    if not context_file.exists():
        return {"status": "not_started", "gates": {}}

    with open(context_file) as f:
        ctx = json.load(f)

    return {
        "status": ctx.get("status", "unknown"),
        "gates": ctx.get("gates", {}),
        "version": ctx.get("version", 0),
        "summary": ctx.get("summary", ""),
    }


def get_agent_spawn_command(agent_name: str, chapter_id: str = "") -> dict:
    """Generate the Agent tool call for spawning a teammate."""
    # System prompts are auto-loaded from .claude/agents/{name}.md by Claude Code.
    # The prompt field below contains ONLY the task assignment for this specific chapter.
    agent_configs = {
        "implementer": {
            "subagent_type": "general-purpose",
            "description": f"Implement {chapter_id}",
            "prompt": f"Task: Implement chapter {chapter_id}. Read the source, complete Source Analysis, write code with # REFERENCE comments. Output impl-notes.md. Mark task complete when done.",
        },
        "tester": {
            "subagent_type": "general-purpose",
            "description": f"Test {chapter_id}",
            "prompt": f"Task: Test chapter {chapter_id}. Write and run pytest tests. ALL must pass. Output test-report.json. Mark task complete when done.",
        },
        "writer": {
            "subagent_type": "general-purpose",
            "description": f"Write {chapter_id}",
            "prompt": f"Task: Write chapter {chapter_id} narrative. Read implementation for line numbers first. Output chapter.md with code walkthrough + proof. Run formula lint. Mark task complete when done.",
        },
        "reviewer": {
            "subagent_type": "general-purpose",
            "description": f"Review {chapter_id}",
            "prompt": f"Task: Review chapter {chapter_id}. Run linters, review 9 dimensions. Output review-report.json. If REVISE: SendMessage to writer. If APPROVED: mark complete.",
        },
        "archivist": {
            "subagent_type": "general-purpose",
            "description": f"Archive {chapter_id}",
            "prompt": f"Task: Archive chapter {chapter_id}. Backup all session transcripts to trace/chapters/{chapter_id}/sessions/. Create delivery record. Update state.json. Terminal stage. Mark complete.",
        },
        "book-editor": {
            "subagent_type": "general-purpose",
            "description": f"Edit {chapter_id}",
            "prompt": f"Task: Orchestrate chapter {chapter_id}. Monitor progress via TaskList. Handle lateral communication. Escalate loops > 3 rounds.",
        },
    }

    config = agent_configs.get(agent_name, {})
    if not config:
        raise ValueError(f"Unknown agent: {agent_name}")

    return {
        "name": f"{agent_name}@{TEAM_NAME}",
        "team_name": TEAM_NAME,
        "subagent_type": config["subagent_type"],
        "description": config["description"],
        "prompt": config["prompt"],
        "model": "inherit",
    }


def _generate_task_id(stage: str, chapter_id: str) -> str:
    """Generate a short task ID."""
    short = chapter_id.replace("artifacts/", "").replace("0", "").replace("-", "")[:6]
    return f"{stage[:4]}-{short}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "create":
        # python3 scripts/team_orchestrator.py create <chapter_id>
        if len(sys.argv) < 3:
            print("Usage: team_orchestrator.py create <chapter_id>")
            sys.exit(1)
        chapter_id = sys.argv[2]
        if not team_exists():
            print(f"Error: Team '{TEAM_NAME}' not found. Create it first:")
            print(f"  Use TeamCreate tool with team_name='{TEAM_NAME}'")
            sys.exit(1)
        create_chapter_tasks(chapter_id)

    elif cmd == "start-team":
        # python3 scripts/team_orchestrator.py start-team
        # One-time: print spawn instructions for ALL persistent agents
        print(f"\n{'='*60}")
        print(f"  START TEAM — Spawn Once, Live Forever")
        print(f"{'='*60}\n")
        print(f"  Team: {TEAM_NAME}")
        print(f"  All agents use backendType: tmux")
        print(f"  Each pane persists across ALL chapters.")
        print(f"\n  Spawn these agents ONCE (they stay alive until shutdown):\n")

        team_order = ["book-editor", "implementer", "tester", "writer", "reviewer", "archivist"]
        for i, stage in enumerate(team_order):
            spawn = get_agent_spawn_command(stage, "")
            print(f"  [{i+1}] Spawn {stage}:")
            print(f"      Agent tool:")
            print(f"        name: {spawn['name']}")
            print(f"        team_name: {TEAM_NAME}")
            print(f"        subagent_type: {spawn['subagent_type']}")
            print(f"        model: {spawn['model']}")
            color = {"book-editor": "purple", "implementer": "blue", "tester": "yellow",
                     "writer": "green", "reviewer": "red", "archivist": "cyan"}.get(stage, "white")
            print(f"        (tmux pane: {color})")
            print()

        print(f"  After spawning: agents go idle, wait for task assignments.")
        print(f"  Then run: python3 scripts/team_orchestrator.py pipeline 14-triton-primer")
        print(f"\n  Never respawn. Only shut down when the entire book is complete:")
        print(f"    SendMessage to each: {{type: 'shutdown_request', reason: 'Book complete'}}")
        print(f"{'='*60}\n")

    elif cmd == "status":
        # python3 scripts/team_orchestrator.py status <chapter_id>
        if len(sys.argv) < 3:
            print("Usage: team_orchestrator.py status <chapter_id>")
            sys.exit(1)
        status = get_chapter_status(sys.argv[2])
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif cmd == "spawn-instructions":
        # python3 scripts/team_orchestrator.py spawn-instructions <agent> <chapter_id>
        if len(sys.argv) < 4:
            print("Usage: team_orchestrator.py spawn-instructions <agent> <chapter_id>")
            sys.exit(1)
        agent = sys.argv[2]
        chapter_id = sys.argv[3]
        spawn = get_agent_spawn_command(agent, chapter_id)
        print(json.dumps(spawn, indent=2, ensure_ascii=False))

    elif cmd == "pipeline":
        # Full pipeline: create tasks + print spawn instructions
        if len(sys.argv) < 3:
            print("Usage: team_orchestrator.py pipeline <chapter_id>")
            sys.exit(1)
        chapter_id = sys.argv[2]

        print(f"\n{'='*60}")
        print(f"  Pipeline: {chapter_id}")
        print(f"  (agents already alive — send tasks, don't respawn)")
        print(f"{'='*60}\n")

        # Step 1: Create tasks
        print("[1/2] Creating pipeline tasks...")
        tasks = create_chapter_tasks(chapter_id)

        # Step 2: Print task assignment instructions
        print(f"\n[2/2] Assign first task:")
        print(f"  ─────────────────────────────────────")
        first = PIPELINE_ORDER[0]
        spawn = get_agent_spawn_command(first, chapter_id)
        print(f"\n  SendMessage to {first}:")
        print(f"    summary: \"Task: {spawn['description']}\"")
        print(f"    message: \"{spawn['prompt']}\"")

        print(f"\n  Pipeline: {' → '.join(PIPELINE_ORDER)}")
        print(f"  Handoff: Hook auto-notifies next agent on task completion.")
        print(f"  Lateral: Reviewer ←→ Writer (direct SendMessage).")
        print(f"  Lead only intervenes for: topology changes, loop >3, user queries.")
        print(f"{'='*60}\n")

    elif cmd == "batch":
        # python3 scripts/team_orchestrator.py batch
        # Full-auto: write all remaining chapters without intervention
        outline_file = CONFIG.get("book", {}).get("outline_file", "instances/vllm/book/book-outline.json")
        outline_path = ROOT / outline_file
        if not outline_path.exists():
            print(f"Outline not found: {outline_path}")
            sys.exit(1)

        with open(outline_path) as f:
            outline = json.load(f)

        # Collect all chapters in order
        all_chapters = []
        for part_name, part in outline.get("parts", {}).items():
            for ch in part.get("chapters", []):
                all_chapters.append({
                    "id": ch["id"],
                    "number": ch.get("number", "?"),
                    "title": ch.get("title", ""),
                    "part": part.get("title", part_name),
                    "difficulty": ch.get("estimated_difficulty", "unknown"),
                })

        # Filter to uncompleted chapters
        pending = []
        for ch in all_chapters:
            status = get_chapter_status(ch["id"])
            if status.get("status") != "published":
                pending.append(ch)

        if not pending:
            print("\n✓ All chapters published. Book complete!\n")
            sys.exit(0)

        print(f"\n{'='*60}")
        print(f"  BATCH MODE — Auto-write {len(pending)} chapters")
        print(f"{'='*60}\n")

        for i, ch in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] [{ch['number']}] {ch['id']}")
            print(f"    Title: {ch['title']}")
            print(f"    Part: {ch['part']}")
            print(f"    Difficulty: {ch['difficulty']}")
            # Propose topology based on difficulty
            if ch["difficulty"] in ("advanced", "expert"):
                topo = "pair"
            elif ch["number"] == pending[0]["number"]:  # First chapter of batch
                topo = "linear"
            else:
                topo = "linear"
            print(f"    Topology: {topo}")

        print(f"\n  Auto-progression:")
        print(f"    Chapter N archivist completes → book-editor creates tasks for N+1")
        print(f"    → implementer spawned for N+1 → pipeline cascades")
        print(f"  Interrupts: test failure, review rejection, loop >3, topology change")
        print(f"  User notified via inbox on each chapter completion")
        print(f"\n  To start: spawn book-editor with Task 'batch-write-all'")
        print(f"  Book-editor will handle everything automatically.")
        print(f"{'='*60}\n")

    elif cmd == "plan":
        # python3 scripts/team_orchestrator.py plan
        # Research pass: analyze all chapters, propose topologies, flag dependencies
        outline_file = CONFIG.get("book", {}).get("outline_file", "instances/vllm/book/book-outline.json")
        outline_path = ROOT / outline_file
        if not outline_path.exists():
            print(f"Outline not found: {outline_path}")
            sys.exit(1)

        with open(outline_path) as f:
            outline = json.load(f)

        plan = {
            "project": CONFIG.get("book", {}).get("title", "Unknown"),
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_chapters": 0,
            "chapters": [],
            "dependency_graph": {},
            "recommended_topologies": {},
        }

        print(f"\n{'='*60}")
        print(f"  PLAN MODE — Research Pass")
        print(f"  Book: {plan['project']}")
        print(f"{'='*60}\n")

        for part_name, part in outline.get("parts", {}).items():
            print(f"  ── {part.get('title', part_name)} ──\n")
            for ch in part.get("chapters", []):
                ch_id = ch["id"]
                difficulty = ch.get("estimated_difficulty", "intermediate")
                deps = ch.get("dependencies", [])

                # Complexity estimate
                if difficulty in ("advanced", "expert"):
                    complexity = "complex"
                    rec_topo = "pair"
                elif ch.get("level") == "core" and not deps:
                    complexity = "medium"
                    rec_topo = "linear"
                elif len(deps) >= 3:
                    complexity = "complex"
                    rec_topo = "pair"
                else:
                    complexity = "medium"
                    rec_topo = "linear"

                # First chapter of a part → panel review
                part_chapters = part.get("chapters", [])
                if ch == part_chapters[0] and part_name != list(outline.get("parts", {}).keys())[0]:
                    rec_topo = "panel"

                # Chapters with has_theory → may need writer_editor
                if ch.get("has_theory"):
                    if rec_topo == "linear":
                        rec_topo = "writer_editor"

                plan["chapters"].append({
                    "id": ch_id,
                    "number": ch.get("number"),
                    "title": ch.get("title"),
                    "part": part.get("title", part_name),
                    "difficulty": difficulty,
                    "complexity": complexity,
                    "dependencies": deps,
                    "recommended_topology": rec_topo,
                })
                plan["dependency_graph"][ch_id] = deps
                plan["recommended_topologies"][ch_id] = rec_topo
                plan["total_chapters"] += 1

                status = get_chapter_status(ch_id)
                s = "✓ published" if status.get("status") == "published" else "○ pending"
                print(f"  [{ch.get('number', '?')}] {ch_id}  [{complexity}]  {s}")
                print(f"    {ch.get('title', '')}")
                print(f"    Dependencies: {deps or 'none'}  →  Topology: {rec_topo}")
                if deps:
                    for d in deps:
                        ds = get_chapter_status(d)
                        if ds.get("status") != "published":
                            print(f"    ⚠ Dependency '{d}' not yet published!")
                print()

        # Save plan
        plan_file = ROOT / "instances" / "vllm" / "trace" / "cross-chapter" / "plan.json"
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_file, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        # Summary
        published = sum(1 for ch in plan["chapters"] if get_chapter_status(ch["id"]).get("status") == "published")
        pending = plan["total_chapters"] - published
        topo_counts = {}
        for ch in plan["chapters"]:
            t = ch["recommended_topology"]
            topo_counts[t] = topo_counts.get(t, 0) + 1

        print(f"  ── Summary ──")
        print(f"  Total: {plan['total_chapters']} chapters")
        print(f"  Published: {published}  |  Pending: {pending}")
        print(f"  Topologies: {topo_counts}")
        print(f"\n  Plan saved: {plan_file.relative_to(ROOT)}")
        print(f"  Review the plan, adjust topologies, then run:")
        print(f"    python3 scripts/team_orchestrator.py batch")
        print(f"{'='*60}\n")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
