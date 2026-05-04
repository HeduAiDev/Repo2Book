#!/usr/bin/env python3
"""
Agent spawn helper — solves the inbox problem at root.

Root cause: subagents are one-shot. They don't poll inboxes.
Fix: deliver task to inbox FIRST, then spawn agent.
     Agent checks inbox as its first action.

Usage:
  python3 scripts/spawn_agent.py writer 02-kv-cache "Write Ch02 narrative..."
  python3 scripts/spawn_agent.py reviewer 02-kv-cache "Review Ch02..."
  python3 scripts/spawn_agent.py --check    # check inbox status
  python3 scripts/spawn_agent.py --clean    # clean idle agents
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

TEAM = "book-factory"
INBOX_DIR = Path.home() / ".claude" / "teams" / TEAM / "inboxes"
CONFIG_FILE = Path.home() / ".claude" / "teams" / TEAM / "config.json"


def deliver_task(agent_name: str, chapter_id: str, task: str) -> dict:
    """Write task to agent's inbox. Agent reads this on startup."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    inbox_file = INBOX_DIR / f"{agent_name}.json"

    msg = {
        "type": "task_assignment",
        "agent": agent_name,
        "chapter": chapter_id,
        "task": task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    with open(inbox_file, "w") as f:
        json.dump(msg, f, indent=2)

    print(f"Task delivered → {agent_name}: {task[:80]}...")
    return msg


def clean_idle():
    """Remove auto-registered agents, keep only templates."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            c = json.load(f)
        before = len(c["members"])
        c["members"] = [m for m in c["members"] if "agentId" not in m]
        with open(CONFIG_FILE, "w") as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
        print(f"Cleaned: {before} → {len(c['members'])} members")


def check_inbox():
    """Show pending tasks in all inboxes."""
    if not INBOX_DIR.exists():
        print("No inboxes")
        return
    for f in sorted(INBOX_DIR.glob("*.json")):
        with open(f) as fh:
            try:
                data = json.load(fh)
                status = data.get("status", "?")
                task = data.get("task", str(data)[:80])
                print(f"  {f.stem}: [{status}] {task}")
            except json.JSONDecodeError:
                print(f"  {f.stem}: unreadable")


def generate_spawn_prompt(agent_name: str, chapter_id: str, task: str) -> str:
    """Generate the standardized 3-phase prompt for any agent."""
    collab_agent = {
        "writer": "reviewer",
        "reviewer": "writer",
    }.get(agent_name, "")

    prompt = f"""PHASE 1 — INBOX: Read your task from ~/.claude/teams/book-factory/inboxes/{agent_name}.json
If found, delete the inbox file after reading.

PHASE 2 — WORK: Execute the task. Output to instances/vllm/artifacts/{chapter_id}/.

PHASE 3 — COLLABORATION LOOP: After completing your work, run:
  bash scripts/inbox_loop.sh {agent_name} 30 10
This checks your inbox every 30s for up to 5 minutes.
- If {collab_agent} sends feedback: apply changes, re-lint, then loop again
- If no message after 5 min: exit"""
    return prompt


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "--clean":
        clean_idle()
    elif cmd == "--check":
        check_inbox()
    elif cmd == "--prompt":
        agent = sys.argv[2]
        chapter = sys.argv[3] if len(sys.argv) > 3 else ""
        task = sys.argv[4] if len(sys.argv) > 4 else ""
        print(generate_spawn_prompt(agent, chapter, task))
    elif cmd == "--pipeline":
        # Full pipeline setup: deliver + print spawn instructions
        chapter = sys.argv[2] if len(sys.argv) > 2 else ""
        agents = ["writer", "reviewer"]
        tasks = {
            "writer": f"Write {chapter} narrative. Read .claude/agents/writer.md. Check researcher brief at instances/vllm/trace/chapters/{chapter}/research-brief.md if exists. Write narrative to instances/vllm/artifacts/{chapter}/narrative/chapter.md. Run lint_formulas.py.",
            "reviewer": f"Review {chapter} narrative. Read .claude/agents/reviewer.md. Wait for instances/vllm/artifacts/{chapter}/narrative/chapter.md to appear. Run lint_formulas.py + lint_source_grounding.py. If issues: write feedback to ~/.claude/teams/book-factory/inboxes/writer.json. If APPROVED: write review-report.json.",
        }
        for agent in agents:
            deliver_task(agent, chapter, tasks[agent])
            print(f"\n# Spawn {agent}:")
            print(f"Agent(team_name='book-factory', name='{agent}',")
            print(f"      prompt='{generate_spawn_prompt(agent, chapter, tasks[agent])}',")
            print(f"      run_in_background=True, subagent_type='general-purpose')")
            print()
    else:
        agent = sys.argv[1]
        chapter = sys.argv[2] if len(sys.argv) > 2 else ""
        task = sys.argv[3] if len(sys.argv) > 3 else ""
        deliver_task(agent, chapter, task)
        print(f"\n# Spawn {agent} with:")
        print(generate_spawn_prompt(agent, chapter, task))


if __name__ == "__main__":
    main()
