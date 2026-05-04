#!/usr/bin/env python3
"""
Pipeline handoff hook for book-factory Agent Teams.

Triggered by TaskCompleted event. Receives event JSON via stdin:
  {"taskId": "...", "status": "completed", "agentName": "...", "teamName": "..."}

Reads the completed task's metadata to determine:
  1. Which pipeline stage just completed (implement/test/write/review)
  2. Which downstream agent should receive the handoff
  3. What action the downstream agent should take

Writes a handoff message to the next agent's inbox so they wake up
and start working — no Lead intervention needed for normal progression.

Pipeline:  Implementer → Tester → Writer → Reviewer
Handoff:   implement   → test   → write  → review

Lateral:   Reviewer ←→ Writer (direct SendMessage, NOT through this hook)
           Writer → Implementer (change requests, NOT through this hook)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Load repo2book config ─────────────────────────────────────────
def load_config() -> dict:
    config_file = Path(__file__).parent.parent / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
TEAM = CONFIG.get("team", {}).get("name", "book-factory")
INBOX_DIR = Path.home() / ".claude" / "teams" / TEAM / "inboxes"
TASKS_DIR = Path.home() / ".claude" / "tasks" / TEAM
PIPELINE_STAGES = CONFIG.get("pipeline", {}).get("stages", ["implementer", "tester", "writer", "reviewer"])

# Build handoff map from pipeline stages config
def build_handoff_map(stages: list) -> list:
    actions = {
        "implementer": ("tester", "Tests must pass before Writer starts. This is THE Backpressure Gate."),
        "tester": ("writer", "Write narrative: code walkthrough + proof + numerical trace + source trail. Every Cell 2-7 must have source file:line ref."),
        "writer": ("reviewer", "Review narrative across 9 dimensions. If REVISE: SendMessage directly to writer (no Lead routing). If APPROVED: chapter published."),
        "reviewer": ("archivist", "APPROVED. Backup session transcripts, create delivery record, update state. Terminal stage."),
        "archivist": (None, None),
    }
    handoff = []
    for stage in stages:
        if stage in actions:
            next_agent, action = actions[stage]
            handoff.append((stage, next_agent, action))
    return handoff

PIPELINE_HANDOFF = build_handoff_map(PIPELINE_STAGES)


def read_task_metadata(task_id: str) -> dict:
    """Read task metadata from the tasks directory."""
    task_file = TASKS_DIR / f"{task_id}.json"
    if task_file.exists():
        with open(task_file) as f:
            return json.load(f)
    return {}


def write_handoff(target_agent: str, message: dict) -> Path:
    """Write a handoff message to the target agent's inbox."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    inbox_file = INBOX_DIR / f"{target_agent}.json"

    # Read existing inbox messages
    messages = []
    if inbox_file.exists():
        try:
            with open(inbox_file) as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    messages = existing
                elif isinstance(existing, dict):
                    messages = [existing]
        except (json.JSONDecodeError, OSError):
            messages = []

    messages.append(message)
    with open(inbox_file, "w") as f:
        json.dump(messages if len(messages) > 1 else messages[0], f, indent=2)

    return inbox_file


def get_event_data() -> dict:
    """Get event data from environment variables (hook) or stdin (manual)."""
    # Try environment variables (set by hook system)
    event = {}
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "")
    if tool_input:
        try:
            event = json.loads(tool_input)
        except json.JSONDecodeError:
            pass
    if event:
        return event

    # Try stdin (for manual testing or different hook modes)
    try:
        if not sys.stdin.isatty():
            event = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        pass
    return event


def main():
    event = get_event_data()
    if not event:
        sys.exit(0)

    task_id = event.get("taskId", "")
    agent_name = event.get("agentName", "")
    task_status = event.get("status", "")

    if task_status != "completed" or not task_id:
        sys.exit(0)

    # Read the task that just completed
    task_meta = read_task_metadata(task_id)
    subject = task_meta.get("subject", "")

    # Find which pipeline stage just completed
    matched_stage = None
    next_agent = None
    handoff_action = None

    for keyword, next_a, action in PIPELINE_HANDOFF:
        if keyword in subject.lower():
            matched_stage = keyword
            next_agent = next_a
            handoff_action = action
            break

    if not matched_stage:
        sys.exit(0)  # Not a pipeline task, nothing to do

    # Terminal stage — signal completion
    if next_agent is None:
        notif = {
            "from": agent_name,
            "type": "pipeline_terminal",
            "task_completed": task_id,
            "subject": subject,
            "action": "Review complete. Chapter pipeline finished. Check review-report.json for APPROVED/REVISE.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_handoff("book-editor", notif)
        print(f"[book-factory] ✓ Pipeline terminal: {agent_name} completed '{subject}'")
        sys.exit(0)

    # Write handoff message to next agent's inbox
    handoff_msg = {
        "from": agent_name,
        "type": "handoff",
        "task_completed": task_id,
        "task_stage": matched_stage,
        "subject": subject,
        "action": handoff_action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    inbox_path = write_handoff(next_agent, handoff_msg)
    print(
        f"[book-factory] {agent_name}({matched_stage}) → {next_agent}: "
        f"{handoff_action}"
    )
    print(f"[book-factory]   inbox: {inbox_path}")


if __name__ == "__main__":
    main()
