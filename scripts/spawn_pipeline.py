#!/usr/bin/env python3
"""
Pipeline spawner — writer and reviewer spawned TOGETHER so inbox loops overlap.

Root cause fix: writer's inbox loop timed out before reviewer finished.
Solution: spawn both agents simultaneously. Writer writes → enters loop.
Reviewer watches for narrative file → reviews → writes feedback to writer inbox.
Writer's loop catches it because reviewer started at the same time.

Usage:
  python3 scripts/spawn_pipeline.py {chapter_id}
"""

import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

TEAM = "book-factory"
INBOX_DIR = Path.home() / ".claude" / "teams" / TEAM / "inboxes"
CONFIG_FILE = Path.home() / ".claude" / "teams" / TEAM / "config.json"
ROOT = Path(__file__).parent.parent


def clean_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            c = json.load(f)
        c['members'] = [m for m in c['members'] if 'agentId' not in m]
        with open(CONFIG_FILE, 'w') as f:
            json.dump(c, f, indent=2, ensure_ascii=False)


def deliver(agent: str, msg: dict):
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INBOX_DIR / f"{agent}.json", "w") as f:
        json.dump(msg, f, indent=2)


def generate(chapter: str):
    """Generate spawn instructions for writer + reviewer (simultaneous)."""
    clean_config()

    chapter_dir = f"instances/vllm/artifacts/{chapter}"
    narrative = f"{chapter_dir}/narrative/chapter.md"
    research = f"instances/vllm/trace/chapters/{chapter}/research-brief.md"
    reviews = f"{chapter_dir}/reviews/review-report.json"

    # Deliver writer task
    deliver("writer", {
        "type": "task",
        "chapter": chapter,
        "phase2": f"Write {chapter} narrative. Read .claude/agents/writer.md. Output to {narrative}. Run lint_formulas.py. Use svg-diagram skill for diagrams.",
        "phase3": "Run: bash scripts/inbox_loop.sh writer 60 40  (60s × 40 = 40 min). Wait for reviewer feedback. Apply and re-lint if received.",
        "research": f"Check {research} if exists."
    })

    # Deliver reviewer task
    deliver("reviewer", {
        "type": "task",
        "chapter": chapter,
        "phase2": f"Wait for {narrative} to appear (check every 30s with: while [ ! -f {narrative} ]; do sleep 30; done). Then review per .claude/agents/reviewer.md. Run lint_formulas.py + lint_source_grounding.py.",
        "phase3_if_issues": f"Write feedback to ~/.claude/teams/book-factory/inboxes/writer.json as {{'type':'review_feedback','blocking':[...],'suggestions':[...]}}. Then run inbox_loop.sh reviewer 45 8 waiting for writer update.",
        "phase3_if_approved": f"Write {reviews} with verdict APPROVED. Notify archivist.",
        "collab": "If you want to discuss with writer: write to writer inbox, wait via inbox_loop.sh. You are COLLABORATORS not upstream/downstream."
    })

    print(f"""
{'='*60}
  Pipeline: {chapter}
  Writer + Reviewer spawned SIMULTANEOUSLY
  Both have overlapping inbox loops (9 min each)
{'='*60}

# Spawn WRITER:
Agent(
  subagent_type="general-purpose",
  team_name="{TEAM}",
  name="writer",
  run_in_background=True,
  prompt="PHASE 1: Read ~/.claude/teams/book-factory/inboxes/writer.json. Delete it.
PHASE 2: Execute the task from the inbox. {narrative}
PHASE 3: bash scripts/inbox_loop.sh writer 60 60  (1 hour loop waiting for reviewer)"
)

# Spawn REVIEWER:
Agent(
  subagent_type="general-purpose",
  team_name="{TEAM}",
  name="reviewer",
  run_in_background=True,
  prompt="PHASE 1: Read ~/.claude/teams/book-factory/inboxes/reviewer.json. Delete it.
PHASE 2: Wait for {narrative} to appear. Review per .claude/agents/reviewer.md.
PHASE 3: If issues → write feedback to writer inbox → bash scripts/inbox_loop.sh reviewer 45 8.
         If APPROVED → write {reviews}."
)
""")


if __name__ == "__main__":
    chapter = sys.argv[1] if len(sys.argv) > 1 else "01-self-attention-fundamentals"
    generate(chapter)
