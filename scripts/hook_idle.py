#!/usr/bin/env python3
"""
TeammateIdle hook — pipeline handoff when agent goes idle.
Article: return exit 2 = keep agent working. Exit 0 = let idle.
"""

import json, os, sys
from pathlib import Path

TEAM = "book-factory"
INBOX_DIR = Path.home() / ".claude" / "teams" / TEAM / "inboxes"
ARTIFACTS = Path("/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts")


def find_active_chapter():
    """Find which chapter has a signal directory."""
    sig_base = Path("/tmp/book-factory")
    for d in sorted(sig_base.glob("*/"), reverse=True):
        return d.name
    # Fallback: look for most recently modified artifact chapter
    chapters = sorted(ARTIFACTS.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    for ch in chapters:
        return ch.name
    return None


def main():
    try:
        event = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    agent = event.get("agentName", "")
    if TEAM not in event.get("teamName", ""):
        sys.exit(0)

    chapter = find_active_chapter()
    if not chapter:
        sys.exit(0)

    narrative = ARTIFACTS / chapter / "narrative" / "chapter.md"
    reviews = ARTIFACTS / chapter / "reviews" / "review-report.json"
    sig_dir = Path(f"/tmp/book-factory/{chapter}")

    INBOX_DIR.mkdir(parents=True, exist_ok=True)

    # Writer just finished → wake reviewer
    if "writer" in agent and narrative.exists():
        reviewer_inbox = INBOX_DIR / "reviewer.json"
        if not reviewer_inbox.exists():
            with open(reviewer_inbox, "w") as f:
                json.dump({
                    "type": "handoff",
                    "chapter": chapter,
                    "action": "Review narrative now",
                    "narrative": str(narrative),
                    "reviews": str(reviews),
                    "from_agent": agent
                }, f)
            print(f"[hook_idle] Writer done → reviewer inbox written")
        sys.exit(0)  # Let writer idle

    # Reviewer just finished → wake archivist
    if "reviewer" in agent and reviews.exists():
        archivist_inbox = INBOX_DIR / "archivist.json"
        if not archivist_inbox.exists():
            with open(archivist_inbox, "w") as f:
                json.dump({
                    "type": "handoff",
                    "chapter": chapter,
                    "action": "Backup sessions + create delivery",
                    "from_agent": agent
                }, f)
            print(f"[hook_idle] Reviewer done → archivist inbox written")
        sys.exit(0)

    # Researcher just finished → notify writer
    if "researcher" in agent and sig_dir.exists():
        research = ARTIFACTS / chapter.replace("artifacts/", "") / "research-brief.md"
        writer_inbox = INBOX_DIR / "writer.json"
        if not writer_inbox.exists():
            with open(writer_inbox, "w") as f:
                json.dump({
                    "type": "handoff",
                    "chapter": chapter,
                    "action": "Research brief ready",
                    "from_agent": agent
                }, f)
            print(f"[hook_idle] Researcher done → writer inbox written")
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
