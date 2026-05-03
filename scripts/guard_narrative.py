#!/usr/bin/env python3
"""
Narrative Guardian — prevents direct edits to chapter narratives.

Each chapter's narrative/chapter.md is OWNED by the Writer agent.
No other agent (including the main orchestrator) may modify it.

This script runs as a pre-check before any operation that might touch
narrative files. If the caller is NOT a Writer subagent, it denies access.

Usage:
    python3 guard_narrative.py check <chapter_id>  — verify access
    python3 guard_narrative.py unlock <chapter_id>  — grant Writer access (writes token)
    python3 guard_narrative.py lock <chapter_id>    — revoke Writer access
    python3 guard_narrative.py status               — show all chapters' lock state
"""

import json, sys, os, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path('/mnt/e/Laboratory/vllm-from-scratch')
LOCK_DIR = ROOT / '.narrative_locks'


def ensure_lock_dir():
    LOCK_DIR.mkdir(exist_ok=True)


def write_lock(chapter_id: str, token: str):
    """Grant Writer access with a token."""
    ensure_lock_dir()
    (LOCK_DIR / f"{chapter_id}.json").write_text(json.dumps({
        "chapter_id": chapter_id,
        "token": hashlib.sha256(token.encode()).hexdigest()[:16],
        "granted_at": datetime.now().isoformat(),
        "granted_by": "orchestrator"
    }, indent=2))


def check_lock(chapter_id: str) -> bool:
    """Check if chapter is locked (no Writer access). Returns True if LOCKED."""
    ensure_lock_dir()
    lock_file = LOCK_DIR / f"{chapter_id}.json"
    if not lock_file.exists():
        return True  # No token = locked by default
    data = json.loads(lock_file.read_text())
    # Token expired after 30 minutes
    granted = datetime.fromisoformat(data["granted_at"])
    if (datetime.now() - granted).total_seconds() > 1800:
        return True  # Expired
    return False  # Writer has active access


def remove_lock(chapter_id: str):
    """Revoke Writer access."""
    (LOCK_DIR / f"{chapter_id}.json").unlink(missing_ok=True)


def is_writer_agent() -> bool:
    """Check if the current process is a Writer subagent.
    Writer subagents set WRITER_AGENT=true in their environment."""
    return os.environ.get("WRITER_AGENT", "").lower() == "true"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: guard_narrative.py <check|unlock|lock|status> [chapter_id]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "status":
        ensure_lock_dir()
        chapters = sorted(LOCK_DIR.glob("*.json"))
        if not chapters:
            print("No chapters have Writer access tokens.")
        for c in chapters:
            data = json.loads(c.read_text())
            granted = datetime.fromisoformat(data["granted_at"])
            elapsed = (datetime.now() - granted).total_seconds()
            status = "ACTIVE" if elapsed < 1800 else "EXPIRED"
            print(f"  {data['chapter_id']}: {status} ({elapsed:.0f}s ago)")
        sys.exit(0)

    if len(sys.argv) < 3:
        print(f"Usage: guard_narrative.py {cmd} <chapter_id>")
        sys.exit(1)

    chapter_id = sys.argv[2]

    if cmd == "unlock":
        # Grant Writer access
        write_lock(chapter_id, f"writer-{datetime.now().timestamp()}")
        print(f"🔓 {chapter_id}: Writer access GRANTED (30 min)")
        sys.exit(0)

    if cmd == "lock":
        remove_lock(chapter_id)
        print(f"🔒 {chapter_id}: Writer access REVOKED")
        sys.exit(0)

    if cmd == "check":
        if is_writer_agent():
            print("✅ Writer agent detected — access allowed")
            sys.exit(0)
        if check_lock(chapter_id):
            print(f"🔒 {chapter_id}: LOCKED — Writer access required")
            print("   Direct edits to narrative/chapter.md are FORBIDDEN.")
            print("   Use Writer agent (set WRITER_AGENT=true) or unlock first.")
            sys.exit(1)
        else:
            print(f"🔓 {chapter_id}: Writer access active")
            sys.exit(0)
