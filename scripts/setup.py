#!/usr/bin/env python3
"""
repo2book Setup — one command to initialize everything after cloning.

Usage:
  python3 scripts/setup.py              # Full setup
  python3 scripts/setup.py --check      # Verify setup status
  python3 scripts/setup.py --team-only  # Only install team config

What it does:
  1. Copies team config from .claude/teams/ to ~/.claude/teams/
  2. Creates task directory at ~/.claude/tasks/book-factory/
  3. Initializes archivist trace system for the current instance
  4. Creates outline snapshot for change detection
  5. Verifies all required files exist
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEAM_NAME = "book-factory"
REPO_TEAM_CONFIG = ROOT / ".claude" / "teams" / f"{TEAM_NAME}.json"
GLOBAL_TEAM_DIR = Path.home() / ".claude" / "teams" / TEAM_NAME
GLOBAL_TASKS_DIR = Path.home() / ".claude" / "tasks" / TEAM_NAME


def step(msg: str):
    print(f"  [{step.count}] {msg}")
    step.count += 1
step.count = 1


def setup_team():
    """Copy team config from repo to global location."""
    if not REPO_TEAM_CONFIG.exists():
        print(f"ERROR: Team config not found at {REPO_TEAM_CONFIG}")
        print("  Expected .claude/teams/book-factory.json in repo root")
        return False

    GLOBAL_TEAM_DIR.mkdir(parents=True, exist_ok=True)
    dest = GLOBAL_TEAM_DIR / "config.json"

    with open(REPO_TEAM_CONFIG) as f:
        team_config = json.load(f)

    # Update cwd to current project path
    for member in team_config.get("members", []):
        member["cwd"] = str(ROOT)

    with open(dest, "w") as f:
        json.dump(team_config, f, indent=2)

    # Create inboxes directory
    (GLOBAL_TEAM_DIR / "inboxes").mkdir(parents=True, exist_ok=True)

    step(f"Team config installed: {dest}")
    step(f"Members: {len(team_config.get('members', []))}")
    for m in team_config["members"]:
        print(f"    - {m['name']} ({m['color']}, {m['backendType']})")
    return True


def setup_tasks():
    """Create task directory for the team."""
    GLOBAL_TASKS_DIR.mkdir(parents=True, exist_ok=True)
    step(f"Task directory: {GLOBAL_TASKS_DIR}")
    return True


def setup_archivist():
    """Initialize trace system for current instance."""
    try:
        from scripts import archivist
        # Initialize if not already done
        state_file = ROOT / "instances" / "vllm" / "trace" / "state.json"
        if not state_file.exists():
            step("Initializing archivist trace system...")
            # Run archivist init
            archivist.init_instance("vllm")
        else:
            step(f"Trace system already initialized: {state_file}")
        return True
    except Exception as e:
        print(f"  WARNING: Archivist init failed: {e}")
        return True  # Non-fatal


def setup_outline_snapshot():
    """Create initial outline snapshot."""
    try:
        from scripts import archivist
        archivist.snapshot_outline()
        step("Outline snapshot created")
        return True
    except Exception as e:
        print(f"  WARNING: Outline snapshot failed: {e}")
        return True  # Non-fatal


def check():
    """Verify setup status."""
    print("\n=== repo2book Setup Check ===\n")

    checks = [
        ("Team config (global)", GLOBAL_TEAM_DIR / "config.json"),
        ("Team config (repo)", REPO_TEAM_CONFIG),
        ("Inboxes", GLOBAL_TEAM_DIR / "inboxes"),
        ("Tasks", GLOBAL_TASKS_DIR),
        ("Instance", ROOT / "instances" / "vllm" / "repo2book.json"),
        ("Trace", ROOT / "instances" / "vllm" / "trace" / "state.json"),
        ("Knowledge", ROOT / "instances" / "vllm" / "knowledge" / "INDEX.md"),
        ("Wisdom", ROOT / "wisdom" / "INDEX.md"),
        ("Agents", ROOT / ".claude" / "agents"),
        ("Outline", ROOT / "instances" / "vllm" / "book" / "book-outline.json"),
        ("Source repo", ROOT / "instances" / "vllm" / "source" / ".git"),
        ("Script: archivist", ROOT / "scripts" / "archivist.py"),
        ("Script: learn", ROOT / "scripts" / "learn.py"),
        ("Script: decide", ROOT / "scripts" / "decide.py"),
    ]

    all_ok = True
    for label, path in checks:
        ok = path.exists()
        status = "✓" if ok else "✗ MISSING"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label}: {path}")

    print()
    if all_ok:
        print("All checks passed. System is ready.")
    else:
        print("Some items missing. Run 'python3 scripts/setup.py' to fix.")

    return all_ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        ok = check()
        sys.exit(0 if ok else 1)

    if len(sys.argv) > 1 and sys.argv[1] == "--team-only":
        print("\n=== repo2book Setup (team only) ===\n")
        setup_team()
        setup_tasks()
        print("\nSetup complete. Run 'python3 scripts/setup.py' for full setup.\n")
        return

    print("\n=== repo2book Setup ===\n")
    print(f"  Project: {ROOT}")
    print()

    ok = True
    ok &= setup_team()
    ok &= setup_tasks()
    ok &= setup_archivist()
    ok &= setup_outline_snapshot()

    print()
    if ok:
        print("Setup complete. System is ready.\n")
        print("Next steps:")
        print("  1. Verify: python3 scripts/setup.py --check")
        print("  2. Start a chapter: python3 scripts/team_orchestrator.py pipeline {chapter_id}")
        print("  3. Query memory: python3 scripts/learn.py query {chapter_id} {role}")
    else:
        print("Setup failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
