#!/usr/bin/env python3
"""
Prune stale `in-process` agents from a team config.

In-process agents are ephemeral (each Agent-tool spawn runs as a one-shot
subprocess and exits at end of turn), but their entries persist in the team
config indefinitely. Over a long session this accumulates dead `<role>-2`,
`<role>-3`, ... entries that:
  - receive task-list broadcasts and emit defensive complaint messages
  - clutter the team registry visually
  - confuse SendMessage routing in some cases

This script prunes them. Tmux-backend entries (the original team skeleton) are
preserved.

Usage:
    python3 scripts/cleanup_stale_agents.py [--team NAME] [--dry-run] [--keep NAME ...]

  --team NAME     Team to clean (default: book-factory)
  --dry-run       Show what would be removed without modifying the file
  --keep NAME     Preserve specific in-process entries (repeatable). Useful
                  when you have a known-active spawn in flight.

Examples:
    # Preview cleanup
    python3 scripts/cleanup_stale_agents.py --dry-run

    # Prune all in-process entries
    python3 scripts/cleanup_stale_agents.py

    # Prune all except the latest active spawns
    python3 scripts/cleanup_stale_agents.py --keep writer-4 --keep reviewer-4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_TEAMS_ROOT = Path.home() / ".claude" / "teams"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default="book-factory", help="Team name (default: book-factory)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying config")
    parser.add_argument("--keep", action="append", default=[], help="Preserve specific in-process entries by name (repeatable)")
    parser.add_argument("--teams-root", default=str(DEFAULT_TEAMS_ROOT), help=f"Teams config root (default: {DEFAULT_TEAMS_ROOT})")
    args = parser.parse_args()

    config_path = Path(args.teams_root) / args.team / "config.json"
    if not config_path.is_file():
        print(f"ERROR: team config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open() as f:
        config = json.load(f)

    members = config.get("members", [])
    if not members:
        print(f"No members in {config_path}; nothing to do.")
        return 0

    keep_set = set(args.keep)
    pruned: list[str] = []
    surviving: list[dict] = []

    for m in members:
        name = m.get("name", "?")
        backend = m.get("backendType", "?")
        if backend == "in-process" and name not in keep_set:
            pruned.append(name)
        else:
            surviving.append(m)

    print(f"Team: {args.team}")
    print(f"Config: {config_path}")
    print(f"Total members: {len(members)}")
    print(f"To prune (in-process, not in --keep): {len(pruned)}")
    for n in pruned:
        print(f"  - {n}")
    print(f"Surviving: {len(surviving)}")
    for m in surviving:
        print(f"  - {m.get('name')} ({m.get('backendType')})")

    if not pruned:
        print("Nothing to remove.")
        return 0

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return 0

    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    shutil.copy2(config_path, backup_path)
    print(f"\nBackup written: {backup_path}")

    config["members"] = surviving
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Updated: {config_path}")
    print(f"Pruned {len(pruned)} stale in-process entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
