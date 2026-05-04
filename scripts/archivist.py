#!/usr/bin/env python3
"""
Archivist CLI — Project Memory Management for repo2book.

The Archivist agent uses this tool to:
  record   — Write a new trace entry (decision, delivery, user_interaction)
  query    — Search trace entries by chapter, type, tags, or date
  brief    — Generate a context rehydration brief for an agent
  summary  — Create a session summary for context continuity
  state    — View or update project state.json
  index    — Update trace/INDEX.md with latest entries
  alert    — Check for context loss risk conditions

Usage:
  python3 scripts/archivist.py record --type decision --chapter 04 --title "..." --what "..." --why "..."
  python3 scripts/archivist.py query --chapter 04 --type decision
  python3 scripts/archivist.py brief --chapter 04 --role implementer
  python3 scripts/archivist.py summary --date 2026-05-04
  python3 scripts/archivist.py state [--set chapters.14.status in_progress]
  python3 scripts/archivist.py index --rebuild
  python3 scripts/archivist.py alert --session-turns 45
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent

def load_config() -> dict:
    config_file = ROOT / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

CONFIG = load_config()

def get_instance_dir(instance_name: str = None) -> Path:
    """Get the instance directory from repo2book config or argument."""
    if instance_name:
        return ROOT / "instances" / instance_name
    # Default: read from main config, resolve from config path
    config_file = ROOT / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        current = config.get("instances", {}).get("current", {})
        # Use config file path to derive instance dir: "instances/vllm/repo2book.json" → "instances/vllm"
        config_path = current.get("config", "")
        if config_path:
            return ROOT / Path(config_path).parent
        instance_name = current.get("name", "vllm")
        return ROOT / "instances" / instance_name
    return ROOT / "instances" / "vllm"

INSTANCE = get_instance_dir()
TRACE_DIR = INSTANCE / "trace"
DECISIONS_DIR = TRACE_DIR / "decisions"
DELIVERIES_DIR = TRACE_DIR / "deliveries"
INTERACTIONS_DIR = TRACE_DIR / "user_interactions"
SUMMARIES_DIR = TRACE_DIR / "context_summaries"
INDEX_FILE = TRACE_DIR / "INDEX.md"
STATE_FILE = TRACE_DIR / "state.json"

# ── Record ───────────────────────────────────────────────────────

def record(entry_type: str, title: str, what: str, why: str,
           chapter: str = None, tags: list = None,
           agents: list = None, user_present: bool = False,
           remember: str = None) -> Path:
    """Record a new trace entry."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine target directory
    dir_map = {
        "decision": DECISIONS_DIR,
        "delivery": DELIVERIES_DIR,
        "user_interaction": INTERACTIONS_DIR,
        "bug": DECISIONS_DIR,       # Bugs are recorded as decisions
        "design_change": DECISIONS_DIR,
    }
    target_dir = dir_map.get(entry_type, DECISIONS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    safe_title = title.lower().replace(" ", "-").replace("/", "-")[:60]
    chapter_prefix = f"ch{chapter}-" if chapter else ""
    filename = f"{date_str}_{chapter_prefix}{safe_title}.md"
    filepath = target_dir / filename

    # Build entry
    tags_str = ", ".join(tags) if tags else ""
    agents_str = ", ".join(agents) if agents else "archivist"

    content = f"""# {title}

- **Type**: {entry_type}
- **Chapter**: {chapter or 'N/A'}
- **Date**: {date_str}
- **Timestamp**: {timestamp}
- **Agents involved**: {agents_str}
- **User present**: {user_present}
- **Tags**: {tags_str}

## What happened

{what}

## Why it matters

{why}

## What to remember

{remember or f"{what[:200]}..."}
"""
    with open(filepath, "w") as f:
        f.write(content)

    # Update INDEX.md
    _update_index(date_str, entry_type, chapter, title, filename)

    # Update state.json if this is a delivery
    if entry_type == "delivery" and chapter:
        _update_state_chapter(chapter, "published")

    print(f"Recorded: {filepath.relative_to(ROOT)}")
    return filepath


# ── Query ────────────────────────────────────────────────────────

def query(chapter: str = None, entry_type: str = None,
          tags: list = None, limit: int = 10) -> list:
    """Search trace entries."""
    results = []
    search_dirs = []

    if entry_type:
        dir_map = {
            "decision": DECISIONS_DIR,
            "delivery": DELIVERIES_DIR,
            "user_interaction": INTERACTIONS_DIR,
            "session_summary": SUMMARIES_DIR,
        }
        if entry_type in dir_map:
            search_dirs = [dir_map[entry_type]]
    else:
        search_dirs = [DECISIONS_DIR, DELIVERIES_DIR, INTERACTIONS_DIR, SUMMARIES_DIR]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for f in sorted(search_dir.glob("*.md"), reverse=True):
            if len(results) >= limit:
                break
            with open(f) as fh:
                content = fh.read()
            # Filter by chapter
            if chapter and f"Chapter**: {chapter}" not in content and f"chapter-{chapter}" not in f.name:
                continue
            # Filter by tags
            if tags:
                if not any(t in content for t in tags):
                    continue
            # Extract title
            title = content.split("\n")[0].replace("# ", "")
            results.append({
                "file": str(f.relative_to(ROOT)),
                "title": title,
                "chapter": _extract_field(content, "Chapter"),
                "date": _extract_field(content, "Date"),
                "type": _extract_field(content, "Type"),
            })

    if not results:
        print(f"No results for chapter={chapter}, type={entry_type}")
        return []

    print(f"\n{'='*60}")
    print(f"  Trace Query Results ({len(results)} entries)")
    print(f"{'='*60}\n")
    for r in results:
        print(f"  [{r['type']}] {r['date']} — {r['title']}")
        if r['chapter']:
            print(f"    Chapter: {r['chapter']}")
        print(f"    File: {r['file']}\n")

    return results


# ── Rehydration Brief ────────────────────────────────────────────

def brief(chapter_id: str, role: str) -> str:
    """Generate a context rehydration brief for an agent starting work."""
    # Collect relevant entries
    decisions = _find_relevant(DECISIONS_DIR, chapter_id, 3)
    deliveries = _find_relevant(DELIVERIES_DIR, chapter_id, 2)
    interactions = _find_relevant(INTERACTIONS_DIR, chapter_id, 2)
    summaries = sorted(SUMMARIES_DIR.glob("*.md"), reverse=True) if SUMMARIES_DIR.exists() else []
    latest_summary = summaries[0] if summaries else None

    # Build brief
    brief_text = f"""## Context Rehydration Brief

**Agent**: {role}
**Chapter**: {chapter_id}
**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC

### Current Project State
"""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        brief_text += f"- Last session: {state.get('last_session', {}).get('date', 'unknown')}\n"
        brief_text += f"- Last summary: {state.get('last_session', {}).get('summary', 'none')}\n"
        ch_status = state.get("chapters", {})
        if chapter_id in str(ch_status):
            brief_text += f"- This chapter status: check state.json\n"

    brief_text += "\n### Relevant Past Decisions\n"
    if decisions:
        for d in decisions:
            brief_text += f"- {d['title']} ({d['date']})\n"
    else:
        brief_text += "- No prior decisions for this chapter.\n"

    brief_text += "\n### Previous Deliveries\n"
    if deliveries:
        for d in deliveries:
            brief_text += f"- {d['title']} ({d['date']})\n"
    else:
        brief_text += "- No prior deliveries for this chapter.\n"

    brief_text += "\n### User Feedback\n"
    if interactions:
        for i in interactions:
            brief_text += f"- {i['title']} ({i['date']})\n"
    else:
        brief_text += "- No user feedback for this chapter.\n"

    if latest_summary:
        brief_text += f"\n### Latest Session Summary\n"
        with open(latest_summary) as f:
            brief_text += f.read()[:500] + "...\n"

    brief_text += "\n### Wisdom Relevant to Your Role\n"
    role_wisdom = {
        "implementer": "wisdom/debugging.md, wisdom/architecture.md",
        "tester": "wisdom/testing.md, wisdom/debugging.md",
        "writer": "wisdom/writing.md, wisdom/debugging.md",
        "reviewer": "wisdom/writing.md, wisdom/architecture.md",
    }
    brief_text += f"- Read: {role_wisdom.get(role, 'wisdom/INDEX.md')}\n"

    brief_text += f"\n### Knowledge Relevant to This Chapter\n"
    knowledge_index = INSTANCE / "knowledge" / "INDEX.md"
    if knowledge_index.exists():
        brief_text += f"- Read: {knowledge_index.relative_to(ROOT)}\n"

    brief_text += "\n---\n*Generated by archivist. For full details, read the trace files listed above.*"

    print(brief_text)
    return brief_text


# ── Session Summary ──────────────────────────────────────────────

def session_summary(date: str = None, accomplishments: str = "",
                    decisions: str = "", user_feedback: str = "",
                    next_steps: str = "", current_state: str = "") -> Path:
    """Create a session summary for context continuity."""
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SUMMARIES_DIR / f"session-{date}.md"

    content = f"""# Session Summary — {date}

## Accomplishments
{accomplishments or '(none recorded)'}

## Decisions Made
{decisions or '(none recorded)'}

## User Feedback & Interactions
{user_feedback or '(none recorded)'}

## Current State
{current_state or 'See state.json'}

## Next Session
{next_steps or '(not specified)'}

---
*Archivist: update trace/state.json with last_session details.*
"""
    with open(filepath, "w") as f:
        f.write(content)

    # Update state.json
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
    else:
        state = {}
    state["last_session"] = {
        "date": date,
        "summary": accomplishments[:200] if accomplishments else "",
        "context_hash": f"session-{date}",
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Session summary: {filepath.relative_to(ROOT)}")
    return filepath


# ── State Management ─────────────────────────────────────────────

def manage_state(action: str = "view", key: str = None, value: str = None) -> dict:
    """View or update project state."""
    if not STATE_FILE.exists():
        print("No state.json found. Creating default.")
        return {}

    with open(STATE_FILE) as f:
        state = json.load(f)

    if action == "view":
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return state

    if action == "set" and key:
        # Support dot notation: "chapters.part3.14"
        keys = key.split(".")
        target = state
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        # Parse value: try JSON, fallback to string
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = value
        target[keys[-1]] = parsed
        state["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        state["updated_by"] = "archivist"

        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        print(f"State updated: {key} = {parsed}")
        return state

    print(f"Unknown action: {action}")
    return state


# ── Context Loss Alert ───────────────────────────────────────────

def alert(session_turns: int = 0, session_hours: float = 0,
          chapters_since_summary: int = 0) -> dict:
    """Check for context loss risk conditions."""
    risks = []
    if session_turns > 50:
        risks.append(f"HIGH: {session_turns} turns in session (context compression likely)")
    if session_hours > 24:
        risks.append(f"HIGH: {session_hours}h session (agent fatigue + context decay)")
    if chapters_since_summary > 3:
        risks.append(f"MEDIUM: {chapters_since_summary} chapters since last summary")

    if risks:
        print("⚠ CONTEXT LOSS RISK ALERT:")
        for r in risks:
            print(f"  {r}")
        print("  → Recommended: create session summary, update state.json")
    else:
        print("✓ No context loss risks detected")

    return {"risks": risks, "alert_level": "high" if len(risks) > 2 else "medium" if risks else "low"}


# ── Helpers ──────────────────────────────────────────────────────

def _update_index(date: str, entry_type: str, chapter: str, title: str, filename: str):
    """Add entry to INDEX.md recent activity."""
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    if INDEX_FILE.exists():
        with open(INDEX_FILE) as f:
            index = f.read()

        # Insert after "## Recent Activity" header
        new_entry = f"| {date} | {entry_type} | {chapter or 'N/A'} | {title} | [{filename}]({entry_type}s/{filename}) |\n"
        marker = "| Date | Type | Chapter | Summary | File |\n"
        if marker in index:
            index = index.replace(marker, marker + new_entry)
            with open(INDEX_FILE, "w") as f:
                f.write(index)


def _update_state_chapter(chapter_id: str, status: str):
    """Update chapter status in state.json."""
    if not STATE_FILE.exists():
        return
    with open(STATE_FILE) as f:
        state = json.load(f)

    # Determine which part the chapter belongs to
    ch_num = int(chapter_id.replace("artifacts/", "").split("-")[0]) if chapter_id else 0
    part_map = {
        range(1, 11): "part1",
        range(11, 14): "part2",
        range(14, 22): "part3",
        range(22, 26): "part4",
        range(26, 29): "part5",
    }
    part = "unknown"
    for r, p in part_map.items():
        if ch_num in r:
            part = p
            break

    chapters = state.get("chapters", {})
    if part in chapters:
        if isinstance(chapters[part], dict):
            chapters[part][str(ch_num).zfill(2)] = status

    state["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["updated_by"] = "archivist"
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _find_relevant(directory: Path, chapter_id: str, limit: int) -> list:
    """Find trace entries relevant to a chapter."""
    if not directory.exists():
        return []
    results = []
    for f in sorted(directory.glob("*.md"), reverse=True):
        with open(f) as fh:
            content = fh.read()
        ch_num = chapter_id.replace("artifacts/", "").split("-")[0][:2] if chapter_id else ""
        if ch_num in f.name or (chapter_id and chapter_id[:2] in f.name):
            title = content.split("\n")[0].replace("# ", "") if content else f.stem
            date = _extract_field(content, "Date") or f.stem[:10]
            results.append({"title": title, "date": date, "file": str(f.relative_to(ROOT))})
            if len(results) >= limit:
                break
    return results


def _extract_field(content: str, field: str) -> str:
    """Extract a metadata field from a trace entry."""
    for line in content.split("\n"):
        if line.strip().startswith(f"- **{field}**:"):
            return line.split(":", 1)[1].strip()
    return ""


# ── Outline Change Detection ─────────────────────────────────────

def snapshot_outline() -> Path:
    """Snapshot the current book outline for future change detection."""
    outline_path = CONFIG.get("book", {}).get("outline_file", "book/book-outline.json")
    outline_file = ROOT / outline_path
    if not outline_file.exists():
        # Fall back to instance dir
        outline_file = INSTANCE / "book" / "book-outline.json"
    if not outline_file.exists():
        print(f"No book-outline.json found at {outline_file}")
        raise SystemExit(1)

    with open(outline_file) as f:
        outline = json.load(f)

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    snapshot = {"snapshot_date": date_str, "chapters": {}}
    for part_name, part in outline.get("parts", {}).items():
        for ch in part.get("chapters", []):
            snapshot["chapters"][ch["id"]] = {
                "number": ch.get("number"),
                "title": ch.get("title"),
                "level": ch.get("level", ""),
                "dependencies": ch.get("dependencies", []),
            }

    cross_dir = INSTANCE / "trace" / "cross-chapter"
    cross_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = cross_dir / f"outline-snapshot-{date_str}.json"
    with open(snapshot_file, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # Update state.json
    _update_state_field("outline_snapshot", {
        "file": f"cross-chapter/outline-snapshot-{date_str}.json",
        "created": date_str,
        "chapter_count": len(snapshot["chapters"]),
    })

    print(f"Outline snapshot: {snapshot_file.relative_to(ROOT)}")
    print(f"  Chapters: {len(snapshot['chapters'])}")
    return snapshot_file


def detect_outline_changes() -> dict:
    """Compare current outline to latest snapshot. Report diffs."""
    outline_path = CONFIG.get("book", {}).get("outline_file", "book/book-outline.json")
    outline_file = ROOT / outline_path
    if not outline_file.exists():
        outline_file = INSTANCE / "book" / "book-outline.json"
    if not outline_file.exists():
        print(f"No book-outline.json found at {outline_file}")
        return {}

    with open(outline_file) as f:
        new_outline = json.load(f)

    # Find latest snapshot
    cross_dir = INSTANCE / "trace" / "cross-chapter"
    snapshots = sorted(cross_dir.glob("outline-snapshot-*.json"), reverse=True)
    if not snapshots:
        print("No previous snapshot. Run 'snapshot-outline' first.")
        return {}

    with open(snapshots[0]) as f:
        old_snapshot = json.load(f)

    old_chapters = old_snapshot.get("chapters", {})
    new_chapters = {}
    for part_name, part in new_outline.get("parts", {}).items():
        for ch in part.get("chapters", []):
            new_chapters[ch["id"]] = {
                "number": ch.get("number"),
                "title": ch.get("title"),
            }

    old_ids = set(old_chapters.keys())
    new_ids = set(new_chapters.keys())

    changes = {
        "added": [],
        "removed": [],
        "renumbered": [],
        "renamed": [],
        "unchanged": [],
        "snapshot_file": str(snapshots[0].name),
        "requires_migration": False,
    }

    # Detect added chapters
    for ch_id in new_ids - old_ids:
        changes["added"].append({"id": ch_id, "info": new_chapters[ch_id]})
        changes["requires_migration"] = True

    # Detect removed chapters
    for ch_id in old_ids - new_ids:
        changes["removed"].append({"id": ch_id, "info": old_chapters[ch_id]})
        changes["requires_migration"] = True

    # Detect renumbered and renamed
    for ch_id in old_ids & new_ids:
        old_n = old_chapters[ch_id]["number"]
        new_n = new_chapters[ch_id]["number"]
        old_title = old_chapters[ch_id]["title"]
        new_title = new_chapters[ch_id]["title"]

        if old_n != new_n:
            changes["renumbered"].append({
                "id": ch_id, "old_number": old_n, "new_number": new_n
            })
            changes["requires_migration"] = True

        if old_title != new_title:
            changes["renamed"].append({
                "id": ch_id, "old_title": old_title, "new_title": new_title
            })

        if old_n == new_n and old_title == new_title:
            changes["unchanged"].append(ch_id)

    # Print report
    print(f"\n{'='*60}")
    print(f"  Outline Change Detection")
    print(f"  Snapshot: {snapshots[0].name}")
    print(f"{'='*60}\n")

    if changes["added"]:
        print(f"  ➕ ADDED ({len(changes['added'])}):")
        for c in changes["added"]:
            print(f"    + [{c['info']['number']}] {c['id']}: {c['info']['title']}")

    if changes["removed"]:
        print(f"  ➖ REMOVED ({len(changes['removed'])}):")
        for c in changes["removed"]:
            print(f"    - [{c['info']['number']}] {c['id']}: {c['info']['title']}")

    if changes["renumbered"]:
        print(f"  🔢 RENUMBERED ({len(changes['renumbered'])}):")
        for c in changes["renumbered"]:
            print(f"    ~ {c['id']}: {c['old_number']} → {c['new_number']}")

    if changes["renamed"]:
        print(f"  ✏ RENAMED ({len(changes['renamed'])}):")
        for c in changes["renamed"]:
            print(f"    ~ {c['id']}: \"{c['old_title']}\" → \"{c['new_title']}\"")

    if changes["unchanged"]:
        print(f"  ✓ UNCHANGED: {len(changes['unchanged'])} chapters")

    print(f"\n  Migration needed: {'YES ⚠' if changes['requires_migration'] else 'NO ✓'}")
    print(f"{'='*60}\n")

    if changes["requires_migration"]:
        print("  Next: Send notification to book-editor via:")
        print(f"    python3 scripts/archivist.py notify-lead --subject \"Outline changed\" --message \"{_format_change_message(changes)}\"")
        print(f"  After approval:")
        print(f"    python3 scripts/archivist.py migrate-chapters --from-snapshot {snapshots[0].name} --approved-by book-editor")

    return changes


def migrate_chapters(from_snapshot: str = "", approved_by: str = "book-editor") -> dict:
    """Execute trace directory migrations after outline change approval."""
    # Reload the changes
    changes = detect_outline_changes()
    if not changes.get("requires_migration"):
        print("No migration needed.")
        return {"status": "no_changes"}

    trace_dir = INSTANCE / "trace" / "chapters"
    log = {"migrations": [], "errors": []}

    # Rename trace directories for renumbered chapters
    for c in changes.get("renumbered", []):
        old_dir = trace_dir / c["id"]
        # Chapter ID doesn't change, only number. Trace dir is keyed by ID.
        # But if the ID includes the number (e.g., "05-xxx"), we need to handle that.
        print(f"  Chapter {c['id']}: number {c['old_number']} → {c['new_number']} (directory unchanged)")

    # Create trace skeletons for added chapters
    for c in changes.get("added", []):
        ch_id = c["id"]
        init_chapter(ch_id)
        print(f"  Created trace skeleton for new chapter: {ch_id}")
        log["migrations"].append(f"init-chapter {ch_id}")

    # Archive trace directories for removed chapters
    for c in changes.get("removed", []):
        ch_id = c["id"]
        ch_dir = trace_dir / ch_id
        if ch_dir.exists():
            archive_dir = trace_dir / ".archived" / ch_id
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(str(ch_dir), str(archive_dir))
            print(f"  Archived: {ch_id} → trace/chapters/.archived/{ch_id}")
            log["migrations"].append(f"archive {ch_id}")

    # Record in outline changelog
    changelog_file = INSTANCE / "trace" / "cross-chapter" / "outline-changelog.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    changelog_entry = f"""
### {now} — Outline migration

- **Approved by**: {approved_by}
- **Snapshot**: {from_snapshot}
- **Added**: {len(changes.get('added', []))} chapters
- **Removed**: {len(changes.get('removed', []))} chapters
- **Renumbered**: {len(changes.get('renumbered', []))} chapters
- **Migrations**: {', '.join(log['migrations']) if log['migrations'] else 'none'}

"""
    with open(changelog_file, "a") as f:
        f.write(changelog_entry)

    # Create new snapshot
    snapshot_outline()

    return {"status": "migrated", "log": log}


# ── Session Backup ───────────────────────────────────────────────

def backup_session(chapter: str, role: str, source: str) -> Path:
    """Backup a raw agent session transcript for reproducibility."""
    ch_dir = INSTANCE / "trace" / "chapters" / chapter / "sessions"
    ch_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backup_file = ch_dir / f"{date_str}_{role}.json"

    # Copy from source path or stdin
    if source and Path(source).exists():
        import shutil
        shutil.copy(source, backup_file)
    else:
        # Try to read session output from environment
        task_output = os.environ.get("CLAUDE_TASK_OUTPUT", "")
        if task_output and Path(task_output).exists():
            import shutil
            shutil.copy(task_output, backup_file)
        else:
            # Create minimal backup record
            backup_data = {
                "chapter": chapter,
                "role": role,
                "date": date_str,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": source or "unknown",
                "note": "Full transcript not available. Record this at session end."
            }
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

    # Update chapter INDEX to reference this backup
    ch_index = ch_dir.parent / "INDEX.md"
    if not ch_index.exists():
        init_chapter(chapter)

    print(f"Session backup: {backup_file.relative_to(ROOT)}")
    return backup_file


# ── Chapter Initialization ───────────────────────────────────────

def init_chapter(chapter_id: str) -> Path:
    """Create trace skeleton for a new chapter."""
    ch_dir = INSTANCE / "trace" / "chapters" / chapter_id
    ch_dir.mkdir(parents=True, exist_ok=True)
    (ch_dir / "sessions").mkdir(exist_ok=True)

    index_file = ch_dir / "INDEX.md"
    if not index_file.exists():
        with open(index_file, "w") as f:
            f.write(f"""# Chapter {chapter_id} — Trace Index

## Delivery
- **Status**: not_started

## Decisions
(no decisions yet)

## User Interactions
(no interactions yet)

## Session Backups
(no backups yet)
""")

    print(f"Initialized: {ch_dir.relative_to(ROOT)}")
    return ch_dir


def init_instance(instance_name: str) -> Path:
    """Initialize a new instance's complete trace system."""
    inst_dir = ROOT / "instances" / instance_name
    trace_dir = inst_dir / "trace"
    for subdir in ["chapters", "cross-chapter", "sessions"]:
        (trace_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Create default state.json with lineage support
    default_state = {
        "project": instance_name,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_by": "archivist init",
        "chapters": {},
        "chapter_lineage": {},
        "outline_snapshot": {},
        "active_team": {},
        "open_issues": [],
        "user_preferences": {},
        "last_session": {},
    }
    with open(trace_dir / "state.json", "w") as f:
        json.dump(default_state, f, indent=2)

    # Create cross-chapter files
    (trace_dir / "cross-chapter" / "decisions.md").write_text(
        "# Cross-Chapter Decisions\n\nFramework-wide decisions.\n")
    (trace_dir / "cross-chapter" / "user-preferences.md").write_text(
        "# User Preferences\n\nAccumulated user preferences.\n")
    (trace_dir / "cross-chapter" / "outline-changelog.md").write_text(
        "# Outline Changelog\n\nEvery structural change to the book outline.\n")

    # Create root INDEX
    (trace_dir / "INDEX.md").write_text(
        f"# Trace Index — {instance_name}\n\n"
        "## Recent Activity\n\n| Date | Type | Chapter | Summary |\n"
        "|------|------|---------|--------|\n"
        f"| {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | init | N/A | Trace system initialized |\n\n"
        "## Chapters\n\n(no chapters yet)\n")

    print(f"Initialized trace system for {instance_name}")
    return trace_dir


def notify_lead(subject: str, message: str) -> dict:
    """Send a structured notification to the book-editor (Lead)."""
    inbox_file = INBOX_DIR / "book-editor.json"
    notification = {
        "type": "archivist_notification",
        "subject": subject,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from": "archivist",
        "requires_response": True,
    }
    _append_inbox(inbox_file, notification)
    print(f"Notified book-editor: {subject}")
    return notification


# ── Helpers ──────────────────────────────────────────────────────

def _format_change_message(changes: dict) -> str:
    """Format change detection results for book-editor notification."""
    parts = []
    if changes.get("added"):
        parts.append(f"Added {len(changes['added'])}: {', '.join(c['id'] for c in changes['added'])}")
    if changes.get("removed"):
        parts.append(f"Removed {len(changes['removed'])}: {', '.join(c['id'] for c in changes['removed'])}")
    if changes.get("renumbered"):
        parts.append(f"Renumbered {len(changes['renumbered'])}")
    return "; ".join(parts)


def _update_state_field(key: str, value):
    """Update a single field in state.json."""
    if not STATE_FILE.exists():
        return
    with open(STATE_FILE) as f:
        state = json.load(f)
    state[key] = value
    state["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["updated_by"] = "archivist"
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = {}
    i = 2
    while i < len(sys.argv):
        if sys.argv[i].startswith("--") and i + 1 < len(sys.argv):
            key = sys.argv[i][2:].replace("-", "_")
            args[key] = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    if cmd == "record":
        record(
            entry_type=args.get("type", "decision"),
            title=args.get("title", "Untitled"),
            what=args.get("what", ""),
            why=args.get("why", ""),
            chapter=args.get("chapter"),
            tags=args.get("tags", "").split(",") if args.get("tags") else None,
            agents=args.get("agents", "").split(",") if args.get("agents") else None,
            user_present=args.get("user_present", "false").lower() == "true",
            remember=args.get("remember"),
        )

    elif cmd == "query":
        query(
            chapter=args.get("chapter"),
            entry_type=args.get("type"),
            tags=args.get("tags", "").split(",") if args.get("tags") else None,
            limit=int(args.get("limit", 10)),
        )

    elif cmd == "brief":
        brief(
            chapter_id=args.get("chapter", ""),
            role=args.get("role", "implementer"),
        )

    elif cmd == "summary":
        session_summary(
            date=args.get("date"),
            accomplishments=args.get("accomplishments", ""),
            decisions=args.get("decisions", ""),
            user_feedback=args.get("user_feedback", ""),
            next_steps=args.get("next_steps", ""),
            current_state=args.get("current_state", ""),
        )

    elif cmd == "state":
        manage_state(
            action=args.get("action", "view"),
            key=args.get("set"),
            value=args.get("value"),
        )

    elif cmd == "index":
        print("INDEX.md updated via record/summary commands.")

    elif cmd == "alert":
        alert(
            session_turns=int(args.get("session_turns", 0)),
            session_hours=float(args.get("session_hours", 0)),
            chapters_since_summary=int(args.get("chapters_since_summary", 0)),
        )

    elif cmd == "init":
        init_instance(args.get("instance", "vllm"))

    elif cmd == "init-chapter":
        init_chapter(args.get("chapter", ""))

    elif cmd == "snapshot-outline":
        snapshot_outline()

    elif cmd == "detect-changes":
        detect_outline_changes()

    elif cmd == "migrate-chapters":
        migrate_chapters(
            args.get("from_snapshot", ""),
            args.get("approved_by", "book-editor"),
        )

    elif cmd == "backup-session":
        backup_session(
            chapter=args.get("chapter", ""),
            role=args.get("role", ""),
            source=args.get("source", ""),
        )

    elif cmd == "notify-lead":
        notify_lead(
            subject=args.get("subject", "Outline change detected"),
            message=args.get("message", ""),
        )

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
