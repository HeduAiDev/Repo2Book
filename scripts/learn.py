#!/usr/bin/env python3
"""
repo2book Self-Learning Engine — agents run this after completing work.

Three phases:
  1. EXTRACT  — Agent identifies what it learned from this task
  2. CLASSIFY — Is this a repo-specific fact (→ knowledge/) or universal pattern (→ wisdom/)?
  3. COMPACT  — If a knowledge module exceeds 15 facts, LLM-summarize oldest 5 into one

Usage:
  python3 scripts/learn.py extract <chapter_id> <role>         # Interactive extraction
  python3 scripts/learn.py query <chapter_id> <role>           # Query before work
  python3 scripts/learn.py compact <module>                    # Run anti-bloat compaction
  python3 scripts/learn.py promote <wisdom_id>                 # Promote knowledge → wisdom
  python3 scripts/learn.py stats                               # Knowledge base statistics
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent


def _resolve_instance_dir() -> Path:
    """Derive current instance dir from framework repo2book.json (source_dir's parent)."""
    fw_config = ROOT / "repo2book.json"
    if fw_config.exists():
        cfg = json.loads(fw_config.read_text())
        sd = cfg.get("source", {}).get("source_dir")
        if sd:
            return ROOT / Path(sd).parent  # e.g. instances/vllm
    return ROOT  # fallback (single-instance dev mode)


INSTANCE_DIR = _resolve_instance_dir()
KNOWLEDGE_DIR = INSTANCE_DIR / "knowledge"
MODULES_DIR = KNOWLEDGE_DIR / "modules"
ARCHIVE_DIR = KNOWLEDGE_DIR / "archive"
WISDOM_DIR = ROOT / "wisdom"  # framework-shared, stays at root per CLAUDE.md

MAX_FACTS_PER_MODULE = 15
COMPACT_COUNT = 5       # Oldest N facts to compact when limit exceeded
TTL_DAYS = 30           # Archive facts unused for this many days


def load_config() -> dict:
    config_file = ROOT / "repo2book.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}

CONFIG = load_config()


# ── Phase 1: Query Before Work ───────────────────────────────────

def query(chapter_id: str, role: str) -> dict:
    """What should this agent read before starting work on a chapter?"""
    role_priorities = {
        "implementer": ["debugging", "architecture", "testing", "writing"],
        "tester": ["testing", "debugging", "architecture", "writing"],
        "writer": ["writing", "architecture", "debugging", "testing"],
        "reviewer": ["writing", "architecture", "testing", "debugging"],
        "book-editor": ["architecture", "debugging", "testing", "writing"],
    }
    priorities = role_priorities.get(role, role_priorities["implementer"])

    result = {
        "role": role,
        "chapter_id": chapter_id,
        "knowledge": _get_relevant_knowledge(chapter_id),
        "wisdom_priority": priorities,
        "checklist": _build_checklist(role, chapter_id),
    }

    # Print guidance
    print(f"\n{'='*60}")
    print(f"  Pre-Work Query: {role} → {chapter_id}")
    print(f"{'='*60}\n")

    if result["knowledge"]:
        print(f"[Knowledge] Relevant modules: {', '.join(result['knowledge'])}")
        for mod in result["knowledge"]:
            print(f"  → Read knowledge/modules/{mod}.md")
    else:
        print("[Knowledge] No existing knowledge for this chapter's modules.")

    print(f"\n[Wisdom] Read in priority order:")
    for i, cat in enumerate(priorities):
        wf = WISDOM_DIR / f"{cat}.md"
        status = "✓" if wf.exists() else "✗ (missing)"
        print(f"  {i+1}. wisdom/{cat}.md {status}")

    print(f"\n[Checklist] Before starting:")
    for item in result["checklist"]:
        print(f"  □ {item}")

    return result


# ── Phase 2: Extract & Classify After Work ───────────────────────

def extract(chapter_id: str, role: str) -> dict:
    """Interactive prompt for an agent to extract what it learned."""
    print(f"\n{'='*60}")
    print(f"  Post-Work Learning: {role} → {chapter_id}")
    print(f"{'='*60}\n")
    print("Extract what you learned. For each insight, classify as:")
    print("  K = Knowledge (repo-specific fact)")
    print("  W = Wisdom (universal pattern, applies to any repo)")
    print("\nAnswer the following:\n")

    questions = [
        ("FACTS", "What specific facts did you learn about this repo's code? "
                  "(file locations, API patterns, gotchas, conventions)"),
        ("PATTERNS", "What universal patterns did you discover? "
                     "(patterns that would apply to ANY repo)"),
        ("SURPRISES", "What surprised you? What didn't work as expected? "
                      "(these often become the most valuable wisdom)"),
        ("GOTCHAS", "What should the NEXT agent working on this module know? "
                    "(things that the code doesn't make obvious)"),
    ]

    learnings = {}
    for key, prompt in questions:
        print(f"\n── {key} ──────────────────────────────────")
        print(f"  {prompt}")
        print(f"  (The agent answers this; in automated mode, use --input file.json)")
        learnings[key] = {"question": prompt, "answer": None}

    return {
        "role": role,
        "chapter_id": chapter_id,
        "learnings": learnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_knowledge(module: str, facts: list, chapter_id: str, role: str) -> Path:
    """Append repo-specific facts to a knowledge module file (never rewrites existing content).

    The module file's existing entries set the heading prefix convention
    (`K`, `T`, `P`, `M`, ...). New entries continue that prefix; if the
    module is empty, default to `K`. Title bodies that already start with
    a `<PREFIX>NN:` token are stripped to avoid `## K01: K01: ...` duplication.
    """
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    module_file = MODULES_DIR / f"{module}.md"

    existing_entries = _parse_module_file(module_file)
    prefix = _detect_module_prefix(module_file, default="K")
    # Continue from the highest existing numeric id under that prefix.
    existing_ids = [
        int(e["id"][len(prefix):]) for e in existing_entries
        if e.get("id", "").startswith(prefix) and e["id"][len(prefix):].isdigit()
    ]
    next_id = (max(existing_ids) + 1) if existing_ids else 1
    existing_count = len(existing_entries)

    if not module_file.exists():
        with open(module_file, "w") as f:
            f.write(f"# {module.replace('-', ' ').title()} Knowledge\n")

    import re
    dup_prefix_re = re.compile(r"^\s*[A-Z]\d{1,3}\s*:\s*")
    decay = (datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).strftime("%Y-%m-%d")
    with open(module_file, "a") as f:
        for fact in facts:
            # Accept either {fact: "..."} (legacy) or {title: "...", body: "..."} (rich).
            raw_title = fact.get("title") or (fact.get("fact", "")[:60] if fact.get("fact") else "(untitled)")
            # Strip a leading `Xnn:` prefix from the title body so the heading
            # never reads `## K01: K01: ...` (the bug observed in kv-cache.md
            # and memory.md prior to this fix).
            title = dup_prefix_re.sub("", raw_title).strip() or "(untitled)"
            body = fact.get("body") or fact.get("fact", "")
            tags = fact.get("tags", [])
            source = fact.get("source", "")

            f.write("\n---\n\n")
            f.write(f"## {prefix}{next_id:02d}: {title}\n\n")
            f.write(f"**Module**: {module}\n")
            f.write(f"**Chapter**: {chapter_id}\n")
            f.write(f"**Discovered by**: {role}\n")
            f.write(f"**TTL**: {decay}\n")
            f.write(f"**Access count**: 0\n")
            f.write(f"**Tags**: {', '.join(tags)}\n\n")
            f.write(f"{body}\n")
            if source:
                f.write(f"\n**Source**: {source}\n")
            next_id += 1

    new_total = existing_count + len(facts)
    try:
        rel = module_file.relative_to(ROOT)
    except ValueError:
        rel = module_file
    print(f"  → Appended {len(facts)} facts to {rel} (total now: {new_total})")

    if new_total > MAX_FACTS_PER_MODULE:
        print(f"  ⚠ Module {module} has {new_total} facts (max {MAX_FACTS_PER_MODULE}).")
        print(f"    Run: python3 scripts/learn.py compact {module}")

    return module_file


def _detect_module_prefix(file_path: Path, default: str = "K") -> str:
    """Return the single-letter heading prefix used by an existing module
    file (e.g. 'K', 'T', 'P', 'M'). Falls back to `default` when the file
    is empty or missing.
    """
    if not file_path.exists():
        return default
    import re
    from collections import Counter
    matches = re.findall(
        r"^##+ ([A-Z])\d+:", file_path.read_text(), flags=re.MULTILINE
    )
    if not matches:
        return default
    # Use the most common prefix to be robust to one-off typos in the file.
    return Counter(matches).most_common(1)[0][0]


def _count_facts(file_path: Path) -> int:
    """Count `##`/`###` `<PREFIX>NN:` sections in a module file.
    `### K01:` sub-entries (preserved during compaction inside a `## K01–K05:`
    parent block) count as full facts — they represent the original distinct
    entries before compaction."""
    if not file_path.exists():
        return 0
    import re
    text = file_path.read_text()
    return len(re.findall(r"^##+ [A-Z]\d+:", text, flags=re.MULTILINE))


def propose_wisdom(category: str, pattern: dict, chapter_id: str, role: str) -> Path:
    """Propose a universal pattern for wisdom/ promotion."""
    wisdom_file = WISDOM_DIR / f"{category}.md"
    if not wisdom_file.exists():
        wisdom_file = WISDOM_DIR / "proposed.md"

    proposal = {
        "category": category,
        "discovered_by": role,
        "chapter": chapter_id,
        "confirmed_in": [CONFIG.get("instances", {}).get("current", {}).get("name", "unknown")],
        "severity": pattern.get("severity", "medium"),
        "applies_to": pattern.get("applies_to", [role]),
        "pattern": pattern.get("pattern", ""),
        "proposed": datetime.now(timezone.utc).isoformat(),
        "status": "proposed",  # Needs book-editor promotion
    }

    print(f"  → Proposed wisdom pattern to {wisdom_file.relative_to(ROOT)}")
    print(f"  ⚠ Pending promotion by book-editor (needs confirmation in 2+ repos)")
    return wisdom_file


# ── Phase 3: Anti-Bloat Compaction ───────────────────────────────

def compact(module: str) -> Optional[Path]:
    """Compact the oldest facts in a module file."""
    module_file = MODULES_DIR / f"{module}.md"
    if not module_file.exists():
        print(f"Module not found: {module}")
        return None

    existing = _parse_module_file(module_file)
    if len(existing) <= MAX_FACTS_PER_MODULE:
        print(f"Module {module}: {len(existing)}/{MAX_FACTS_PER_MODULE} — no compaction needed")
        return None

    # Sort by access_count ascending (least-used first)
    sorted_facts = sorted(existing, key=lambda f: f.get("access_count", 0))
    to_compact = sorted_facts[:COMPACT_COUNT]
    to_keep = sorted_facts[COMPACT_COUNT:]

    # Preserve the module's existing heading prefix (T/P/M/K/...) when
    # naming the synthetic summary entry, and use the next available
    # numeric id under that prefix.
    prefix = _detect_module_prefix(module_file, default="K")
    used_ids = [
        int(e["id"][len(prefix):]) for e in to_keep
        if e.get("id", "").startswith(prefix) and e["id"][len(prefix):].isdigit()
    ]
    summary_num = (max(used_ids) + 1) if used_ids else 1
    summary_id = f"{prefix}{summary_num:02d}"

    # Create a summary fact from the compacted ones
    compacted_titles = [f.get("title") or f.get("id", "") for f in to_compact]
    compacted_ids = [f.get("id", "?") for f in to_compact]
    summary_body = (
        f"**Module**: {module}\n"
        f"**Chapter**: compacted\n"
        f"**Discovered by**: learn.py (auto-compact)\n"
        f"**TTL**: {(datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).strftime('%Y-%m-%d')}\n"
        f"**Access count**: 0\n"
        f"**Tags**: compacted\n\n"
        f"[COMPACTED from {len(to_compact)} facts: {', '.join(compacted_ids)}]\n\n"
        + "; ".join(compacted_titles[:3])
        + ("..." if len(compacted_titles) > 3 else "")
    )
    summary_fact = {
        "id": summary_id,
        "title": f"[COMPACTED] {', '.join(compacted_ids)}",
        "module": module,
        "chapter": "compacted",
        "discovered_by": "learn.py (auto-compact)",
        "ttl_days": TTL_DAYS,
        "decay_after": (datetime.now(timezone.utc) + timedelta(days=TTL_DAYS)).strftime("%Y-%m-%d"),
        "access_count": 0,
        "body": summary_body,
        "fact": summary_body,
        "source": "auto-compaction",
        "tags": ["compacted"],
    }

    # Also archive the compacted facts
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_file = ARCHIVE_DIR / f"{module}-{datetime.now().strftime('%Y%m%d')}.json"
    with open(archive_file, "w") as f:
        json.dump({
            "module": module,
            "compacted_at": datetime.now(timezone.utc).isoformat(),
            "facts": to_compact,
            "summary": summary_fact["fact"],
        }, f, indent=2)

    # Write compacted file
    new_facts = to_keep + [summary_fact]
    _write_module_file(module_file, new_facts)

    print(f"Compacted {module}: {len(existing)} → {len(new_facts)} facts")
    try:
        print(f"  Archived: {archive_file.relative_to(ROOT)}")
    except ValueError:
        print(f"  Archived: {archive_file}")
    return module_file


# ── Promotion: Knowledge → Wisdom ────────────────────────────────

def promote(wisdom_id: str, approved_by: str = "book-editor") -> Optional[Path]:
    """Promote a proposed wisdom entry to the actual wisdom category file."""
    proposed_file = WISDOM_DIR / "proposed.md"
    if not proposed_file.exists():
        print("No proposed wisdom entries to promote.")
        return None

    # Find the entry by ID
    existing = _parse_module_file(proposed_file)
    target = None
    for entry in existing:
        if entry.get("id") == wisdom_id:
            target = entry
            break

    if not target:
        print(f"Wisdom entry {wisdom_id} not found in proposed.md")
        return None

    # Move to the appropriate category file
    category = target.get("category", "debugging")
    wisdom_file = WISDOM_DIR / f"{category}.md"

    target["status"] = "promoted"
    target["promoted_by"] = approved_by
    target["promoted_at"] = datetime.now(timezone.utc).isoformat()

    _append_module_file(wisdom_file, target)
    print(f"Promoted {wisdom_id} → wisdom/{category}.md")
    return wisdom_file


# ── Statistics ────────────────────────────────────────────────────

def stats() -> dict:
    """Print knowledge base statistics."""
    stats = {
        "knowledge_modules": 0,
        "knowledge_facts": 0,
        "wisdom_categories": 0,
        "wisdom_entries": 0,
        "archived_facts": 0,
        "needs_compaction": [],
        "stale_facts": [],
    }

    if MODULES_DIR.exists():
        for mf in MODULES_DIR.glob("*.md"):
            n = _count_facts(mf)
            stats["knowledge_modules"] += 1
            stats["knowledge_facts"] += n
            if n > MAX_FACTS_PER_MODULE:
                stats["needs_compaction"].append(mf.stem)

    if WISDOM_DIR.exists():
        for wf in WISDOM_DIR.glob("*.md"):
            if wf.stem == "INDEX":
                continue
            entries = _parse_module_file(wf)
            if entries:
                stats["wisdom_categories"] += 1
                stats["wisdom_entries"] += len(entries)

    if ARCHIVE_DIR.exists():
        for af in ARCHIVE_DIR.glob("*.json"):
            with open(af) as f:
                data = json.load(f)
            stats["archived_facts"] += len(data.get("facts", []))

    print(f"\n{'='*40}")
    print(f"  repo2book Knowledge Base Stats")
    print(f"{'='*40}")
    print(f"  Knowledge modules:  {stats['knowledge_modules']}")
    print(f"  Knowledge facts:    {stats['knowledge_facts']}")
    print(f"  Wisdom categories:  {stats['wisdom_categories']}")
    print(f"  Wisdom entries:     {stats['wisdom_entries']}")
    print(f"  Archived facts:     {stats['archived_facts']}")
    if stats["needs_compaction"]:
        print(f"  Needs compaction:   {', '.join(stats['needs_compaction'])}")
    print(f"{'='*40}\n")

    return stats


# ── Helpers ───────────────────────────────────────────────────────

def _get_relevant_knowledge(chapter_id: str) -> list:
    """Find which knowledge modules are relevant for a chapter."""
    # Read INDEX.md to find module→chapter mapping
    index_file = KNOWLEDGE_DIR / "INDEX.md"
    if not index_file.exists():
        return []

    with open(index_file) as f:
        content = f.read()

    # Simple extraction: modules that mention the chapter prefix
    relevant = []
    if MODULES_DIR.exists():
        for mf in MODULES_DIR.glob("*.md"):
            with open(mf) as f:
                if chapter_id[:2] in f.read() or chapter_id in f.read():
                    relevant.append(mf.stem)
    return relevant


def _parse_module_file(file_path: Path) -> list:
    """Parse a module file into a list of structured entries.

    Each entry is delimited by a `## <PREFIX>NN: <title>` heading (any
    uppercase letter prefix: K/T/P/M/W/...). Returns dicts with keys:
    `id`, `title`, `body`, `module`, `chapter`, `discovered_by`,
    `decay_after`, `access_count`, `tags`, `source`, `fact`.

    `fact` is set to `body` for backward compatibility with compact()'s
    summary pipeline. The bullet-style `**Field**: value` lines are
    extracted when present; missing fields default to "unknown" / 0 / [].
    """
    if not file_path.exists():
        return []
    import re
    text = file_path.read_text()

    # Match both `## K01:` (top-level fact) and `### K01:` (preserved
    # sub-entry inside a compacted parent like `## K01–K05: [COMPACTED]`).
    # The compacted parent itself doesn't match because its ID range
    # `K01–K05` contains an en-dash, not a `:`-terminated single ID.
    heading_re = re.compile(r"^##+ ([A-Z]\d+):\s*(.*?)\s*$", re.MULTILINE)
    headings = list(heading_re.finditer(text))
    if not headings:
        return []

    entries: list = []
    for i, m in enumerate(headings):
        body_start = m.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[body_start:body_end]
        # Trim a trailing `---` separator (and surrounding whitespace) belonging
        # to the *next* entry, not this one.
        section = re.sub(r"\n\s*---\s*\n?\s*$", "\n", section).strip("\n")

        entry = {
            "id": m.group(1),
            "title": m.group(2).strip(),
            "body": section,
            "fact": section,  # back-compat alias used by compact()
        }

        # Pull out the `**Field**: value` bullet lines if they're present.
        for field, key in (
            ("Module", "module"),
            ("Chapter", "chapter"),
            ("Discovered by", "discovered_by"),
            ("TTL", "decay_after"),
            ("Source", "source"),
        ):
            mm = re.search(rf"^\*\*{re.escape(field)}\*\*:\s*(.+)$", section, re.MULTILINE)
            if mm:
                entry[key] = mm.group(1).strip()

        ac = re.search(r"^\*\*Access count\*\*:\s*(\d+)", section, re.MULTILINE)
        entry["access_count"] = int(ac.group(1)) if ac else 0

        tags_m = re.search(r"^\*\*Tags\*\*:\s*(.*)$", section, re.MULTILINE)
        if tags_m:
            entry["tags"] = [t.strip() for t in tags_m.group(1).split(",") if t.strip()]
        else:
            entry["tags"] = []

        entries.append(entry)

    return entries


def _write_module_file(file_path: Path, entries: list):
    """Write structured entries to a module file, preserving each entry's
    original heading id (so a T-prefixed module stays T-prefixed after
    compaction) and the full body text returned by `_parse_module_file`."""
    title_h1 = f"# {file_path.stem.replace('-', ' ').title()} Knowledge"
    lines = [title_h1, ""]
    for entry in entries:
        eid = entry.get("id", "K??")
        title = entry.get("title") or (entry.get("body", "")[:60] if entry.get("body") else "")
        lines.append("---")
        lines.append("")
        lines.append(f"## {eid}: {title}".rstrip())
        lines.append("")
        body = entry.get("body") or entry.get("fact", "")
        lines.append(body.rstrip("\n"))
        lines.append("")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def _append_module_file(file_path: Path, entry: dict):
    """Append a single entry to a module file."""
    # Re-read existing, append, rewrite
    existing = _parse_module_file(file_path)
    # For wisdom files, just append text
    with open(file_path, "a") as f:
        f.write(f"\n---\n")
        f.write(f"\n## {entry.get('id', 'W??')}: {entry.get('pattern', '')[:60]}\n")
        f.write(f"\n**Discovered by**: {entry.get('discovered_by', 'unknown')}\n")
        f.write(f"**Confirmed in**: {', '.join(entry.get('confirmed_in', []))}\n")
        f.write(f"**Severity**: {entry.get('severity', 'medium')}\n")
        f.write(f"**Applies to**: {', '.join(entry.get('applies_to', []))}\n")
        f.write(f"\n{entry.get('pattern', '')}\n")


def _build_checklist(role: str, chapter_id: str) -> list:
    """Build a pre-work checklist for a specific role and chapter."""
    checklists = {
        "implementer": [
            "Read wisdom/debugging.md — check for common shape/import bugs",
            "Read wisdom/architecture.md — understand backpressure gates",
            "Query knowledge/INDEX.md for relevant module → read module file",
            "Complete 5-item Source Analysis in impl-notes.md BEFORE writing code",
            "Identify all relevant source files with absolute paths",
        ],
        "tester": [
            "Read wisdom/testing.md — check preemption/OOM test patterns",
            "Read knowledge/modules/ for the relevant module — note gotchas",
            "Read the implementation code BEFORE writing tests",
            "Design tests: unit (core logic) + integration (cross-chapter) + teaching examples",
            "Verify Docker image tag and command for containerized testing",
        ],
        "writer": [
            "Read wisdom/writing.md — formula rules, code walkthrough checks",
            "Read wisdom/debugging.md — SVG/diagram gotchas",
            "Read knowledge/modules/ for the relevant module — note file:line references",
            "Read the implementation file RIGHT BEFORE writing walkthrough (correct line numbers)",
            "Run lint_formulas.py after writing, fix all BLOCKING issues",
        ],
        "reviewer": [
            "Read wisdom/writing.md — know the auto-REJECT triggers",
            "Read wisdom/architecture.md — understand lateral communication rules",
            "Read knowledge/modules/ for the relevant module — verify line number references",
            "Review from 0-basis reader perspective: assume NO prior vLLM knowledge",
            "Run BOTH linters: lint_formulas.py AND lint_source_grounding.py",
        ],
        "book-editor": [
            "Read wisdom/architecture.md — pipeline patterns, lateral comm rules",
            "Read all wisdom/ category files — you need the full picture",
            "Check context.json for chapter state before dispatching",
            "Create tasks with proper blockedBy dependencies",
            "Spawn agents with correct backendType (tmux for writer/reviewer, in-process for others)",
        ],
    }
    return checklists.get(role, checklists["implementer"])


# ── CLI ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "query":
        chapter_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        role = sys.argv[3] if len(sys.argv) > 3 else "implementer"
        query(chapter_id, role)

    elif cmd == "extract":
        chapter_id = sys.argv[2] if len(sys.argv) > 2 else "unknown"
        role = sys.argv[3] if len(sys.argv) > 3 else "implementer"
        # Check for --input flag for automated extraction
        input_file = None
        for i, arg in enumerate(sys.argv):
            if arg == "--input" and i + 1 < len(sys.argv):
                input_file = sys.argv[i + 1]
                break
        if input_file:
            with open(input_file) as f:
                data = json.load(f)
            for module, facts in data.get("knowledge", {}).items():
                save_knowledge(module, facts, chapter_id, role)
            for category, patterns in data.get("wisdom", {}).items():
                for pattern in patterns:
                    propose_wisdom(category, pattern, chapter_id, role)
        else:
            extract(chapter_id, role)

    elif cmd == "compact":
        module = sys.argv[2] if len(sys.argv) > 2 else "all"
        if module == "all":
            if MODULES_DIR.exists():
                for mf in MODULES_DIR.glob("*.md"):
                    compact(mf.stem)
        else:
            compact(module)

    elif cmd == "promote":
        wisdom_id = sys.argv[2] if len(sys.argv) > 2 else ""
        approved_by = sys.argv[3] if len(sys.argv) > 3 else "book-editor"
        promote(wisdom_id, approved_by)

    elif cmd == "stats":
        stats()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
