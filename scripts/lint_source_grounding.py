#!/usr/bin/env python3
"""
Source Grounding Linter — verify chapter is anchored to vLLM source code.

Usage:
    python scripts/lint_source_grounding.py artifacts/02-kv-cache/

Checks:
    1. Chapter narrative has vLLM file:line references per section
    2. Implementation code has # REFERENCE: comments
    3. Source Mapping Table has 5+ rows
    4. impl-notes.md has vLLM file list
"""

import re, sys, json
from pathlib import Path


def lint_source_grounding(chapter_dir: str) -> dict:
    """Run all grounding checks."""
    base = Path(chapter_dir)
    results = {}

    narrative = base / "narrative" / "chapter.md"
    impl_dir = base / "implementation"
    impl_notes = impl_dir / "impl-notes.md"
    ctx_path = base / "context.json"

    # ── Check 1: vLLM references in chapter narrative ──
    issues = []
    if narrative.exists():
        text = narrative.read_text(encoding="utf-8")
        sections = re.split(r'^## ', text, flags=re.MULTILINE)

        vllm_ref_pattern = re.compile(
            r'(?:vllm/)?[\w/]+\.(?:py|cu|cuh|h)(?:[\s:]*L?\d+(?:-L?\d+)?)?', re.IGNORECASE
        )

        refs_per_section = {}
        for sec in sections:
            title = sec.split('\n')[0].strip() if sec.strip() else '(intro)'
            refs = vllm_ref_pattern.findall(sec)
            refs_per_section[title] = len(refs)

        sections_without_refs = [t for t, n in refs_per_section.items() if n == 0]
        if sections_without_refs:
            issues.append(
                f"  Sections without vLLM source references: {sections_without_refs}"
            )
    results["narrative_vllm_refs"] = issues

    # ── Check 2: REFERENCE comments in implementation ──
    issues = []
    ref_count = 0
    if impl_dir.exists():
        for py_file in impl_dir.glob("*.py"):
            code = py_file.read_text(encoding="utf-8")
            refs = re.findall(r'# REFERENCE:\s*(.+)', code)
            ref_count += len(refs)
        if ref_count < 3:
            issues.append(
                f"  Only {ref_count} REFERENCE comments found (need >= 3)"
            )
    results["implementation_references"] = issues

    # ── Check 3: Source Mapping Table in impl-notes ──
    issues = []
    if impl_notes.exists():
        notes = impl_notes.read_text(encoding="utf-8")
        # Count rows in source mapping table
        table_rows = len(re.findall(r'^\|.*\|.*\|.*\|$', notes, re.MULTILINE))
        if table_rows < 5:
            issues.append(
                f"  Source Mapping Table has {table_rows} rows (need >= 5)"
            )
    results["source_mapping_table"] = issues

    # ── Check 4: vLLM files listed in impl-notes ──
    issues = []
    if impl_notes.exists():
        notes = impl_notes.read_text(encoding="utf-8")
        vllm_files = re.findall(r'vllm/[\w/]+\.py', notes)
        if len(vllm_files) < 3:
            issues.append(
                f"  Only {len(vllm_files)} vLLM files mentioned (need >= 3)"
            )
    results["vllm_files_listed"] = issues

    return results


def print_report(results: dict, chapter_dir: str):
    total = sum(len(v) for v in results.values())
    print(f"Source Grounding Lint: {chapter_dir}")
    print(f"{'=' * 60}")

    if total == 0:
        print("✓ All grounding checks passed!")
        return

    for check, issues in results.items():
        if issues:
            print(f"\n❌ {check} ({len(issues)} issue(s)):")
            for issue in issues:
                print(issue)

    blocking = len(results["narrative_vllm_refs"]) + len(results["implementation_references"])
    print(f"\n{'=' * 60}")
    if blocking > 0:
        print(f"🔴 {blocking} BLOCKING issue(s) — auto-REJECT")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lint_source_grounding.py <chapter_dir>")
        sys.exit(1)
    results = lint_source_grounding(sys.argv[1])
    print_report(results, sys.argv[1])
