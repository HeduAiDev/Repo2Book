#!/usr/bin/env python3
"""
Formula Linter — auto-detect LaTeX rendering issues in chapter narratives.

Usage:
    python scripts/lint_formulas.py artifacts/01-self-attention-fundamentals/narrative/chapter.md

Checks:
    1. \text{} in formulas → use \mathrm{} instead
    2. \boxed{} → not supported in basic renderers
    3. \tag*{} → not supported in basic renderers
    4. \frac inside inline $...$ (should be block $$...$$)
    5. $$ on same line as content (needs separate lines)
    6. Too many inline formulas per paragraph
    7. Underscores in inline math that could collide with markdown
"""

import re
import sys
from pathlib import Path


def lint_formulas(filepath: str) -> dict:
    """Run all formula checks. Returns {check_name: [issues]}."""
    text = Path(filepath).read_text(encoding="utf-8")
    lines = text.split("\n")

    results = {}

    # ── Check 1: \text{} usage ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\text{" in line:
            issues.append(f"  Line {i}: \\text{{...}} found — use \\mathrm{{...}} instead")
    results["text_instead_of_mathrm"] = issues

    # ── Check 2: \boxed{} ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\boxed{" in line:
            issues.append(
                f"  Line {i}: \\boxed{{...}} found — requires amsmath, "
                f"use bold markdown header above formula instead"
            )
    results["boxed_requires_amsmath"] = issues

    # ── Check 3: \tag*{} ──
    issues = []
    for i, line in enumerate(lines, 1):
        if "\\tag{" in line or "\\tag*{" in line:
            issues.append(
                f"  Line {i}: \\tag{{}} found — requires amsmath, "
                f"put annotation outside $$ block"
            )
    results["tag_requires_amsmath"] = issues

    # ── Check 4: \frac inside inline $ ──
    issues = []
    # Find inline math spans: $...$ where ... contains \frac
    for i, line in enumerate(lines, 1):
        # Remove $$ blocks first (they're fine)
        cleaned = re.sub(r'\$\$[^$]*\$\$', '', line)
        # Find $...$ with \frac inside
        for m in re.finditer(r'\$([^$]+)\$', cleaned):
            if '\\frac' in m.group(1):
                issues.append(
                    f"  Line {i}: \\frac inside inline $...$ → "
                    f'"{m.group()[:50]}..." — promote to $$...$$ block'
                )
    results["frac_in_inline_math"] = issues

    # ── Check 5: $$ on same line as formula content ──
    issues = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("$$") and len(line.strip()) > 2:
            # $$ followed by content on same line
            content = line.strip()[2:].strip()
            if content and not content.startswith("$$"):
                issues.append(
                    f"  Line {i}: $$ on same line as formula content — "
                    f"move content to next line"
                )
    results["block_math_on_separate_lines"] = issues

    # ── Check 6: Inline formulas with ≥3 $...$ per paragraph ──
    issues = []
    paragraph_lines = []
    current_start = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#") or stripped.startswith("```"):
            if paragraph_lines:
                para = " ".join(paragraph_lines)
                inline_count = len(re.findall(r'(?<!\$)\$(?!\$)[^$]+\$(?!\$)', para))
                if inline_count >= 3:
                    issues.append(
                        f"  Lines {current_start}-{i-1}: {inline_count} inline formulas "
                        f"in one paragraph — consider promoting some to block formulas"
                    )
                paragraph_lines = []
                current_start = i + 1
        else:
            if not paragraph_lines:
                current_start = i
            # Remove $$ blocks from count
            cleaned = re.sub(r'\$\$[^$]*\$\$', '', stripped)
            paragraph_lines.append(cleaned)
    results["too_many_inline_formulas"] = issues

    # ── Check 7: Complex inline formulas (>30 chars inside $...$) ──
    issues = []
    for i, line in enumerate(lines, 1):
        cleaned = re.sub(r'\$\$[^$]*\$\$', '', line)
        for m in re.finditer(r'\$(?!\$)([^$]+)\$(?!\$)', cleaned):
            content = m.group(1)
            if len(content) > 30:
                issues.append(
                    f"  Line {i}: Complex inline formula ({len(content)} chars) → "
                    f'"{content[:60]}..." — should be block formula'
                )
    results["complex_inline_formulas"] = issues

    return results


def print_report(results: dict, filepath: str):
    """Pretty-print the lint results."""
    total_issues = sum(len(v) for v in results.values())
    print(f"Formula Lint: {filepath}")
    print(f"{'=' * 60}")

    if total_issues == 0:
        print("✓ All formula checks passed!")
        return

    for check_name, issues in results.items():
        if issues:
            label = check_name.replace("_", " ").title()
            print(f"\n❌ {label} ({len(issues)} issue(s)):")
            for issue in issues:
                print(issue)

    print(f"\n{'=' * 60}")
    print(f"Total: {total_issues} issue(s) found")
    print()

    # Severity
    blocking = (
        len(results.get("text_instead_of_mathrm", []))
        + len(results.get("boxed_requires_amsmath", []))
        + len(results.get("tag_requires_amsmath", []))
        + len(results.get("frac_in_inline_math", []))
        + len(results.get("block_math_on_separate_lines", []))
    )
    if blocking > 0:
        print(f"🔴 {blocking} BLOCKING issue(s) — auto-REJECT")
    else:
        print("🟢 No blocking issues")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lint_formulas.py <chapter.md>")
        sys.exit(1)

    filepath = sys.argv[1]
    results = lint_formulas(filepath)
    print_report(results, filepath)
