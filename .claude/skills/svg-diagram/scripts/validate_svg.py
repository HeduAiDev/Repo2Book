#!/usr/bin/env python3
"""
SVG Semantic Validator — catches rendering bugs that xmllint can't.

xmllint validates XML syntax (well-formedness).
This script validates SVG semantics (will it render correctly?).

Usage: python3 validate_svg.py file.svg
Exit code: 0 = clean, 1 = errors found
"""

import sys, re
from pathlib import Path

def check_svg(svg_path: str) -> list[str]:
    """Run all semantic checks. Returns list of error messages."""
    text = Path(svg_path).read_text(encoding='utf-8')
    errors = []

    # —— Check 1: Double-escaping ——
    # esc() called on already-escaped text → &amp;lt; &amp;gt; &amp;amp;
    double_escapes = re.findall(r'&amp;(lt|gt|amp|quot|apos);', text)
    if double_escapes:
        errors.append(f"DOUBLE-ESCAPING: found &amp;{double_escapes[0]}; — "
                       "Python source had pre-escaped text. Write RAW text "
                       "in Python strings, let esc() handle escaping ONCE.")

    # —— Check 2: Unescaped < or > in text content ——
    # xmllint catches this as XML error, but we double-check
    # (skipped — xmllint handles this)

    # —— Check 3: Text clipping risk ——
    # text-anchor="end" at small x values → text extends beyond viewBox
    viewbox_match = re.search(r'viewBox="([^"]+)"', text)
    if viewbox_match:
        parts = viewbox_match.group(1).split()
        vbx_min = float(parts[0])
        # Find text-anchor="end" elements and check their x
        for m in re.finditer(r'text-anchor="end"[^>]*x="([^"]+)"', text):
            x = float(m.group(1))
            if x < 30:
                errors.append(f"TEXT CLIPPING RISK: text-anchor='end' at x={x}. "
                               "Text extends LEFT from x. Use x>=50 for row labels. "
                               f"Context: ...{text[max(0,m.start()-40):m.end()+40]}...")

    # —— Check 4: Arrowhead markers ——
    # Lines between elements should have marker-end
    markers_defined = set(re.findall(r'id="([^"]+)"', text))
    lines = re.findall(r'<(line|path)[^>]*>', text)
    for tag in lines:
        if 'marker-end' not in tag and 'stroke' in tag:
            m = re.search(r'id="([^"]+)"', tag)
            eid = m.group(1) if m else "?"
            errors.append(f"MISSING ARROWHEAD: <{tag.split()[0]} id='{eid}'...> "
                           "has stroke but no marker-end. Add marker-end='url(#arrowId)'.")

    # —— Check 5: PNG reference ——
    # Embedded images should reference .png, not .svg
    # (this check is for the markdown file, not SVG)

    return errors


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 validate_svg.py file.svg")
        sys.exit(1)

    errors = check_svg(sys.argv[1])
    if errors:
        print(f"SEMANTIC ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("SEMANTIC CHECK: clean")
