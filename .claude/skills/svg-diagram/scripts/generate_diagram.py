#!/usr/bin/env python3
"""
SVG Diagram Generator — Template Script.
Usage: python3 generate_diagram.py <output_path_without_ext>
Produces: <output_path>.svg + <output_path>.png
"""

import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s: str) -> str:
    """Escape ALL text for XML safety. Call on every text string."""
    return xs.escape(s)

def validate(svg_path: str) -> bool:
    """Run xmllint. Returns True if valid XML."""
    r = subprocess.run(['xmllint', '--noout', svg_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"XML INVALID: {r.stderr[:300]}")
        return False
    return True

def convert_to_png(svg_path: str, png_path: str) -> bool:
    """Convert SVG to PNG with ImageMagick."""
    r = subprocess.run(['convert', '-density', '150', svg_path, png_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"PNG conversion failed: {r.stderr[:300]}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════
# DIAGRAM DEFINITION — adapt this section for your diagram
# ═══════════════════════════════════════════════════════════════

def build_svg() -> str:
    """Build SVG content. Replace with your diagram logic."""
    w, h = 600, 300
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

    # --- Add your diagram elements here ---
    L.append(f'<text x="20" y="30" font-family="monospace" font-size="16" fill="#1e40af" font-weight="bold">{esc("Diagram Title")}</text>')
    L.append(f'<text x="20" y="55" font-family="monospace" font-size="12" fill="#64748b">{esc("Subtitle or description")}</text>')
    # Example box + arrow:
    L.append(f'<rect x="100" y="100" width="150" height="40" rx="3" fill="#3b82f6" stroke="#1e3a5f" stroke-width="2"/>')
    L.append(f'<text x="175" y="125" text-anchor="middle" font-family="monospace" font-size="13" fill="white">{esc("Box Label")}</text>')
    L.append(f'<line x1="250" y1="120" x2="350" y2="120" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    L.append('</svg>')
    return '\n'.join(L)


# ═══════════════════════════════════════════════════════════════
# MAIN — do not modify below this line
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate_diagram.py <output_path_without_ext>")
        sys.exit(1)

    base = sys.argv[1]
    svg_path = base + '.svg'
    png_path = base + '.png'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    # Generate
    svg = build_svg()
    with open(svg_path, 'w') as f:
        f.write(svg)
    print(f"SVG: {svg_path} ({len(svg)} bytes)")

    # Validate XML syntax
    if not validate(svg_path):
        sys.exit(1)
    print("xmllint: VALID")

    # Validate SVG semantics
    val_script = Path(__file__).parent / 'validate_svg.py'
    r2 = subprocess.run([sys.executable, str(val_script), svg_path],
                        capture_output=True, text=True)
    if r2.returncode != 0:
        print(r2.stdout)
        sys.exit(1)
    print("semantics: CLEAN")

    # Convert to PNG
    if not convert_to_png(svg_path, png_path):
        sys.exit(1)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)//1024} KB)")
