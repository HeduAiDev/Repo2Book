#!/usr/bin/env python3
"""Diagram: Auto-regressive waste pattern — why KV Cache is inevitable."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 960, 600
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    L.append('<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('<marker id="arrRed" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#fafafa"/>')

    # Colors
    RED = "#ef4444"      # recompute
    RED_BG = "#fef2f2"   # light red bg
    GREEN = "#22c55e"    # new compute
    GREEN_BG = "#f0fdf4" # light green bg
    BLUE = "#3b82f6"     # cache read
    BLUE_BG = "#eff6ff"  # light blue bg
    GRAY = "#64748b"

    # ── Title ──
    L.append(f'<text x="{w//2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1e293b" font-weight="bold">{esc("Auto-Regressive Generation: With vs Without KV Cache")}</text>')
    L.append(f'<text x="{w//2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#64748b">{esc("L = 4 tokens generated  |  Each colored box = one token position")}</text>')

    # ── Panel geometry ──
    left_x = 40
    right_x = 500
    panel_w = 410
    start_y = 80
    row_h = 60
    gap = 12
    sq = 36  # square size
    label_x = 100  # Step label x offset

    # ── Left panel: Without KV Cache ──
    L.append(f'<rect x="{left_x}" y="{start_y}" width="{panel_w}" height="{5*row_h + 20}" rx="6" fill="#fef2f2" stroke="#fca5a5" stroke-width="2"/>')
    L.append(f'<text x="{left_x + panel_w//2}" y="{start_y - 14}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#dc2626" font-weight="bold">{esc("Without KV Cache — Recompute all previous Q,K,V each step")}</text>')

    for step in range(4):
        base_y = start_y + 30 + step * row_h
        tokens = step + 1
        # Step label
        L.append(f'<text x="{left_x + 10}" y="{base_y + sq//2 + 5}" font-family="monospace" font-size="12" fill="#1e293b" font-weight="bold">{esc(f"Step {step+1}")}</text>')
        # Draw squares for each token
        for t in range(tokens):
            sx = left_x + 70 + t * (sq + 4)
            if t < tokens - 1:
                # Recompute old K,V
                L.append(f'<rect x="{sx}" y="{base_y}" width="{sq}" height="{sq}" rx="3" fill="{RED}" opacity="0.85"/>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq//2 + 4}" text-anchor="middle" font-family="monospace" font-size="10" fill="white" font-weight="bold">K{t+1}V</text>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq + 12}" text-anchor="middle" font-family="monospace" font-size="9" fill="#dc2626">{esc("re算")}</text>')
            else:
                # New compute
                L.append(f'<rect x="{sx}" y="{base_y}" width="{sq}" height="{sq}" rx="3" fill="{GREEN}" opacity="0.85"/>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq//2 + 4}" text-anchor="middle" font-family="monospace" font-size="10" fill="white" font-weight="bold">K{t+1}V</text>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq + 12}" text-anchor="middle" font-family="monospace" font-size="9" fill="#22c55e">{esc("新算")}</text>')

        # Arrow showing attention over N tokens
        end_x = left_x + 70 + tokens * (sq + 4) - 4
        L.append(f'<line x1="{left_x + 250}" y1="{base_y + sq//2}" x2="{end_x}" y2="{base_y + sq//2}" stroke="#f87171" stroke-width="1.5" stroke-dasharray="4,3"/>')

    # Total counts for left
    total_y = start_y + 30 + 4 * row_h + 14
    L.append(f'<text x="{left_x + panel_w//2}" y="{total_y}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#dc2626" font-weight="bold">{esc("Total: 1+2+3+4 = 10 computations  |  O(L²)")}</text>')

    # ── Speedup annotation between panels ──
    sp_y = start_y + 30 + 2 * row_h
    L.append(f'<text x="472" y="{sp_y}" text-anchor="middle" font-family="sans-serif" font-size="32" fill="#64748b" font-weight="bold">{esc("→")}</text>')
    L.append(f'<text x="472" y="{sp_y + 24}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{esc("Speedup =")}</text>')
    L.append(f'<text x="472" y="{sp_y + 40}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{esc("(L+1)/2 = 2.5x (L=4)")}</text>')
    L.append(f'<text x="472" y="{sp_y + 56}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{esc("= 2048x (L=4096)")}</text>')

    # ── Right panel: With KV Cache ──
    L.append(f'<rect x="{right_x}" y="{start_y}" width="{panel_w}" height="{5*row_h + 20}" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="2"/>')
    L.append(f'<text x="{right_x + panel_w//2}" y="{start_y - 14}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#16a34a" font-weight="bold">{esc("With KV Cache — Compute only new, read old from cache")}</text>')

    for step in range(4):
        base_y = start_y + 30 + step * row_h
        tokens = step + 1
        L.append(f'<text x="{right_x + 10}" y="{base_y + sq//2 + 5}" font-family="monospace" font-size="12" fill="#1e293b" font-weight="bold">{esc(f"Step {step+1}")}</text>')
        for t in range(tokens):
            sx = right_x + 70 + t * (sq + 4)
            if t < tokens - 1:
                # Cache read
                L.append(f'<rect x="{sx}" y="{base_y}" width="{sq}" height="{sq}" rx="3" fill="{BLUE}" opacity="0.85"/>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq//2 + 4}" text-anchor="middle" font-family="monospace" font-size="10" fill="white" font-weight="bold">K{t+1}V</text>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq + 12}" text-anchor="middle" font-family="monospace" font-size="9" fill="#3b82f6">{esc("缓存读")}</text>')
            else:
                # New compute
                L.append(f'<rect x="{sx}" y="{base_y}" width="{sq}" height="{sq}" rx="3" fill="{GREEN}" opacity="0.85"/>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq//2 + 4}" text-anchor="middle" font-family="monospace" font-size="10" fill="white" font-weight="bold">K{t+1}V</text>')
                L.append(f'<text x="{sx + sq//2}" y="{base_y + sq + 12}" text-anchor="middle" font-family="monospace" font-size="9" fill="#22c55e">{esc("新算")}</text>')

        # Cache read arrow
        if tokens > 1:
            cache_start = right_x + 70
            cache_end = right_x + 70 + (tokens - 1) * (sq + 4)
            L.append(f'<line x1="{cache_start}" y1="{base_y - 8}" x2="{cache_end}" y2="{base_y - 8}" stroke="#93c5fd" stroke-width="2"/>')
            L.append(f'<text x="{(cache_start + cache_end)//2}" y="{base_y - 14}" text-anchor="middle" font-family="monospace" font-size="8" fill="#3b82f6">{esc("cache read")}</text>')

    total_y2 = start_y + 30 + 4 * row_h + 14
    L.append(f'<text x="{right_x + panel_w//2}" y="{total_y2}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#16a34a" font-weight="bold">{esc("Total: 4 computations  |  O(L)")}</text>')

    # ── Legend ──
    leg_y = start_y + 5 * row_h + 50
    L.append(f'<text x="{w//2}" y="{leg_y}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1e293b" font-weight="bold">{esc("Legend")}</text>')
    items = [
        (GREEN, "新算 = Compute new K,V (once per token)"),
        (RED, "重算 = Recompute old K,V (waste!)"),
        (BLUE, "缓存读 = Read K,V from cache"),
    ]
    leg_start_x = w//2 - 160
    for i, (color, label) in enumerate(items):
        ix = leg_start_x + i * 120
        L.append(f'<rect x="{ix}" y="{leg_y + 14}" width="14" height="14" rx="2" fill="{color}"/>')
        L.append(f'<text x="{ix + 20}" y="{leg_y + 26}" font-family="sans-serif" font-size="11" fill="#334155">{esc(label)}</text>')

    # ── Bottom insight ──
    insight_y = leg_y + 60
    L.append(f'<rect x="60" y="{insight_y}" width="{w-120}" height="48" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="{w//2}" y="{insight_y + 30}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#1e40af" font-weight="bold">{esc("Key insight: K,V for past tokens are deterministic (causal mask) — they never change.")}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/02-kv-cache/diagrams/02-kv-cache-waste-pattern'
    svg_path = base + '.svg'
    png_path = base + '.png'
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    svg = build_svg()
    with open(svg_path, 'w') as f:
        f.write(svg)
    print(f"SVG: {svg_path} ({len(svg)} bytes)")

    # Validate
    import subprocess as sp
    r = sp.run(['xmllint', '--noout', svg_path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"XML INVALID: {r.stderr[:300]}")
        sys.exit(1)
    print("xmllint: VALID")

    # Semantic check
    val_script = Path(__file__).parent.parent.parent.parent.parent / '.claude/skills/svg-diagram/scripts/validate_svg.py'
    if val_script.exists():
        r2 = sp.run([sys.executable, str(val_script), svg_path], capture_output=True, text=True)
        if r2.returncode != 0:
            print(r2.stdout)
            sys.exit(1)
        print("semantics: CLEAN")

    # PNG
    r3 = sp.run(['convert', '-density', '150', svg_path, png_path], capture_output=True, text=True)
    if r3.returncode != 0:
        print(f"PNG conversion failed: {r3.stderr[:300]}")
        sys.exit(1)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)//1024} KB)")
