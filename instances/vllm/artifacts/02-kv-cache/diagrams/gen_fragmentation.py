#!/usr/bin/env python3
"""Diagram: External vs Internal Fragmentation — contiguous vs block-based allocation."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 960, 520
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    L.append('<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#fafafa"/>')

    RED = "#ef4444"; RED_BG = "#fef2f2"
    GREEN = "#22c55e"; GREEN_BG = "#f0fdf4"
    GRAY = "#64748b"; DARK = "#1e293b"
    BLUE = "#3b82f6"
    ORANGE = "#f97316"; ORANGE_BG = "#fff7ed"

    # ── Title ──
    L.append(f'<text x="{w//2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="20" fill="{DARK}" font-weight="bold">{esc("External vs Internal Fragmentation: Why Block-Based Wins")}</text>')

    # ── Top: Contiguous Allocation ──
    cy = 70
    L.append(f'<text x="30" y="{cy}" font-family="sans-serif" font-size="16" fill="#dc2626" font-weight="bold">{esc("Contiguous Allocation — External Fragmentation")}</text>')

    # Memory bar: 100 units total
    bar_x, bar_y, bar_w, bar_h = 30, cy + 20, 900, 40
    total_units = 100
    px_per_unit = bar_w / total_units

    # Scenario: A=30, B=20, A leaves → hole of 30, C needs 50 → can't fit!
    L.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="2" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1"/>')

    # Phase 1: A(30) + B(20) + free(50)
    ph1_y = bar_y + bar_h + 16
    L.append(f'<text x="30" y="{ph1_y}" font-family="monospace" font-size="11" fill="{GRAY}">{esc("Phase 1: Request A(30 tokens) arrives → alloc [0..29]")}</text>')
    L.append(f'<rect x="{bar_x}" y="{bar_y}" width="{30*px_per_unit}" height="{bar_h}" rx="2" fill="{BLUE}" opacity="0.7"/>')
    L.append(f'<text x="{bar_x + 15*px_per_unit}" y="{bar_y + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="white" font-weight="bold">A = 30</text>')
    L.append(f'<text x="{bar_x + 30*px_per_unit + 4}" y="{bar_y + bar_h//2 + 5}" font-family="monospace" font-size="10" fill="{GRAY}">Free = 70</text>')

    ph2_y = ph1_y + 50
    L.append(f'<text x="30" y="{ph2_y}" font-family="monospace" font-size="11" fill="{GRAY}">{esc("Phase 2: Request B(20 tokens) arrives → alloc [30..49]")}</text>')
    L.append(f'<rect x="{bar_x}" y="{bar_y+60}" width="{30*px_per_unit}" height="{bar_h}" rx="2" fill="{BLUE}" opacity="0.4"/>')
    L.append(f'<text x="{bar_x + 15*px_per_unit}" y="{bar_y+60 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{GRAY}">A=30</text>')
    L.append(f'<rect x="{bar_x + 30*px_per_unit}" y="{bar_y+60}" width="{20*px_per_unit}" height="{bar_h}" rx="2" fill="{BLUE}" opacity="0.7"/>')
    L.append(f'<text x="{bar_x + 40*px_per_unit}" y="{bar_y+60 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="white" font-weight="bold">B=20</text>')
    L.append(f'<text x="{bar_x + 50*px_per_unit + 4}" y="{bar_y+60 + bar_h//2 + 5}" font-family="monospace" font-size="10" fill="{GRAY}">Free = 50</text>')

    ph3_y = ph2_y + 50
    L.append(f'<text x="30" y="{ph3_y}" font-family="monospace" font-size="11" fill="{GRAY}">{esc("Phase 3: A finishes → hole [0..29] appears")}</text>')
    L.append(f'<rect x="{bar_x}" y="{bar_y+120}" width="{30*px_per_unit}" height="{bar_h}" rx="2" fill="{RED}" opacity="0.5" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5,3"/>')
    L.append(f'<text x="{bar_x + 15*px_per_unit}" y="{bar_y+120 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{RED}" font-weight="bold">HOLE=30</text>')
    L.append(f'<rect x="{bar_x + 30*px_per_unit}" y="{bar_y+120}" width="{20*px_per_unit}" height="{bar_h}" rx="2" fill="{BLUE}" opacity="0.7"/>')
    L.append(f'<text x="{bar_x + 40*px_per_unit}" y="{bar_y+120 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="white">B=20</text>')
    L.append(f'<text x="{bar_x + 50*px_per_unit + 4}" y="{bar_y+120 + bar_h//2 + 5}" font-family="monospace" font-size="10" fill="{GRAY}">Free = 50</text>')

    ph4_y = ph3_y + 50
    L.append(f'<text x="30" y="{ph4_y}" font-family="monospace" font-size="11" fill="{GRAY}">{esc("Phase 4: Request C(50 tokens) arrives → Total free = 30+50 = 80 > 50")}</text>')
    L.append(f'<text x="30" y="{ph4_y + 18}" font-family="monospace" font-size="11" fill="#dc2626" font-weight="bold">{esc("          BUT hole=30 < 50, tail=50 — C cannot fit! EXTERNAL FRAGMENTATION")}</text>')
    L.append(f'<rect x="{bar_x}" y="{bar_y+180}" width="{30*px_per_unit}" height="{bar_h}" rx="2" fill="{RED}" opacity="0.5" stroke="{RED}" stroke-width="1.5" stroke-dasharray="5,3"/>')
    L.append(f'<text x="{bar_x + 15*px_per_unit}" y="{bar_y+180 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{RED}" font-weight="bold">HOLE=30</text>')
    L.append(f'<rect x="{bar_x + 30*px_per_unit}" y="{bar_y+180}" width="{20*px_per_unit}" height="{bar_h}" rx="2" fill="{BLUE}" opacity="0.7"/>')
    L.append(f'<text x="{bar_x + 40*px_per_unit}" y="{bar_y+180 + bar_h//2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="white">B=20</text>')
    L.append(f'<text x="{bar_x + 50*px_per_unit + 4}" y="{bar_y+180 + bar_h//2 + 5}" font-family="monospace" font-size="10" fill="{GRAY}">Free = 50</text>')
    # X mark
    L.append(f'<text x="{bar_x + 15*px_per_unit}" y="{bar_y+180 + bar_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#dc2626" font-weight="bold">{esc("TOO SMALL")}</text>')
    L.append(f'<text x="{bar_x + 65*px_per_unit}" y="{bar_y+180 + bar_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#dc2626" font-weight="bold">{esc("CANNOT USE")}</text>')

    # ── Bottom: Block-Based Allocation ──
    by = 370
    L.append(f'<text x="30" y="{by}" font-family="sans-serif" font-size="16" fill="#16a34a" font-weight="bold">{esc("Block-Based Allocation (vLLM) — Zero External Fragmentation")}</text>')
    L.append(f'<text x="30" y="{by + 22}" font-family="monospace" font-size="11" fill="{GRAY}">{esc("Same scenario, block_size=16 tokens. A=30→2 blocks, B=20→2 blocks, C=50→4 blocks.")}</text>')

    # Draw 10 blocks in a row
    nb = 10
    bw = 66; bh = 36; bgap = 6
    bx_start = 30; by_pos = by + 38
    blocks_used = [("A", BLUE, 0.7, None), ("A", GREEN, 0.7, "最后一个"), ("B", BLUE, 0.7, None), ("B", GREEN, 0.7, "最后一个"),
                   ("", GRAY, 0.2, "free"), ("", GRAY, 0.2, "free"), ("", GRAY, 0.2, "free"), ("", GRAY, 0.2, "free"),
                   ("", GRAY, 0.2, "free"), ("", GRAY, 0.2, "free")]
    for i, (label, color, opacity, note) in enumerate(blocks_used[:10]):
        ix = bx_start + i * (bw + bgap)
        if color == GRAY:
            L.append(f'<rect x="{ix}" y="{by_pos}" width="{bw}" height="{bh}" rx="3" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1"/>')
        else:
            L.append(f'<rect x="{ix}" y="{by_pos}" width="{bw}" height="{bh}" rx="3" fill="{color}" opacity="{opacity}"/>')
        if label:
            L.append(f'<text x="{ix + bw//2}" y="{by_pos + bh//2 + 4}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="white" font-weight="bold">{esc(label)}</text>')
        # Block ID
        L.append(f'<text x="{ix + bw//2}" y="{by_pos + bh + 13}" text-anchor="middle" font-family="monospace" font-size="9" fill="{GRAY}">blk{i}</text>')
        if note:
            L.append(f'<text x="{ix + bw//2}" y="{by_pos - 6}" text-anchor="middle" font-family="monospace" font-size="8" fill="{GRAY}">{esc(note)}</text>')

    # Internal fragmentation note
    frag_y = by_pos + bh + 40
    L.append(f'<rect x="30" y="{frag_y}" width="{w-60}" height="48" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>')
    L.append(f'<text x="{w//2}" y="{frag_y + 18}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#166534" font-weight="bold">{esc("Internal fragmentation only: each block can waste ≤ (block_size-1) tokens")}</text>')
    L.append(f'<text x="{w//2}" y="{frag_y + 36}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#166534">{esc("For block_size=16: max waste = 15 tokens/request. For 4096 tokens: ≤ 0.37% — negligible!")}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/02-kv-cache/diagrams/02-kv-cache-fragmentation'
    svg_path = base + '.svg'; png_path = base + '.png'
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    svg = build_svg()
    with open(svg_path, 'w') as f: f.write(svg)
    print(f"SVG: {svg_path} ({len(svg)} bytes)")
    r = subprocess.run(['xmllint', '--noout', svg_path], capture_output=True, text=True)
    if r.returncode != 0: print(f"XML INVALID: {r.stderr[:300]}"); sys.exit(1)
    print("xmllint: VALID")
    r3 = subprocess.run(['convert', '-density', '150', svg_path, png_path], capture_output=True, text=True)
    if r3.returncode != 0: print(f"PNG failed: {r3.stderr[:300]}"); sys.exit(1)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)//1024} KB)")
