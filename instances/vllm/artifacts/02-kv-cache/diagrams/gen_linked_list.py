#!/usr/bin/env python3
"""Diagram: Doubly-linked free block list with sentinels — why not deque."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 960, 480
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="arrR" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker><marker id="arrL" viewBox="0 0 10 6" refX="1" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M10,0 L0,3 L10,6 Z" fill="#94a3b8"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#fafafa"/>')

    GRAY = "#64748b"; DARK = "#1e293b"
    BLUE = "#3b82f6"; RED = "#ef4444"; GREEN = "#22c55e"; PURPLE = "#8b5cf6"

    # ── Title ──
    L.append(f'<text x="{w//2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="20" fill="{DARK}" font-weight="bold">{esc("FreeKVCacheBlockQueue: Hand-Rolled Doubly-Linked List")}</text>')
    L.append(f'<text x="{w//2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{GRAY}">{esc("Why not collections.deque? Because deque.remove(item) is O(n). We need O(1) for prefix cache touch().")}</text>')

    # ── Node drawing utility ──
    def draw_node(x, y, label, fill, txt_color="white", w=100, h=40):
        L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{DARK}" stroke-width="1.5"/>')
        L.append(f'<text x="{x + w//2}" y="{y + h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="11" fill="{txt_color}" font-weight="bold">{esc(label)}</text>')
        return x + w, y + h//2  # right edge center

    def draw_arrow(x1, y1, x2, y2, color=GRAY, label="", label_offset=0):
        L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#arrR)"/>')
        if label:
            L.append(f'<text x="{(x1+x2)//2}" y="{y1 - 10 - label_offset}" text-anchor="middle" font-family="monospace" font-size="9" fill="{color}">{esc(label)}</text>')

    def draw_back_arrow(x1, y1, x2, y2, color="#94a3b8"):
        L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5" marker-end="url(#arrL)" stroke-dasharray="4,3"/>')

    # ── Main list: head → free blocks → tail ──
    ny = 100  # node y
    # Head sentinel
    hx = 50
    draw_node(hx, ny, "HEAD\n(sentinel)", PURPLE)
    # Arrow head→first_free
    draw_arrow(hx + 100, ny + 20, hx + 160, ny + 20, GRAY, "next")

    # Free blocks (3 shown)
    free_labels = ["Block 42\n(free)", "Block 73\n(free)", "Block 15\n(free)"]
    node_positions = []
    for i, lbl in enumerate(free_labels):
        bx = hx + 170 + i * 150
        draw_node(bx, ny, lbl, GREEN)
        node_positions.append((bx, ny))
        if i < len(free_labels) - 1:
            draw_arrow(bx + 100, ny + 20, bx + 150, ny + 20, GRAY, "next")
        draw_back_arrow(bx + 100, ny + 24, bx - 56, ny + 24, "#94a3b8")

    # Arrow last_free→tail
    last_x = hx + 170 + 2 * 150
    tx = last_x + 130
    draw_arrow(last_x + 100, ny + 20, tx, ny + 20, GRAY, "next")

    # Tail sentinel
    draw_node(tx, ny, "TAIL\n(sentinel)", PURPLE)
    # Back arrow tail→last_free
    draw_back_arrow(tx, ny + 24, last_x + 104, ny + 24)

    # Labels below
    L.append(f'<text x="{hx + 50}" y="{ny + 62}" text-anchor="middle" font-family="monospace" font-size="10" fill="{RED}">{esc("LRU ← evict first")}</text>')
    L.append(f'<text x="{last_x + 50}" y="{ny + 62}" text-anchor="middle" font-family="monospace" font-size="10" fill="{BLUE}">{esc("MRU → evict last")}</text>')

    # ── O(1) remove from middle demonstration ──
    dem_y = ny + 100
    L.append(f'<text x="50" y="{dem_y}" font-family="sans-serif" font-size="14" fill="{DARK}" font-weight="bold">{esc("O(1) remove from middle (when prefix cache touches Block 73):")}</text>')

    # Show remove operation
    mid_x = hx + 170 + 150  # Block 73
    L.append(f'<text x="{mid_x + 50}" y="{dem_y + 30}" text-anchor="middle" font-family="monospace" font-size="12" fill="{RED}" font-weight="bold">{esc("touch(block_73)")}</text>')
    L.append(f'<text x="{mid_x + 50}" y="{dem_y + 50}" text-anchor="middle" font-family="monospace" font-size="11" fill="{DARK}">{esc("Step 1: block_73.prev.next = block_73.next")}</text>')
    L.append(f'<text x="{mid_x + 50}" y="{dem_y + 67}" text-anchor="middle" font-family="monospace" font-size="11" fill="{DARK}">{esc("Step 2: block_73.next.prev = block_73.prev")}</text>')
    L.append(f'<text x="{mid_x + 50}" y="{dem_y + 84}" text-anchor="middle" font-family="monospace" font-size="11" fill="{DARK}">{esc("Step 3: block_73.prev = block_73.next = None")}</text>')
    L.append(f'<text x="{mid_x + 50}" y="{dem_y + 104}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{GREEN}" font-weight="bold">{esc("Done: O(1) — no search, no traversal!")}</text>')

    # ── Comparison table ──
    comp_y = dem_y + 130
    L.append(f'<text x="50" y="{comp_y}" font-family="sans-serif" font-size="14" fill="{DARK}" font-weight="bold">{esc("Operation Complexity Comparison")}</text>')

    # Table header
    th_y = comp_y + 18
    col_x = [50, 260, 420, 580]
    col_w = [200, 150, 150, 200]
    headers = ["Operation", "collections.deque", "Hand-rolled LL", "Why?"]
    for i, (cx, cw, hdr) in enumerate(zip(col_x, col_w, headers)):
        fill = "#1e40af" if i == 0 else "#1e40af"
        L.append(f'<rect x="{cx}" y="{th_y}" width="{cw}" height="26" rx="2" fill="{fill}"/>')
        L.append(f'<text x="{cx + cw//2}" y="{th_y + 18}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="white" font-weight="bold">{esc(hdr)}</text>')

    rows = [
        ("popleft() (LRU)", "O(1)", "O(1)", "Both pop head"),
        ("append() (MRU)", "O(1)", "O(1)", "Both append tail"),
        ("remove(item) (touch)", "O(n) — SCAN", "O(1)", "Has prev/next ptrs"),
    ]
    for ri, (op, dq, ll, why) in enumerate(rows):
        ry = th_y + 28 + ri * 24
        bg = "#f8fafc" if ri % 2 == 0 else "white"
        for i, cx, cw in zip(range(4), col_x, col_w):
            L.append(f'<rect x="{cx}" y="{ry}" width="{cw}" height="22" fill="{bg}" stroke="#e2e8f0" stroke-width="0.5"/>')
        vals = [op, dq, ll, why]
        colors = [DARK, RED if "O(n)" in dq else GREEN, GREEN, GRAY]
        for val, clr, cx, cw in zip(vals, colors, col_x, col_w):
            L.append(f'<text x="{cx + cw//2}" y="{ry + 15}" text-anchor="middle" font-family="monospace" font-size="11" fill="{clr}">{esc(val)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/mnt/e/Laboratory/vllm-from-scratch/instances/vllm/artifacts/02-kv-cache/diagrams/02-kv-cache-linked-list'
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
