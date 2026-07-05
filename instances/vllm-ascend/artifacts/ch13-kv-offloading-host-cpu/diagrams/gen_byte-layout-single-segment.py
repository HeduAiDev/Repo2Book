#!/usr/bin/env python3
"""layout 模板：单段 storage 的字节布局。
展示 storage.nbytes()（全长，含前导对齐偏移 + 尾部 padding）与
shape/stride 精确裁出的 num_blocks×page_size_bytes 数据区之间的错位。
数字来自 explainer/traces/build_block_views.json#A_single_segment。"""
import os
import subprocess
import xml.sax.saxutils as xs

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return xs.escape(s)


def text(x, y, s, size=13, fill="#1e293b", anchor="middle", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


def rect(x, y, w, h, fill, stroke, rx=4, sw=2, fill_opacity=None):
    fo = f' fill-opacity="{fill_opacity}"' if fill_opacity is not None else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}"{fo} stroke="{stroke}" stroke-width="{sw}"/>')


# ── numbers (all sourced from traces/build_block_views.json#A_single_segment) ──
LEADING_BYTES = 6
DATA_BYTES = 32
TRAILING_BYTES = 10
STORAGE_NBYTES = 48
PAGE_SIZE_BYTES = 8
NUM_BLOCKS = 4
NAIVE_BLOCKS = 6
PHANTOM_BLOCKS = 2
STRIDE0_ELEMS = 4
ELEMENT_SIZE = 2

assert LEADING_BYTES + DATA_BYTES + TRAILING_BYTES == STORAGE_NBYTES
assert DATA_BYTES == NUM_BLOCKS * PAGE_SIZE_BYTES
assert STRIDE0_ELEMS * ELEMENT_SIZE == PAGE_SIZE_BYTES

SCALE = 8  # px per byte
BAR_H = 70
PAD = 50
TOP = 118

leading_w = LEADING_BYTES * SCALE
data_w = DATA_BYTES * SCALE
trailing_w = TRAILING_BYTES * SCALE
bar_w = leading_w + data_w + trailing_w
MIN_W = 760
W = max(MIN_W, PAD * 2 + bar_w + 40)
H = TOP + BAR_H + 230

bar_x = (W - bar_w) / 2
bar_y = TOP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" '
    'patternUnits="userSpaceOnUse">'
    '<line x1="0" y1="0" x2="0" y2="7" stroke="#94a3b8" stroke-width="3"/>'
    '</pattern>'
    '<marker id="tick" viewBox="0 0 10 6" refX="5" refY="3" markerWidth="6" markerHeight="6" orient="auto">'
    '<path d="M5,0 L5,6" stroke="#dc2626" stroke-width="1.4"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(text(W / 2, 30, "block 视图的字节布局：storage.nbytes() 把对齐残渣也数进去了",
              17, "#0f172a", weight="bold"))
L.append(text(W / 2, 52,
              f"page_size_bytes = stride(0)×element_size = {STRIDE0_ELEMS}×{ELEMENT_SIZE} = {PAGE_SIZE_BYTES}B",
              13, "#475569"))

# ── top bracket: full storage.nbytes() span (wrong view) ──
top_bk_y = bar_y - 34
L.append(f'<line x1="{bar_x}" y1="{top_bk_y}" x2="{bar_x + bar_w}" y2="{top_bk_y}" '
          'stroke="#dc2626" stroke-width="2"/>')
for xx in (bar_x, bar_x + bar_w):
    L.append(f'<line x1="{xx}" y1="{top_bk_y}" x2="{xx}" y2="{top_bk_y + 8}" stroke="#dc2626" stroke-width="2"/>')
L.append(text(bar_x + bar_w / 2, top_bk_y - 10,
              f"storage.nbytes()={STORAGE_NBYTES}B → ÷{PAGE_SIZE_BYTES}={NAIVE_BLOCKS} 块"
              f"（含 {PHANTOM_BLOCKS} phantom）✗",
              13.5, "#b91c1c", weight="bold"))

# ── the storage bar itself: leading | 4 blocks | trailing ──
L.append(rect(bar_x, bar_y, leading_w, BAR_H, "url(#hatch)", "#94a3b8", rx=0))
for i in range(NUM_BLOCKS):
    bx = bar_x + leading_w + i * PAGE_SIZE_BYTES * SCALE
    bw = PAGE_SIZE_BYTES * SCALE
    L.append(rect(bx, bar_y, bw, BAR_H, "#bbf7d0", "#16a34a", rx=0, sw=1.5))
    L.append(text(bx + bw / 2, bar_y + BAR_H / 2 - 4, f"block{i}", 12.5, "#166534", weight="bold"))
    L.append(text(bx + bw / 2, bar_y + BAR_H / 2 + 14, f"{PAGE_SIZE_BYTES}B", 11, "#166534"))
L.append(rect(bar_x + leading_w + data_w, bar_y, trailing_w, BAR_H, "url(#hatch)", "#94a3b8", rx=0))
L.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{BAR_H}" fill="none" stroke="#334155" stroke-width="2"/>')

L.append(text(bar_x + leading_w / 2, bar_y + BAR_H + 20,
              f"前导偏移 {LEADING_BYTES}B", 12, "#475569"))
L.append(text(bar_x + leading_w + data_w / 2, bar_y + BAR_H + 20,
              f"数据区 {DATA_BYTES}B（{NUM_BLOCKS}×{PAGE_SIZE_BYTES}B）", 12, "#166534", weight="bold"))
L.append(text(bar_x + leading_w + data_w + trailing_w / 2, bar_y + BAR_H + 20,
              f"尾部 padding {TRAILING_BYTES}B", 12, "#475569"))

# ── bottom bracket: data-area-only span (correct view) ──
bot_bk_y = bar_y + BAR_H + 46
data_x0 = bar_x + leading_w
data_x1 = data_x0 + data_w
L.append(f'<line x1="{data_x0}" y1="{bot_bk_y}" x2="{data_x1}" y2="{bot_bk_y}" stroke="#16a34a" stroke-width="2.5"/>')
for xx in (data_x0, data_x1):
    L.append(f'<line x1="{xx}" y1="{bot_bk_y - 8}" x2="{xx}" y2="{bot_bk_y}" stroke="#16a34a" stroke-width="2.5"/>')
L.append(text((data_x0 + data_x1) / 2, bot_bk_y + 26,
              f"set_(storage_offset={LEADING_BYTES}B, data_bytes={DATA_BYTES}B) "
              f"→ view[{NUM_BLOCKS},{PAGE_SIZE_BYTES}] ✓",
              13.5, "#15803d", weight="bold"))

# ── legend (stacked rows to avoid any width-estimate overlap) ──
leg_y0 = bot_bk_y + 56
leg_items = [("#bbf7d0", "数据区（精确覆盖，正确）"), ("url(#hatch)", "前后 padding（被 nbytes() 误纳）")]
lx = bar_x
for i, (color, label) in enumerate(leg_items):
    ly = leg_y0 + i * 30
    L.append(rect(lx, ly, 18, 18, color, "#64748b", rx=3, sw=1.2))
    L.append(text(lx + 26, ly + 14, label, 12.5, "#334155", anchor="start"))

L.append('</svg>')

svg_path = os.path.join(HERE, "byte-layout-single-segment.svg")
png_path = os.path.join(HERE, "byte-layout-single-segment.png")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
assert subprocess.run(["xmllint", "--noout", svg_path]).returncode == 0
subprocess.run(["rsvg-convert", "-z", "2", svg_path, "-o", png_path], check=True)
print("wrote", png_path)
