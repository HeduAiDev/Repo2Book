#!/usr/bin/env python3
"""before-after 模板：两种物理布局（单段 / 多段 K|V 堆叠）都被
_build_block_views 规约成同一个 [num_blocks, block_bytes] int8 视图。
数字来自 explainer/traces/build_block_views.json#A_single_segment / #B_multi_segment。"""
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


def rect(x, y, w, h, fill, stroke, rx=4, sw=2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y1, x2, y2, color="#64748b", sw=2.2, marker="a"):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"/>')


# ── numbers (sourced from traces/build_block_views.json) ──
NUM_BLOCKS = 4          # A.num_blocks / B (connector 只用 4 块/段)
BLOCK_BYTES = 8         # A.page_size_bytes / B.seg_page_size_bytes
N_SEGMENTS = 2          # B.n_segments
PHYS_BLOCKS_PER_SEG = 6  # B.phys_blocks_per_seg
SEG_STRIDE = 48         # B.seg_stride_bytes
SEG_DATA = 32           # B.seg_data_bytes
TOTAL_BYTES = 80        # B.total_bytes_shape_stride
SEG1_START = 48         # B.seg1_start_byte

W = 1220
H = 430
PAD = 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(text(W / 2, 30, "两种物理布局，规约成同一个 [num_blocks, block_bytes] 视图",
              17, "#0f172a", weight="bold"))

col_hdr_y = 62
L.append(text(230, col_hdr_y, "之前：物理布局各异", 14, "#475569", weight="bold"))
L.append(text(870, col_hdr_y, "之后：统一视图", 14, "#475569", weight="bold"))

BLK_W = 46
BLK_H = 44

# ── Row A: single segment ──────────────────────────────────────────
rowA_y = 100
L.append(text(PAD, rowA_y - 10, "单段 · shape=(4,4)，blocks-outermost", 13, "#166534", weight="bold", anchor="start"))
bx0 = PAD
for i in range(NUM_BLOCKS):
    bx = bx0 + i * BLK_W
    L.append(rect(bx, rowA_y, BLK_W, BLK_H, "#bbf7d0", "#16a34a", rx=3, sw=1.5))
    L.append(text(bx + BLK_W / 2, rowA_y + BLK_H / 2 + 5, f"b{i}", 13, "#166534", weight="bold"))
L.append(text(bx0 + NUM_BLOCKS * BLK_W / 2, rowA_y + BLK_H + 20,
              f"一条连续 storage，{NUM_BLOCKS} 块顺排在最外维", 12, "#475569"))

arrA_y = rowA_y + BLK_H / 2
arrA_x0 = bx0 + NUM_BLOCKS * BLK_W + 20
arrA_x1 = 700
L.append(arrow(arrA_x0, arrA_y, arrA_x1, arrA_y))
L.append(text((arrA_x0 + arrA_x1) / 2, arrA_y - 12,
              f"set_(offset, {SEG_DATA}B) + view", 12, "#334155"))

# unified view A (top-right)
uvA_x = 730
uvA_y = rowA_y
for i in range(NUM_BLOCKS):
    bx = uvA_x + i * BLK_W
    L.append(rect(bx, uvA_y, BLK_W, BLK_H, "#bbf7d0", "#16a34a", rx=3, sw=1.5))
    L.append(text(bx + BLK_W / 2, uvA_y + BLK_H / 2 + 5, f"b{i}", 13, "#166534", weight="bold"))
L.append(text(uvA_x + NUM_BLOCKS * BLK_W / 2, uvA_y + BLK_H + 20,
              f"视图 [{NUM_BLOCKS}, {BLOCK_BYTES}] int8", 12.5, "#166534", weight="bold"))

# ── Row B: multi segment ───────────────────────────────────────────
rowB_y = 260
L.append(text(PAD, rowB_y - 10,
              "多段 · shape=(2,6,4)，K|V 堆叠（connector 只用前 4/6 块每段）",
              13, "#7c3aed", weight="bold", anchor="start"))

seg_gap = 26
for seg in range(N_SEGMENTS):
    sy = rowB_y + seg * (BLK_H + 16)
    bx0s = PAD
    for i in range(NUM_BLOCKS):
        bx = bx0s + i * BLK_W
        L.append(rect(bx, sy, BLK_W, BLK_H, "#ddd6fe", "#7c3aed", rx=3, sw=1.5))
        L.append(text(bx + BLK_W / 2, sy + BLK_H / 2 + 5, f"b{i}", 12.5, "#5b21b6", weight="bold"))
    # unused physical blocks (PHYS_BLOCKS_PER_SEG - NUM_BLOCKS) per segment
    for i in range(NUM_BLOCKS, PHYS_BLOCKS_PER_SEG):
        bx = bx0s + i * BLK_W
        L.append(rect(bx, sy, BLK_W, BLK_H, "#f1f5f9", "#94a3b8", rx=3, sw=1.2))
        L.append(text(bx + BLK_W / 2, sy + BLK_H / 2 + 5, "未用", 10.5, "#64748b"))
    L.append(text(bx0s - 8, sy + BLK_H / 2 + 5, f"seg{seg}", 12, "#5b21b6", weight="bold", anchor="end"))

L.append(text(bx0s, rowB_y + N_SEGMENTS * (BLK_H + 16) + 8,
              f"每段物理 {PHYS_BLOCKS_PER_SEG} 块（数据 {SEG_DATA}B），seg_stride={SEG_STRIDE}B "
              f"跨过每段末 {PHYS_BLOCKS_PER_SEG - NUM_BLOCKS} 个未用块", 12, "#475569", anchor="start"))

arrB_y = rowB_y + (BLK_H + 16) * N_SEGMENTS / 2 - 8
arrB_x0 = bx0s + PHYS_BLOCKS_PER_SEG * BLK_W + 20
arrB_x1 = 700
L.append(arrow(arrB_x0, arrB_y, arrB_x1, arrB_y, color="#7c3aed"))
L.append(text((arrB_x0 + arrB_x1) / 2, arrB_y - 12,
              f"set_(offset, {TOTAL_BYTES}B) + 按 seg_stride={SEG_STRIDE}B 切 "
              f"n_segments={N_SEGMENTS} 段", 11.5, "#334155"))

# unified view B: seg0 / seg1 (bottom-right)
uvB_x = 730
for seg in range(N_SEGMENTS):
    uy = rowB_y + seg * (BLK_H + 16)
    for i in range(NUM_BLOCKS):
        bx = uvB_x + i * BLK_W
        L.append(rect(bx, uy, BLK_W, BLK_H, "#ddd6fe", "#7c3aed", rx=3, sw=1.5))
        L.append(text(bx + BLK_W / 2, uy + BLK_H / 2 + 5, f"b{i}", 12.5, "#5b21b6", weight="bold"))
    label = f"seg{seg} [{NUM_BLOCKS},{BLOCK_BYTES}]" if seg == 0 else \
            f"seg{seg} [{NUM_BLOCKS},{BLOCK_BYTES}]（第 {SEG1_START}B 起）"
    L.append(text(uvB_x + NUM_BLOCKS * BLK_W + 12, uy + BLK_H / 2 + 5, label, 12, "#5b21b6", anchor="start"))

L.append('</svg>')

svg_path = os.path.join(HERE, "layout-reduction-single-vs-multi.svg")
png_path = os.path.join(HERE, "layout-reduction-single-vs-multi.png")
with open(svg_path, "w", encoding="utf-8") as f:
    f.write("\n".join(L))
assert subprocess.run(["xmllint", "--noout", svg_path]).returncode == 0
subprocess.run(["rsvg-convert", "-z", "2", svg_path, "-o", png_path], check=True)
print("wrote", png_path)
