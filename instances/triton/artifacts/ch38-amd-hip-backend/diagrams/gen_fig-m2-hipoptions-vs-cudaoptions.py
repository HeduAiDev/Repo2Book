#!/usr/bin/env python3
"""fig-m2-hipoptions-vs-cudaoptions：HIPOptions 与 CUDAOptions 共有字段一致，
差异全在后端专属项。三组：共有字段 / AMD 专属 / NVIDIA 专属。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "HIPOptions vs CUDAOptions：共有字段对齐，差异全在专属项"
SUBTITLE = "同一套编译选项 dataclass，parse_options 契约的两种填法"

# (field, HIPOptions值, CUDAOptions值)
SHARED = [
    ("num_warps 默认", "4", "4"),
    ("num_stages 默认", "2", "3"),
]
AMD_ONLY = [
    ("waves_per_eu 默认", "1", "—"),
    ("matrix_instr_nonkdim 默认", "0", "—"),
    ("kpack 默认", "1", "—"),
    ("warp_size（按 gfx 档）", "32（RDNA gfx10/11/12）\n64（CDNA gfx9 等）", "—"),
]
NVIDIA_ONLY = [
    ("warp（硬编码字面量，非 dataclass 字段）", "—", "32"),
    ("maxnreg / ptx_version", "—", "NVIDIA 专属字段"),
]

NAME_W, COL_W, ROW_H, HEADER_H, GROUP_GAP = 320, 260, 34, 32, 26
PAD, TOP = 40, 128

groups = [
    ("共有字段", "两边一致/同字段不同默认值", SHARED, ("#e0f2fe", "#0369a1")),
    ("AMD 专属", "调 mfma / wavefront 32-64", AMD_ONLY, ("#fee2e2", "#b91c1c")),
    ("NVIDIA 专属", "warp 恒 32，无 warp_size 字段", NVIDIA_ONLY, ("#dcfce7", "#15803d")),
]

table_w = NAME_W + COL_W * 2
w = PAD * 2 + table_w


def group_height(rows):
    # rows with multi-line cell (contains \n) need taller row
    extra = 0
    for _, hv, nv in rows:
        lines = max(hv.count("\n"), nv.count("\n")) + 1
        if lines > 1:
            extra += (lines - 1) * 14
    return HEADER_H + len(rows) * ROW_H + extra


tops = []
y = TOP
for label, note, rows, colors in groups:
    tops.append(y)
    y += group_height(rows) + GROUP_GAP
h = y + 40 + PAD

col_x = [PAD + NAME_W, PAD + NAME_W + COL_W]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for gi, (label, note, rows, colors) in enumerate(groups):
    top = tops[gi]
    fill, stroke = colors
    L.append(f'<rect x="{PAD}" y="{top}" width="{NAME_W}" height="{HEADER_H-4}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD+12}" y="{top+(HEADER_H-4)/2+4}" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{stroke}">{esc(label)}（{esc(note)}）</text>')
    for j, colname in enumerate(["HIPOptions（AMD）", "CUDAOptions（NVIDIA）"]):
        x = col_x[j]
        L.append(f'<rect x="{x}" y="{top}" width="{COL_W-6}" height="{HEADER_H-4}" rx="4" '
                  'fill="#334155" stroke="#1e293b" stroke-width="1"/>')
        L.append(f'<text x="{x+(COL_W-6)/2}" y="{top+(HEADER_H-4)/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="white" '
                  f'font-weight="bold">{esc(colname)}</text>')
    ry = top + HEADER_H
    for name, hv, nv in rows:
        lines_h = hv.split("\n")
        lines_n = nv.split("\n")
        nlines = max(len(lines_h), len(lines_n))
        row_h = ROW_H + (nlines - 1) * 14
        row_fill = "#f8fafc"
        L.append(f'<rect x="{PAD}" y="{ry}" width="{table_w}" height="{row_h}" '
                  f'fill="{row_fill}" stroke="#e2e8f0" stroke-width="1"/>')
        L.append(f'<text x="{PAD+12}" y="{ry+row_h/2+4}" font-family="monospace" '
                  f'font-size="12" fill="#0f172a">{esc(name)}</text>')
        for j, lines in enumerate((lines_h, lines_n)):
            x = col_x[j]
            y0 = ry + row_h / 2 - (len(lines) - 1) * 8 + 4
            for k, line in enumerate(lines):
                L.append(f'<text x="{x+(COL_W-6)/2}" y="{y0+k*15}" text-anchor="middle" '
                          f'font-family="sans-serif" font-size="11.5" fill="#1e293b">{esc(line)}</text>')
        ry += row_h

foot_y = h - PAD + 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">"—"= 该字段在此后端不存在（third_party/amd/backend/compiler.py:L28-L72，'
          f'third_party/nvidia/backend/compiler.py:L93-L218）。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m2-hipoptions-vs-cudaoptions.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
