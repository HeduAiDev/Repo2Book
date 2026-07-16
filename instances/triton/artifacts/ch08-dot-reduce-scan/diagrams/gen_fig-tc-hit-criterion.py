#!/usr/bin/env python3
"""state-table 模板：命中 Tensor Core 的三把锁——形状锁(min_dot_size)、
精度锁(input_precision)、dtype 锁(白名单+acc 相容)。5 行=5 种常见写法，
3 列锁 + 1 列结果，逐格上色(通过=绿/破锁=红)。
数字来自 explainer fig-tc-hit-criterion.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "命中 Tensor Core 的三把锁：形状 + 精度 + dtype"
SUBTITLE = "任一锁未开：形状/dtype 锁 → 追踪期 AssertionError（不生成 IR）；精度锁选 ieee → 能编译但退化到非 TC 路径"

ROWS = [
    {"写法": "fp16 (128,128)@(128,128)\n默认",  "形状锁": ("M,N,K=128≥16", "pass"),
     "精度锁": ("fp16 与精度无关", "pass"), "dtype锁": ("fp16 受支持", "pass"), "结果": ("命中 Tensor Core", "hit")},
    {"写法": "f32 (128,128)@(128,128)\n默认",   "形状锁": ("128≥16", "pass"),
     "精度锁": ("tf32(默认)", "pass"), "dtype锁": ("f32 受支持", "pass"), "结果": ("命中 Tensor Core(tf32)", "hit")},
    {"写法": "f32 同上\nallow_tf32=False",       "形状锁": ("128≥16", "pass"),
     "精度锁": ("ieee", "fail"), "dtype锁": ("f32 受支持", "pass"), "结果": ("退化：ieee 非 TC 路径", "degrade")},
    {"写法": "fp16 (8,128)@(128,128)",           "形状锁": ("M=8<16", "fail"),
     "精度锁": ("—", "na"), "dtype锁": ("—", "na"), "结果": ("追踪期报错", "err")},
    {"写法": "int8 (128,128)@(128,16)",          "形状锁": ("N=16<32(int8)", "fail"),
     "精度锁": ("—", "na"), "dtype锁": ("int8 受支持", "pass"), "结果": ("追踪期报错", "err")},
]
COLS = ["形状锁\n(min_dot_size)", "精度锁\n(input_precision)", "dtype 锁\n(白名单/acc)", "结果"]
KEYS = ["形状锁", "精度锁", "dtype锁", "结果"]

COLOR = {
    "pass": ("#dcfce7", "#15803d"), "fail": ("#fee2e2", "#b91c1c"),
    "na": ("#f1f5f9", "#64748b"), "hit": ("#dcfce7", "#15803d"),
    "degrade": ("#fef9c3", "#a16207"), "err": ("#fee2e2", "#b91c1c"),
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 230, 66, 52, 118, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 46
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

sub_lines = [SUBTITLE[:52], SUBTITLE[52:]]
for i, line in enumerate(sub_lines):
    L.append(f'<text x="{PAD}" y="{PAD+20+i*16}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')

# 列头 "写法" label (row label column header)
L.append(f'<text x="{PAD+LABEL_W-16}" y="{TOP+HEADER_H/2+4}" text-anchor="end" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#374151">写法</text>')
for j, name in enumerate(COLS):
    x = col_x[j]
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    n = len(lines)
    y0 = TOP + (HEADER_H - 6) / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    label_lines = row["写法"].split("\n")
    n = len(label_lines)
    y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(label_lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{y0+k*16}" text-anchor="end" '
                  f'font-family="monospace" font-size="11.5" font-weight="bold" '
                  f'fill="#374151">{esc(line)}</text>')
    for j, key in enumerate(KEYS):
        cx = col_x[j]
        text, status = row[key]
        fill, stroke = COLOR[status]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{stroke}" '
                  f'font-weight="bold">{esc(text)}</text>')

foot_y0 = h - PAD - 30
FOOT = [
    "三锁自查：形状每维 ≥ min_dot_size(int8 的 N≥32)；f32 用默认或显式 tf32；操作数 dtype 属白名单且同型、acc 相容。",
    "绿=通过/命中，黄=能编译但退化到非 TC 路径，红=破锁(报错或退化)，灰=该锁在此行不适用(已在更早的锁被拦下)。",
]
for i, line in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-tc-hit-criterion.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
