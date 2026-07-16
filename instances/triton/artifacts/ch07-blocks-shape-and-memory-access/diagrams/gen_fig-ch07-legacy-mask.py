#!/usr/bin/env python3
"""state-table 模板:legacy 逐元素路径的 mask 裁边界。列=8 个 lane,行=
offs/mask/访存动作/载入值,越界的两个 lane(6,7)整列高亮为『关断』语义色。
数字全部来自 dossier m6-legacy-ptr-mask(pid=0, BLOCK=8, n=6, other=0.0)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "legacy 路径：mask 逐 lane 裁边界"
SUBTITLE = "pid=0, BLOCK=8, n=6(尾块)；mask = offs < 6；越界 lane 不发内存请求，填 other=0.0"

N_LANE = 8
N = 6
COLS = [str(i) for i in range(N_LANE)]
ROW_LABELS = ["offs = pid*8+i", "mask = offs<6", "访存动作", "载入值"]

def row(i):
    offs = i
    m = offs < N
    action = "真访存 x[i]" if m else "不发请求"
    val = f"x[{i}]" if m else "other=0.0"
    return offs, m, action, val

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 110, 46, 34, 96, 36
w = PAD * 2 + LABEL_W + COL_W * N_LANE
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 150

col_x = [PAD + LABEL_W + i * COL_W for i in range(N_LANE)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+12}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

COLOR_ON = ("#dcfce7", "#15803d")
COLOR_OFF = ("#fee2e2", "#b91c1c")

for j in range(N_LANE):
    x = col_x[j]
    on = j < N
    fill, stroke = COLOR_ON if on else COLOR_OFF
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{stroke}">lane {j}</text>')

for i, rowname in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-14}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="#374151">{esc(rowname)}</text>')
    for j in range(N_LANE):
        cx = col_x[j]
        offs, m, action, val = row(j)
        on = j < N
        fill, stroke = COLOR_ON if on else COLOR_OFF
        text_fill = stroke
        cell = [str(offs), "true" if m else "false", action, val][i]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" fill-opacity="0.55" stroke="{stroke}" stroke-width="1.2"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                  f'font-weight="bold">{esc(cell)}</text>')

legend_y = row_y[-1] + ROW_H + 30
L.append(f'<rect x="{PAD}" y="{legend_y}" width="16" height="16" rx="3" '
          f'fill="{COLOR_ON[0]}" stroke="{COLOR_ON[1]}"/>')
L.append(f'<text x="{PAD+24}" y="{legend_y+13}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("mask=true：真访存（6 个 lane，利用率 6/8=75%）")}</text>')
L.append(f'<rect x="{PAD+380}" y="{legend_y}" width="16" height="16" rx="3" '
          f'fill="{COLOR_OFF[0]}" stroke="{COLOR_OFF[1]}"/>')
L.append(f'<text x="{PAD+404}" y="{legend_y+13}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("mask=false：不解引用内存，取 other=0.0 兜底")}</text>')

foot_y0 = legend_y + 40
FOOT = [
    "结论:legacy 指针本身是一个 block 形状的地址张量;mask = arith.cmpi slt 生成的越界判据。",
    "mask 为真的 6 个 lane 真取数,越界的 2 个 lane 填 other 且绝不碰内存——灵活但编译器只看到",
    "一堆独立地址、不知道边界在哪。",
]
for i, line in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*19}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-legacy-mask.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
