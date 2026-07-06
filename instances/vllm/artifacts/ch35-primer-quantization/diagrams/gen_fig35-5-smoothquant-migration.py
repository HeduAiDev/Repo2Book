#!/usr/bin/env python3
"""fig35-5-smoothquant-migration: SmoothQuant 把离群通道的量化难度从激活搬到权重：
激活 absmax 骤降、权重 absmax 上升，二者在 s_2 处相等，整层量化误差减半。
before-after：channel 2（60x 离群通道）的激活/权重 absmax 迁移前后对比。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "SmoothQuant 迁移：outlier channel 2 的量化难度从激活搬到权重"
SUBTITLE = "s_2 = max|X_2|^0.5 / max|W_2|^0.5 = 16.5362（alpha=0.5，60x 激活离群通道）"

PAD = 40
TOP = 110
ROW_H = 130
COL_LABEL_W = 90
BOX_W = 260
GAP_X = 140
ANNOT_W = 190  # room for the "两者相等" side annotation after the last column
w = PAD + COL_LABEL_W + BOX_W + GAP_X + BOX_W + ANNOT_W + PAD
h = TOP + ROW_H + 90 + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

before_x = PAD + COL_LABEL_W
after_x = before_x + BOX_W + GAP_X

L.append(f'<text x="{before_x+BOX_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" fill="#0f172a">迁移前</text>')
L.append(f'<text x="{after_x+BOX_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" fill="#0f172a">迁移后</text>')

ROWS = [
    {"label": "激活 |X_2|", "before": "121.1992", "after": "7.3293", "op": "÷ s_2 = 16.5362"},
    {"label": "权重 |W_2|", "before": "0.4432", "after": "7.3293", "op": "× s_2 = 16.5362"},
]

for i, row in enumerate(ROWS):
    ry = TOP + i * ROW_H
    L.append(f'<text x="{PAD+COL_LABEL_W-14}" y="{ry+52}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" fill="#334155">{esc(row["label"])}</text>')
    # before box
    L.append(f'<rect x="{before_x}" y="{ry}" width="{BOX_W}" height="90" rx="8" '
              f'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{before_x+BOX_W/2}" y="{ry+52}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="20" fill="#b45309">{esc(row["before"])}</text>')
    # arrow
    amy = ry + 45
    L.append(f'<line x1="{before_x+BOX_W+8}" y1="{amy}" x2="{after_x-8}" y2="{amy}" '
              'stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<text x="{(before_x+BOX_W+after_x)/2}" y="{amy-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#1d4ed8">{esc(row["op"])}</text>')
    # after box
    L.append(f'<rect x="{after_x}" y="{ry}" width="{BOX_W}" height="90" rx="8" '
              f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2" stroke-dasharray="0"/>')
    L.append(f'<text x="{after_x+BOX_W/2}" y="{ry+52}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="20" fill="#1d4ed8">{esc(row["after"])}</text>')

# equality annotation between the two "after" boxes
L.append(f'<line x1="{after_x+BOX_W/2}" y1="{TOP+90}" x2="{after_x+BOX_W/2}" y2="{TOP+ROW_H}" '
          'stroke="#047857" stroke-width="2" stroke-dasharray="4,3"/>')
L.append(f'<text x="{after_x+BOX_W+16}" y="{TOP+ROW_H/2+40}" font-family="sans-serif" '
          f'font-size="12" fill="#047857">两者相等</text>')
L.append(f'<text x="{after_x+BOX_W+16}" y="{TOP+ROW_H/2+58}" font-family="sans-serif" '
          f'font-size="12" fill="#047857">(alpha=0.5 均分)</text>')

foot_y = TOP + ROW_H + 90 + 44
foot_lines = [
    "恒等检验：变换后 X_hat·W_hat 与原始 X·W 的最大差 = 0.0——迁移只改可量化性，不改浮点结果。",
    "整层 per-tensor W8A8 量化误差 1.1515 -> 0.5273（alpha=0.75 为此层最优），降 54.21%。",
    "s 在离线阶段折进前一层权重，运行期激活已被抹平，vllm 只消费迁移后的定点数据。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-5-smoothquant-migration.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
