#!/usr/bin/env python3
"""figure m5-key-cache: autotune 缓存键 = key 参数值 + 各张量实参 dtype，
4 次调用的 key/判定/cache 项数演化表。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "Autotuner 缓存键：key 参数 + 各实参 dtype"
SUBTITLE = "python/triton/runtime/autotuner.py:L170-L189 —— 同(尺寸,dtype)命中复用，换一样就另建键重搜"

COLS = ["调用", "key 元组", "判定", "cache 项数"]
ROWS = [
    ["N=1024 fp16（首次）", "(1024, float16, float16)", "MISS → 搜索并缓存", "1"],
    ["N=1024 fp16（重复）", "(1024, float16, float16)", "HIT → 复用 best_config", "1"],
    ["N=1024 fp32", "(1024, float32, float32)", "MISS → 搜索（fp32 不共享结果）", "2"],
    ["N=2048 fp16", "(2048, float16, float16)", "MISS → 搜索（尺寸变）", "3"],
]
JUDGE_COL = 2
STATUS = ["MISS", "HIT", "MISS", "MISS"]
COLOR = {"MISS": ("#fee2e2", "#b91c1c"), "HIT": ("#dcfce7", "#15803d")}

PAD = 40
HEADER_H = 40
ROW_H = 56
col_gap = 14

col_widths = []
for j, name in enumerate(COLS):
    max_content = max([cjk_w(name, 13)] + [cjk_w(ROWS[i][j], 12.5) for i in range(len(ROWS))])
    col_widths.append(max_content + 36)

w = PAD * 2 + sum(col_widths) + col_gap * (len(COLS) - 1)
TOP = 96
h = TOP + HEADER_H + ROW_H * len(ROWS) + 90

col_x = []
cx = PAD
for cw in col_widths:
    col_x.append(cx)
    cx += cw + col_gap

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    cw = col_widths[j]
    L.append(f'<rect x="{x:.0f}" y="{TOP}" width="{cw:.0f}" height="{HEADER_H}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+cw/2:.0f}" y="{TOP+HEADER_H/2+5:.0f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    status = STATUS[i]
    fill, stroke = COLOR[status]
    for j, cell in enumerate(row):
        x = col_x[j]
        cw = col_widths[j]
        is_judge = (j == JUDGE_COL)
        if is_judge:
            L.append(f'<rect x="{x:.0f}" y="{ry+4:.0f}" width="{cw:.0f}" height="{ROW_H-8}" rx="6" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        elif i % 2 == 1:
            L.append(f'<rect x="{x:.0f}" y="{ry+4:.0f}" width="{cw:.0f}" height="{ROW_H-8}" '
                      'fill="#f8fafc"/>')
        text_fill = stroke if is_judge else "#334155"
        weight = 'font-weight="bold" ' if is_judge else ''
        L.append(f'<text x="{x+cw/2:.0f}" y="{ry+ROW_H/2+5:.0f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                  f'{weight}>{esc(cell)}</text>')
    # row separator
    L.append(f'<line x1="{PAD}" y1="{ry+ROW_H:.0f}" x2="{w-PAD:.0f}" y2="{ry+ROW_H:.0f}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = TOP + HEADER_H + ROW_H * len(ROWS) + 30
note = "红=MISS(新建键并搜索)，绿=HIT(命中既有键直接复用)；指针地址等不入键，故同(尺寸,dtype)重复调用必稳定 HIT。"
L.append(f'<text x="{PAD}" y="{foot_y:.0f}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(note)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("m5-key-cache.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
