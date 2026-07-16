#!/usr/bin/env python3
"""figure m7-hook-guard: pre_hook/post_hook 保证两个 config 从同一起点起跑——
高亮两个 config 的 pre_hook 行完全相同(acc=0, x=5)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "pre_hook / post_hook：跨 config 复位输入的一致起点"
SUBTITLE = "python/triton/runtime/autotuner.py:L65-L85 —— reset_to_zero 清零、restore_value 拍照存档再还原"

GROUPS = ["configA", "configB"]
PHASES = ["pre_hook", "kernel 后", "post_hook"]
COLS = ["阶段", "acc", "x", "restore_copies['x']"]
DATA = {
    "configA": [
        ["pre_hook", "0", "5", "5"],
        ["kernel 后", "10", "999", "5"],
        ["post_hook", "10", "5", "已清空"],
    ],
    "configB": [
        ["pre_hook", "0", "5", "5"],
        ["kernel 后", "10", "999", "5"],
        ["post_hook", "10", "5", "已清空"],
    ],
}
HIGHLIGHT_PHASE = "pre_hook"

PAD = 40
GROUP_COL_W = 110
col_widths = [max(cjk_w(c, 13) for c in [name] + [DATA[g][i][j] for g in GROUPS for i in range(3)]) + 34
              for j, name in enumerate(COLS)]
HEADER_H = 40
ROW_H = 46
TOP = 100

note_lines_probe = [
    "黄底高亮两个 config 的 pre_hook 行——acc=0、x=5 完全相同：configB 未被 configA 的累加",
    "(acc=10)/覆盖(x=999)污染。搜索结束后还有一次 pre_hook(reset_only=True) 只清零、不再存副本。",
]
table_w = GROUP_COL_W + sum(col_widths)
subtitle_w = cjk_w(SUBTITLE, 12) + 20
note_w = max(cjk_w(s, 12.5) for s in note_lines_probe) + 32
w = PAD * 2 + max(table_w, subtitle_w, note_w)
h = TOP + HEADER_H + ROW_H * 6 + 110

group_x = PAD
col_x = []
cx = PAD + GROUP_COL_W
for cw in col_widths:
    col_x.append(cx)
    cx += cw

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# header row
L.append(f'<rect x="{group_x}" y="{TOP}" width="{GROUP_COL_W}" height="{HEADER_H}" rx="4" '
          'fill="#334155" stroke="#1e293b" stroke-width="1.5"/>')
L.append(f'<text x="{group_x+GROUP_COL_W/2:.0f}" y="{TOP+HEADER_H/2+5:.0f}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12.5" fill="white" font-weight="bold">bench 轮</text>')
for j, name in enumerate(COLS):
    x = col_x[j]
    cw = col_widths[j]
    L.append(f'<rect x="{x:.0f}" y="{TOP}" width="{cw:.0f}" height="{HEADER_H}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+cw/2:.0f}" y="{TOP+HEADER_H/2+5:.0f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for gi, g in enumerate(GROUPS):
    gy = TOP + HEADER_H + gi * 3 * ROW_H
    group_fill = "#eff6ff" if gi == 0 else "#fdf4ff"
    group_stroke = "#93c5fd" if gi == 0 else "#e9d5ff"
    L.append(f'<rect x="{group_x}" y="{gy:.0f}" width="{GROUP_COL_W}" height="{3*ROW_H}" '
              f'fill="{group_fill}" stroke="{group_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{group_x+GROUP_COL_W/2:.0f}" y="{gy+3*ROW_H/2+5:.0f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#1e3a5f">{esc(g)}</text>')
    for pi, row in enumerate(DATA[g]):
        ry = gy + pi * ROW_H
        is_hl = (row[0] == HIGHLIGHT_PHASE)
        for j, cell in enumerate(row):
            x = col_x[j]
            cw = col_widths[j]
            if is_hl:
                L.append(f'<rect x="{x:.0f}" y="{ry+3:.0f}" width="{cw:.0f}" height="{ROW_H-6}" '
                          'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>')
            text_fill = "#854d0e" if is_hl else "#334155"
            weight = 'font-weight="bold" ' if is_hl else ''
            L.append(f'<text x="{x+cw/2:.0f}" y="{ry+ROW_H/2+5:.0f}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                      f'{weight}>{esc(cell)}</text>')
        L.append(f'<line x1="{group_x}" y1="{ry+ROW_H:.0f}" x2="{w-PAD:.0f}" y2="{ry+ROW_H:.0f}" '
                  'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<line x1="{group_x}" y1="{gy:.0f}" x2="{w-PAD:.0f}" y2="{gy:.0f}" '
              'stroke="#94a3b8" stroke-width="1.5"/>')

note_lines = [
    "黄底高亮两个 config 的 pre_hook 行——acc=0、x=5 完全相同：configB 未被 configA 的累加",
    "(acc=10)/覆盖(x=999)污染。搜索结束后还有一次 pre_hook(reset_only=True) 只清零、不再存副本。",
]
note_top = TOP + HEADER_H + ROW_H * 6 + 30
note_h = 24 * len(note_lines) + 24
L.append(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
          'fill="#f0fdf4" stroke="#86efac"/>')
for i, line in enumerate(note_lines):
    L.append(f'<text x="{PAD+16}" y="{note_top+26+i*24:.0f}" font-family="sans-serif" '
              f'font-size="12.5" fill="#166534">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("m7-hook-guard.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
