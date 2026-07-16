#!/usr/bin/env python3
"""state-table 模板：argmax 的带索引树归约——(值,索引) 成对 combine，
输入 [3,1,4,1] 经 2 层归约得 value=4/index=2；末行附平局样例(取最左索引)。
数字全部来自 explainer fig-argmax-tree.numbers（交叉验证 traces/argmax_tree_reduce.out）。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "argmax 树归约：(值,索引) 成对 combine"
SUBTITLE = "输入 [3,1,4,1]，n=4，log2(4)=2 层归约；combine 用 where(v1>v2) 让索引跟随值一起被选，平局取最左"

COLS = ["轮次", "combine 输入 (v1,i1) × (v2,i2)", "gt 判定", "where 选出 (value, index)"]
ROWS = [
    ["1", "(3,0) × (1,1)", "3>1 = True", "(3, 0)"],
    ["2", "(4,2) × (1,3)", "4>1 = True", "(4, 2)"],
    ["3", "(3,0) × (4,2)", "3>4 = False", "(4, 2)  → 最终 value=4, index=2"],
    ["平局例", "(4,0) × (4,1)  输入[4,4]", "tie(4==4 and 0<1)=True → gt=True", "(4, 0)  → 左侧索引 0 胜"],
]
ROW_KIND = ["normal", "normal", "result", "tie"]
COLOR = {
    "normal": ("#f8fafc", "#334155"),
    "result": ("#dcfce7", "#15803d"),
    "tie": ("#fef9c3", "#a16207"),
}
COL_W = [90, 320, 300, 340]
ROW_H = 56
HEADER_H = 42
PAD, TOP = 30, 108
w = PAD * 2 + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 54
col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)
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

for j, name in enumerate(COLS):
    x, cw = col_x[j], COL_W[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(cw-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    kind = ROW_KIND[i]
    fill, stroke = COLOR[kind]
    for j, cell in enumerate(row):
        x, cw = col_x[j], COL_W[j]
        L.append(f'<rect x="{x}" y="{ry+4}" width="{cw-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{"2" if kind!="normal" else "1"}"/>')
        fw = 'font-weight="bold" ' if kind != "normal" else ''
        ff = "monospace" if j in (1, 2, 3) else "sans-serif"
        fs = "11.5" if j in (1, 2, 3) else "13"
        L.append(f'<text x="{x+(cw-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="{ff}" font-size="{fs}" {fw}'
                  f'fill="{stroke if kind!="normal" else "#374151"}">{esc(cell)}</text>')

legend_y = h - PAD - 34
L.append(f'<rect x="{PAD}" y="{legend_y-14}" width="16" height="16" rx="3" fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y-2}" font-family="sans-serif" font-size="12" fill="#374151">最终结果</text>')
L.append(f'<rect x="{PAD+120}" y="{legend_y-14}" width="16" height="16" rx="3" fill="#fef9c3" stroke="#a16207" stroke-width="2"/>')
L.append(f'<text x="{PAD+142}" y="{legend_y-2}" font-family="sans-serif" font-size="12" fill="#374151">平局样例(取最左索引)</text>')

foot_y = h - PAD - 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">argmax = reduce 的免费高级用法：_reduce_with_indices 把 (值,索引) 成对喂给同一个 create_reduce，值那路做 max，索引那路跟随值被选。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-argmax-tree.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
