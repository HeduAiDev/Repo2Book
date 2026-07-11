#!/usr/bin/env python3
"""fig-m6-loop-statetable：state-table 模板。列=块内位置 i=0,1,2（玩具 V=4/N=3/r=2），
行=prev/嵌入/偏置/基础logits/合成logits/base argmax/draft，高亮末行(draft)标注"翻转"。
数据取自 explainer.json worked_example（traces/m6_sequential.out 复现）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "序列 Markov 头：玩具 V=4/N=3 逐位翻转 argmax"
SUBTITLE = "锚点 A 出发；base_logits 只算 1 次，每步用 prev token 生成偏置逐位修正（traces/m6_sequential.out 复现）"
COLS = ["i=0（prev=A 锚点）", "i=1（prev=B）", "i=2（prev=C）"]
ROW_LABELS = [
    "prev（前驱 token）",
    "e=W1[prev]",
    "bias [A,B,C,D]",
    "U_i [A,B,C,D]",
    "logits_i=U_i+bias",
    "base argmax（仅 U_i）",
    "draft_i=argmax(logits_i)",
]
CELLS = {
    "prev（前驱 token）":        ["A（锚点）", "B", "C"],
    "e=W1[prev]":               ["[1.0, 0.0]", "[0.0, 1.0]", "[1.0, 1.0]"],
    "bias [A,B,C,D]":           ["[0.0, 2.0, 0.0, 1.0]", "[0.0, 0.0, 2.0, 1.0]", "[0.0, 2.0, 2.0, 2.0]"],
    "U_i [A,B,C,D]":            ["[1.0, 0.5, 0.0, 0.0]", "[0.0, 1.5, 0.5, 0.0]", "[1.5, 0.0, 0.0, 0.5]"],
    "logits_i=U_i+bias":        ["[1.0, 2.5, 0.0, 1.0]", "[0.0, 1.5, 2.5, 1.0]", "[1.5, 2.0, 2.0, 2.5]"],
    "base argmax（仅 U_i）":     ["A", "B", "A"],
    "draft_i=argmax(logits_i)": ["B（翻转）", "C（翻转）", "D（翻转）"],
}
HIGHLIGHT_ROW = "draft_i=argmax(logits_i)"
STATUS = {HIGHLIGHT_ROW: ["changed", "changed", "changed"]}
COLOR = {"changed": ("#fee2e2", "#b91c1c"), "stable": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 250, 46, 40, 108, 32
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 40
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

foot_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#374151">draft 序列 = B, C, D（锚点 A 出发）——三步 base argmax 本是 A/B/A，'
          f'Markov 偏置逐位翻转成 B/C/D。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">红=本步 argmax 相对 base_logits 发生翻转（changed）；'
          f'玩具值 V=4/N=3/r=2 便于心算，真实 r=256（file 摘要）</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m6-loop-statetable.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
