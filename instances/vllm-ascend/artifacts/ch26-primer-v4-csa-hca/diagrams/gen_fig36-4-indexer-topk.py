#!/usr/bin/env python3
"""state-table 模板改造:lightning indexer 对 5 个候选压缩块打分,
ReLU 加权求和后 top-2 入选。列=5 个候选块,行=两头点积/ReLU 后加权分/是否入选。
数字来自 traces/indexer_topk.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Lightning Indexer:ReLU 加权内积打分,top-k 只留最高的 k 个块"
SUBTITLE = "n_heads_idx=2, k=2, num_candidates=5;score = Σ_head ReLU(head_dot) (等权求和示例)"

COLS = ["块0", "块1", "块2", "块3", "块4"]
ROW_LABELS = ["head0·key", "head1·key", "ReLU 后加权分 I(t,s)", "入选 top-2?"]
CELLS = {
    "head0·key":            ["2.0", "0.0", "-1.0", "1.0", "4.0"],
    "head1·key":            ["0.0", "3.0", "-1.0", "1.0", "1.0"],
    "ReLU 后加权分 I(t,s)":  ["2.0", "3.0", "0.0", "2.0", "5.0"],
    "入选 top-2?":          ["否", "是", "否", "否", "是"],
}
STATUS = {
    "ReLU 后加权分 I(t,s)": ["neutral", "win", "zero", "neutral", "win"],
    "入选 top-2?":          ["no", "yes", "no", "no", "yes"],
}
COLOR = {"win": ("#ecfdf5", "#047857"), "zero": ("#fee2e2", "#b91c1c"),
         "neutral": ("#f8fafc", "#64748b"), "yes": ("#ecfdf5", "#047857"),
         "no": ("#f1f5f9", "#94a3b8")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 230, 130, 46, 36, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 100 + PAD
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    hi = name in ("块1", "块4")
    fill = "#059669" if hi else "#3b82f6"
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
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
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
box_x, box_w = PAD, w - PAD * 2
L.append(f'<rect x="{box_x}" y="{callout_y}" width="{box_w}" height="66" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{box_x+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">块4(分 5.0)与块1(分 3.0)胜出;块2 两头点积皆为 -1.0,ReLU 清零后得分 0.0 被淘汰</text>')
L.append(f'<text x="{box_x+16}" y="{callout_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">top-k=2 让每个 query 实际参与核注意力的块数固定为 min(k, 候选数),不随上下文长度增长</text>')
L.append(f'<text x="{box_x+16}" y="{callout_y+60}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">选中率 2/5 —— 核注意力代价从 O(L/m) 个候选降到固定 O(k),与序列长完全解耦</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-4-indexer-topk.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
