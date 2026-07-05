#!/usr/bin/env python3
"""fig34-1-straggler — state-table 变体：两张卡的槽位负载条形对比。
朴素放置把热专家 3(60)、4(55) 同放 card0，par=最热/平均=1.543。
全坐标由 SCALE * heat 计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "朴素放置：两张卡的槽位负载对比"
SUBTITLE = "card0 槽位=[3,4,1,2,1]（热度[60,55,10,10,0]），card1 槽位=[0,5,6,7,5]（热度[10,10,10,10,0]）"

CARD0_SLOTS = [(3, 60.0), (4, 55.0), (1, 10.0), (2, 10.0), (1, 0.0)]
CARD1_SLOTS = [(0, 10.0), (5, 10.0), (6, 10.0), (7, 10.0), (5, 0.0)]
MEAN = 87.5
PAR_BEFORE = 1.543
CARD0_LOAD = 135
CARD1_LOAD = 40

SCALE = 4.0          # px per heat unit
BAR_X0 = 150
BAR_H = 56
ROW_GAP = 46
TOP = 100
PAD = 30

HOT_FILL, HOT_STROKE = "#fecaca", "#b91c1c"
COLD_FILL, COLD_STROKE = "#bfdbfe", "#1e40af"

def seg_color(heat):
    return (HOT_FILL, HOT_STROKE) if heat >= 40 else (COLD_FILL, COLD_STROKE)

max_bar_w = max(CARD0_LOAD, CARD1_LOAD) * SCALE
w = BAR_X0 + max_bar_w + 260
row1_y = TOP
row2_y = TOP + BAR_H + ROW_GAP
summary_y = row2_y + BAR_H + 70
h = summary_y + 90

mean_x = BAR_X0 + MEAN * SCALE

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

def draw_row(y, label, slots, total):
    L.append(f'<text x="{PAD}" y="{y+BAR_H/2+5}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    x = BAR_X0
    for expert_id, heat in slots:
        wseg = heat * SCALE
        if wseg <= 0:
            continue
        fill, stroke = seg_color(heat)
        L.append(f'<rect x="{x:.1f}" y="{y}" width="{wseg:.1f}" height="{BAR_H}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        if wseg >= 30:
            L.append(f'<text x="{x+wseg/2:.1f}" y="{y+BAR_H/2-3}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" font-weight="bold" '
                      f'fill="{stroke}">e{expert_id}</text>')
            L.append(f'<text x="{x+wseg/2:.1f}" y="{y+BAR_H/2+13}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" fill="{stroke}">{heat:g}</text>')
        x += wseg
    L.append(f'<text x="{x+12:.1f}" y="{y+BAR_H/2+5}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">= {total}</text>')

# mean dashed line drawn BEFORE the bars so opaque bar fills knock it out
# where it would otherwise cross segment labels (no_overlap) — the line stays
# visible above row1, in the row gap, and past card1's shorter bar.
line_top = row1_y - 14
line_bot = row2_y + BAR_H + 10
L.append(f'<line x1="{mean_x:.1f}" y1="{line_top}" x2="{mean_x:.1f}" y2="{line_bot}" '
          'stroke="#059669" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{mean_x:.1f}" y="{line_top-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#059669">mean={MEAN:g}</text>')

draw_row(row1_y, "card0", CARD0_SLOTS, CARD0_LOAD)
draw_row(row2_y, "card1", CARD1_SLOTS, CARD1_LOAD)

# legend
leg_y = row2_y + BAR_H + 34
L.append(f'<rect x="{BAR_X0}" y="{leg_y}" width="16" height="16" fill="{HOT_FILL}" stroke="{HOT_STROKE}"/>')
L.append(f'<text x="{BAR_X0+22}" y="{leg_y+13}" font-family="sans-serif" font-size="11" '
          f'fill="#374151">热专家（heat≥40）</text>')
L.append(f'<rect x="{BAR_X0+180}" y="{leg_y}" width="16" height="16" fill="{COLD_FILL}" stroke="{COLD_STROKE}"/>')
L.append(f'<text x="{BAR_X0+202}" y="{leg_y+13}" font-family="sans-serif" font-size="11" '
          f'fill="#374151">冷专家（heat&lt;40）</text>')

# summary box
box_x, box_y, box_w, box_h = PAD, summary_y, w - PAD * 2, 78
L.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="8" '
          'fill="#fff7ed" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{box_x+18}" y="{box_y+28}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#92400e">'
          f'card0={CARD0_LOAD}　card1={CARD1_LOAD}　mean=(135+40)/2={MEAN:g}</text>')
L.append(f'<text x="{box_x+18}" y="{box_y+52}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#b91c1c">'
          f'par = 最热卡/平均 = 135/87.5 = {PAR_BEFORE:g}　→ 整批被最热卡拖慢约 {PAR_BEFORE:g} 倍</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-1-straggler.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
