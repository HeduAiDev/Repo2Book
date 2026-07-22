#!/usr/bin/env python3
"""fig-m8-greedy-earlystop：state-table 模板。列=录取到第几位 k=1..4，
行=c_k/累计存活/期望接受/SPS/吞吐Θ/决策，决策行按语义色（录取绿/达峰金/早停红）。
数据取自 explainer.json worked_example（已用 host 纯 Python 复现）。论文 Algorithm 1 机制，
本 PR #46995 快照无对应调度器代码——图注需诚实标注。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "硬件感知动态调度 Algorithm 1：贪心录取，吞吐首次下降即早停（论文机制）"
SUBTITLE = "玩具 c=[.9,.8,.5,.4]（已用 host 纯 Python 复现）——本 PR #46995 快照无此调度器代码"
COLS = ["k=1", "k=2", "k=3", "k=4"]
ROW_LABELS = [
    "c_k（置信度）",
    "P_k=∏c_i（累计存活）",
    "τ(k)=ΣP_i（期望接受数）",
    "SPS(k)（每秒步数）",
    "Θ=τ·SPS",
    "决策",
]
CELLS = {
    "c_k（置信度）":              ["0.9", "0.8", "0.5", "0.4"],
    "P_k=∏c_i（累计存活）":       ["0.9", "0.72", "0.36", "0.144"],
    "τ(k)=ΣP_i（期望接受数）":    ["0.9", "1.62", "1.98", "2.124"],
    "SPS(k)（每秒步数）":         ["100.0", "90.0", "75.0", "55.0"],
    "Θ=τ·SPS":                   ["90", "145.8", "148.5", "116.8"],
    "决策":                       ["录取（Θ↑）", "录取（Θ↑）", "录取（Θ↑，达峰）", "早停（Θ↓）→ 录取 3"],
}
STATUS = {"决策": ["accept", "accept", "peak", "stop"]}
COLOR = {"accept": ("#dcfce7", "#15803d"), "peak": ("#fef3c7", "#b45309"), "stop": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 230, 220, 46, 40, 108, 32
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 96 + 90
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
        fs = "11.5" if status else "12.5"
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

# 折线：Θ 随 k 变化的趋势（复用 Θ 行数值，标出达峰点）
theta_vals = [90, 145.8, 148.5, 116.8]
theta_row_y = row_y[4] + ROW_H / 2
chart_y0 = row_y[-1] + ROW_H + 40
chart_h = 70
tmax = max(theta_vals)
pts = []
for j, v in enumerate(theta_vals):
    cx = col_x[j] + (COL_W - 8) / 2
    cy = chart_y0 + chart_h - (v / tmax) * chart_h
    pts.append((cx, cy))
L.append(f'<text x="{PAD}" y="{chart_y0-14}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#374151">吞吐 Θ 趋势（k=3 达峰，k=4 跌落触发早停）</text>')
poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
L.append(f'<polyline points="{poly}" fill="none" stroke="#1d4ed8" stroke-width="2"/>')
for j, (x, y) in enumerate(pts):
    color = COLOR[STATUS["决策"][j]][1]
    L.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
    L.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="{color}">{theta_vals[j]}</text>')

foot_y = chart_y0 + chart_h + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#374151">动态早停 148.5 / 盲取满 N=4 的 116.8 ≈ 1.271×；/ 保守只取 1 的 90 ≈ 1.65×。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">此为论文 Algorithm 1 机制；本 PR #46995 快照未实现调度器，'
          f'DSparkSpeculator 每步固定生产 N 个草稿 token，无早停分支。</text>')
L.append(f'<rect x="{PAD}" y="{foot_y+34}" width="{w-2*PAD}" height="40" rx="6" '
          f'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5,3"/>')
L.append(f'<text x="{PAD+12}" y="{foot_y+56}" font-family="sans-serif" font-size="11" '
          f'fill="#475569">因果闸门（论文 Appendix A 反例，未独立复现）：若调度允许回看目标模型验证结果，'
          f'会把目标分布 (0.7, 0.3) 系统性偏成 (0.85, 0.15)——故早停只能用已产生的单调信息。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m8-greedy-earlystop.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
