#!/usr/bin/env python3
"""fig36-6-calibration: state-table 模板。5 个置信度分桶,列=分桶,
行=mean_confidence/empirical_accept_rate,两行几乎相等(良好校准)。
数字来自 explainer.json fig36-6 numbers(traces/eagle2_tree.json calibration_curve)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "校准曲线：草稿头置信度 ≈ 实测接受率"
SUBTITLE = "4000 个 (置信度, 是否接受) 样本按置信度分 5 桶；每桶两行几乎相等 → 良好校准"
BINS = ["桶 1\n(~0.1)", "桶 2\n(~0.3)", "桶 3\n(~0.5)", "桶 4\n(~0.7)", "桶 5\n(~0.9)"]
ROW_LABELS = ["mean confidence", "empirical accept rate", "偏差"]
MEAN_CONF = [0.101, 0.305, 0.499, 0.702, 0.897]
ACCEPT_RATE = [0.113, 0.312, 0.518, 0.713, 0.888]
DIFF = [round(a - m, 3) for m, a in zip(MEAN_CONF, ACCEPT_RATE)]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 200, 190, 44, 46, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(BINS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 70

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(BINS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(BINS):
    x = col_x[j]
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    ny = TOP + (HEADER_H-6)/2 - (len(lines)-1)*8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{ny+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(BINS)):
        cx = col_x[j]
        if row == "mean confidence":
            text = f"{MEAN_CONF[j]:.3f}"
            fill, stroke = "#eff6ff", "#1e40af"
        elif row == "empirical accept rate":
            text = f"{ACCEPT_RATE[j]:.3f}"
            fill, stroke = "#ecfdf5", "#047857"
        else:
            d = DIFF[j]
            text = f"{'+' if d>=0 else ''}{d:.3f}"
            fill, stroke = "#f8fafc", "#64748b"
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{stroke}">{esc(text)}</text>')

foot_y = row_y[-1] + ROW_H + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("结论：草稿头良好校准（confidence ≈ 实测接受率），故 V_i=∏c_j 可代替 ∏(真实接受率) 排序，无需目标前向。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-6-calibration.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
