#!/usr/bin/env python3
"""fig-constexpr-forward: state-table 模板 —— constexpr 把每步算术转发给内层值
再重新包裹，追踪期常量性闭合传播；__index__/__bool__ 是仅有的出壳口。
列=表达式，行=(触发 dunder / 内层求值 / 结果)，结果行按"仍是 constexpr"vs
"出壳为真值"两种语义上色。全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "constexpr：转发 dunder，追踪期常数性质闭合传播"
SUBTITLE = "c = constexpr(256)，L = [10, 11, 12, 13]（pin triton==3.2.0 实测，traces/constexpr.txt）"

COLS = ["c + 8", "c * 4", "c // 64", "L[c2]  (c2=constexpr(2))"]
ROW_LABELS = ["触发 dunder", "内层求值", "结果"]
CELLS = {
    "触发 dunder": ["__add__", "__mul__", "__floordiv__", "__index__"],
    "内层求值": ["256 + 8", "256 * 4", "256 // 64", "当下标 2 用"],
    "结果": ["constexpr[264]", "constexpr[1024]", "constexpr[4]", "12"],
}
# 前三列结果仍裹在 constexpr 里（stays），第四列出壳为真值（unwraps）
STATUS_ROW = "结果"
STATUS = {"结果": ["stays", "stays", "stays", "unwraps"]}
COLOR = {"stays": ("#eff6ff", "#1d4ed8"), "unwraps": ("#fef3c7", "#b45309")}

LABEL_W, COL_W, HEADER_H, TOP, PAD = 110, 210, 40, 118, 34
ROW_HS = [50, 56, 56]
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + sum(ROW_HS) + PAD + 96
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = []
y = TOP + HEADER_H
for rh in ROW_HS:
    row_y.append(y)
    y += rh

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# 起点：c 方框
L.append(f'<rect x="{PAD}" y="{TOP-52}" width="150" height="34" rx="6" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+75}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          'font-size="13" font-weight="bold" fill="#1e3a5f">constexpr(256)</text>')

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')
    # 起点方框 -> 各列头的箭头
    L.append(f'<line x1="{PAD+75}" y1="{TOP-18}" x2="{x+(COL_W-8)/2}" y2="{TOP-2}" '
              'stroke="#94a3b8" stroke-width="1.3" marker-end="url(#a)"/>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    rh = ROW_HS[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+rh/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+rh/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')
    if i > 0:
        L.append(f'<line x1="{PAD}" y1="{ry}" x2="{w-PAD}" y2="{ry}" '
                  'stroke="#e2e8f0" stroke-width="1"/>')

legend_y = row_y[-1] + ROW_HS[-1] + 34
L.append(f'<rect x="{PAD}" y="{legend_y-14}" width="18" height="18" rx="3" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{PAD+26}" y="{legend_y}" font-family="sans-serif" font-size="12" '
          'fill="#374151">仍裹在 constexpr 里（常量性未丢失）</text>')
L.append(f'<rect x="{PAD+330}" y="{legend_y-14}" width="18" height="18" rx="3" '
          'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
L.append(f'<text x="{PAD+356}" y="{legend_y}" font-family="sans-serif" font-size="12" '
          'fill="#374151">__index__/__bool__ 出壳为宿主真值</text>')

foot_y = h - PAD + 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          'fill="#64748b">core.py:L151-L154 每个算术 dunder 形如 constexpr(self.value ⊕ _constexpr_to_value(other))——内层用真值算，结果重新裹壳。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-constexpr-forward.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
