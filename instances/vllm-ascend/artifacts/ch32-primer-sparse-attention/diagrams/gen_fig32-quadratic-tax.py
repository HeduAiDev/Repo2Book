#!/usr/bin/env python3
"""state-table 模板改造:O(L^2) 注意力税——序列长 L 翻倍,点积数增至约 4 倍。
列 = 4 个 L 取值,行 = 点积总数 / 相对 L=4 倍数 / 翻倍增长比;
末行高亮 L=131072 落地数字。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "O(L^2) 注意力税:点积总数随序列长二次增长"
SUBTITLE = "标准因果注意力 T(L) = L(L+1)/2 次 q·k 点积;每次 L 翻倍,代价约变 4 倍(非线性 2 倍)"

COLS = ["L=4", "L=8", "L=16", "L=32"]
ROW_LABELS = ["点积总数 T(L)", "相对 L=4 倍数", "翻倍增长比"]
CELLS = {
    "点积总数 T(L)":  ["10", "36", "136", "528"],
    "相对 L=4 倍数":  ["1.0x", "3.6x", "13.6x", "52.8x"],
    "翻倍增长比":     ["—", "3.6x", "3.78x", "3.88x"],
}
HIGHLIGHT_ROW = "翻倍增长比"
STATUS = {"翻倍增长比": [None, "changed", "changed", "changed"]}
COLOR = {"changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 150, 50, 36, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 90 + PAD
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

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
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

# 落地数字 callout
callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
box_x, box_w = PAD, w - PAD * 2
L.append(f'<rect x="{box_x}" y="{callout_y}" width="{box_w}" height="46" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{box_x+16}" y="{callout_y+20}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#92400e">落地 L=131072(128K 上下文):T(L) = 8.59×10⁹ 次 q·k 点积/层</text>')
L.append(f'<text x="{box_x+16}" y="{callout_y+38}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">论文估 64K 解码时注意力已占端到端 70–80% 延迟 —— 必须稀疏化的定量动机</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">每翻一倍 L,点积数趋于变 4 倍(3.6→3.78→3.88x)——Θ(L²) 的指纹,不可能靠常数因子降成线性</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig32-quadratic-tax.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
