#!/usr/bin/env python3
"""state-table 模板:IO 复杂度账,三个 N 下标准注意力 vs FlashAttention 的
元素级 HBM 访存计数与比值,比值随 N 单调上升。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "IO 复杂度账 — d=64, SRAM M≈25600 元素(block_c=100)"
SUBTITLE = "标准注意力访存 Θ(Nd+N²)(含 N×N 物化);FlashAttention 消去 N² 物化项 —— 比值随 N 增大单调上升"
COLS = ["N = 1024", "N = 2048", "N = 4096"]
ROW_LABELS = ["标准 HBM 访问(元素)", "FlashAttn HBM 访问(元素)", "比值 标准/Flash"]
CELLS = {
    "标准 HBM 访问(元素)":     ["4,456,448", "17,301,504", "68,157,440"],
    "FlashAttn HBM 访问(元素)": ["2,338,816", "8,691,712", "33,439,744"],
    "比值 标准/Flash":          ["1.91×", "1.99×", "2.04×"],
}
HIGHLIGHT_ROW = "比值 标准/Flash"
STATUS = {"比值 标准/Flash": ["stable", "stable", "changed"]}
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 260, 200, 58, 34, 96, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 96
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
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(CELLS[row][j])}</text>')

# 趋势箭头:比值行,1024->2048->4096 递增
arrow_y = row_y[-1] + ROW_H / 2
for j in range(len(COLS) - 1):
    x1 = col_x[j] + COL_W - 8
    x2 = col_x[j + 1]
    L.append(f'<line x1="{x1+4}" y1="{arrow_y}" x2="{x2-4}" y2="{arrow_y}" '
              'stroke="#b91c1c" stroke-width="1.5" marker-end="url(#a)"/>')

box_y = row_y[-1] + ROW_H + 30
L.append(f'<rect x="{PAD}" y="{box_y}" width="{w-2*PAD}" height="34" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
FINAL = "标准 Θ(Nd+N²) vs Flash Θ(N²d²/M) —— d²=4096 ≪ M=25600,序列越长、N² 物化项收益越大"
L.append(f'<text x="{w/2}" y="{box_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc(FINAL)}</text>')

foot_y = h - 16
FOOT = "注:元素级访存计数(不含 wall-clock 融合收益);论文图 2 报告的实测加速可达 9×。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-4-io-accounting.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
