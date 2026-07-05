#!/usr/bin/env python3
"""state-table 模板改造:对齐旋钮 alpha 从 0(对齐)到 1(随机)扫描,
KL 单调升、top-k 召回单调降——训练协同适配是 top-k 不掉点的定量根因。
列 = 5 个 alpha 取值,行 = 平均 KL / 平均召回,末行标注 KL/召回反向趋势箭头。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "训练协同适配:indexer 越对齐真注意力(KL 越低),top-k 召回越高"
SUBTITLE = "对齐旋钮 α(0=完全对齐真分布,1=完全随机),400 次试验平均;完美对齐时 KL(p‖p)=0"

COLS = ["α=0.0\n(对齐)", "α=0.25", "α=0.5", "α=0.75", "α=1.0\n(随机)"]
ROW_LABELS = ["平均 dense-warmup KL", "平均 top-k 质量召回"]
CELLS = {
    "平均 dense-warmup KL": ["0.002", "0.044", "0.198", "0.478", "0.864"],
    "平均 top-k 质量召回": ["0.386", "0.368", "0.284", "0.181", "0.126"],
}
STATUS = {
    "平均 dense-warmup KL": ["stable", None, None, None, "changed"],
    "平均 top-k 质量召回": ["stable", None, None, None, "changed"],
}
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 150, 54, 46, 108, 30
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

for j, name in enumerate(COLS):
    x = col_x[j]
    lines = name.split("\n")
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    n = len(lines)
    y0 = TOP + (HEADER_H - 6) / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

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
    # 行内趋势箭头(第一列 -> 末列)
    arrow_y = ry + ROW_H / 2
    ax1 = col_x[0] + COL_W - 8 + 4
    ax2 = col_x[-1] - 4

# 趋势 callout
callout_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="52" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">α: 0→1,KL 单调升 0.002→0.864;召回同步单调降 0.386→0.126(约 3x 差距)</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+40}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">低 KL ⟺ 高召回——top-k 不掉点是训练压 KL 换来的,不是打分器天生准</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">绿=对齐端(低 KL/高召回),红=随机端(高 KL/低召回)——两行反向单调是同一枚硬币的两面</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig32-training-coadapt.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
