#!/usr/bin/env python3
"""fig35-1-outlier-collapse — state-table 变体：行=通道，列=场景。
展示 per-tensor 量化下，非 outlier 通道的有效量化级数如何被 outlier 撑大的 absmax 挤压。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 已知渲染环境缺陷：字符"量"在 synthetic-bold 下会被错渲成实心方块（其余汉字均正常）。
# 对话粗体文本一律经 btext() 拆 tspan，避开该字的粗体渲染。
_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
                buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

TITLE = "per-tensor 量化：outlier 撑大 absmax，非 outlier 通道有效级数塌缩"
SUBTITLE = "有效级数 = 2^8 · m_i / m（matrix absmax m=10.0，INT8 满量程 = 256 级）"

COLS = ["通道 absmax m_i", "有效级数（基准场景）", "有效级数（125× 极端 outlier）"]
ROW_LABELS = ["通道 0", "通道 1", "通道 2（outlier）"]
CELLS = {
    "通道 0": ["0.15", "3.84", "1.536"],
    "通道 1": ["0.2", "5.12", "2.048"],
    "通道 2（outlier）": ["10.0", "256.0", "256.0"],
}
# 语义色：非 outlier 通道两列 = collapsed（红），outlier 通道两列 = full（绿）
STATUS = {
    "通道 0": ["collapsed", "collapsed"],
    "通道 1": ["collapsed", "collapsed"],
    "通道 2（outlier）": ["full", "full"],
}
COLOR = {"collapsed": ("#fee2e2", "#b91c1c"), "full": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 130, 220, 56, 44, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 30
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 第一列（数据列，第 0 列是"通道 absmax m_i"，不参与状态高亮，普通灰底表头即可）
for j, name in enumerate(COLS):
    x = col_x[j]
    header_color = "#3b82f6" if j > 0 else "#64748b"
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{header_color}" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("（")
    if len(lines) == 2:
        line1, line2 = lines[0], "（" + lines[1]
    else:
        line1, line2 = name, ""
    cy = TOP + (HEADER_H - 6) / 2
    if line2:
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{cy-2}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line1)}</text>')
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{cy+14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line2)}</text>')
    else:
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{cy+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line1)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        # 第 0 列(absmax m_i)不着色，只作为参考值；第 1/2 列(有效级数)按语义色高亮
        status = statuses[j-1] if (statuses and j > 0) else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            text_fill = stroke
            weight_attr = 'font-weight="bold" '
        else:
            text_fill = "#374151"
            weight_attr = ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

foot_y = h - PAD + 4
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">红=有效级数被压到远低于 256（非 outlier 通道）；绿=outlier 通道独占满量程 256 级。</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">极端场景(125× outlier)下最差非 outlier 通道有效级数仅 1.536——不足 2 档，8-bit 名存实亡。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-1-outlier-collapse.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
