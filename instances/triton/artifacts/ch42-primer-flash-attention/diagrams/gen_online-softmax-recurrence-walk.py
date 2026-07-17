#!/usr/bin/env python3
"""figure: online-softmax-recurrence-walk (state-table 模板改)
claim: 在线 softmax 一遍扫过 [1,3,2,5],running max 每次被刷新(1->3、3->5)时
旧分母 d 就被 e^{m_old-m_new} 因子降标度,扫完得 d_N=1.203438,与三遍法逐位相等。
数据来源: explainer/explainer.json mechanism m02-online-softmax-recurrence
(explainer/traces/online_softmax.json)。全坐标计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "在线 softmax 递推:一遍扫过 [1, 3, 2, 5]"
SUBTITLE = "初始 m_0=-inf, d_0=0;每格给出该轮的 4 个量——刷新最大值时旧账先降标度,再加新项"

COLS = ["j=1  x=1", "j=2  x=3", "j=3  x=2", "j=4  x=5"]
ROW_LABELS = ["running max m_j", "旧账降标度", "新增 e^{x_j-m_j}", "running 分母 d_j"]

CELLS = {
    "running max m_j": ["m_1 = 1\n(首元素)", "m_2 = 3\n(1 -> 3 刷新)", "m_3 = 3\n(不变)", "m_4 = 5\n(3 -> 5 刷新)"],
    "旧账降标度":       ["d_0=0\n(无历史)", "d_1*e^{1-3}\n= 0.135335", "d_2*e^{3-3}\n= 1.135335", "d_3*e^{3-5}\n= 0.203438"],
    "新增 e^{x_j-m_j}": ["e^{1-1}=1.0", "e^{3-3}=1.0", "e^{2-3}\n=0.367879", "e^{5-5}=1.0"],
    "running 分母 d_j": ["d_1 = 1.0", "d_2 = 1.135335", "d_3 = 1.503215", "d_4 = 1.203438"],
}
HIGHLIGHT_ROW = "running max m_j"
STATUS = {"running max m_j": ["stable", "changed", "stable", "changed"]}
COLOR = {"stable": ("#ecfdf5", "#047857"), "changed": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 168, 218, 60, 40, 118, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 96
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
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
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-10}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 9 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

# 竖向箭头连接每列 d_j 到下一列的"旧账降标度",体现"一遍扫过、逐步传递"
dj_row_idx = ROW_LABELS.index("running 分母 d_j")
old_row_idx = ROW_LABELS.index("旧账降标度")
for j in range(len(COLS) - 1):
    x1 = col_x[j] + (COL_W - 10) / 2
    y1 = row_y[dj_row_idx] + ROW_H - 4
    x2 = col_x[j + 1] + (COL_W - 10) / 2
    y2 = row_y[old_row_idx] + 4
    midy = (y1 + y2) / 2 + 20
    L.append(f'<path d="M {x1} {y1} Q {(x1+x2)/2} {midy} {x2} {y2}" fill="none" '
              f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# 底部对照:三遍法一次性分母 vs 在线法 d_N
foot_top = row_y[-1] + ROW_H + 26
box_w = 340
box1_x = PAD + LABEL_W
box2_x = w - PAD - box_w
L.append(f'<rect x="{box1_x}" y="{foot_top}" width="{box_w}" height="52" rx="6" '
          'fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>')
L.append(f'<text x="{box1_x+box_w/2}" y="{foot_top+22}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" fill="#1e40af" '
          f'font-weight="bold">{esc("在线法一遍扫过:d_N = 1.203438")}</text>')
L.append(f'<text x="{box1_x+box_w/2}" y="{foot_top+40}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("只维护 2 个 running 标量(m, d),访存 1 遍")}</text>')
L.append(f'<rect x="{box2_x}" y="{foot_top}" width="{box_w}" height="52" rx="6" '
          'fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>')
L.append(f'<text x="{box2_x+box_w/2}" y="{foot_top+22}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" fill="#334155" '
          f'font-weight="bold">{esc("三遍法一次性分母:1.203438(对照)")}</text>')
L.append(f'<text x="{box2_x+box_w/2}" y="{foot_top+40}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("先扫 max、再扫分母、再归一化——访存 3 遍")}</text>')
eq_x = (box1_x + box_w + box2_x) / 2
L.append(f'<text x="{eq_x}" y="{foot_top+30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" font-weight="bold" fill="#047857">{esc("=")}</text>')

foot_y = foot_top + 52 + 22
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc("红=本轮刷新 running max、旧账被降标度(changed);绿=running max 不变(stable)。虚线箭头=分母逐轮向后传递,全程一遍过。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("online-softmax-recurrence-walk.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
