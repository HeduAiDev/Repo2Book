#!/usr/bin/env python3
"""fig-remove-layout-four-phase (state-table 模板)
RemoveLayoutConversions 四阶段状态轨迹:convert 数 2 -> 2 -> 2 -> 0。
列 = 四阶段,行 = [源码锚点 / 动作 / convert 数(高亮)]。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "RemoveLayoutConversions 四阶段:前三阶段只做分析,真正的删除发生在第四阶段"
SUBTITLE = "输入 IR:load#blocked -> convert->#mma -> convert->#blocked -> addf -> store#blocked(冗余 convert 往返)"

STAGES = [
    ("①initAnchorLayout", "L168-L206"),
    ("②propagateLayout", "L208-L230"),
    ("③resolveConflicts", "L311-L332"),
    ("④rewrite+canonicalize", "L666-L717"),
]
ACTIONS = [
    "锚点 = load %0->#blocked、\n函数参数、store 期望#blocked",
    "遇 convert 令 dst:=src\n->%1:=#blocked,%2:=#blocked;\naddf inferDst->%3:=#blocked",
    "每个值只剩单编码\n(全 #blocked)->无冲突,\n不插新 convert",
    "两个 convert 都成\n#blocked->#blocked no-op\n-> canonicalize 折掉死 convert",
]
CONVERT_COUNTS = [2, 2, 2, 0]
STATUS = ["flat", "flat", "flat", "dropped"]
COLOR = {"flat": ("#fee2e2", "#b91c1c"), "dropped": ("#dcfce7", "#16a34a")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 240, 60, 46, 168, 50
n = len(STAGES)
w = PAD * 2 + LABEL_W + COL_W * n
ROWS = ["源码", "动作", "convert 数"]
ROW_H_LIST = [34, 74, 60]
h = TOP + HEADER_H + sum(ROW_H_LIST) + 70

col_x = [PAD + LABEL_W + i * COL_W for i in range(n)]
row_y = []
y_cursor = TOP + HEADER_H
for rh in ROW_H_LIST:
    row_y.append(y_cursor)
    y_cursor += rh

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17.5" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc(SUBTITLE)}</text>')

# 列头
for j, (name, anchor) in enumerate(STAGES):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.4"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+35}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#dbeafe">{esc("RemoveLayoutConversions.cpp:"+anchor)}</text>')
    if j < n - 1:
        mid_y = TOP + (HEADER_H - 6) / 2
        L.append(f'<line x1="{x+COL_W-8}" y1="{mid_y}" x2="{x+COL_W+2}" y2="{mid_y}" '
                  'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#a)"/>')

# 行标签
row_labels = ["源码", "动作", "convert 数"]
for i, label in enumerate(row_labels):
    ry = row_y[i]
    rh = ROW_H_LIST[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+rh/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(label)}</text>')

# 行 0:源码(其实已在列头展示,这里放简写留白省略——改放"锚点/传播/坍缩/重写"分类词)
CATEGORY = ["纯普查(锚点)", "纯普查(传播)", "纯普查(坍缩)", "真正改写 IR"]
for j in range(n):
    x = col_x[j]
    ry = row_y[0]
    rh = ROW_H_LIST[0]
    L.append(f'<rect x="{x}" y="{ry+3}" width="{COL_W-8}" height="{rh-6}" '
              'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{ry+rh/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(CATEGORY[j])}</text>')

# 行 1:动作
for j in range(n):
    x = col_x[j]
    ry = row_y[1]
    rh = ROW_H_LIST[1]
    L.append(f'<rect x="{x}" y="{ry+3}" width="{COL_W-8}" height="{rh-6}" '
              'fill="white" stroke="#e2e8f0" stroke-width="1"/>')
    lines = ACTIONS[j].split("\n")
    y0 = ry + rh / 2 - (len(lines) - 1) * 7 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#334155">{esc(ln)}</text>')

# 行 2:convert 数(高亮)
for j in range(n):
    x = col_x[j]
    ry = row_y[2]
    rh = ROW_H_LIST[2]
    status = STATUS[j]
    fill, stroke = COLOR[status]
    L.append(f'<rect x="{x}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2.2"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{ry+rh/2+9}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="24" font-weight="bold" '
              f'fill="{stroke}">{esc(str(CONVERT_COUNTS[j]))}</text>')
    if j < n - 1:
        mid_y = ry + rh / 2
        L.append(f'<line x1="{x+COL_W-8}" y1="{mid_y}" x2="{x+COL_W+2}" y2="{mid_y}" '
                  'stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')

foot_y = h - 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("红=数值未变(convert 仍在,只是被『染色』待删);绿=真正被删除。convert 令 dst=src 发生在 L208-L230。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("本例是纯往返(blocked->mma->blocked),消除率 100%;matmul 场景通常残留 4 个必要 convert(见正文)。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-remove-layout-four-phase.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
