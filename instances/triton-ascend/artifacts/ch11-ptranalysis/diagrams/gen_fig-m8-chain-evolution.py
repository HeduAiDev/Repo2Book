#!/usr/bin/env python3
"""fig-m8-chain-evolution: b_ptrs 子链 %14→%27 逐节点 PtrState 演化（state-table 模板）。
13 个节点分三组（左侧色条标注，非装饰）：行索引子链(绿)/列索引+rem 子链(蓝)/
两侧合并+addptr(琥珀，末行终态高亮)。列 = [节点/算子/stateInfo/offset/source]。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "b_ptrs 子链 %14→%27 逐节点 PtrState 演化（matmul_kernel，simplify_for_loop.mlir）"
SUBTITLE = "14 个地址算子把散落的 stride/offset 一步步归并成 base=%arg1、strides=[%arg4,1]、offset=rem(%9,1024) 的结构化 2D 块"

ROWS = [
    ("%14", "make_range(0,64)", "[(1,64,d0)]", "0", "∅", "row"),
    ("%19", "expand_dims(ax=1)", "[(1,64,d0),(0,1,d1)]", "0", "∅", "row"),
    ("%20", "splat(%arg4:i32)", "[(0,64,d0),(0,1,d1)]", "%arg4", "∅", "row"),
    ("%21", "muli(%19,%20)", "[(%arg4,64,d0),(0,1,d1)]", "0", "∅", "row"),
    ("%23", "broadcast(%21)", "[(%arg4,64,d0),(0,256,d1)]", "0", "∅", "row"),
    ("%10", "make_range(0,256)", "[(1,256,d0)]", "0", "∅", "col"),
    ("%12", "addi(splat %9,%10)", "[(1,256,d0)]", "%9", "∅", "col"),
    ("%13", "remsi(%12,1024)", "[(1,256,d0)]", "rem(%9,1024)", "∅ (shouldLinearize=true)", "col"),
    ("%22", "expand_dims(ax=0)", "[(0,1,d0),(1,256,d1)]", "rem(%9,1024)", "∅", "col"),
    ("%24", "broadcast(%22)", "[(0,64,d0),(1,256,d1)]", "rem(%9,1024)", "∅", "col"),
    ("%25", "addi(%23,%24)", "[(%arg4,64,d0),(1,256,d1)]", "rem(%9,1024)", "∅", "merge"),
    ("%26", "splat(%arg1:!ptr)", "[(0,64,d0),(0,256,d1)]", "0", "%arg1", "merge"),
    ("%27", "addptr(%26,%25)", "[(%arg4,64,d0),(1,256,d1)]", "rem(%9,1024)", "%arg1", "final"),
]

GROUP_COLOR = {
    "row": "#16a34a", "col": "#2563eb", "merge": "#d97706", "final": "#d97706",
}
GROUP_LABEL = {
    "row": "行索引子链（%14→%23）", "col": "列索引 + rem 子链（%10→%24）",
    "merge": "合并 + base 指针（%25/%26）", "final": "addptr 终态",
}

COLS = ["算子 (IR)", "stateInfo(stride,shape,dim)", "offset", "source"]
LABEL_W = 66
STRIPE_W = 8
COL_W = [220, 320, 200, 170]
ROW_H, HEADER_H, TOP, PAD = 40, 46, 118, 30

w = PAD * 2 + STRIPE_W + LABEL_W + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 60

col_x = [PAD + STRIPE_W + LABEL_W]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
label_x0 = PAD + STRIPE_W
L.append(f'<rect x="{label_x0}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#1e3a5f"/>')
L.append(f'<text x="{label_x0+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">节点</text>')
for j, name in enumerate(COLS):
    x = col_x[j]
    cw = COL_W[j] - 8
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+cw/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

# 行
for i, (node, op, state, offset, source, group) in enumerate(ROWS):
    ry = row_y[i]
    is_final = group == "final"
    stripe_color = GROUP_COLOR[group]
    L.append(f'<rect x="{PAD}" y="{ry+2}" width="{STRIPE_W}" height="{ROW_H-4}" '
              f'fill="{stripe_color}"/>')
    if is_final:
        L.append(f'<rect x="{label_x0}" y="{ry+2}" width="{sum(COL_W)+LABEL_W-8}" '
                  f'height="{ROW_H-4}" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
    text_fill = "#92400e" if is_final else "#374151"
    L.append(f'<text x="{label_x0+(LABEL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{text_fill}">{esc(node)}</text>')
    cells = [op, state, offset, source]
    for j, cell in enumerate(cells):
        cx = col_x[j]
        cw = COL_W[j] - 8
        weight = 'font-weight="bold" ' if is_final else ''
        fs = "10" if j == 1 else "10.5"
        L.append(f'<text x="{cx+cw/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="{text_fill}" '
                  f'{weight}>{esc(cell)}</text>')

# 分隔线
tot_h = HEADER_H + len(ROWS) * ROW_H
for i in range(len(ROWS) + 1):
    y = TOP + HEADER_H + i * ROW_H
    L.append(f'<line x1="{label_x0}" y1="{y}" x2="{label_x0+LABEL_W+sum(COL_W)-16}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
# 三组之间加粗分隔线
group_seq = [g for *_, g in ROWS]
for i in range(1, len(ROWS)):
    if group_seq[i] != group_seq[i-1] and not (group_seq[i-1] == "merge" and group_seq[i] == "final"):
        y = row_y[i]
        L.append(f'<line x1="{PAD}" y1="{y}" x2="{label_x0+LABEL_W+sum(COL_W)-16}" y2="{y}" '
                  'stroke="#64748b" stroke-width="2"/>')

# 图例（左侧色条 = 三条子链分组，非装饰）
legend_y = h - 42
lx = PAD
for key in ("row", "col", "merge"):
    L.append(f'<rect x="{lx}" y="{legend_y}" width="14" height="14" fill="{GROUP_COLOR[key]}"/>')
    L.append(f'<text x="{lx+20}" y="{legend_y+12}" font-family="sans-serif" font-size="10.5" '
              f'fill="#334155">{esc(GROUP_LABEL[key])}</text>')
    lx += 20 + int(len(GROUP_LABEL[key]) * 7.6) + 26

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("终态 %27：strides=[%arg4,1]、sizes=[64,256]、offset=rem(%9,1024)、source=%arg1——14 算子/13 演化行（splat %9 折入 %12 行）")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m8-chain-evolution.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
