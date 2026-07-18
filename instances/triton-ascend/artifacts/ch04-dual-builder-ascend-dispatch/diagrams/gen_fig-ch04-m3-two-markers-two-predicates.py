#!/usr/bin/env python3
"""fig-ch04-m3-two-markers-two-predicates — state-table 模板（双标记表 + 集合图）。
上半：三个函数各自的两个布尔标记(__triton_builtin__/__ascend_builtin__)，两个下游
谓词各读一个。下半：集合关系 A(ascend 内建)⊆ B(全体内建)——@al.builtin 同时打两个
标记是这条子集关系的根源。全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "双标记 × 双谓词：A（ascend 内建）⊆ B（全体内建）"
SUBTITLE = "@al.builtin 同时打两个标记；入口门只读 __triton_builtin__，选路只读 __ascend_builtin__"

# ── 上半：标记表 ──
COLS = ["__triton_builtin__\nlanguage.core.is_builtin 读", "__ascend_builtin__\nextension.is_builtin 读"]
ROWS = ["al.sub_vec_id", "tl_load", "plain_python"]
MARK = {
    "al.sub_vec_id": ["有", "有"],
    "tl_load": ["有", "无"],
    "plain_python": ["无", "无"],
}
LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 230, 46, 52, 108, 40
w_table = PAD * 2 + LABEL_W + COL_W * len(COLS)
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

# 表格区域高度
table_bottom = row_y[-1] + ROW_H + 20

# ── 下半：集合图 A ⊆ B（几何显式分区，留足头部/尾部边距，避免与文字碰撞）──
VENN_TOP = table_bottom + 60
TOP_MARGIN, BOTTOM_MARGIN = 70, 100
B_R = 180
A_R = 95
RIGHT_DOT_GAP, RIGHT_LABEL_W = 60, 100  # plain_python 落在 B 之外，右侧留够点+标签的宽度
B_CX = PAD + B_R + 40
A_CX, A_CY_OFF = B_CX + 50, 10
w_venn = B_CX + B_R + RIGHT_DOT_GAP + 14 + RIGHT_LABEL_W + PAD
w = max(w_table, int(w_venn))

B_CY = VENN_TOP + TOP_MARGIN + B_R
A_CY = B_CY + A_CY_OFF
VENN_H = TOP_MARGIN + 2 * B_R + BOTTOM_MARGIN

h = VENN_TOP + VENN_H + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>',
     f'<text x="{PAD}" y="{TOP-16}" font-family="sans-serif" font-size="13.5" '
     f'font-weight="bold" fill="#334155">{esc("① 三例的双标记（实测）")}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    n = len(lines)
    y0 = TOP + (HEADER_H - 6) / 2 - (n - 1) * 8 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(ln)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        val = MARK[row][j]
        fill = "#dcfce7" if val == "有" else "#f1f5f9"
        stroke = "#15803d" if val == "有" else "#94a3b8"
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{stroke}" '
                  f'font-weight="bold">{esc(val)}</text>')

# 装饰器/谓词读取说明（① 表格下方脚注）
note_y = table_bottom + 4
L.append(f'<text x="{PAD}" y="{note_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("装饰器 @al.builtin 一次 setattr 两个标记（extension/core.py:L82-83：TRITON_BUILTIN + ASCEND_BUILTIN）。")}</text>')
L.append(f'<text x="{PAD}" y="{note_y+18}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("language.core.is_builtin：code_generator.py:L1179　·　extension.is_builtin：extension/core.py:L88-90")}</text>')

# ── ② 集合图（标签放在圆外的头部留白区，圆内只留点，避免文字被圆截断）──
L.append(f'<text x="{PAD}" y="{VENN_TOP-16}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#334155">{esc("② 由此推出的集合关系")}</text>')

label_y0 = VENN_TOP + 26
label_y1 = VENN_TOP + 46
L.append(f'<text x="{B_CX-B_R+30}" y="{label_y0}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#1e3a8a">{esc("B：全体内建")}</text>')
L.append(f'<text x="{B_CX-B_R+30}" y="{label_y1}" font-family="sans-serif" font-size="10.5" '
          f'fill="#1e40af">{esc("入口门读 __triton_builtin__")}</text>')

L.append(f'<text x="{A_CX+10}" y="{label_y0}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#14532d">{esc("A：ascend 内建")}</text>')
L.append(f'<text x="{A_CX+10}" y="{label_y1}" font-family="sans-serif" font-size="10.5" '
          f'fill="#166534">{esc("选路读 __ascend_builtin__")}</text>')

L.append(f'<circle cx="{B_CX}" cy="{B_CY}" r="{B_R}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<circle cx="{A_CX}" cy="{A_CY}" r="{A_R}" fill="#dcfce7" stroke="#15803d" stroke-width="2.2"/>')

# 三个函数落点（点 + 标签）：al.sub_vec_id 深入 A 内部；tl_load 落 B\A（在 B 内、A 外）；
# plain_python 落 B 之外（RIGHT_DOT_GAP 处，与 w_venn 的预留宽度一致）
DOTS = [
    ("al.sub_vec_id", A_CX - 10, A_CY + 25, "#15803d", "middle", 0, -14),
    ("tl_load", B_CX - 50, B_CY + 110, "#1d4ed8", "middle", 0, -14),
    ("plain_python", B_CX + B_R + RIGHT_DOT_GAP, B_CY, "#b91c1c", "start", 14, -14),
]
for name, dx, dy, color, anchor, tdx, tdy in DOTS:
    L.append(f'<circle cx="{dx}" cy="{dy}" r="6" fill="{color}"/>')
    L.append(f'<text x="{dx+tdx}" y="{dy+tdy}" text-anchor="{anchor}" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{color}">{esc(name)}</text>')

L.append(f'<text x="{B_CX+B_R+RIGHT_DOT_GAP+14}" y="{B_CY+22}" font-family="sans-serif" '
          f'font-size="10.5" fill="#b91c1c">{esc("（B 之外，兜底裸调用）")}</text>')

L.append(f'<text x="{PAD}" y="{h-24}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("A ⊆ B：al.builtin 同时打两标记 ⇒ 任何 ascend 内建必先过入口门（∈B）、再被选路命中（∈A）路由到 ascend_builder。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch04-m3-two-markers-two-predicates.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
