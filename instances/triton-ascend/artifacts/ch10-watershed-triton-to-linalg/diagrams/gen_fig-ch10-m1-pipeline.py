#!/usr/bin/env python3
"""fig-ch10-m1-pipeline: ttir_to_linalg 的 ttadapter pass 管线全景（flow 模板）。
18 趟 pass = 11 趟必挂（两行主链）+ 可选 auto_scheduling 段 7 趟（虚线内嵌小链）。
triton_to_structure 在主链中出现两次（同色高亮），收官 add_triton_to_linalg 单独配色 + named_ops 标注。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

# ---------- 数据 ----------
TITLE = "ttir_to_linalg：ttadapter pass 管线全景（compiler.py:L96-L171）"
SUBTITLE = "18 趟 pass 依序挂载 = 11 趟必挂 + 可选 auto_scheduling 段 7 趟；add_triton_to_structure 出现 2 次"

# 主链第一行（含入口 + 可选块的分岔/合流）
ROW1 = [
    ("add_auto_blockify", "normal"),
    ("add_triton_to_structure", "structure"),   # 第 1 次
    ("add_discrete_mask_\naccess_conversion", "normal"),
    ("add_triton_to_\nannotation", "normal"),
    ("add_triton_to_\nunstructure", "normal"),
    ("add_triton_to_hivm", "normal"),
]
# 主链第二行
ROW2 = [
    ("add_triton_to_hfusion", "normal"),
    ("add_triton_to_llvm", "normal"),
    ("add_bubble_up_\noperation", "normal"),
    ("add_triton_to_structure", "structure"),   # 第 2 次
    ("add_triton_to_linalg", "final"),
]

# 可选 auto_scheduling 段（7 趟，仅 metadata['add_auto_scheduling'] 为真时挂）
OPTIONAL_CHAIN = [
    "add_dag_sync", "add_dag_scope", "add_cse", "add_canonicalizer",
    "add_dag_ssbuffer", "add_cse", "add_canonicalizer",
]
OPTIONAL_LABEL = "可选：auto_scheduling 段（metadata['add_auto_scheduling'] 为真才挂）× 7 趟"

COLOR = {
    "normal":    ("#e2e8f0", "#64748b", "#0f172a"),
    "structure": ("#dbeafe", "#2563eb", "#1e3a8a"),
    "final":     ("#fef3c7", "#d97706", "#78350f"),
}

BOX_W, BOX_H = 168, 56
HGAP, VGAP = 22, 46
PAD, TOP = 36, 96
OPT_H = 74

n_row1, n_row2 = len(ROW1), len(ROW2)
row1_w = n_row1 * BOX_W + (n_row1 - 1) * HGAP
row2_w = n_row2 * BOX_W + (n_row2 - 1) * HGAP
w = PAD * 2 + max(row1_w, row2_w)
opt_top = TOP + BOX_H + 34
row2_top = opt_top + OPT_H + 60
h = row2_top + BOX_H + 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ad" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']


def box_x_positions(n, row_w):
    start = PAD + (max(row1_w, row2_w) - row_w) / 2
    return [start + i * (BOX_W + HGAP) for i in range(n)]


def draw_box(x, y, label, kind):
    fill, stroke, text_fill = COLOR[kind]
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if kind != "normal" else 1.5}"/>')
    lines = label.split("\n")
    n = len(lines)
    y0 = y + BOX_H / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                  f'font-weight="{"bold" if kind != "normal" else "normal"}">{esc(line)}</text>')


x1 = box_x_positions(n_row1, row1_w)
x2 = box_x_positions(n_row2, row2_w)

# 行 1
for i, (label, kind) in enumerate(ROW1):
    draw_box(x1[i], TOP, label, kind)
    if i < n_row1 - 1:
        y_mid = TOP + BOX_H / 2
        L.append(f'<line x1="{x1[i]+BOX_W}" y1="{y_mid}" x2="{x1[i+1]}" y2="{y_mid}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1[i]+BOX_W+x1[i+1])/2}" y="{y_mid-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#94a3b8">{i+1}</text>')

# 可选段（虚线容器，挂在 add_auto_blockify 之后、第一次 add_triton_to_structure 之前）
opt_box_w = min(w - PAD * 2, 7 * 118 + 6 * 14 + 40)
opt_x = PAD + (max(row1_w, row2_w) - opt_box_w) / 2
L.append(f'<rect x="{opt_x}" y="{opt_top}" width="{opt_box_w}" height="{OPT_H}" rx="10" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')
L.append(f'<text x="{opt_x+14}" y="{opt_top+18}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="#475569">{esc(OPTIONAL_LABEL)}</text>')
chip_w, chip_gap = 118, 14
chip_start = opt_x + 20
chip_y = opt_top + 30
for i, name in enumerate(OPTIONAL_CHAIN):
    cx = chip_start + i * (chip_w + chip_gap)
    L.append(f'<rect x="{cx}" y="{chip_y}" width="{chip_w}" height="30" rx="6" '
              'fill="#eef2f7" stroke="#94a3b8" stroke-width="1"/>')
    L.append(f'<text x="{cx+chip_w/2}" y="{chip_y+19}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.5" fill="#334155">{esc(name)}</text>')
    if i < len(OPTIONAL_CHAIN) - 1:
        yy = chip_y + 15
        L.append(f'<line x1="{cx+chip_w}" y1="{yy}" x2="{cx+chip_w+chip_gap-3}" y2="{yy}" '
                  'stroke="#94a3b8" stroke-width="1" marker-end="url(#ad)"/>')

# 连线：add_auto_blockify(row1[0]) 下沉到可选段，可选段合流回第一次 add_triton_to_structure(row1[1])
ab_x = x1[0] + BOX_W / 2
st1_x = x1[1] + BOX_W / 2
L.append(f'<path d="M {ab_x} {TOP+BOX_H} L {ab_x} {opt_top+OPT_H/2} L {opt_x-4} {opt_top+OPT_H/2}" '
          'fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3"/>')
L.append(f'<path d="M {opt_x+opt_box_w+4} {opt_top+OPT_H/2} L {st1_x} {opt_top+OPT_H/2} '
          f'L {st1_x} {TOP+BOX_H}" fill="none" stroke="#94a3b8" stroke-width="1.5" '
          'stroke-dasharray="5,3" marker-end="url(#ad)"/>')

# 行 1 末尾折到行 2 开头（add_triton_to_hivm → add_triton_to_hfusion，编号 6，
# 承接行 1 的 1-5 号、衔接行 2 的 7-10 号，凑满 10 条必挂 pass 间转移箭头）
last1_x = x1[-1] + BOX_W / 2
first2_x = x2[0] + BOX_W / 2
turn_y = row2_top - 24
L.append(f'<path d="M {last1_x} {TOP+BOX_H} L {last1_x} {turn_y} L {first2_x} {turn_y} '
          f'L {first2_x} {row2_top}" fill="none" stroke="#64748b" stroke-width="1.5" '
          'marker-end="url(#a)"/>')
L.append(f'<text x="{(last1_x+first2_x)/2}" y="{turn_y-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">6</text>')

# 行 2
for i, (label, kind) in enumerate(ROW2):
    draw_box(x2[i], row2_top, label, kind)
    if i < n_row2 - 1:
        y_mid = row2_top + BOX_H / 2
        L.append(f'<line x1="{x2[i]+BOX_W}" y1="{y_mid}" x2="{x2[i+1]}" y2="{y_mid}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x2[i]+BOX_W+x2[i+1])/2}" y="{y_mid-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#94a3b8">{n_row1+i+1}</text>')

# named_ops 标注挂在收官 pass 下方
final_x = x2[-1] + BOX_W / 2
final_y = row2_top + BOX_H
L.append(f'<line x1="{final_x}" y1="{final_y}" x2="{final_x}" y2="{final_y+22}" '
          'stroke="#d97706" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{final_x}" y="{final_y+38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#92400e">named_ops=True（第 3 位参数）</text>')

# 图例（结构分析高亮 + 收官高亮）
legend_y = h - 46
L.append(f'<rect x="{PAD}" y="{legend_y}" width="16" height="16" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y+13}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">add_triton_to_structure：指针分析，管线中出现 2 次</text>')
L.append(f'<rect x="{PAD+430}" y="{legend_y}" width="16" height="16" rx="3" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{PAD+452}" y="{legend_y+13}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">add_triton_to_linalg：收官 pass，named_ops 落点</text>')

# 底部计数小结
foot_y = h - 18
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#0f172a" font-weight="bold">'
          f'{esc("总趟数 = 18（11 趟必挂 + 可选 auto_scheduling 段 7 趟）")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch10-m1-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
