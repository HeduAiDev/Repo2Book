#!/usr/bin/env python3
"""tiling 模板改造:CSA 压缩块的窗口 = 本块 m 个 token + 上一块借来的 m 个 token。
块0 前半是 -inf padding(因果起点无上块);块1 借块0 的 token 1.0-4.0 + 本块 token
5.0-8.0。每格下方标 softmax 权重,凸组合汇入 C_comp 输出框。
数字来自 traces/csa_overlap.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "CSA 压缩窗口:本块 m 个 token + 借上一块 m 个 token,净压缩率仍 1/m"
SUBTITLE = "n=8, m=4, c=2;块0 前半是因果起点的 -inf padding(权重 0),块1 借块0 的 token 1.0-4.0"

CELL, GAP, PAD, TOP = 62, 6, 40, 130
ROW_GAP = 190

BLOCK0 = [
    (None, "0", "pad"), (None, "0", "pad"), (None, "0", "pad"), (None, "0", "pad"),
    ("1.0", "0.032", "own"), ("2.0", "0.087", "own"), ("3.0", "0.237", "own"), ("4.0", "0.644", "own"),
]
BLOCK1 = [
    ("1.0", "0.001", "borrow"), ("2.0", "0.002", "borrow"), ("3.0", "0.004", "borrow"), ("4.0", "0.012", "borrow"),
    ("5.0", "0.031", "own"), ("6.0", "0.086", "own"), ("7.0", "0.233", "own"), ("8.0", "0.632", "own"),
]
COLOR = {"pad": ("#f1f5f9", "#94a3b8", "#94a3b8"),
         "own": ("#3b82f6", "#1e3a5f", "white"),
         "borrow": ("#fcd34d", "#b45309", "#78350f")}

row_w = len(BLOCK0) * (CELL + GAP) - GAP
w = PAD * 2 + row_w + 260
h = TOP + ROW_GAP + CELL + 34 + 90

def draw_row(L, y, label, cells, result, note):
    L.append(f'<text x="{PAD}" y="{y-14}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    for i, (val, wgt, kind) in enumerate(cells):
        x = PAD + i * (CELL + GAP)
        fill, stroke, tcol = COLOR[kind]
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        disp = val if val else "pad"
        L.append(f'<text x="{x+CELL/2}" y="{y+CELL/2-2}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="{tcol}" '
                  f'font-weight="bold">{esc(disp)}</text>')
        L.append(f'<text x="{x+CELL/2}" y="{y+CELL+16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#64748b">w={esc(wgt)}</text>')
    # 汇入结果框的箭头 + 结果框
    arrow_x0 = PAD + row_w + 14
    arrow_x1 = arrow_x0 + 40
    box_x = arrow_x1 + 6
    L.append(f'<line x1="{arrow_x0}" y1="{y+CELL/2}" x2="{arrow_x1}" y2="{y+CELL/2}" '
              'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<rect x="{box_x}" y="{y}" width="150" height="{CELL}" rx="6" '
              'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
    L.append(f'<text x="{box_x+75}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#047857">C_comp[0] =</text>')
    L.append(f'<text x="{box_x+75}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#047857">{esc(result)}</text>')
    L.append(f'<text x="{PAD}" y="{y+CELL+34}" font-family="sans-serif" font-size="11" '
              f'fill="#475569">{esc(note)}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="a-orange" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

y0 = TOP
draw_row(L, y0, "压缩块 0(窗口)", BLOCK0, "3.493",
         "前 4 位 -inf padding 权重恒 0(因果起点无上一块);仅本块 4 个 token 参与,权重降序 0.644/0.237/0.087/0.032")

y1 = TOP + ROW_GAP
draw_row(L, y1, "压缩块 1(窗口)", BLOCK1, "7.421",
         "前 4 位借自块0 的 token 1.0-4.0(黄=借用),后 4 位本块 token 5.0-8.0;8 个位置参与但净压缩率仍 1/m=1/4")

# 借用箭头:块0 own(黄色借用来源) -> 块1 borrow 区,起讫点都卡在两行文字之间的空白带内
b0_own_x = PAD + 4 * (CELL + GAP) + CELL / 2
b1_borrow_x = PAD + 1.5 * (CELL + GAP)
note0_y = y0 + CELL + 34          # 块0 注释行 baseline
label1_y = y1 - 14                # 块1 行标题 baseline
band_top = note0_y + 12
band_bot = label1_y - 22
mid_y = (band_top + band_bot) / 2
L.append(f'<path d="M {b0_own_x} {band_top} C {b0_own_x} {mid_y}, {b1_borrow_x} {mid_y}, '
          f'{b1_borrow_x} {band_bot}" fill="none" stroke="#d97706" stroke-width="2" '
          'stroke-dasharray="5,3" marker-end="url(#a-orange)"/>')
label_txt = "借用(overlap)"
label_cx = (b0_own_x + b1_borrow_x) / 2
label_cy = mid_y + 4
# 白底遮罩,让文字"盖住"虚线而不是被虚线穿过
L.append(f'<rect x="{label_cx-58}" y="{label_cy-13}" width="116" height="18" '
          f'fill="white"/>')
L.append(f'<text x="{label_cx}" y="{label_cy}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#d97706">{esc(label_txt)}</text>')

# 图例
ly = h - 34
LEGEND = [("own", "本块 token"), ("borrow", "借上一块 token"), ("pad", "因果 padding(权重0)")]
for j, (key, label) in enumerate(LEGEND):
    lx = PAD + j * 220
    fill, stroke, _ = COLOR[key]
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-3-csa-overlap-window.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
