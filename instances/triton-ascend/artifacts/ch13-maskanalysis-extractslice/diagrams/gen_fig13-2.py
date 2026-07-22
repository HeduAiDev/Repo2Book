#!/usr/bin/env python3
"""fig13-2 flow 模板:parse 沿 use-def 链递归下降 + TypeSwitch 分派,叶子返回后
parseCmp 熔合成矩形。数据取自 explainer m2.worked_example / figure_specs.numbers。
注:CJK 标题避免 font-weight=bold(环境字体对"量"等密集字加粗会糊成实心块,
本图未含该字但沿用同一保守约定,强调改用颜色/边框)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

ROOT = ("%m = cmpi slt %r, %c", "TypeSwitch → parseCmp (L113-L114)")
LEFT = ("%r = make_range(0,16)", "parseMakeRange", "start=0,end=16,dims=[16]")
RIGHT = ("%c = constant dense<10>", "parseConstant", "scalar=10")
MERGE = ("parseCmp 熔合 slt, bound=10", "offsets=[0],dims=[10]")
FOOT_LINES = [
    "TypeSwitch 覆盖 11 个可解析 op(L106-L129):",
    "ConstantOp/AddIOp/AndIOp/CmpIOp/MakeRangeOp/BroadcastOp/SplatOp/ExpandDimsOp/ExtSIOp/DivSIOp/SelectOp",
]

BOX_W, BOX_H = 260, 56
PAD, TOP = 60, 90
ROW_GAP = 110
COL_GAP = 60

w = PAD * 2 + BOX_W * 2 + COL_GAP
h = TOP + BOX_H * 3 + ROW_GAP * 2 + 110

root_x = PAD + (BOX_W * 2 + COL_GAP) / 2 - BOX_W / 2
root_y = TOP
left_x = PAD
right_x = PAD + BOX_W + COL_GAP
mid_y = TOP + BOX_H + ROW_GAP
merge_x = root_x
merge_y = mid_y + BOX_H + ROW_GAP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{38}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="16" fill="#0f172a" font-weight="bold">'
     f'{esc("parse 递归下降:cmpi 向两个操作数各要一份 MaskState")}</text>']

# root box
L.append(f'<rect x="{root_x}" y="{root_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
L.append(f'<text x="{root_x+BOX_W/2}" y="{root_y+22}" text-anchor="middle" '
          f'font-family="monospace" font-size="12" fill="#1e1b4b">{esc(ROOT[0])}</text>')
L.append(f'<text x="{root_x+BOX_W/2}" y="{root_y+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#4338ca">{esc(ROOT[1])}</text>')

# left leaf
L.append(f'<rect x="{left_x}" y="{mid_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#ecfdf5" stroke="#059669" stroke-width="2"/>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{mid_y+20}" text-anchor="middle" '
          f'font-family="monospace" font-size="12" fill="#064e3b">{esc(LEFT[0])}</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{mid_y+37}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#047857">{esc(LEFT[1])}</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{mid_y+52}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#065f46">{esc(LEFT[2])}</text>')

# right leaf
L.append(f'<rect x="{right_x}" y="{mid_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{mid_y+20}" text-anchor="middle" '
          f'font-family="monospace" font-size="12" fill="#78350f">{esc(RIGHT[0])}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{mid_y+37}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#b45309">{esc(RIGHT[1])}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{mid_y+52}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#92400e">{esc(RIGHT[2])}</text>')

# merge box
L.append(f'<rect x="{merge_x}" y="{merge_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          f'fill="#dbeafe" stroke="#2563eb" stroke-width="2.5"/>')
L.append(f'<text x="{merge_x+BOX_W/2}" y="{merge_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#1e3a8a">{esc(MERGE[0])}</text>')
L.append(f'<text x="{merge_x+BOX_W/2}" y="{merge_y+41}" text-anchor="middle" '
          f'font-family="monospace" font-size="13" font-weight="bold" fill="#1e40af">{esc(MERGE[1])}</text>')

# arrows: root -> left leaf, root -> right leaf
root_bottom_x, root_bottom_y = root_x + BOX_W * 0.25, root_y + BOX_H
root_bottom_x2 = root_x + BOX_W * 0.75
left_top_x, left_top_y = left_x + BOX_W / 2, mid_y
right_top_x, right_top_y = right_x + BOX_W / 2, mid_y
L.append(f'<path d="M{root_bottom_x},{root_bottom_y} C{root_bottom_x-40},{(root_bottom_y+left_top_y)/2} '
          f'{left_top_x+30},{(root_bottom_y+left_top_y)/2} {left_top_x},{left_top_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(root_bottom_x+left_top_x)/2-60}" y="{(root_bottom_y+left_top_y)/2}" '
          f'font-family="sans-serif" font-size="11" fill="#334155">{esc("parse(lhs)")}</text>')
L.append(f'<path d="M{root_bottom_x2},{root_bottom_y} C{root_bottom_x2+40},{(root_bottom_y+right_top_y)/2} '
          f'{right_top_x-30},{(root_bottom_y+right_top_y)/2} {right_top_x},{right_top_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(root_bottom_x2+right_top_x)/2+10}" y="{(root_bottom_y+right_top_y)/2}" '
          f'font-family="sans-serif" font-size="11" fill="#334155">{esc("parse(rhs)")}</text>')

# arrows: left leaf -> merge, right leaf -> merge (converge, 真实数据回填,非虚假因果)
left_bottom_x, left_bottom_y = left_x + BOX_W / 2, mid_y + BOX_H
right_bottom_x, right_bottom_y = right_x + BOX_W / 2, mid_y + BOX_H
merge_top_x1, merge_top_y = merge_x + BOX_W * 0.25, merge_y
merge_top_x2 = merge_x + BOX_W * 0.75
L.append(f'<path d="M{left_bottom_x},{left_bottom_y} C{left_bottom_x-30},{(left_bottom_y+merge_top_y)/2} '
          f'{merge_top_x1-30},{(left_bottom_y+merge_top_y)/2} {merge_top_x1},{merge_top_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<path d="M{right_bottom_x},{right_bottom_y} C{right_bottom_x+30},{(right_bottom_y+merge_top_y)/2} '
          f'{merge_top_x2+30},{(right_bottom_y+merge_top_y)/2} {merge_top_x2},{merge_top_y}" '
          f'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{w/2}" y="{(mid_y+BOX_H+merge_y)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("两侧递归返回后回填(真实数据流,非独立汇聚)")}</text>')

foot_y = h - 40
for i, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{w/2}" y="{foot_y + i*18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig13-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
