#!/usr/bin/env python3
"""state-machine 模板改写:tl.load 的分派判据。一个入口节点(tl.load 调用)按
指针类型嵌套方向分岔成两条互斥路径,双向都标注各自拒收的参数与 IR 类型。
数字全部来自 dossier m5-load-dispatch。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "load 的分派判据：指针类型的嵌套方向"
SUBTITLE = "判据：ptr.type.is_ptr() and ptr.type.element_ty.is_block()"

ENTRY = ("tl.load(ptr, ...)", 0, 0)
BOX_W, BOX_H = 220, 64
PAD, TOP = 44, 130
HGAP = 260

w = PAD * 2 + BOX_W * 2 + HGAP + 60
h = TOP + 20 + BOX_H + 46 + 92 + 32 + 3 * 19 + 14

entry_x = (w - BOX_W) / 2
entry_y = 40

left_x = PAD + 30
right_x = w - PAD - BOX_W - 30
branch_y = TOP + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="18" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

# entry box
L.append(f'<rect x="{entry_x}" y="{entry_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{entry_x+BOX_W/2}" y="{entry_y+BOX_H/2+5}" text-anchor="middle" '
          f'font-family="monospace" font-size="14" font-weight="bold" '
          f'fill="#0f172a">{esc("tl.load(ptr,...)")}</text>')
L.append(f'<text x="{entry_x+BOX_W/2}" y="{entry_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#475569">{esc(SUBTITLE)}</text>')

ey = entry_y + BOX_H
ecx = entry_x + BOX_W / 2

# 分支线到左右两个状态框
lcx, rcx = left_x + BOX_W / 2, right_x + BOX_W / 2
L.append(f'<path d="M {ecx} {ey} L {lcx} {branch_y}" stroke="#334155" stroke-width="1.8" '
          'fill="none" marker-end="url(#a)"/>')
L.append(f'<path d="M {ecx} {ey} L {rcx} {branch_y}" stroke="#334155" stroke-width="1.8" '
          'fill="none" marker-end="url(#a)"/>')
mid_l = ((ecx + lcx) / 2, (ey + branch_y) / 2)
mid_r = ((ecx + rcx) / 2, (ey + branch_y) / 2)
L.append(f'<text x="{mid_l[0]-10}" y="{mid_l[1]-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11.5" fill="#334155">{esc("否(block<pointer> 或标量)")}</text>')
L.append(f'<text x="{mid_r[0]+10}" y="{mid_r[1]-6}" font-family="sans-serif" '
          f'font-size="11.5" fill="#334155">{esc("是(pointer<block>)")}</text>')

# 左框:legacy
box_y = branch_y
L.append(f'<rect x="{left_x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#7f1d1d">_load_legacy</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+46}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#991b1b">tensor&lt;8x!tt.ptr&lt;f32&gt;&gt;</text>')

# 右框:block pointer
L.append(f'<rect x="{right_x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#14532d">_load_block_pointer</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+46}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#166534">!tt.ptr&lt;tensor&lt;16x16xf32&gt;&gt;</text>')

# 底部:各自拒收的参数(两张卡片)
card_y = box_y + BOX_H + 46
card_h = 92
L.append(f'<rect x="{left_x}" y="{card_y}" width="{BOX_W}" height="{card_h}" rx="8" '
          'fill="#fff1f2" stroke="#fca5a5" stroke-width="1.3"/>')
left_lines = ["拒收:boundary_check / padding", "mask/other 广播到 ptr 形状",
              "有 mask→create_masked_load", "无 mask→create_load"]
for i, line in enumerate(left_lines):
    L.append(f'<text x="{left_x+12}" y="{card_y+20+i*18}" font-family="sans-serif" '
              f'font-size="11.5" fill="#7f1d1d">{esc(line)}</text>')

L.append(f'<rect x="{right_x}" y="{card_y}" width="{BOX_W}" height="{card_h}" rx="8" '
          'fill="#f0fdf4" stroke="#86efac" stroke-width="1.3"/>')
right_lines = ["拒收:mask / other", "boundary_check 归一化后传入",
               "create_tensor_pointer_load", "(带 boundary_check + padding)"]
for i, line in enumerate(right_lines):
    L.append(f'<text x="{right_x+12}" y="{card_y+20+i*18}" font-family="sans-serif" '
              f'font-size="11.5" fill="#14532d">{esc(line)}</text>')

foot_y0 = card_y + card_h + 32
FOOT = [
    "结论:load 靠指针类型的『里外方向』一分为二——『指针的块』(block<pointer>)灵活、",
    "走逐元素 mask 那条;『块的指针』(pointer<block>)信息足、走 make_block_ptr 那条。",
    "两条路各拒收对方的边界表达法,泾渭分明。",
]
for i, line in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*19}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-load-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
