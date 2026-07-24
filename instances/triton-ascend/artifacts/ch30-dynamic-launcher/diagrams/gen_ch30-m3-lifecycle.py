#!/usr/bin/env python3
"""ch30-m3-lifecycle：发射器一生只编译一次——构造期生成并按 sha256 缓存编译成 .so 再
dlopen，之后每次调用只走 __call__ → self.launch。flow 模板，纵向步骤 + 括注一次性/每次。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "NPULauncher 生命周期：只编译一次，之后每次只转调"
SUBTITLE = "driver.py —— 构造期生成+编译+缓存+dlopen（一次性）；此后每次 kernel 调用只转调（每次）"

PAD = 40
BOX_W = 680
GAP = 26
TOP = 96

elems = []


def add(s):
    elems.append(s)


def step_box(y, lines, fill, stroke, text_fill, box_w=None, cx=None):
    bw = box_w if box_w is not None else BOX_W
    ccx = cx if cx is not None else x_center
    bx = ccx - bw / 2
    n = len(lines)
    box_h = 30 + 22 * (n - 1) + 30
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{bw}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if k == 0 else ''
        fs = 13.5 if k == 0 else 11.5
        fill_c = text_fill if k == 0 else "#334155"
        add(f'<text x="{ccx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="{fs}" {fw}fill="{fill_c}">{esc(line)}</text>')
    return box_h


def arrow(x, y1, y2):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')


x_center = PAD + 90 + BOX_W / 2
w = PAD * 2 + 180 + BOX_W

STEPS = [
    (["① NPULauncher.__init__ —— 构造期入口", "driver.py:L105"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["② generate_npu_wrapper_src 现拼 wrapper 源码",
      "调用于 L115，定义于 L403"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["③ make_npu_launcher_stub：以 sha256(wrapper_src) 为缓存键",
      "命中 → 直接用旧 .so；未命中 → 就地编译 (driver.py:L253)"], "#fef3c7", "#b45309", "#78350f"),
    (["④ importlib dlopen → self.launch = getattr(mod, \"launch\")",
      "driver.py:L124-126"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["⑤ 每次 kernel 调用：__call__ → self.launch(*args)",
      "driver.py:L128 / L139-140"], "#ede9fe", "#6d28d9", "#3730a3"),
]

y = TOP
bh_list = []
y_list = []
for i, (lines, fill, stroke, tfill) in enumerate(STEPS):
    y_list.append(y)
    bh = step_box(y, lines, fill, stroke, tfill)
    bh_list.append(bh)
    if i < len(STEPS) - 1:
        arrow(x_center, y + bh, y + bh + GAP)
    y += bh + GAP

content_bottom = y_list[-1] + bh_list[-1]

# 括注：①-④ 一次性 / ⑤ 每次
bracket_x = x_center + BOX_W / 2 + 30
b1_top, b1_bottom = y_list[0], y_list[3] + bh_list[3]
b2_top, b2_bottom = y_list[4], y_list[4] + bh_list[4]
for top, bottom, label, color in [
    (b1_top, b1_bottom, "只发生一次", "#0369a1"),
    (b2_top, b2_bottom, "每次调用", "#6d28d9"),
]:
    add(f'<path d="M {bracket_x} {top} L {bracket_x+14} {top} L {bracket_x+14} {bottom} '
        f'L {bracket_x} {bottom}" fill="none" stroke="{color}" stroke-width="2"/>')
    mid = (top + bottom) / 2
    for k, ch in enumerate(label):
        add(f'<text x="{bracket_x+30}" y="{mid-((len(label)-1)*13)/2+k*13+5:.0f}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13" '
            f'font-weight="bold" fill="{color}">{esc(ch)}</text>')

w = bracket_x + 60 + 60

note_lines = [
    "wrapper_src 完全由 (constants, signature, metadata) 三者确定，相同签名逐字节相同 → sha256 相同 →",
    "复用同一个 .so，永不重复编译；不同签名的 kernel 各自只在第一次调用时付一次编译代价。",
]
note_top = content_bottom + 30
note_h = 24 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch30-m3-lifecycle.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
