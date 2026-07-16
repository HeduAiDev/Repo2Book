#!/usr/bin/env python3
"""scf-for-arg-layout: layout 模板。scf.for body 参数槽像座位表：arg(0)=归纳变量，
arg(1..)=各 loop-carried（第 i 个在 arg(i+1)）。下方弧线回指 names 列表 [acc, m]，
标注 i→i+1 偏移。底部引官方 SCF dialect 文档一句 + 源码锚点。全部坐标由循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

SLOTS = [
    ("arg(0)", "归纳变量 k", "(induction var)", "#fde68a", "#d97706"),
    ("arg(1)", "acc = iter_arg₀", "(i=0)", "#bfdbfe", "#2563eb"),
    ("arg(2)", "m = iter_arg₁", "(i=1)", "#bbf7d0", "#16a34a"),
]
CELL, GAP, PAD, TOP = 200, 24, 40, 110
w = PAD * 2 + len(SLOTS) * (CELL + GAP) - GAP
NAMES_Y = TOP + 150
NAMES_W, NAMES_H = 90, 46
h = NAMES_Y + NAMES_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("scf.for body 参数槽布局：arg(0)=归纳变量，第 i 个 loop-carried 在 arg(i+1)")}</text>')
L.append(f'<text x="{PAD}" y="46" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("块参数总数 = 1 归纳 + 2 loop-carried = 3")}</text>')

L.append(f'<text x="{PAD}" y="{TOP-20}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("scf.for body 入口块（像列车座位表，从左到右按槽位号排列）")}</text>')

CELL_H = 90
for i, (slot, label, sub, fill, stroke) in enumerate(SLOTS):
    x = PAD + i * (CELL + GAP)
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL}" height="{CELL_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(slot)}</text>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#0f172a">{esc(label)}</text>')
    L.append(f'<text x="{x+CELL/2}" y="{TOP+70}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#475569">{esc(sub)}</text>')

# names 列表框（loop-carried 登记序 [acc, m]），弧线从 arg(1)/arg(2) 回指
names_x = PAD + (SLOTS.__len__() * (CELL + GAP) - GAP) / 2 - NAMES_W / 2
L.append(f'<rect x="{names_x}" y="{NAMES_Y}" width="{NAMES_W}" height="{NAMES_H}" rx="8" '
          'fill="#f1f5f9" stroke="#64748b" stroke-width="1.4"/>')
L.append(f'<text x="{names_x+NAMES_W/2}" y="{NAMES_Y+20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#334155">{esc("names 登记序")}</text>')
L.append(f'<text x="{names_x+NAMES_W/2}" y="{NAMES_Y+36}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#0f172a">{esc("[acc, m]")}</text>')

# 弧线：arg(1) i=0 -> names, arg(2) i=1 -> names，标注 i -> i+1
for idx, i in enumerate([1, 2]):
    slot_x = PAD + i * (CELL + GAP) + CELL / 2
    slot_bottom = (slot_x, TOP + CELL_H)
    names_top = (names_x + NAMES_W * (0.35 if idx == 0 else 0.65), NAMES_Y)
    midx = (slot_bottom[0] + names_top[0]) / 2
    midy_ctrl = (slot_bottom[1] + names_top[1]) / 2 + 6
    path = (f'M {slot_bottom[0]},{slot_bottom[1]} Q {midx},{midy_ctrl} '
            f'{names_top[0]},{names_top[1]}')
    L.append(f'<path d="{path}" fill="none" stroke="#7c3aed" stroke-width="1.6" '
              'marker-end="url(#a)"/>')
    lbl = "0 → 1" if i == 1 else "1 → 2"
    lx = midx + (18 if idx == 0 else -18)
    L.append(f'<text x="{lx}" y="{midy_ctrl-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#6d28d9">{esc(lbl)}</text>')

# 说明：+1 偏移
note_y = NAMES_Y + NAMES_H + 34
L.append(f'<text x="{w/2}" y="{note_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#0f172a">'
          f'{esc("第 i 个（0-based）loop-carried 的槽位 = 1 + i = arg(i+1)——+1 正是跳过 arg(0) 的归纳变量")}</text>')

# 底部引用框
cite_y = note_y + 30
L.append(f'<rect x="{PAD}" y="{cite_y-18}" width="{w-2*PAD}" height="46" rx="6" '
          'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
L.append(f'<text x="{w/2}" y="{cite_y+2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">'
          f'{esc("官方 SCF dialect 文档：“an argument for the induction variable, followed by")}</text>')
L.append(f'<text x="{w/2}" y="{cite_y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">'
          f'{esc("one argument for each loop-carried variable” —— 源码 for_op.get_body(0).arg(i+1)，code_generator.py:L1002")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('scf-for-arg-layout.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
