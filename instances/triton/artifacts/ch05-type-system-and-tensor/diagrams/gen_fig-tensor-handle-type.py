#!/usr/bin/env python3
"""layout 模板：tensor = (handle, type) 二元组解剖图。中心两个真字段，
外围三个派生只读视图（shape/numel/dtype），用虚线框+浅色区分"派生"与"真字段"。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 1220
PAD, TOP = 40, 100

CORE_W, CORE_H = 340, 190
core_x = PAD + 60
core_y = TOP + 40

TYPE_W, TYPE_H = 320, 100
type_x = core_x + CORE_W + 130
type_y = core_y + 20

DER_W, DER_H = 300, 78
der_x = type_x + TYPE_W + 130
der_gap = 26
der_y0 = TOP - 10

H = max(core_y + CORE_H, der_y0 + 3 * DER_H + 2 * der_gap) + 90
W = der_x + DER_W + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{TOP-58}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("tensor 解剖：只有 2 个真字段，其余全是派生只读视图")}</text>',
     f'<text x="{PAD}" y="{TOP-34}" font-family="sans-serif" font-size="13" '
     f'fill="#64748b">'
     f'{esc("数值本身不在这张图里——它在 IR / 寄存器里；tensor 只是指向它的把手")}</text>']

# 核心 tensor box（真字段 ×2）
L.append(f'<rect x="{core_x}" y="{core_y}" width="{CORE_W}" height="{CORE_H}" rx="10" '
         f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{core_x+CORE_W/2}" y="{core_y+30}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="15" font-weight="bold" '
         f'fill="#1e3a8a">{esc("tensor 对象（真字段 ×2）")}</text>')
L.append(f'<rect x="{core_x+20}" y="{core_y+50}" width="{CORE_W-40}" height="{56}" rx="6" '
         f'fill="white" stroke="#1d4ed8" stroke-width="1.2"/>')
L.append(f'<text x="{core_x+40}" y="{core_y+74}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#1e3a8a">{esc("handle")}</text>')
L.append(f'<text x="{core_x+40}" y="{core_y+96}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">{esc("IR 里某个 SSA 值 —— 真实数值所在")}</text>')
L.append(f'<rect x="{core_x+20}" y="{core_y+118}" width="{CORE_W-40}" height="{56}" rx="6" '
         f'fill="white" stroke="#1d4ed8" stroke-width="1.2"/>')
L.append(f'<text x="{core_x+40}" y="{core_y+142}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#1e3a8a">{esc("type")}</text>')
L.append(f'<text x="{core_x+40}" y="{core_y+164}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">{esc("这个 SSA 值的类型（可以是 block_type）")}</text>')

# type -> block_type 示例框
L.append(f'<rect x="{type_x}" y="{type_y}" width="{TYPE_W}" height="{TYPE_H}" rx="10" '
         f'fill="#f1f5f9" stroke="#475569" stroke-width="1.8"/>')
L.append(f'<text x="{type_x+TYPE_W/2}" y="{type_y+28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
         f'fill="#0f172a">{esc("示例：block_type(fp16, [128, 64])")}</text>')
L.append(f'<text x="{type_x+TYPE_W/2}" y="{type_y+52}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#334155">'
         f'{esc("element_ty = fp16，shape = [128, 64]")}</text>')
L.append(f'<text x="{type_x+TYPE_W/2}" y="{type_y+76}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#64748b">'
         f'{esc("3 个视图都从这个 type 现算，不额外存字段")}</text>')

core_type_y = core_y + 118 + 28
L.append(f'<line x1="{core_x+CORE_W}" y1="{core_type_y}" x2="{type_x-8}" y2="{type_y+TYPE_H/2}" '
         'stroke="#1d4ed8" stroke-width="1.8" marker-end="url(#a)"/>')

# 三个派生视图
DER = [
    ("shape（派生）", "shape = [constexpr(128), constexpr(64)]", "block_type.shape，逐元素裹 constexpr"),
    ("numel（派生）", "numel = 128 × 64 = 8192", "shape 各维之积"),
    ("dtype（派生）", "dtype = type.scalar", "block_type 剥到最里层的标量 dtype"),
]
for i, (name, val, note) in enumerate(DER):
    dy = der_y0 + i * (DER_H + der_gap)
    L.append(f'<rect x="{der_x}" y="{dy}" width="{DER_W}" height="{DER_H}" rx="9" '
             f'fill="#fdf4ff" stroke="#a21caf" stroke-width="1.6" stroke-dasharray="5,4"/>')
    L.append(f'<text x="{der_x+18}" y="{dy+24}" font-family="sans-serif" font-size="13" '
             f'font-weight="bold" fill="#86198f">{esc(name)}</text>')
    L.append(f'<text x="{der_x+18}" y="{dy+46}" font-family="sans-serif" font-size="12" '
             f'fill="#0f172a">{esc(val)}</text>')
    L.append(f'<text x="{der_x+18}" y="{dy+64}" font-family="sans-serif" font-size="10.5" '
             f'fill="#64748b">{esc(note)}</text>')
    L.append(f'<line x1="{type_x+TYPE_W}" y1="{type_y+TYPE_H/2}" x2="{der_x-8}" y2="{dy+DER_H/2}" '
             'stroke="#a21caf" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# 图例
ly = H - 46
legend = [("#dbeafe", "#1d4ed8", "真字段（handle / type）"),
          ("#f1f5f9", "#475569", "type 具体取值示例"),
          ("#fdf4ff", "#a21caf", "派生只读视图（虚线=非独立存储）")]
lx = PAD
for fill, stroke, label in legend:
    dash = ' stroke-dasharray="5,4"' if stroke == "#a21caf" else ''
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" '
             f'stroke="{stroke}"{dash}/>')
    L.append(f'<text x="{lx+22}" y="{ly+13}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')
    lx += 22 + 12 * len(label) + 34

L.append('</svg>')
out = Path(__file__).with_name("fig-tensor-handle-type.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
