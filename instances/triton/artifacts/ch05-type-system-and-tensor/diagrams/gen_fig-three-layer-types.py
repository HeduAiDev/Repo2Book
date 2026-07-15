#!/usr/bin/env python3
"""layout 模板：三层类型体系——dtype 是最里层标量，pointer_type / block_type
各裹一层 element_ty（都=dtype），block_type 再加 shape。三者都实现 to_ir，
逐层下降到 builder.get_*_ty。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, PAD, TOP = 1180, 40, 78
CORE_W, CORE_H = 220, 90
WRAP_W, WRAP_H = 330, 118
IR_W, IR_H = 300, 60

core_x = PAD
core_y = TOP + 70

wrap_y = TOP + 40
ptr_x = core_x + CORE_W + 110
blk_x = ptr_x
blk_y = wrap_y + WRAP_H + 56

ir_x = ptr_x + WRAP_W + 110
ptr_ir_y = wrap_y + (WRAP_H - IR_H) / 2
blk_ir_y = blk_y + (WRAP_H - IR_H) / 2

W = ir_x + IR_W + PAD
H = blk_y + WRAP_H + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{TOP-46}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("三层类型体系：dtype 被 pointer_type / block_type 各包一层 element_ty")}</text>',
     f'<text x="{PAD}" y="{TOP-22}" font-family="sans-serif" font-size="13" '
     f'fill="#64748b">{esc("三者都实现 to_ir()，逐层下降到 builder.get_*_ty —— 内层先降，外层再套壳")}</text>']

# 核心 dtype box（最里层）
core_cy = core_y + CORE_H / 2
L.append(f'<rect x="{core_x}" y="{core_y}" width="{CORE_W}" height="{CORE_H}" rx="10" '
         f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{core_x+CORE_W/2}" y="{core_y+28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" '
         f'fill="#1e3a8a">{esc("dtype")}</text>')
L.append(f'<text x="{core_x+CORE_W/2}" y="{core_y+50}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#1e40af">'
         f'{esc("标量类型，例：fp32")}</text>')
L.append(f'<text x="{core_x+CORE_W/2}" y="{core_y+70}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#1e40af">'
         f'{esc("primitive_bitwidth = 32")}</text>')

# pointer_type box（第二层，裹 element_ty）
L.append(f'<rect x="{ptr_x}" y="{wrap_y}" width="{WRAP_W}" height="{WRAP_H}" rx="10" '
         f'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{ptr_x+WRAP_W/2}" y="{wrap_y+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" '
         f'fill="#14532d">{esc("pointer_type（第二层）")}</text>')
L.append(f'<text x="{ptr_x+20}" y="{wrap_y+50}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc("element_ty = dtype（如 fp32）")}</text>')
L.append(f'<text x="{ptr_x+20}" y="{wrap_y+72}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc("address_space（默认）= 1")}</text>')
L.append(f'<text x="{ptr_x+20}" y="{wrap_y+94}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc("to_ir() → builder.get_ptr_ty(...)")}</text>')

# block_type box（第三层，裹 element_ty + shape）
L.append(f'<rect x="{blk_x}" y="{blk_y}" width="{WRAP_W}" height="{WRAP_H}" rx="10" '
         f'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
L.append(f'<text x="{blk_x+WRAP_W/2}" y="{blk_y+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" '
         f'fill="#78350f">{esc("block_type（第三层）")}</text>')
L.append(f'<text x="{blk_x+20}" y="{blk_y+50}" font-family="sans-serif" font-size="12" '
         f'fill="#92400e">{esc("element_ty = dtype（如 fp32）+ shape")}</text>')
L.append(f'<text x="{blk_x+20}" y="{blk_y+72}" font-family="sans-serif" font-size="12" '
         f'fill="#92400e">{esc("validate_block_shape：numel ≤ 1048576")}</text>')
L.append(f'<text x="{blk_x+20}" y="{blk_y+94}" font-family="sans-serif" font-size="12" '
         f'fill="#92400e">{esc(".scalar 属性 → element_ty")}</text>')

# core -> ptr / blk 箭头
L.append(f'<line x1="{core_x+CORE_W}" y1="{core_cy}" x2="{ptr_x-8}" y2="{wrap_y+WRAP_H/2}" '
         'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<line x1="{core_x+CORE_W}" y1="{core_cy}" x2="{blk_x-8}" y2="{blk_y+WRAP_H/2}" '
         'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# ptr -> ir box
ptr_ir_x = ir_x
L.append(f'<rect x="{ptr_ir_x}" y="{ptr_ir_y}" width="{IR_W}" height="{IR_H}" rx="8" '
         f'fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{ptr_ir_x+IR_W/2}" y="{ptr_ir_y+24}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" '
         f'fill="#0f172a">{esc("builder.get_ptr_ty(elem, addr_space)")}</text>')
L.append(f'<text x="{ptr_ir_x+IR_W/2}" y="{ptr_ir_y+44}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#475569">'
         f'{esc("IR 指针类型（element_ty 先递归下降）")}</text>')
L.append(f'<line x1="{ptr_x+WRAP_W}" y1="{wrap_y+WRAP_H/2}" x2="{ptr_ir_x-8}" y2="{ptr_ir_y+IR_H/2}" '
         'stroke="#15803d" stroke-width="1.6" marker-end="url(#a)"/>')

# blk -> ir box
blk_ir_x = ir_x
L.append(f'<rect x="{blk_ir_x}" y="{blk_ir_y}" width="{IR_W}" height="{IR_H}" rx="8" '
         f'fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{blk_ir_x+IR_W/2}" y="{blk_ir_y+24}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" '
         f'fill="#0f172a">{esc("builder.get_block_ty(elem, shape)")}</text>')
L.append(f'<text x="{blk_ir_x+IR_W/2}" y="{blk_ir_y+44}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#475569">'
         f'{esc("IR 块类型（shape 已通过 numel 校验）")}</text>')
L.append(f'<line x1="{blk_x+WRAP_W}" y1="{blk_y+WRAP_H/2}" x2="{blk_ir_x-8}" y2="{blk_ir_y+IR_H/2}" '
         'stroke="#b45309" stroke-width="1.6" marker-end="url(#a)"/>')

# 图例（>2 语义色）
ly = H - 34
legend = [("#dbeafe", "#1d4ed8", "dtype 标量"), ("#dcfce7", "#15803d", "pointer_type"),
          ("#fef3c7", "#b45309", "block_type"), ("#f1f5f9", "#475569", "to_ir() 下降结果")]
lx = PAD
for fill, stroke, label in legend:
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+22}" y="{ly+13}" font-family="sans-serif" font-size="12" '
             f'fill="#334155">{esc(label)}</text>')
    lx += 20 + 12 * len(label) + 30

L.append('</svg>')
out = Path(__file__).with_name("fig-three-layer-types.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
