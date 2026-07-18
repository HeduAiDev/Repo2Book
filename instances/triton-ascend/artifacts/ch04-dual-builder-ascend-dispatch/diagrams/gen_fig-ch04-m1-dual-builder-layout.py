#!/usr/bin/env python3
"""fig-ch04-m1-dual-builder-layout — layout 模板（CodeGenerator 持有两支 builder）。
同一个 CodeGenerator、同一个 MLIR context 上并挂 self.builder(标准 Triton IR)与
self.ascend_builder(ascendnpu_ir_builder,emit 昇腾方言)——两个不同对象，经
setup_unified_builder 反向接线。全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, fs):
    return fs * sum((0.98 if '一' <= c <= '鿿' else 0.58) for c in s)


TITLE = "CodeGenerator 上并挂两支 IR builder"
SUBTITLE = "fork 在同一 MLIR context 上加挂第二支笔——两个不同对象，经 setup_unified_builder 接线"

OUTER_W, OUTER_H = 760, 360
PAD = 40
TOP = 96

BOX_W, BOX_H = 300, 120
GAP = 60

w = OUTER_W + PAD * 2
h = TOP + OUTER_H + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 外层容器：CodeGenerator 实例
outer_x = PAD
outer_y = TOP
L.append(f'<rect x="{outer_x}" y="{outer_y}" width="{OUTER_W}" height="{OUTER_H}" rx="14" '
         f'fill="#f8fafc" stroke="#334155" stroke-width="2" stroke-dasharray="7,5"/>')
L.append(f'<text x="{outer_x+18}" y="{outer_y+28}" font-family="sans-serif" font-size="13.5" '
         f'font-weight="bold" fill="#334155">{esc("CodeGenerator 实例")}</text>')

# 两支 builder
box_y = outer_y + 130
left_x = outer_x + 40
right_x = outer_x + OUTER_W - 40 - BOX_W

L.append(f'<rect x="{left_x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
         f'fill="#dbeafe" stroke="#2563eb" stroke-width="2.2"/>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#1e3a8a">{esc("self.builder")}</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#1e3a8a">{esc("ir.builder(compile_mode=simd/simt)")}</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#1e40af">{esc("emit 标准 Triton IR")}</text>')
L.append(f'<text x="{left_x+BOX_W/2}" y="{box_y+96}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#2563eb" font-weight="bold">{esc("code_generator.py:L215-L219")}</text>')

L.append(f'<rect x="{right_x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
         f'fill="#fef3c7" stroke="#b45309" stroke-width="2.4"/>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#78350f">{esc("self.ascend_builder")}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#78350f">{esc("ascendnpu_ir_builder(context, arch)")}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#92400e">{esc("emit 昇腾方言（hivm/ascend）")}</text>')
L.append(f'<text x="{right_x+BOX_W/2}" y="{box_y+96}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#b45309" font-weight="bold">{esc("code_generator.py:L226")}</text>')

# 共享 context 条
ctx_y = box_y + BOX_H + 34
ctx_x = outer_x + 40
ctx_w = OUTER_W - 80
L.append(f'<rect x="{ctx_x}" y="{ctx_y}" width="{ctx_w}" height="42" rx="8" '
         f'fill="#e2e8f0" stroke="#475569" stroke-width="1.6"/>')
L.append(f'<text x="{ctx_x+ctx_w/2}" y="{ctx_y+27}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#1e293b">'
         f'{esc("共享同一个 MLIR context（same_context = true）")}</text>')

# 两条竖线：builder -> context
for bx in (left_x + BOX_W / 2, right_x + BOX_W / 2):
    L.append(f'<line x1="{bx}" y1="{box_y+BOX_H}" x2="{bx}" y2="{ctx_y}" '
              f'stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')

# 反挂弧线：setup_unified_builder(self.builder, self.ascend_builder)
arc_y = box_y - 52
mid_x = (left_x + BOX_W + right_x) / 2
L.append(f'<path d="M {right_x} {box_y+4} C {mid_x} {arc_y} {mid_x} {arc_y} {left_x+BOX_W} {box_y+4}" '
          f'fill="none" stroke="#b45309" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#b)"/>')
L.append(f'<text x="{mid_x}" y="{arc_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#92400e">'
          f'{esc("setup_unified_builder(self.builder, self.ascend_builder)")}</text>')
L.append(f'<text x="{mid_x}" y="{arc_y+14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#92400e">{esc("反挂：self.builder._ascend_builder = self.ascend_builder（code_generator.py:L228）")}</text>')

foot_y = outer_y + OUTER_H + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("两支笔是两个不同对象（is_second_distinct_builder = true）；tl.* 落 self.builder，al.* 落 self.ascend_builder。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("AST 遍历/符号表全套复用基座——fork 只在这两支笔的构造与接线上做原位改动。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch04-m1-dual-builder-layout.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
