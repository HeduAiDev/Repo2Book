#!/usr/bin/env python3
"""f17-7-range-attrs: tl.range(...,num_stages=3,loop_unroll_factor=2) 追踪期即把两个
流水线意图挂成 scf.for 的属性——前瞻第 29/30 章 pass 的输入。before-after:左=源码调用,
右=挂上属性的 scf.for(带 tt.num_stages/tt.loop_unroll_factor)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

BOX_W, PANEL_W, PAD, TOP = 320, 400, 44, 96
BOX_H = 50
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + 130 + 50 + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="15.5" font-weight="bold" '
          f'fill="#0f172a">{esc("tl.range 的流水线参数,追踪期就钉成 scf.for 的属性——第 29/30 章 pass 的输入")}</text>')

# 左面板:源码
lx = PAD
L.append(f'<text x="{lx+PANEL_W/2}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">{esc("源码:tl.range(...)")}</text>')
src_lines = [
    "for i in tl.range(",
    "    0, N,",
    "    num_stages=3,",
    "    loop_unroll_factor=2):",
]
cy = TOP
box_h_src = (len(src_lines)) * 22 + 20
L.append(f'<rect x="{lx}" y="{cy}" width="{PANEL_W}" height="{box_h_src}" rx="8" '
          'fill="#e2e8f0" stroke="#64748b" stroke-width="1.4"/>')
for k, line in enumerate(src_lines):
    L.append(f'<text x="{lx+24}" y="{cy+22+k*22}" font-family="monospace" '
              f'font-size="13" fill="#0f172a">{esc(line)}</text>')

# 箭头 下降
mid_y = cy + box_h_src / 2
L.append(f'<line x1="{lx+PANEL_W+10}" y1="{mid_y}" x2="{lx+PANEL_W+90}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{lx+PANEL_W+50}" y="{mid_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#b45309">{esc("追踪期下降")}</text>')

# 右面板:IR
rx = lx + PANEL_W + 100
L.append(f'<text x="{rx+PANEL_W/2}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">{esc("下降后:scf.for 挂上两个属性")}</text>')
ir_lines = [
    "%5 = scf.for %arg2 = %1 to %2",
    "    step %3 iter_args(...)",
    "  { ... }",
    "{ tt.loop_unroll_factor = 2 : i32,",
    "  tt.num_stages = 3 : i32 }",
]
box_h_ir = len(ir_lines) * 22 + 20
L.append(f'<rect x="{rx}" y="{cy}" width="{PANEL_W}" height="{box_h_ir}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.6"/>')
for k, line in enumerate(ir_lines):
    hl = (k >= 3)
    fill = "#b45309" if hl else "#0f172a"
    fw = 'font-weight="bold" ' if hl else ''
    L.append(f'<text x="{rx+24}" y="{cy+22+k*22}" font-family="monospace" '
              f'font-size="12.5" fill="{fill}" {fw}>{esc(line)}</text>')
# 高亮框圈住属性两行(k=3,4 的文本基线分别在 cy+88 / cy+110)
attr_y = cy + 22 + 3*22 - 18
L.append(f'<rect x="{rx+16}" y="{attr_y}" width="{PANEL_W-32}" height="50" rx="6" '
          'fill="none" stroke="#b45309" stroke-width="1.8" stroke-dasharray="5,3"/>')

foot_y = cy + max(box_h_src, box_h_ir) + 50
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("op_counts.K1_for_range_attrs: tt.num_stages 计数=1, tt.loop_unroll_factor 计数=1")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("挂载点 code_generator.py:L991-L994；num_stages/loop_unroll_factor 由 tl.range 构造记下(language/core.py:L2570-L2582)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+42}" font-family="sans-serif" font-size="11" '
          f'fill="#94a3b8">{esc("traces/ch17_traces.json -> ir.K1_for_range_attrs")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-7-range-attrs.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
