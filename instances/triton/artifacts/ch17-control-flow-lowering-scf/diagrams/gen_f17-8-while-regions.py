#!/usr/bin/env python3
"""f17-8-while-regions: while 下降成 scf.while 双区域——before 区判条件+scf.condition
带出 loop-carried,after 区跑体+scf.yield 回传。两区各自持一份块参数(仿 phi-vs-block-arg
的左右对照写法,上下两个"房间"而非左右)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

BOX_W = 560
PAD, TOP = 44, 96
BEFORE_H = 100
AFTER_H = 130
VGAP = 60
RPAD = 210  # 右侧留给回边折线标注,避免文字越界

w = PAD + BOX_W + RPAD + PAD
h = TOP + BEFORE_H + VGAP + AFTER_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#0f172a">{esc("while 下降成 scf.while 双区域:上房间判条件,下房间跑体")}</text>')
L.append(f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("%1:2 = scf.while (%arg2=%0, %arg3=%c0_i32):(tensor<4xi32>, i32) -> (...)  —— 2 个 loop-carried:acc, i")}</text>')

cx = PAD + BOX_W / 2

# before 区(上房间)
by = TOP
L.append(f'<rect x="{PAD}" y="{by}" width="{BOX_W}" height="{BEFORE_H}" rx="12" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.8"/>')
L.append(f'<text x="{cx}" y="{by+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1d4ed8">{esc("before 区(条件区)—— 块参数 %arg2, %arg3")}</text>')
L.append(f'<text x="{cx}" y="{by+48}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" fill="#0f172a">{esc("%5 = arith.cmpi slt, %arg3, %arg1")}</text>')
L.append(f'<text x="{cx}" y="{by+70}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" font-weight="bold" fill="#1e3a8a">{esc("scf.condition(%5) %arg2, %arg3")}</text>')
L.append(f'<text x="{cx}" y="{by+90}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("为真则把 loop-carried 带过门,进 after 区;为假则跳出整个 while")}</text>')

# 箭头 before -> after
ay = by + BEFORE_H + VGAP
L.append(f'<line x1="{cx}" y1="{by+BEFORE_H}" x2="{cx}" y2="{ay-6}" '
          'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{cx+14}" y="{(by+BEFORE_H+ay)/2}" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc("scf.condition 为真,带 loop-carried 过门")}</text>')

# after 区(下房间)
L.append(f'<rect x="{PAD}" y="{ay}" width="{BOX_W}" height="{AFTER_H}" rx="12" '
          'fill="#dcfce7" stroke="#15803d" stroke-width="1.8"/>')
L.append(f'<text x="{cx}" y="{ay+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#15803d">{esc("after 区(循环体区)—— 块参数 %arg2, %arg3(同名,独立一份)")}</text>')
L.append(f'<text x="{cx}" y="{ay+48}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" fill="#0f172a">{esc("%12 = arith.addi %arg2, ...   (acc 累加)")}</text>')
L.append(f'<text x="{cx}" y="{ay+68}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" fill="#0f172a">{esc("%19 = arith.addi %arg3, %c1_i32   (i 自增)")}</text>')
L.append(f'<text x="{cx}" y="{ay+90}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" font-weight="bold" fill="#15803d">{esc("scf.yield %12, %19")}</text>')
L.append(f'<text x="{cx}" y="{ay+112}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("跑完体,把新值 yield 回 before 区,再判一次条件")}</text>')

# 回边(after -> before,虚线,表示循环回去)
loop_x = PAD + BOX_W + 40
L.append(f'<path d="M {PAD+BOX_W},{ay+AFTER_H/2} L {loop_x},{ay+AFTER_H/2} '
          f'L {loop_x},{by+BEFORE_H/2} L {PAD+BOX_W},{by+BEFORE_H/2}" '
          'fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4" marker-end="url(#a)"/>')
L.append(f'<text x="{loop_x+8}" y="{(by+BEFORE_H/2+ay+AFTER_H/2)/2 - 8}" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("每轮回")}</text>')
L.append(f'<text x="{loop_x+8}" y="{(by+BEFORE_H/2+ay+AFTER_H/2)/2 + 10}" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("before 区重判")}</text>')

foot_y = ay + AFTER_H + 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("before/after 各自 create_block_with_parent 绑块参数;对照 for:")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("while 的诱导逻辑全在 before 区用户条件里,无 poison/负步长那套")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+44}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("traces/ch17_traces.json -> ir.K5_while(scf.while=1, scf.condition=1, scf.yield=1)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+62}" font-family="sans-serif" font-size="10.5" '
          f'fill="#94a3b8">{esc("code_generator.py:L847-L875")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-8-while-regions.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
