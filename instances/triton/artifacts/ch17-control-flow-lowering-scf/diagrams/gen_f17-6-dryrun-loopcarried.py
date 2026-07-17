#!/usr/bin/env python3
"""f17-6-dryrun-loopcarried: for 下降靠一次 dry-run(建临时块->visit body->erase)
先探 loop-carried=local_defs∩liveins,再正式 create_for_op + 重 visit。单泳道两阶段
时序,第一阶段末尾用红色叉号强调"erase 丢弃,不留痕"。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PHASE1 = [
    ("建临时块", "create_block()"),
    ("visit body", "抄一遍 local_defs"),
    ("erase() 丢弃", "不留痕(仅为侦察)"),
]
PHASE2 = [
    ("loop-carried", "local_defs ∩ liveins"),
    ("create_for_op", "iter_args(%arg3=%0)"),
    ("正式 visit body", "for_op arg(i+1) 接块参数"),
]
BOX_W, BOX_H, HGAP, PAD, TOP = 210, 58, 46, 40, 130
PHASE_VGAP = 56
w = PAD * 2 + 3 * BOX_W + 2 * HGAP
h = TOP + BOX_H * 2 + PHASE_VGAP + 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#0f172a">{esc("for 下降靠一次 dry-run 先探 loop-carried,再正式建 for_op(见第 15 章识别机理,本章落地)")}</text>')

# 阶段标签
L.append(f'<text x="{PAD}" y="{TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#a16207">{esc("阶段一(dry-run,临时块,最终 erase)")}</text>')

X = [PAD + i * (BOX_W + HGAP) for i in range(3)]
Y1 = TOP
for i, (l1, l2) in enumerate(PHASE1):
    x = X[i]
    is_erase = (i == 2)
    fill, stroke = ("#fee2e2", "#b91c1c") if is_erase else ("#fef9c3", "#a16207")
    dash = ' stroke-dasharray="6,4"' if is_erase else ''
    L.append(f'<rect x="{x}" y="{Y1}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.7"{dash}/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{Y1+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(l1)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{Y1+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(l2)}</text>')
    if i < 2:
        L.append(f'<line x1="{x+BOX_W}" y1="{Y1+BOX_H/2}" x2="{X[i+1]}" y2="{Y1+BOX_H/2}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# 转折箭头:phase1 -> phase2(垂直,红色虚线表 erase 后重新开始)
Y2 = Y1 + BOX_H + PHASE_VGAP
turn_x = X[2] + BOX_W / 2
L.append(f'<path d="M {turn_x},{Y1+BOX_H} L {turn_x},{Y1+BOX_H+PHASE_VGAP/2} '
          f'L {X[0]+BOX_W/2},{Y1+BOX_H+PHASE_VGAP/2} L {X[0]+BOX_W/2},{Y2}" '
          'fill="none" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#r)"/>')
L.append(f'<text x="{(turn_x+X[0]+BOX_W/2)/2}" y="{Y1+BOX_H+PHASE_VGAP/2-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#b91c1c">{esc("拿到 loop-carried 名单后重来")}</text>')

L.append(f'<text x="{PAD}" y="{Y2-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#15803d">{esc("阶段二(正式建 for_op,真实生成体)")}</text>')

for i, (l1, l2) in enumerate(PHASE2):
    x = X[i]
    L.append(f'<rect x="{x}" y="{Y2}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              'fill="#dcfce7" stroke="#15803d" stroke-width="1.7"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{Y2+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(l1)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{Y2+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(l2)}</text>')
    if i < 2:
        L.append(f'<line x1="{x+BOX_W}" y1="{Y2+BOX_H/2}" x2="{X[i+1]}" y2="{Y2+BOX_H/2}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

foot_y = Y2 + BOX_H + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("本例(acc 累加循环):acc 为唯一 loop-carried -> scf.for iter_args(%arg3 = %0)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("code_generator.py:L964-L971(dry-run) / L978-L986(交集) / L1001-L1003(正式 visit)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-6-dryrun-loopcarried.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
