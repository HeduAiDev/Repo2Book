#!/usr/bin/env python3
"""state-machine 模板:状态机/生命周期。主线横排 + 分支态下挂,转移边带触发条件标签。
改造点:CHAIN(主线态)、SIDE(分支态: (挂在哪个主线态下, 名字, 去边标签, 回边标签))。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CHAIN = ["WAITING", "RUNNING", "FINISHED"]
CHAIN_LBL = ["schedule()", "全部 token 生成完"]          # CHAIN 相邻边标签
SIDE = [("RUNNING", "PREEMPTED", "块不足,LIFO 换出", "重新调度")]
BOX_W, BOX_H, HGAP, PAD, TOP, SIDE_DY = 150, 46, 120, 50, 90, 120
w = PAD * 2 + len(CHAIN) * BOX_W + (len(CHAIN) - 1) * HGAP
h = TOP + BOX_H + SIDE_DY + BOX_H + PAD
X = {s: (PAD + i * (BOX_W + HGAP), TOP) for i, s in enumerate(CHAIN)}
for anchor, name, _, _ in SIDE:
    X[name] = (X[anchor][0], TOP + BOX_H + SIDE_DY)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for name, (x, y) in X.items():
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="22" '
             'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" '
             f'fill="#0c4a6e">{esc(name)}</text>')
for i in range(len(CHAIN) - 1):  # 主线转移:右边缘 → 左边缘
    (x1, y1), (x2, y2) = X[CHAIN[i]], X[CHAIN[i + 1]]
    ay = y1 + BOX_H / 2
    L.append(f'<line x1="{x1+BOX_W}" y1="{ay}" x2="{x2}" y2="{ay}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+BOX_W+x2)/2}" y="{ay-8}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" fill="#334155">{esc(CHAIN_LBL[i])}</text>')
for anchor, name, down_lbl, up_lbl in SIDE:  # 分支:双向竖边,左右错开避免重叠
    (ax, ay), (sx, sy) = X[anchor], X[name]
    xl, xr = ax + BOX_W * 0.3, ax + BOX_W * 0.7
    L.append(f'<line x1="{xl}" y1="{ay+BOX_H}" x2="{xl}" y2="{sy}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<line x1="{xr}" y1="{sy}" x2="{xr}" y2="{ay+BOX_H}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    my = (ay + BOX_H + sy) / 2
    L.append(f'<text x="{xl-8}" y="{my}" text-anchor="end" font-family="sans-serif" '
             f'font-size="11" fill="#334155">{esc(down_lbl)}</text>')
    L.append(f'<text x="{xr+8}" y="{my}" font-family="sans-serif" '
             f'font-size="11" fill="#334155">{esc(up_lbl)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-state-machine.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
