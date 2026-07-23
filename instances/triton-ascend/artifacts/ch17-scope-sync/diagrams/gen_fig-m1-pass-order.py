#!/usr/bin/env python3
"""flow 模板:add_auto_scheduling 下三个 pass 的固定顺序,及其原因(sync 先跑,scope 后切)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
BLUE = "#3b82f6"
BLUE_DK = "#1e3a5f"

TITLE = "add_auto_scheduling 的 pass 顺序(compiler.py:L123-127)"
SUBTITLE = "先在扁平 IR 上同步,再切 scope——顺序颠倒会让「找跨核边」跨 region,复杂得多"

STEPS = [
    ("add_dag_sync", "compiler.py:L123", "扁平 IR 上插 set/wait + 搬运"),
    ("add_dag_scope", "compiler.py:L124", "切成 AIC/AIV 两个 scope"),
    ("add_dag_ssbuffer", "compiler.py:L127", "收尾:静态共享缓冲分配"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 340, 92, 70, 40, 110
W = PAD * 2 + BOX_W * len(STEPS) + GAP * (len(STEPS) - 1)
H = TOP + BOX_H + 156

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">'
     f'{esc(SUBTITLE)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(len(STEPS))]
for i, (name, loc, desc) in enumerate(STEPS):
    x = xs_[i]
    y = TOP
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="#dbeafe" stroke="{BLUE_DK}" stroke-width="1.6"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="{BLUE_DK}">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+53}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="{GRAY}">{esc(loc)}</text>')
    # desc below box
    L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" fill="{INK}">{esc(desc)}</text>')
    if i < len(STEPS) - 1:
        ax1 = x + BOX_W
        ax2 = xs_[i + 1]
        ay = y + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-6}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# step number badges (reading order)
for i, x in enumerate(xs_):
    cx, cy = x + 20, TOP - 16
    L.append(f'<circle cx="{cx}" cy="{cy}" r="13" fill="{BLUE}"/>')
    L.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="white">{i+1}</text>')

cap_y = TOP + BOX_H + 60
CAP = "先在线性数据流上找跨核边插 set/wait+搬运最直观；切 scope 会把 cube/vector 拆进两个 region，再找跨核边要跨 region——所以 DAGSync 必须先跑。"
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP)}</text>')
CAP2 = "本图仅标 3 个 pass 调用行(L123/L124/L127)；完整 if 判断块见正文 compiler.py:L122-L129。"
L.append(f'<text x="{PAD}" y="{cap_y+24}" font-family="sans-serif" font-size="12" '
          f'fill="{GRAY}">{esc(CAP2)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m1-pass-order.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
