#!/usr/bin/env python3
"""fig-remove-layout-thrice (flow/timeline 模板)
make_ttgir 里 RemoveLayoutConversions 出现三次(L225/L228/L243),
夹在 coalesce/plan_cta、accelerate_matmul、pipeline/prefetch 之间——
每次结构变换引入新 convert,故要反复消一轮才收敛。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD = 50
TOP = 170
w = 1300
h = 380

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="42" font-family="sans-serif" font-size="18.5" '
          f'font-weight="bold" fill="#0f172a">{esc("make_ttgir 里 RemoveLayoutConversions 跑三次:每次结构变换后消一轮")}</text>')
L.append(f'<text x="{PAD}" y="66" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("third_party/nvidia/backend/compiler.py:L225-L243——布局优化是个多轮不动点过程,不是一次到位")}</text>')

# 时间线节点:交替 结构变换(灰) / RemoveLayoutConversions(蓝,高亮编号)
NODES = [
    ("coalesce\nplan_cta", None, "struct"),
    ("RemoveLayout\nConversions #1", "L225", "pass"),
    ("accelerate\n_matmul", "L227,引入 3 类新 convert", "struct"),
    ("RemoveLayout\nConversions #2", "L228,消上一步引入的 convert", "pass"),
    ("pipeline\nprefetch", None, "struct"),
    ("RemoveLayout\nConversions #3", "L243", "pass"),
]

n = len(NODES)
NODE_W = (w - 2 * PAD) / n
cy = TOP + 60

for i, (name, sub, kind) in enumerate(NODES):
    cx = PAD + NODE_W * i + NODE_W / 2
    is_pass = kind == "pass"
    box_w = NODE_W - 24
    box_h = 92 if sub else 56
    bx = cx - box_w / 2
    by = cy - box_h / 2
    fill, stroke, txtcol = ("#dbeafe", "#1d4ed8", "#1e3a5f") if is_pass else ("#f1f5f9", "#64748b", "#334155")
    L.append(f'<rect x="{bx}" y="{by}" width="{box_w}" height="{box_h}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if is_pass else 1.4}"/>')
    lines = name.split("\n")
    fsize = 11.5 if is_pass else 12.5
    y0 = by + 24 if not sub else by + 22
    for li, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{y0+li*17}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fsize}" font-weight="bold" fill="{txtcol}">{esc(ln)}</text>')
    if sub:
        sy = y0 + len(lines) * 17 + 20
        L.append(f'<text x="{cx}" y="{sy}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="{"#3730a3" if is_pass else "#64748b"}">{esc(sub)}</text>')
    if i < n - 1:
        next_cx = PAD + NODE_W * (i + 1) + NODE_W / 2
        y_arrow = cy
        L.append(f'<line x1="{cx+box_w/2}" y1="{y_arrow}" x2="{next_cx-box_w/2}" y2="{y_arrow}" '
                  'stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

# 图例
leg_y = cy + 90
L.append(f'<rect x="{PAD}" y="{leg_y}" width="18" height="18" rx="3" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{PAD+26}" y="{leg_y+14}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("RemoveLayoutConversions(同一个 pass,跑 3 次)")}</text>')
L.append(f'<rect x="{PAD+380}" y="{leg_y}" width="18" height="18" rx="3" fill="#f1f5f9" stroke="#64748b" stroke-width="1.4"/>')
L.append(f'<text x="{PAD+406}" y="{leg_y+14}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("引入新 convert 的结构性变换")}</text>')

foot_y = leg_y + 50
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("出现次数 = 3:每次结构性变换(coalesce/plan_cta、accelerate_matmul、pipeline/prefetch)都会引入新 convert_layout,单趟消不净。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-remove-layout-thrice.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
