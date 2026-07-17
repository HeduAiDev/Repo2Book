#!/usr/bin/env python3
"""fig-prefetch-loop-rewrite: before-after 双面板——Prefetch 把『整片 local_load 再 dot』的循环
重写成『沿 K 逐片、下一片搬运与本片 dot 重叠』的循环，iter_args 增量 +2。
数字来源见 explainer.json mechanism prefetch-loop-rewrite。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Prefetch 改写单-dot 循环 —— shared→register 搬运与计算重叠"

BEFORE_STEPS = [
    ("local_load 整片 K=64", False),
    ("dot(a, b, c)  — 累加一次", False),
    ("scf.yield a_buf, b_buf", False),
]
AFTER_STEPS = [
    ("prologue: 预取片0 [0,16)", None),
    ("dot(片0, C_in)  ← 用 iter_arg 里已备好的片0", True),
    ("本轮内 local_load 片1/2/3", False),
    ("dot(片1..3, 累加)  ← 与下一片搬运错峰重叠", True),
    ("尾部预取下轮片0 → yield", False),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 42, 20, 340, 44, 116
GAP = 90
w = PAD * 2 + PANEL_W * 2 + GAP
h = TOP + max(len(BEFORE_STEPS), len(AFTER_STEPS)) * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

PANELS = [
    ("改写前:整片搬运再算(1 个 iter_arg)", BEFORE_STEPS),
    ("改写后:逐片搬运与算重叠(+2 个 iter_arg)", AFTER_STEPS),
]

for p, (title, steps) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + GAP)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e40af">{esc(title)}</text>')
    # scf.for 外框
    body_h = len(steps) * (BOX_H + VGAP) - VGAP
    L.append(f'<rect x="{px-14}" y="{TOP-10}" width="{PANEL_W+28}" height="{body_h+30}" '
              f'rx="10" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,4"/>')
    L.append(f'<text x="{px-6}" y="{TOP+8}" font-family="sans-serif" font-size="11" '
              f'fill="#64748b">scf.for</text>')
    for i, (step, hot) in enumerate(steps):
        y = TOP + 22 + i * (BOX_H + VGAP)
        fill = "#fef3c7" if hot else "#e2e8f0"
        stroke = "#d97706" if hot else "#64748b"
        sw = 2 if hot else 1
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="7" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            y2 = TOP + 22 + (i + 1) * (BOX_H + VGAP)
            col = "#d97706" if hot else "#64748b"
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y2-4}" '
                      f'stroke="{col}" stroke-width="1.5" marker-end="url(#a)"/>')

# 中间总箭头
midy = TOP + 22 + (max(len(BEFORE_STEPS), len(AFTER_STEPS)) * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+GAP-10}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#ah)"/>')
L.append(f'<text x="{PAD+PANEL_W+GAP/2}" y="{midy-12}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" font-weight="bold" '
          'fill="#d97706">Prefetch pass</text>')

# 底部数字条
NUMS = [
    ("prefetchWidth", "16"),
    ("BLOCK_K", "64"),
    ("K 片数", "4"),
    ("iter_args 增量", "+2"),
]
ny = h - 92
L.append(f'<rect x="{PAD}" y="{ny}" width="{w-2*PAD}" height="56" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
seg_w = (w - 2*PAD) / len(NUMS)
for i, (label, val) in enumerate(NUMS):
    cx = PAD + seg_w * i + seg_w / 2
    L.append(f'<text x="{cx}" y="{ny+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="19" font-weight="bold" fill="#1e40af">{esc(val)}</text>')
    L.append(f'<text x="{cx}" y="{ny+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#334155">{esc(label)}</text>')

CAPTION_LINES = [
    "单-dot 循环经 Prefetch 后:local_load 被切成 4 片、错开一拍藏进相邻片的 dot 计算里——",
    "这是软件流水线(见第 29/30 章,重叠 global→shared)之外、专门重叠 shared→register 的第二层旋钮,两级重叠正交叠加。",
]
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{w/2}" y="{h-30+i*17}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-prefetch-loop-rewrite.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
