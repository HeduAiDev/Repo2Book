#!/usr/bin/env python3
"""before-after 模板:FlashAttention -> FlashAttention-2 三处改进对照。
同构双面板,高亮差异步骤;右侧标 warp 分工与整体加速。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "FlashAttention → FlashAttention-2 — 三处工程改进"
PANELS = [
    ("FA(v1)", ["外层循环:遍历 KV 块", "内层每步除 l 归一化 O", "存 (m, l) 两个标量", "warp 分工:split-K"], [0, 1, 2, 3]),
    ("FA-2", ["外层循环:遍历 Q 行块", "收尾只除一次 l(非每步)", "只存 L = m+log(l) 一个标量", "warp 分工:split-Q"], [0, 1, 2, 3]),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP, GAP = 270, 46, 26, 330, 40, 90, 230
w = PAD * 2 + PANEL_W * 2 + GAP
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + PAD + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>']

for p, (title, steps, hot_idxs) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + GAP)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = i in hot_idxs
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                  f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 中间大箭头,横向对齐每一行,标注差异要点(长注释拆两行,避免与两侧方框重叠)
mid_labels = [
    ["序列并行 ↑ occupancy"],
    ["省 non-matmul FLOP", "(A100 上贵 ~16×)"],
    ["省一半标量存储"],
    ["免 warp 间", "shared-memory 通信"],
]
for i, lines in enumerate(mid_labels):
    y = TOP + i * (BOX_H + VGAP) + BOX_H / 2
    x1 = PAD + PANEL_W + 10
    x2 = PAD + PANEL_W + GAP - 10
    n = len(lines)
    text_top = y - 12 - (n - 1) * 13
    for k, line in enumerate(lines):
        L.append(f'<text x="{(x1+x2)/2}" y="{text_top+k*13}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#92400e">{esc(line)}</text>')
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
              'stroke="#d97706" stroke-width="2.5" marker-end="url(#b)"/>')

box_y = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 10
L.append(f'<rect x="{PAD}" y="{box_y}" width="{w-2*PAD}" height="36" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
FINAL = "三处改进综合:FA-2 相对 FA(v1)约快 2×"
L.append(f'<text x="{w/2}" y="{box_y+23}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#1e3a8a">{esc(FINAL)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-5-fa1-vs-fa2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
