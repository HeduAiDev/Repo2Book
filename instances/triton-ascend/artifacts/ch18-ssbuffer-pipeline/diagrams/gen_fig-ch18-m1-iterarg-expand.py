#!/usr/bin/env python3
"""before-after 模板(改造:支持多个高亮下标):double-buffer 变换前后 scf.for
iterArgs 布局对比。左panel=变换前(2 个 iterArg),右panel=变换后 N=2(5 个),
新增的 3 个(buf1/frontCnt/postCnt)高亮标黄。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
NEW_BG, NEW_FG = "#fef3c7", "#b45309"
OLD_BG, OLD_FG = "#e2e8f0", "#334155"

TITLE = "scf.for 迭代参数(iterArgs)扩容:double-buffer 的空间代价物化在这里"
SUB = "DAGSSBuffer.cpp:L4528-L4547 —— 每个被多缓冲的 dep,追加 (N-1) 份 buffer 副本 + 2 个计数器"

PANELS = [
    ("变换前(2 个 iterArg)", ["%acc", "%buf"], []),
    ("变换后 N=2(5 个 iterArg)", ["%acc", "%buf0", "%buf1", "%frontCnt", "%postCnt"], [2, 3, 4]),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 230, 42, 20, 300, 44, 130

max_steps = max(len(steps) for _, steps, _ in PANELS)
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + max_steps * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB)}</text>']

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="{INK}">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        is_hot = i in hot
        bg, fg = (NEW_BG, NEW_FG) if is_hot else (OLD_BG, OLD_FG)
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{bg}" stroke="{fg}" stroke-width="{2.5 if is_hot else 1.2}"/>')
        L.append(f'<text x="{cx-BOX_W/2+16}" y="{y+BOX_H/2+5}" text-anchor="start" '
                 f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
                 f'fill="{fg}">{esc(step)}</text>')
        L.append(f'<text x="{cx+BOX_W/2-14}" y="{y+BOX_H/2+5}" text-anchor="end" '
                 f'font-family="sans-serif" font-size="11" fill="{fg}">'
                 f'{"新增" if is_hot else "原有"}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    total_y = TOP + len(steps) * (BOX_H + VGAP)
    L.append(f'<text x="{cx}" y="{total_y+8}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{INK}">iterArg 总数 = {len(steps)}</text>')

# 中间大箭头 + 增量标注
mid_top = TOP + (max_steps * (BOX_H + VGAP) - VGAP) / 2 - 10
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{mid_top}" x2="{PAD+PANEL_W+80}" y2="{mid_top}" '
         'stroke="#b45309" stroke-width="3" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{mid_top-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#b45309">+3</text>')

foot_y = TOP + max_steps * (BOX_H + VGAP) + 60
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
         f'fill="{INK}">每 dep 新增 iterArg = (N-1) 份 buffer 副本 + 2 个计数器 = (2-1)+2 = 3</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
         f'fill="{GRAY}">黄框=本次新增(buffer 副本 / frontCnt / postCnt);灰框=原有 iterArg 原样保留</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch18-m1-iterarg-expand.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
