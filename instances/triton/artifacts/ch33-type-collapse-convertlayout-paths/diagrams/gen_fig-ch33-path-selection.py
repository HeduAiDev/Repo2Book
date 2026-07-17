#!/usr/bin/env python3
"""fig-ch33-path-selection: flow 模板——block->warp->lane->register 逐维相除,
菱形判据链竖排,每级"参与(非恒等)"分支出右侧结果盒,"不参与"分支继续往下一级。
minimalCvtLayout: lib/Analysis/Utility.cpp:L661。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 1080
PAD = 40
DIA_W, DIA_H = 210, 76
DIA_CX = PAD + 230
LEAF_X = DIA_CX + DIA_W/2 + 90
LEAF_W = 430
TOP = 175
GAP = 130
N_LEVELS = 4
H = TOP + (N_LEVELS - 1) * GAP + DIA_H + 50 + 42 + 40

LEVELS = [
    ("block 维参与?", "跨 CTA,v3.2.0 NYI", "notifyMatchFailure()", "#fee2e2", "#b91c1c",
     "ConvertLayoutOpToLLVM.cpp:L295-L299"),
    ("warp 维参与?", "共享内存往返", "transferWithinBlock()", "#fed7aa", "#c2410c",
     "ConvertLayoutOpToLLVM.cpp:L300-L307"),
    ("lane 维参与?", "warp shuffle(专用实现 TODO,\n暂落共享内存往返)", "transferWithinBlock()", "#fef3c7", "#b45309",
     "ConvertLayoutOpToLLVM.cpp:L308-L318"),
    ("register 维参与?", "纯寄存器重排(最便宜)", "transferWithinThread()", "#dcfce7", "#15803d",
     "ConvertLayoutOpToLLVM.cpp:L319-L322"),
]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f766e"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     'fill="#0f172a">ConvertLayoutOp 选路:相除维序 block-&gt;warp-&gt;lane-&gt;register</text>',
     f'<text x="{PAD}" y="55" font-family="sans-serif" font-size="12" fill="#64748b">'
     '第一个非恒等(参与)维决定四条路径与代价档位(Utility.cpp:L661)</text>']

# 入口(留出与副标题的垂直间距,避免重叠)
entry_y = TOP - 32
L.append(f'<rect x="{DIA_CX-100}" y="{entry_y}" width="200" height="28" rx="14" '
          'fill="#0f766e"/>')
L.append(f'<text x="{DIA_CX}" y="{entry_y+19}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="white">minimalCvtLayout(src,dst)</text>')
L.append(f'<line x1="{DIA_CX}" y1="{entry_y+28}" x2="{DIA_CX}" y2="{TOP-4}" '
          'stroke="#0f766e" stroke-width="2" marker-end="url(#g)"/>')

for i, (q, leaf_title, leaf_sub, bg, stroke, anchor) in enumerate(LEVELS):
    y = TOP + i * GAP
    # 菱形判据
    cx, cy = DIA_CX, y + DIA_H/2
    pts = f"{cx},{y} {cx+DIA_W/2},{cy} {cx},{y+DIA_H} {cx-DIA_W/2},{cy}"
    L.append(f'<polygon points="{pts}" fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#312e81">{esc(q)}</text>')

    # "是(参与)" 分支 -> 右侧结果盒
    leaf_y = cy - 40
    L.append(f'<line x1="{cx+DIA_W/2}" y1="{cy}" x2="{LEAF_X-6}" y2="{cy}" '
              'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<text x="{(cx+DIA_W/2+LEAF_X)/2}" y="{cy-8}" text-anchor="middle" '
              'font-family="sans-serif" font-size="11" font-weight="bold" '
              'fill="#334155">是(参与)</text>')
    L.append(f'<rect x="{LEAF_X}" y="{leaf_y}" width="{LEAF_W}" height="80" rx="8" '
              f'fill="{bg}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{LEAF_X+16}" y="{leaf_y+22}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{stroke}">{esc(leaf_title)}</text>')
    for k, line in enumerate(leaf_sub.split("\n")):
        L.append(f'<text x="{LEAF_X+16}" y="{leaf_y+42+k*15}" font-family="sans-serif" '
                  f'font-size="12" fill="{stroke}">{esc(line)}</text>')
    L.append(f'<text x="{LEAF_X+16}" y="{leaf_y+80-8}" font-family="sans-serif" font-size="10" '
              f'fill="{stroke}">{esc(anchor)}</text>')

    # "否" 分支 -> 下一级(竖直向下),最后一级"否"表示两布局等价
    if i < len(LEVELS) - 1:
        y2 = y + GAP  # 下一级菱形的顶点 y(修复此前误加 DIA_H 导致标签落进下一菱形内部)
        L.append(f'<line x1="{cx}" y1="{y+DIA_H}" x2="{cx}" y2="{y2-4}" '
                  'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
        lab_y = (y + DIA_H + y2) / 2
        L.append(f'<text x="{cx-DIA_W/2-14}" y="{lab_y-4}" text-anchor="end" '
                  'font-family="sans-serif" font-size="11" '
                  'fill="#334155">否(恒等)</text>')
        L.append(f'<text x="{cx-DIA_W/2-14}" y="{lab_y+11}" text-anchor="end" '
                  'font-family="sans-serif" font-size="11" '
                  'fill="#334155">quotient 消去该维</text>')
    else:
        y2 = y + DIA_H + 50
        L.append(f'<line x1="{cx}" y1="{y+DIA_H}" x2="{cx}" y2="{y2-4}" '
                  'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<rect x="{cx-150}" y="{y2}" width="300" height="42" rx="8" '
                  'fill="#e2e8f0" stroke="#475569" stroke-width="1.2"/>')
        L.append(f'<text x="{cx}" y="{y2+27}" text-anchor="middle" font-family="sans-serif" '
                  'font-size="12" fill="#334155">四维全恒等 -&gt; 两布局等价,直接 replaceOp</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch33-path-selection.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
