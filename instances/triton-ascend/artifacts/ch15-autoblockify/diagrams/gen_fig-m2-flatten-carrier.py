#!/usr/bin/env python3
"""flow 模板:m2 网格拍平 + blockifiedId 载体构造(worked example:3x2x1 网格,第0号物理块,size=5)。
主链纵向 5 个框 + 侧支 mask 框合流进最终载体框。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

MAIN = [
    ("三维网格", "numX=3, numY=2, numZ=1"),
    ("拍平总块数 G", "G = numX·numY·numZ = 6"),
    ("第0号物理块 → logicalBlockId", "idX=idY=idZ=0 → logicalBlockId = 0"),
    ("连号折叠 blockifiedId", "splat(0) + range(0,5) = [0,1,2,3,4]"),
    ("双输入载体", "UnrealizedConversionCastOp(blockifiedId, mask)"),
]
ARROW_LABELS = [
    "拍平 (AutoBlockify.cpp:L204-205)",
    "定位物理块0 (L217-220)",
    "splat+range(0,size=5) (L222-231)",
    "打包 2 输入 cast (L245-249)",
]
SIDE = ("mask (ori 合成)", "upper ORI lower = [T,T,T,T,T]  (L235-243;honesty: 恒全 True)")

BOX_W, BOX_H = 460, 60
SIDE_W, SIDE_H = 300, 58
VGAP = 58
PAD, TOP = 40, 70
cx = PAD + BOX_W / 2
w = PAD * 2 + BOX_W + SIDE_W + 60
main_y = [TOP + i * (BOX_H + VGAP) for i in range(len(MAIN))]
side_y = main_y[3] + (BOX_H + VGAP) / 2 - SIDE_H / 2
h = main_y[-1] + BOX_H + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-10}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("网格拍平 + 载体构造 — worked example:3×2×1 网格,第 0 号物理块,size=5")}</text>']

for i, (title, sub) in enumerate(MAIN):
    y = main_y[i]
    hl = (i == len(MAIN) - 1)
    L.append(f'<rect x="{PAD}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{"#dbeafe" if hl else "#e2e8f0"}" '
              f'stroke="{"#1d4ed8" if hl else "#64748b"}" stroke-width="{2.2 if hl else 1.4}"/>')
    L.append(f'<text x="{cx}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{cx}" y="{y+44}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" fill="#334155">{esc(sub)}</text>')
    if i < len(MAIN) - 1:
        y1, y2 = y + BOX_H, y + BOX_H + VGAP
        L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-3}" '
                  'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
        L.append(f'<text x="{cx+14}" y="{(y1+y2)/2+4}" font-family="sans-serif" '
                  f'font-size="11.5" fill="#475569">{esc(ARROW_LABELS[i])}</text>')

# side mask box, forked off box index 3 (blockifiedId), merges into box index 4 (cast)
sx = PAD + BOX_W + 60
L.append(f'<rect x="{sx}" y="{side_y}" width="{SIDE_W}" height="{SIDE_H}" rx="9" '
          f'fill="#fef3c7" stroke="#b45309" stroke-width="1.6"/>')
L.append(f'<text x="{sx+SIDE_W/2}" y="{side_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#92400e">{esc(SIDE[0])}</text>')
L.append(f'<text x="{sx+SIDE_W/2}" y="{side_y+41}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#78350f">{esc(SIDE[1])}</text>')
# fork arrow: blockifiedId box right edge -> side box left edge
fork_y = main_y[3] + BOX_H / 2
L.append(f'<line x1="{PAD+BOX_W}" y1="{fork_y}" x2="{sx-4}" y2="{side_y+SIDE_H/2}" '
          'stroke="#b45309" stroke-width="1.6" marker-end="url(#b)"/>')
L.append(f'<text x="{PAD+BOX_W+8}" y="{fork_y-8}" font-family="sans-serif" font-size="11" '
          f'fill="#92400e">{esc("upper/lower 谓词均由 blockifiedId 求出")}</text>')
# merge arrow: side box bottom -> cast box top-right area
merge_y = main_y[4]
L.append(f'<line x1="{sx+SIDE_W/2}" y1="{side_y+SIDE_H}" x2="{PAD+BOX_W-60}" y2="{merge_y-4}" '
          'stroke="#b45309" stroke-width="1.6" marker-end="url(#b)"/>')

L.append(f'<text x="{w/2}" y="{h-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#64748b">'
          f'{esc("blockifiedId 与 mask 一起被双输入 cast 包成——类型不变、语义已批处理的载体,交给 PropagateUnrealizedCastDown 下推")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m2-flatten-carrier.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f"wrote {out}")
