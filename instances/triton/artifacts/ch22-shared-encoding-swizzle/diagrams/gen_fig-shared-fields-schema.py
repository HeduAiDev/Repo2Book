#!/usr/bin/env python3
"""layout 模板:SharedEncodingAttr 六字段说明书。
三组语义色:swizzle 标量(蓝,3 个)/ 轴序(绿,1 个)/ CTA·Hopper(橙,2 个)。
底部一条 print 输出样例(真实 IR dump 形态),坐实『这就是你在 TRITON_KERNEL_DUMP 里看到的东西』。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# (字段名, 一句话职责, 组key)
FIELDS = [
    ("vec",              "向量化粒度", "swizzle"),
    ("perPhase",         "共相位行数", "swizzle"),
    ("maxPhase",         "错位图案周期", "swizzle"),
    ("order",             "最快变化轴", "axis"),
    ("CTALayout",         "跨 CTA 怎么切", "cta"),
    ("hasLeadingOffset",  "Hopper GMMA 开关", "cta"),
]
GROUP_COLOR = {
    "swizzle": ("#dbeafe", "#3b82f6", "#1e3a5f"),
    "axis":    ("#dcfce7", "#16a34a", "#14532d"),
    "cta":     ("#fef3c7", "#d97706", "#78350f"),
}
LEGEND = [
    ("swizzle", "swizzle 标量(3):定义 xor-swizzle"),
    ("axis",    "轴序(1):最快变化维"),
    ("cta",     "CTA·Hopper(2):跨 CTA 切分 / GMMA 开关"),
]

COLS = 3
BOX_W, BOX_H, HGAP, VGAP = 220, 100, 24, 24
PAD, TOP = 40, 78
rows = [FIELDS[i:i + COLS] for i in range(0, len(FIELDS), COLS)]

grid_w = COLS * BOX_W + (COLS - 1) * HGAP
grid_h = len(rows) * BOX_H + (len(rows) - 1) * VGAP

BANNER_TOP_GAP = 36
BANNER_H = 68
LEGEND_H = 26

w = PAD * 2 + grid_w
h = TOP + grid_h + BANNER_TOP_GAP + BANNER_H + LEGEND_H + PAD

SUBTITLE = ("三个 swizzle 标量描述元素怎么错位打散,其余三个管轴序 / CTA 切分 / Hopper 开关"
            "(TritonGPUAttrDefs.td:L243-L249)")

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD - 6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("SharedEncodingAttr 的六个字段")}</text>',
     f'<text x="{PAD}" y="{PAD + 16}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

grid_x0 = PAD
for r, row in enumerate(rows):
    for c, (name, desc, grp) in enumerate(row):
        x = grid_x0 + c * (BOX_W + HGAP)
        y = TOP + r * (BOX_H + VGAP)
        fill, stroke, text_c = GROUP_COLOR[grp]
        L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{x + BOX_W/2}" y="{y + 38}" text-anchor="middle" '
                  f'font-family="monospace" font-size="16" font-weight="bold" '
                  f'fill="{text_c}">{esc(name)}</text>')
        L.append(f'<text x="{x + BOX_W/2}" y="{y + 64}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" '
                  f'fill="{text_c}">{esc(desc)}</text>')

# 图例
ly = TOP + grid_h + 16
lx = PAD
for grp, label in LEGEND:
    fill, stroke, _ = GROUP_COLOR[grp]
    L.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx + 20}" y="{ly + 12}" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(label)}</text>')
    lx += 20 + 8 + len(label) * 6.6 + 26

# 底部:print 输出样例(读者在 IR dump 里真会看到的形态)
by = TOP + grid_h + BANNER_TOP_GAP + LEGEND_H
L.append(f'<rect x="{PAD}" y="{by}" width="{grid_w}" height="{BANNER_H}" rx="8" '
          f'fill="#0f172a" stroke="#334155"/>')
L.append(f'<text x="{PAD + 16}" y="{by + 24}" font-family="sans-serif" font-size="11" '
          f'fill="#94a3b8">{esc("序列化样例(TRITON_KERNEL_DUMP 里能看到的真实形态,Dialect.cpp:L1550-L1557)")}</text>')
sample = "#shared<{vec=2, perPhase=1, maxPhase=4, order=[1,0], hasLeadingOffset=false}>"
L.append(f'<text x="{PAD + 16}" y="{by + 48}" font-family="monospace" font-size="13" '
          f'fill="#e2e8f0">{esc(sample)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-shared-fields-schema.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
