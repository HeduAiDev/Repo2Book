#!/usr/bin/env python3
"""fig-ch23-address-space-hierarchy — layout 模板。
HIVM 把达芬奇六级显式内存做成 AddressSpace 枚举挂到 memref 类型上；C++ 枚举名与 IR 助记符
不同名（L1→cbuf、L0A→ca、L0B→cb、L0C→cc）。Zero 哨兵（枚举值 0）单独标出、不计入六级。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)

def fit(s, maxw, base, floor=9.0):
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size

TITLE = "AddressSpace：把达芬奇六级显式内存做成 memref 类型上的枚举"
SUBTITLE = "枚举值 1..6 排布（Zero=0 是哨兵，不计入六级）；C++ 枚举名与 IR 助记符不同名 —— HIVMAttrs.td:L188-L194"

# (枚举名, 值, 助记符, 分类, 说明)
ENUM = [
    ("Zero", 0, "zero", "sentinel", "未标注（哨兵，不算一级）"),
    ("GM", 1, "gm", "global", "全局内存（片外）"),
    ("L1", 2, "cbuf", "shared", "片上缓冲（共享暂存）"),
    ("L0A", 3, "ca", "cube", "Cube 输入 A"),
    ("L0B", 4, "cb", "cube", "Cube 输入 B"),
    ("L0C", 5, "cc", "cube", "Cube 累加器"),
    ("UB", 6, "ub", "vector", "Vector 工作缓冲"),
]
COLOR = {
    "sentinel": ("#f1f5f9", "#94a3b8", "#64748b"),
    "global":   ("#e0e7ff", "#4338ca", "#3730a3"),
    "shared":   ("#fef9c3", "#a16207", "#78350f"),
    "cube":     ("#fed7aa", "#c2410c", "#7c2d12"),
    "vector":   ("#bbf7d0", "#15803d", "#14532d"),
}
LEGEND = [
    ("sentinel", "Zero 哨兵（不计入六级）"),
    ("global",   "GM：片外全局"),
    ("shared",   "L1：片上共享暂存"),
    ("cube",     "Cube 侧独占：L0A / L0B / L0C"),
    ("vector",   "Vector 侧独占：UB"),
]

CHIP_W, CHIP_H, GAP, PAD, TOP = 160, 108, 18, 50, 172
n = len(ENUM)
w = PAD * 2 + n * CHIP_W + (n - 1) * GAP
h = TOP + CHIP_H + 380

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 值轴标签
L.append(f'<text x="{PAD}" y="{TOP-16}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#475569">{esc("枚举值 →")}</text>')

xs_ = [PAD + i * (CHIP_W + GAP) for i in range(n)]
for i, (name, val, mnem, cat, desc) in enumerate(ENUM):
    x = xs_[i]
    fill, stroke, tf = COLOR[cat]
    dash = ' stroke-dasharray="6,4"' if cat == "sentinel" else ''
    L.append(f'<rect x="{x}" y="{TOP}" width="{CHIP_W}" height="{CHIP_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{TOP+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" fill="{tf}">{esc(f"枚举值 {val}")}</text>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{TOP+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{TOP+72}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" fill="{tf}">{esc(f"助记符 {mnem}")}</text>')
    dfit = fit(desc, CHIP_W - 16, 10.5)
    L.append(f'<text x="{x+CHIP_W/2}" y="{TOP+92}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{dfit}" fill="{tf}">{esc(desc)}</text>')

# 相邻箭头（枚举值递增方向）
cy = TOP + CHIP_H / 2
for i in range(n - 1):
    x1, x2 = xs_[i] + CHIP_W, xs_[i+1]
    L.append(f'<line x1="{x1+2}" y1="{cy}" x2="{x2-2}" y2="{cy}" '
              f'stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')

# 物理位置分组框：GM(片外) | L1(片上共享) | Cube(L0A/L0B/L0C) | Vector(UB)
group_y = TOP + CHIP_H + 30
group_h = 30
GROUPS = [
    (xs_[1], xs_[1]+CHIP_W, "片外(HBM)", "#4338ca"),
    (xs_[2], xs_[2]+CHIP_W, "片上·共享暂存", "#a16207"),
    (xs_[3], xs_[5]+CHIP_W, "片上·Cube 独占", "#c2410c"),
    (xs_[6], xs_[6]+CHIP_W, "片上·Vector 独占", "#15803d"),
]
for x0, x1, label, color in GROUPS:
    gw = x1 - x0
    L.append(f'<rect x="{x0}" y="{group_y}" width="{gw}" height="{group_h}" rx="6" '
              f'fill="none" stroke="{color}" stroke-width="1.8" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{x0+gw/2}" y="{group_y+group_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fit(label, gw-10, 11.5)}" '
              f'font-weight="bold" fill="{color}">{esc(label)}</text>')

# 图例
leg_y = group_y + group_h + 44
L.append(f'<text x="{PAD}" y="{leg_y-14}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#475569">{esc("图例")}</text>')
for i, (cat, label) in enumerate(LEGEND):
    fill, stroke, tf = COLOR[cat]
    ly = leg_y + i * 26
    L.append(f'<rect x="{PAD}" y="{ly-13}" width="24" height="16" rx="4" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{PAD+34}" y="{ly}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')

# IR 实际长相
ir_y = leg_y + len(LEGEND) * 26 + 20
ir_text = "IR 里实际长相：memref<128x128xf16, #hivm.address_space<cbuf>>"
L.append(f'<rect x="{PAD}" y="{ir_y}" width="{w-2*PAD}" height="46" rx="8" '
          f'fill="#0f172a"/>')
L.append(f'<text x="{PAD+16}" y="{ir_y+29}" font-family="monospace" '
          f'font-size="{fit(ir_text, w-2*PAD-32, 15)}" fill="#e2e8f0">{esc(ir_text)}</text>')

foot_y = ir_y + 46 + 30
FOOT = "六级内存层级(Zero 哨兵不算)按枚举值 1..6 排布：GM 在片外、其余五级在片上；Cube 侧独占 L0A/L0B/L0C、" \
       "Vector 侧独占 UB、L1 是二者共享的暂存 —— memref 类型尾巴上的 <cbuf>/<cc>/<gm> 就是这套枚举的助记符。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(FOOT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-address-space-hierarchy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
