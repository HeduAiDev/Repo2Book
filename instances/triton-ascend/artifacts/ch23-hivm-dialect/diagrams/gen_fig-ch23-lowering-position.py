#!/usr/bin/env python3
"""fig-ch23-lowering-position — flow 模板。
HIVM 在 bishengir 下降链上的位置：convert-hfusion-to-hivm 把 linalg+hfusion 判 illegal
（全部消灭）、把 hivm 判 legal（下降目标）；memref/tensor/arith/scf/func 等宿主方言在
HIVM 之下继续存活（不被转换）；HIVM 再往下（超本章）落 Standard/AscendC。
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

TITLE = "HIVM 在 bishengir 下降链上的位置：convert-hfusion-to-hivm"
SUBTITLE = "入口 pass 的 ConversionTarget 把 linalg + hfusion 判 illegal（必须消灭），把 hivm 判 legal（下降目标）"

BOX_W, BOX_H, GAP, PAD, TOP = 300, 116, 100, 50, 150
STAGES = [
    ("linalg + hfusion", ["融合张量 IR", "内存层级隐式"], "#fee2e2", "#b91c1c", "#7f1d1d", "illegal（必须消灭）"),
    ("hivm(HIVMDialect)", ["Cube/Vector 算子", "带 address_space 类型"], "#dbeafe", "#1d4ed8", "#1e3a8a", "legal（下降目标）"),
    ("Standard / AscendC", ["下一层，超本章范围"], "#f1f5f9", "#64748b", "#334155", None),
]
EDGE_LABELS = [
    "convert-hfusion-to-hivm",
    "HIVMToStandard 等（超本章）",
]

n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = TOP + BOX_H + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(n)]
cy = TOP + BOX_H / 2

for i, (name, lines, fill, stroke, tf, badge) in enumerate(STAGES):
    x = xs_[i]
    sw = 2.6 if i < 2 else 1.6
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(name, BOX_W-24, 16)}" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+56+k*20}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fit(ln, BOX_W-28, 12.5)}" fill="{tf}">{esc(ln)}</text>')
    if badge:
        bw = text_w(badge, 11) + 22
        bx = x + BOX_W/2 - bw/2
        by = TOP - 32
        L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="20" rx="10" fill="white" '
                  f'stroke="{stroke}" stroke-width="1.4"/>')
        L.append(f'<text x="{x+BOX_W/2}" y="{by+14}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="{stroke}">{esc(badge)}</text>')

for i in range(n - 1):
    x1, x2 = xs_[i] + BOX_W, xs_[i+1]
    marker = "url(#a)" if i == 0 else "url(#gray)"
    color = "#334155" if i == 0 else "#94a3b8"
    dash = '' if i == 0 else ' stroke-dasharray="6,5"'
    L.append(f'<line x1="{x1+4}" y1="{cy}" x2="{x2-4}" y2="{cy}" '
              f'stroke="{color}" stroke-width="2"{dash} marker-end="{marker}"/>')
    lbl = EDGE_LABELS[i]
    L.append(f'<text x="{(x1+x2)/2}" y="{cy-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(lbl, x2-x1-8, 11.5)}" font-weight="bold" fill="{color}">{esc(lbl)}</text>')

# provenance strip under boxes（只印文件:行号，源码符号留给正文，图上不堆长符号）
prov_y = TOP + BOX_H + 30
PROV = [
    "HFusionToHIVM.cpp:L1183",
    "HFusionToHIVM.cpp:L1178",
    None,
]
for i, p in enumerate(PROV):
    if p is None:
        continue
    x = xs_[i] + BOX_W/2
    L.append(f'<text x="{x}" y="{prov_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(p, BOX_W-20, 11)}" fill="#94a3b8">{esc(p)}</text>')

# 宿主方言存活说明框
box_y = prov_y + 40
box_h = 96
L.append(f'<rect x="{PAD}" y="{box_y}" width="{w-2*PAD}" height="{box_h}" rx="9" '
          f'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.4"/>')
L.append(f'<text x="{PAD+18}" y="{box_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#166534">'
          f'{esc("HIVM 之下仍存活的宿主方言（未被 ConversionTarget 判 illegal，不参与本次转换）")}</text>')
L.append(f'<text x="{PAD+18}" y="{box_y+50}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#15803d">memref / tensor / arith / scf / func</text>')
NOTE_TEXT = "HFusionToHIVM.cpp:L1178-L1182 —— 这些方言的算子原样保留在 HIVM 层的函数体里，和 hivm.* 算子混排。"
L.append(f'<text x="{PAD+18}" y="{box_y+72}" font-family="sans-serif" font-size="11" '
          f'fill="#4d7c0f">{esc(NOTE_TEXT)}</text>')

foot_y = box_y + box_h + 30
FOOT = "融合张量 IR 经 convert-hfusion-to-hivm 落到 HIVM 层：linalg/hfusion 判 illegal 全部消灭，hivm 成为唯一合法目标——" \
       "这一步之后，内存层级从「隐式」变成写在 memref 类型上的「可静态检查对象」。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(FOOT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-lowering-position.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
