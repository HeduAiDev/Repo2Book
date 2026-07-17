#!/usr/bin/env python3
"""fig-ch32-dot-legality: before-after 模板。
ConversionTarget 声明 tt.dot 的合法性条件——两操作数皆 DotOperand 编码才合法。
声明合法性、让框架自己驱动重写(dialect-conversion 的精髓)。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PANEL_W, PAD, TOP = 320, 44, 128
BOX_W, BOX_H, VGAP = 280, 46, 20
w = PAD * 2 + PANEL_W * 2 + 110
h = TOP + 3 * (BOX_H + VGAP) + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker></defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(
    f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("tt.dot 的合法性:ConversionTarget 只声明条件,不写怎么改")}</text>'
)
L.append(
    f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
    f'fill="#475569">{esc("TritonGPUConversion.cpp:L112-L121 addDynamicallyLegalOp<DotOp>")}</text>'
)

PANELS = [
    ("非法(make_ttir 产物)", "#dc2626", "#fee2e2", [
        ("%14 = tt.dot(结果无编码)", False),
        ("A 操作数:#blocked1(无 dot_op)", False),
        ("B 操作数:#blocked1(无 dot_op)", False),
    ], "isa<DotOperandEncodingAttr> → false"),
    ("合法(第一跳收敛后)", "#16a34a", "#dcfce7", [
        ("%22 = tt.dot(结果 #blocked1)", False),
        ("A 操作数:dot_op(opIdx=0)", True),
        ("B 操作数:dot_op(opIdx=1)", True),
    ], "两侧皆 dot_op → true"),
]

for p, (title, tcolor, bgc, rows, tag) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 110)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14.5" font-weight="bold" fill="{tcolor}">{esc(title)}</text>')
    for i, (line, hot) in enumerate(rows):
        y = TOP + i * (BOX_H + VGAP)
        fill = bgc if hot else "#f1f5f9"
        stroke = tcolor if hot else "#94a3b8"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hot else 1.3}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" font-family="monospace" '
                 f'font-size="12.5" fill="#0f172a">{esc(line)}</text>')
    # 判定标签
    tag_y = TOP + len(rows) * (BOX_H + VGAP) + 6
    L.append(f'<rect x="{cx-BOX_W/2}" y="{tag_y}" width="{BOX_W}" height="30" rx="8" '
             f'fill="{bgc}" stroke="{tcolor}" stroke-width="1.6"/>')
    L.append(f'<text x="{cx}" y="{tag_y+20}" text-anchor="middle" font-family="monospace" '
             f'font-size="11.5" font-weight="bold" fill="{tcolor}">{esc(tag)}</text>')

# 中间箭头:非法 -> 合法,标 TritonDotPattern
midy = TOP + (3 * (BOX_H + VGAP) - VGAP) / 2
x1 = PAD + PANEL_W + 8
x2 = PAD + PANEL_W + 102
L.append(f'<line x1="{x1}" y1="{midy}" x2="{x2}" y2="{midy}" stroke="#2563eb" '
         f'stroke-width="2.4" marker-end="url(#g)"/>')
L.append(f'<text x="{(x1+x2)/2}" y="{midy-16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#1d4ed8">{esc("TritonDotPattern")}</text>')
L.append(f'<text x="{(x1+x2)/2}" y="{midy+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#475569">{esc("插 convert_layout")}</text>')

# 底部:驱动入口 + 合法条件全文
cond_y = TOP + 3 * (BOX_H + VGAP) + 46
L.append(f'<rect x="{PAD}" y="{cond_y}" width="{w-2*PAD}" height="34" rx="8" '
         f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.4"/>')
L.append(
    f'<text x="{w/2}" y="{cond_y+22}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="10.5" fill="#1d4ed8">'
    f'{esc("合法条件: aEncoding && isa<DotOperandEncodingAttr>(aEncoding) && bEncoding && isa<...>(bEncoding) → true,否则 false")}</text>'
)

drive_y = cond_y + 52
L.append(
    f'<text x="{w/2}" y="{drive_y}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11.5" fill="#64748b">'
    f'{esc("驱动入口 applyPartialConversion(TritonToTritonGPUPass.cpp:L800):反复应用 pattern 至所有 op 合法,否则 pass 失败")}</text>'
)

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-dot-legality.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
