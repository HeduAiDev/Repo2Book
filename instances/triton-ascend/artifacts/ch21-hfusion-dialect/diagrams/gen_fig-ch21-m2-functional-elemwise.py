#!/usr/bin/env python3
"""state-table 变体：函数化 elementwise 词汇表——枚举属性数 vs op 数。
行=5 个函数枚举（UnaryFn/BinaryFn/CompareFn/TernaryFn/TypeFn），
列=[枚举值数, 举例（首尾/全列, 非穷举列表标 …）, .td 定位]。
枚举值数按大小上色（>=10 蓝 / <10 琥珀），呼应 claim：
少数几个 op 靠枚举覆盖大量函数，而非一函数一 op。
底部 callout 说明枚举如何被 op 携带（fun 属性，非 op 本身）。
全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "HFusion 函数化 elementwise 词汇表——枚举属性，不是逐函数建 op"
SUBTITLE = "elemwise_unary/elemwise_binary/compare/cast 各携带一个枚举属性，一个 op 参数化出整族函数"
COLS = ["枚举值数", "举例（非穷举，… 表示中间省略）", ".td 定位"]
ROW_LABELS = ["UnaryFn", "BinaryFn", "CompareFn", "TernaryFn", "TypeFn"]
CELLS = {
    "UnaryFn":    ["18", "relu, sqrt, … , ilogb", "HFusionEnums.td:L29-L51"],
    "BinaryFn":   ["18", "vor, vand, … , maxf", "HFusionEnums.td:L53-L75"],
    "CompareFn":  ["10", "(10 个 case)", "HFusionEnums.td:L77-L91"],
    "TernaryFn":  ["1", "仅 select", "HFusionEnums.td:L93-L98"],
    "TypeFn":     ["3", "cast_signed / cast_unsigned / bitcast", "HFusionEnums.td:L100-L107"],
}
# 枚举值数按量级上色：>=10 视为"大词汇表"(蓝)，<10 视为"小词汇表"(琥珀)——
# 直观对比"覆盖整族函数"与"个别专属函数"的规模差异
MAGNITUDE = {"UnaryFn": "large", "BinaryFn": "large", "CompareFn": "large",
             "TernaryFn": "small", "TypeFn": "small"}
COLOR = {"large": ("#eff6ff", "#1d4ed8"), "small": ("#fffbeb", "#b45309")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 100, 220, 48, 34, 100, 30
CALLOUT_H = 82
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 24 + CALLOUT_H + PAD + 30
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    mag = MAGNITUDE[row]
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        # 只在"枚举值数"列(j=0)上色，突出规模对比；其余列保持中性
        if j == 0:
            fill, stroke = COLOR[mag]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            text_fill, weight_attr = stroke, 'font-weight="bold" '
        else:
            text_fill, weight_attr = "#374151", ''
        fsize = 12 if len(text) <= 26 else 10
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fsize}" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

# callout：强调"这是属性，不是 op"
co_y = row_y[-1] + ROW_H + 24
L.append(f'<rect x="{PAD}" y="{co_y}" width="{w-2*PAD}" height="{CALLOUT_H}" rx="6" '
          'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+14}" y="{co_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#166534">属性携带方式（非 op）:'
          f'</text>')
CALLOUT_LINE2 = "elemwise_unary op 的 fun 参数 = unary_fn_attr（默认 sqrt）——"
CALLOUT_LINE3 = "HFusionNamedStructuredOps.yaml:L74-L123，打印形态即 #hfusion.unary_fn<relu>"
L.append(f'<text x="{PAD+14}" y="{co_y+42}" font-family="sans-serif" font-size="12" '
          f'fill="#166534">{esc(CALLOUT_LINE2)}</text>')
L.append(f'<text x="{PAD+14}" y="{co_y+62}" font-family="sans-serif" font-size="12" '
          f'fill="#166534">{esc(CALLOUT_LINE3)}</text>')

foot_y = h - PAD + 4
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">蓝=覆盖多个函数的"大词汇表"枚举，琥珀=只覆盖 1-3 个函数的"小词汇表"枚举</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch21-m2-functional-elemwise.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
