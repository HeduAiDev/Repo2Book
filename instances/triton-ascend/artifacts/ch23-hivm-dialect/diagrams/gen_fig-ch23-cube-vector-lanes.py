#!/usr/bin/env python3
"""fig-ch23-cube-vector-lanes — swimlane 模板（改造为两条静态属性泳道）。
Cube 与 Vector 双核分工是算子的静态属性（写死在基类 Trait 上），不靠运行时判断；
func_core_type(AIC/AIV/MIX) 在函数级决定整核落哪个物理核。
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

TITLE = "Cube / Vector 双核分工：写死在算子基类 Trait 上的静态属性"
SUBTITLE = "func_core_type(AIC/AIV/MIX) 决定整核落哪个物理核；单个算子的核归属由其基类 Trait 编译期钉死，不靠运行时判断"

PAD, TOP = 50, 130
LANE_W, LANE_GAP = 460, 60
w = PAD * 2 + 2 * LANE_W + LANE_GAP

# ── 顶部:func_core_type 三态选择器 ──────────────────────────────────────
SEL_Y, SEL_H = TOP, 58
CHIPS = [("AIV", "纯 Vector"), ("MIX", "混合(拆子核)"), ("AIC", "纯 Cube")]
CHIP_W = 150
chips_w = len(CHIPS) * CHIP_W + (len(CHIPS)-1) * 20
sel_x0 = PAD + (2*LANE_W+LANE_GAP)/2 - chips_w/2

lane_y = SEL_Y + SEL_H + 70
row_h = 96
rows = ["核 Trait", "流水 + 缓冲", "代表算子"]
lane_h = len(rows) * row_h + 30

h = lane_y + lane_h + 210

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
     '<marker id="dash" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

L.append(f'<text x="{w/2}" y="{SEL_Y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#475569">'
          f'{esc("func_core_type 三态（函数级，决定整核落哪个物理核）")}</text>')

lane_cx = {"cube": PAD + LANE_W + LANE_GAP + LANE_W/2, "vector": PAD + LANE_W/2}
chip_cx = []
for i, (name, sub) in enumerate(CHIPS):
    x = sel_x0 + i * (CHIP_W + 20)
    ccx = x + CHIP_W/2
    chip_cx.append((name, ccx))
    fill, stroke, tf = ("#f1f5f9", "#64748b", "#334155") if name == "MIX" else \
                       (("#fed7aa", "#c2410c", "#7c2d12") if name == "AIC" else ("#bbf7d0", "#15803d", "#14532d"))
    dash = ' stroke-dasharray="5,4"' if name == "MIX" else ''
    L.append(f'<rect x="{x}" y="{SEL_Y}" width="{CHIP_W}" height="{SEL_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    L.append(f'<text x="{ccx}" y="{SEL_Y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    L.append(f'<text x="{ccx}" y="{SEL_Y+44}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="{tf}">{esc(sub)}</text>')

# 三态 -> 两条泳道 的分派箭头(AIC->Cube实线, AIV->Vector实线, MIX->两条都虚线)
arrow_y0 = SEL_Y + SEL_H
arrow_y1 = lane_y
for name, ccx in chip_cx:
    if name == "AIC":
        L.append(f'<line x1="{ccx}" y1="{arrow_y0}" x2="{lane_cx["cube"]}" y2="{arrow_y1}" '
                  f'stroke="#c2410c" stroke-width="2" marker-end="url(#a)"/>')
    elif name == "AIV":
        L.append(f'<line x1="{ccx}" y1="{arrow_y0}" x2="{lane_cx["vector"]}" y2="{arrow_y1}" '
                  f'stroke="#15803d" stroke-width="2" marker-end="url(#a)"/>')
    else:
        for target in ("cube", "vector"):
            L.append(f'<path d="M {ccx} {arrow_y0} L {lane_cx[target]} {arrow_y1}" fill="none" '
                      f'stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="5,4" marker-end="url(#dash)"/>')

# ── 两条泳道头 ───────────────────────────────────────────────────────
LANES = [
    ("vector", PAD, "Vector 泳道（AIV）", "#bbf7d0", "#15803d", "#14532d"),
    ("cube", PAD + LANE_W + LANE_GAP, "Cube 泳道（AIC）", "#fed7aa", "#c2410c", "#7c2d12"),
]
head_h = 40
for key, x, name, fill, stroke, tf in LANES:
    L.append(f'<rect x="{x}" y="{lane_y}" width="{LANE_W}" height="{head_h}" rx="8" '
              f'fill="{stroke}"/>')
    L.append(f'<text x="{x+LANE_W/2}" y="{lane_y+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="white">{esc(name)}</text>')
    L.append(f'<line x1="{x+LANE_W/2}" y1="{lane_y+head_h}" x2="{x+LANE_W/2}" y2="{lane_y+lane_h}" '
              f'stroke="{stroke}" stroke-width="1.2" stroke-dasharray="3,4"/>')

ROWDATA = {
    "vector": [
        ("VectorCoreTypeTrait(AIV 向量核)", "HIVMVectorOps.td:L34"),
        ("PIPE_V / 落 UB", "HIVMVectorOps.td:L34；InferHIVMMemScope.cpp:L457 AIV→UB 兜底"),
        ("vadd / vexp / vreduce…(40+)", "HIVM 向量算子族，全部继承 HIVM_VectorOp"),
    ],
    "cube": [
        ("CubeCoreTypeTrait(AIC 矩阵核)", "HIVMMacroOps.td:L60"),
        ("MTE1+M / 走 L1→L0A/L0B→L0C", "HIVMMacroOps.td:L62 MacroOpPipeTrait<PIPE_MTE1,PIPE_M>"),
        ("mmadL1(C=C+A×B)", "HIVMMacroOps.td:L163"),
    ],
}
for i, row_label in enumerate(rows):
    ry = lane_y + head_h + 14 + i * row_h
    for key, x, name, fill, stroke, tf in LANES:
        value, prov = ROWDATA[key][i]
        L.append(f'<rect x="{x}" y="{ry}" width="{LANE_W}" height="{row_h-14}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        L.append(f'<text x="{x+16}" y="{ry+20}" font-family="sans-serif" font-size="11" '
                  f'font-weight="bold" fill="{tf}">{esc(row_label)}</text>')
        L.append(f'<text x="{x+LANE_W/2}" y="{ry+46}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fit(value, LANE_W-32, 14)}" font-weight="bold" fill="{tf}">{esc(value)}</text>')
        L.append(f'<text x="{x+LANE_W/2}" y="{ry+68}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fit(prov, LANE_W-32, 10)}" fill="{tf}">{esc(prov)}</text>')

foot_y0 = lane_y + lane_h + 40
FOOT = [
    "两条泳道的核归属是编译期就钉死的静态 Trait：向量算子一律 VectorCoreTypeTrait+PIPE_V 落 UB，"
    "mmad 一律 CubeCoreTypeTrait+MTE1/M 走 L1/L0x。",
    "正因为核归属是算子属性而非运行时判断，编译器才能据此把 MIX 核安全拆成 AIC/AIV 子核。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*22}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-cube-vector-lanes.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
