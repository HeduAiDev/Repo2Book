#!/usr/bin/env python3
"""fig-ch23-mem-scope-resolution — state-table 模板（改造为带优先级的竖排阶梯表）。
内存层级推断按四步优先级把每个 memref 定型：mmadL1 硬约束 > func 参数 > pointer cast >
核类型兜底(AIC/AIV)，并沿 use-def 级联；高优先级先定的不被兜底覆盖。
级联传播不是第五个优先级，是附加在任意步骤结果之上的后续过程，单独一行标出。
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

TITLE = "内存层级推断：四步优先级 + use-def 级联"
SUBTITLE = "高优先级先定的地址空间，兜底步骤不再覆盖 —— InferHIVMMemScope.cpp:L436-L467"

PAD, TOP = 50, 130
BADGE_W = 64
ROW_W = 980
ROW_H = 66
ROW_GAP = 14

# (优先级徽标, 规则(spec.numbers.label), 结果(spec.numbers.value 原文), provenance, 结果分类色)
ROWS = [
    ("①", "步① · mmadL1 约束", "mmadL1：A/B→cbuf(L1)、C→cc(L0C)",
     "InferHIVMMemScope.cpp:L215/L222/L229；夹具 L32/L43/L48", "hard"),
    ("②", "步② · func 参数 GM", "func memref 参数→gm(GM)",
     "InferHIVMMemScope.cpp:L371 helper.Run(arg, gmSpaceAttr)；夹具 L26-27", "boundary"),
    ("③", "步③ · pointer cast", "pointer cast 标记→gm",
     "InferHIVMMemScope.cpp:L449 walk(PointerCastOp)；夹具 L179", "boundary"),
    ("④", "步④ · 核类型兜底（AIC）", "剩余 alloc→cbuf(L1)",
     "InferHIVMMemScope.cpp:L458-459 funcCoreType==AIC → L1；夹具 L260", "fallback"),
    ("④", "步④ · 核类型兜底（AIV）", "剩余 alloc→ub(UB)",
     "InferHIVMMemScope.cpp:L457 space=UB（默认）；夹具 L94", "fallback"),
]
CASCADE_ROW = ("级联传播：scf.for iter_arg/结果 → 随 C 变 cc",
               "propagateMemScopeToUsers：InferHIVMMemScope.cpp:L65-L144；夹具 L38-39")

CAT_COLOR = {
    "hard":     ("#fed7aa", "#c2410c", "#7c2d12"),
    "boundary": ("#e0e7ff", "#4338ca", "#3730a3"),
    "fallback": ("#f1f5f9", "#94a3b8", "#475569"),
}

n = len(ROWS)
table_h = n * (ROW_H + ROW_GAP) - ROW_GAP
w = PAD * 2 + BADGE_W + 20 + ROW_W
h = TOP + table_h + 220

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="casc" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7e22ce"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 左侧"优先级(高→低)"箭头
arrow_x = PAD + BADGE_W/2
L.append(f'<text x="{arrow_x}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#475569">{esc("优先级")}</text>')
L.append(f'<line x1="{arrow_x}" y1="{TOP}" x2="{arrow_x}" y2="{TOP+table_h}" '
          f'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{arrow_x}" y="{TOP+table_h+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#94a3b8">{esc("高 → 低")}</text>')

row_x = PAD + BADGE_W + 20
for i, (badge, rule, result, prov, cat) in enumerate(ROWS):
    ry = TOP + i * (ROW_H + ROW_GAP)
    fill, stroke, tf = CAT_COLOR[cat]
    # 徽标
    L.append(f'<circle cx="{arrow_x}" cy="{ry+ROW_H/2}" r="19" fill="white" stroke="{stroke}" stroke-width="2.2"/>')
    L.append(f'<text x="{arrow_x}" y="{ry+ROW_H/2+6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="16" font-weight="bold" fill="{stroke}">{esc(badge)}</text>')
    # 行卡片
    L.append(f'<rect x="{row_x}" y="{ry}" width="{ROW_W}" height="{ROW_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{row_x+18}" y="{ry+26}" font-family="sans-serif" '
              f'font-size="{fit(rule, 320, 13.5)}" font-weight="bold" fill="{tf}">{esc(rule)}</text>')
    L.append(f'<text x="{row_x+18}" y="{ry+48}" font-family="sans-serif" '
              f'font-size="{fit(prov, 360, 9.5)}" fill="{tf}">{esc(prov)}</text>')
    # 结果（右对齐胶囊）
    res_w = text_w(result, 13) + 28
    res_x = row_x + ROW_W - res_w - 16
    res_y = ry + ROW_H/2 - 15
    L.append(f'<rect x="{res_x}" y="{res_y}" width="{res_w}" height="30" rx="15" '
              f'fill="white" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{res_x+res_w/2}" y="{res_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(result, res_w-16, 13)}" font-weight="bold" fill="{stroke}">{esc(result)}</text>')

# 兜底不覆盖：④ 两行间的"平级"横线标注
tie_y = TOP + 3 * (ROW_H + ROW_GAP) + ROW_H + ROW_GAP/2
L.append(f'<text x="{row_x}" y="{tie_y+4}" font-family="sans-serif" font-size="10" '
          f'fill="#94a3b8">{esc("④ 的 AIC/AIV 两支同优先级 —— 按 func_core_type 择一分支，不叠加")}</text>')

# 级联行（视觉上独立于优先级阶梯，虚线连接，标"附加过程"）
casc_y = TOP + table_h + 60
L.append(f'<line x1="{row_x+40}" y1="{TOP+table_h}" x2="{row_x+40}" y2="{casc_y}" '
          f'stroke="#7e22ce" stroke-width="1.8" stroke-dasharray="5,4" marker-end="url(#casc)"/>')
L.append(f'<rect x="{row_x}" y="{casc_y}" width="{ROW_W}" height="{ROW_H}" rx="10" '
          f'fill="#f3e8ff" stroke="#7e22ce" stroke-width="2" stroke-dasharray="7,5"/>')
casc_rule, casc_prov = CASCADE_ROW
L.append(f'<text x="{row_x+18}" y="{casc_y+26}" font-family="sans-serif" '
          f'font-size="{fit(casc_rule, ROW_W-36, 13)}" font-weight="bold" fill="#581c87">{esc(casc_rule)}</text>')
L.append(f'<text x="{row_x+18}" y="{casc_y+48}" font-family="sans-serif" '
          f'font-size="{fit(casc_prov, ROW_W-36, 9.5)}" fill="#7e22ce">{esc(casc_prov)}</text>')
L.append(f'<text x="{PAD}" y="{casc_y-10}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="#7e22ce">'
          f'{esc("级联传播（不是第五个优先级，是附加在任意步骤结果之上的后续过程）")}</text>')

foot_y0 = casc_y + ROW_H + 40
FOOT = [
    "把「内存墙摆到类型里」是一次带优先级的类型推断：硬约束（矩阵三件套）最先钉死、边界数据（参数）其次，",
    "剩下的按「这是 Cube 核还是 Vector 核」兜底，再沿数据流级联到 scf.for —— 次序不可乱，先定的 L1/L0C 不会被后面的 UB 兜底覆盖。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*22}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-mem-scope-resolution.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
