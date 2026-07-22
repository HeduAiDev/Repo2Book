#!/usr/bin/env python3
"""fig-m2-dispatch-tree: visitOperand 三层决策树（flow 模板）。
第一层 3 个前置快门（缓存命中/标量/指针）顺序判定，命中即终止；
三个都未命中落到 getDefiningOp<> 分派，分三路：12 个产状态算子（绿）、
2 个显式保守失败算子（红）、裸 block-arg/未知算子兜底（灰，内部再分叉）。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "visitOperand 三层决策树（PtrAnalysis.cpp:L1280-L1355）"
SUBTITLE = "前置 3 个快门顺序判定；未命中落到 getDefiningOp<> 分派，14 个 defining-op 分支中 12 产状态、2 保守失败"

GATES = [
    ("knownPtrs 命中？", "复用缓存 PtrState", "PtrAnalysis.cpp:L1280-L1291"),
    ("operandIsScalar？\n(仅 Integer/Index)", "initStateByScalar\noffset=index_cast(operand)", "PtrAnalysis.cpp:L1280-L1291"),
    ("isa<PointerType>？", "initStateByPointer\nsource=operand, offset=0\n（裸指针 block-arg 走这里）", "PtrAnalysis.cpp:L1280-L1291"),
]

PRODUCE_OPS = ["Add", "Mul", "Sub", "MakeRange", "Broadcast", "Splat",
               "ExpandDims", "AddPtr", "ConstSplat", "Rem", "Div", "ExtSI"]
FAIL_OPS = ["LoadOp", "FPToSIOp"]

W = 1360
PAD, TOP = 40, 96
ROOT_W, ROOT_H = 460, 56
GATE_W, GATE_H, GATE_GAP = 300, 62, 20
TERM_W, TERM_H = 380, 62
DISPATCH_W, DISPATCH_H = 480, 56

root_x = W / 2 - ROOT_W / 2
root_y = TOP

gate_x = 60
gate_y0 = root_y + ROOT_H + 46
gates_y = [gate_y0 + i * (GATE_H + GATE_GAP) for i in range(len(GATES))]
term_x = gate_x + GATE_W + 70

dispatch_y = gates_y[-1] + GATE_H + 60
dispatch_x = W / 2 - DISPATCH_W / 2

branch_y = dispatch_y + DISPATCH_H + 56
BR_W = (W - PAD * 2 - 2 * 40) / 3
br_x = [PAD + i * (BR_W + 40) for i in range(3)]

chip_w, chip_h, chip_gap_x, chip_gap_y = BR_W / 4 - 10, 34, 10, 10
chip_rows = 3
produce_top = branch_y + 74
fail_top = branch_y + 74

fallback_lines = [
    "裸 block-arg 或未知 op",
    "在 knownPtrs（loop iter-arg）→ 复用",
    "否则 → return failure()",
    "「input parameters not supported」",
]

h = branch_y + 74 + chip_rows * (chip_h + chip_gap_y) + 56

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
     '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>',
     f'<rect width="{W}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 根节点
L.append(f'<rect x="{root_x}" y="{root_y}" width="{ROOT_W}" height="{ROOT_H}" rx="10" '
          'fill="#1e3a5f" stroke="#0f172a" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{root_y+ROOT_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="white">{esc("visitOperand(operand)：判定一个 SSA 值的地址来源")}</text>')

# 根 -> 第一个快门
L.append(f'<line x1="{gate_x+GATE_W/2}" y1="{root_y+ROOT_H}" x2="{gate_x+GATE_W/2}" y2="{gates_y[0]}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# 三个快门顺序判定
for i, (cond, result, prov) in enumerate(GATES):
    gy = gates_y[i]
    L.append(f'<rect x="{gate_x}" y="{gy}" width="{GATE_W}" height="{GATE_H}" rx="10" '
              'fill="#e0e7ff" stroke="#4338ca" stroke-width="1.8"/>')
    lines = cond.split("\n")
    y0 = gy + GATE_H / 2 - (len(lines) - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{gate_x+GATE_W/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#3730a3">{esc(line)}</text>')
    # 命中 -> 右侧终点
    L.append(f'<line x1="{gate_x+GATE_W}" y1="{gy+GATE_H/2}" x2="{term_x}" y2="{gy+GATE_H/2}" '
              'stroke="#16a34a" stroke-width="1.6" marker-end="url(#ag)"/>')
    L.append(f'<text x="{(gate_x+GATE_W+term_x)/2}" y="{gy+GATE_H/2-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#16a34a">是</text>')
    L.append(f'<rect x="{term_x}" y="{gy}" width="{TERM_W}" height="{GATE_H}" rx="10" '
              'fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>')
    rl = result.split("\n")
    ry0 = gy + GATE_H / 2 - (len(rl) - 1) * 8 + 4
    for k, line in enumerate(rl):
        L.append(f'<text x="{term_x+TERM_W/2}" y="{ry0+k*13.5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
                  f'fill="#166534">{esc(line)}</text>')
    # 未命中 -> 继续往下
    if i < len(GATES) - 1:
        ny = gates_y[i + 1]
        L.append(f'<line x1="{gate_x+GATE_W/2}" y1="{gy+GATE_H}" x2="{gate_x+GATE_W/2}" y2="{ny}" '
                  'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
        L.append(f'<text x="{gate_x+GATE_W/2-10}" y="{(gy+GATE_H+ny)/2+4}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="10" fill="#94a3b8">否</text>')

# 第三个快门未命中 -> 汇入 dispatch box
last_gate_bottom = gates_y[-1] + GATE_H
L.append(f'<path d="M {gate_x+GATE_W/2} {last_gate_bottom} L {gate_x+GATE_W/2} {dispatch_y+DISPATCH_H/2} '
          f'L {dispatch_x} {dispatch_y+DISPATCH_H/2}" fill="none" stroke="#64748b" stroke-width="1.6" '
          'marker-end="url(#a)"/>')
L.append(f'<text x="{gate_x+GATE_W/2-10}" y="{last_gate_bottom+16}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">否</text>')

# dispatch box
L.append(f'<rect x="{dispatch_x}" y="{dispatch_y}" width="{DISPATCH_W}" height="{DISPATCH_H}" rx="10" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{dispatch_y+DISPATCH_H/2+5}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#92400e">{esc("getDefiningOp<> 分派：14 个 defining-op 分支")}</text>')

# 三路分支
BR_LABELS = [
    ("产状态：12 个分支", "#16a34a", "#dcfce7"),
    ("保守失败：2 个分支", "#dc2626", "#fee2e2"),
    ("兜底：裸 block-arg / 未知 op", "#64748b", "#f1f5f9"),
]
for i, (label, stroke, fill) in enumerate(BR_LABELS):
    bx_c = br_x[i] + BR_W / 2
    L.append(f'<line x1="{W/2}" y1="{dispatch_y+DISPATCH_H}" x2="{bx_c}" y2="{branch_y}" '
              f'stroke="{stroke}" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<rect x="{br_x[i]}" y="{branch_y}" width="{BR_W}" height="46" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{bx_c}" y="{branch_y+29}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="{stroke}">{esc(label)}</text>')

# 分支 A：12 个产状态算子做 chip 网格（4x3）
gx0 = br_x[0]
for i, name in enumerate(PRODUCE_OPS):
    r, c = divmod(i, 4)
    cx = gx0 + c * (chip_w + chip_gap_x)
    cy = produce_top + r * (chip_h + chip_gap_y)
    L.append(f'<rect x="{cx}" y="{cy}" width="{chip_w}" height="{chip_h}" rx="6" '
              'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.2"/>')
    L.append(f'<text x="{cx+chip_w/2}" y="{cy+chip_h/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#166534">{esc(name)}</text>')
prod_bottom = produce_top + chip_rows * (chip_h + chip_gap_y) - chip_gap_y
L.append(f'<text x="{br_x[0]+BR_W/2}" y="{prod_bottom+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#166534">→ 递归/终止产生 PtrState（成功）</text>')

# 分支 B：2 个保守失败算子
fx0 = br_x[1] + (BR_W - len(FAIL_OPS) * chip_w - (len(FAIL_OPS) - 1) * chip_gap_x) / 2
for i, name in enumerate(FAIL_OPS):
    cx = fx0 + i * (chip_w + chip_gap_x)
    cy = fail_top
    L.append(f'<rect x="{cx}" y="{cy}" width="{chip_w}" height="{chip_h}" rx="6" '
              'fill="#fef2f2" stroke="#dc2626" stroke-width="1.2"/>')
    L.append(f'<text x="{cx+chip_w/2}" y="{cy+chip_h/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#991b1b">{esc(name)}</text>')
L.append(f'<text x="{br_x[1]+BR_W/2}" y="{fail_top+chip_h+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#991b1b">→ 显式 return failure()</text>')

# 分支 C：兜底文本块
fb_y = branch_y + 74
L.append(f'<rect x="{br_x[2]}" y="{fb_y}" width="{BR_W}" height="{len(fallback_lines)*17+16}" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="5,3"/>')
for i, line in enumerate(fallback_lines):
    weight = "bold" if i in (1, 3) else "normal"
    col = "#166534" if i == 1 else ("#991b1b" if i == 3 else "#475569")
    L.append(f'<text x="{br_x[2]+14}" y="{fb_y+20+i*17}" font-family="sans-serif" font-size="10.5" '
              f'font-weight="{weight}" fill="{col}">{esc(line)}</text>')

# 底部计数小结
foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#0f172a" font-weight="bold">'
          f'{esc("14 个 defining-op 分支 = 12 产状态 + 2 保守失败；加前置 3 快门与裸 block-arg / 未知 op 兜底")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m2-dispatch-tree.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
