#!/usr/bin/env python3
"""fig-ch32-dot-operand-glue: before-after 模板。
TritonDotPattern 把无编码 tt.dot 重写:A/B/C 各插一个 convert_layout,操作数钉成 DotOperand。
数据取自 worked_example(traces/dump_ir.json ttgir_first_hop_only 逐 SSA 编号)。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# 三个操作数:(标签, 前态, 插入的 convert_layout SSA, 目标编码)
OPERANDS = [
    ("A(load 结果 %15)", "tensor<16x16xf16> #blocked1", "%19", "dot_op(opIdx=0, parent=#blocked1)"),
    ("B(load 结果 %18)", "tensor<16x16xf16> #blocked1", "%20", "dot_op(opIdx=1, parent=#blocked1)"),
    ("C(常量 %cst_0)", "tensor<16x16xf32> #blocked1", "%21", "#blocked1(结果布局)"),
]

ROW_H, VGAP = 56, 18
PAD, TOP = 46, 128
COL_BEFORE_W = 300
COL_ARROW_W = 150
COL_AFTER_W = 340
w = PAD * 2 + COL_BEFORE_W + COL_ARROW_W + COL_AFTER_W
h = TOP + len(OPERANDS) * (ROW_H + VGAP) + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker></defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(
    f'<text x="{w/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("TritonDotPattern:三条 convert_layout 焊上 tt.dot 的操作数")}</text>'
)
L.append(
    f'<text x="{w/2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
    f'fill="#475569">{esc("lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:L215-L279")}</text>'
)
L.append(
    f'<text x="{w/2}" y="74" text-anchor="middle" font-family="sans-serif" font-size="12" '
    f'fill="#64748b">{esc("实测 · Triton v3.2.0 headless 编译 · ttgir_first_hop_only")}</text>'
)

x_before = PAD
x_arrow_c = PAD + COL_BEFORE_W + COL_ARROW_W / 2
x_after = PAD + COL_BEFORE_W + COL_ARROW_W

# 列标题
L.append(f'<text x="{x_before+COL_BEFORE_W/2}" y="{TOP-16}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#334155">{esc("第一跳前")}</text>')
L.append(f'<text x="{x_after+COL_AFTER_W/2}" y="{TOP-16}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#1d4ed8">{esc("第一跳后(插入 convert_layout)")}</text>')

for i, (label, before, ssa, after) in enumerate(OPERANDS):
    y = TOP + i * (ROW_H + VGAP)
    cy = y + ROW_H / 2
    # 前态框
    L.append(f'<rect x="{x_before}" y="{y}" width="{COL_BEFORE_W}" height="{ROW_H}" rx="8" '
             f'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.4"/>')
    L.append(f'<text x="{x_before+14}" y="{y+22}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{x_before+14}" y="{y+42}" font-family="monospace" font-size="12" '
             f'fill="#0f172a">{esc(before)}</text>')
    # 箭头 + convert_layout SSA 标签
    ax1 = x_before + COL_BEFORE_W + 6
    ax2 = x_after - 6
    L.append(f'<line x1="{ax1}" y1="{cy}" x2="{ax2}" y2="{cy}" stroke="#2563eb" '
             f'stroke-width="2.2" marker-end="url(#a)"/>')
    L.append(f'<text x="{x_arrow_c}" y="{cy-10}" text-anchor="middle" font-family="monospace" '
             f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc(ssa)}</text>')
    L.append(f'<text x="{x_arrow_c}" y="{cy+18}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10" fill="#64748b">{esc("convert_layout")}</text>')
    # 后态框
    L.append(f'<rect x="{x_after}" y="{y}" width="{COL_AFTER_W}" height="{ROW_H}" rx="8" '
             f'fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/>')
    L.append(f'<text x="{x_after+14}" y="{y+22}" font-family="sans-serif" font-size="11.5" '
             f'fill="#1e40af">{esc(label)}</text>')
    L.append(f'<text x="{x_after+14}" y="{y+42}" font-family="monospace" font-size="11.5" '
             f'font-weight="bold" fill="#0f172a">{esc(after)}</text>')

# 汇合到新 tt.dot
join_y = TOP + len(OPERANDS) * (ROW_H + VGAP) - VGAP + 10
join_cx = x_after + COL_AFTER_W / 2
L.append(f'<rect x="{join_cx-190}" y="{join_y}" width="380" height="40" rx="10" '
         f'fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{join_cx}" y="{join_y+26}" text-anchor="middle" font-family="monospace" '
         f'font-size="13" font-weight="bold" fill="#1d4ed8">'
         f'{esc("%22 = tt.dot %19, %20, %21 → 结果 #blocked1")}</text>')
# 底部结论(拆两行,避免超出画布)
concl_y1 = join_y + 66
concl_y2 = concl_y1 + 20
L.append(f'<text x="{PAD}" y="{concl_y1}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'{esc("retSizePerThread(结果布局) 由 TritonToTritonGPUPass.cpp:L228-L237 按 numElements/(numWarps×threadsPerWarp) 阈值取 1/2/4")}</text>')
L.append(f'<text x="{PAD}" y="{concl_y2}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'{esc("代入本例:256/128=2,2<4 → retSizePerThread 保持 [1,1],与实测 #blocked1 一致")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-dot-operand-glue.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
