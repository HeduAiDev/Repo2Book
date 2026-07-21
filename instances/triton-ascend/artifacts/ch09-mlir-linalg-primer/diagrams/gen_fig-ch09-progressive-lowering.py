#!/usr/bin/env python3
"""fig-ch09-progressive-lowering — m02 渐进式下降 + 维持高层语义。
三列对照阶梯:Open64 WHIRL(5 级,实线钉死) / Clang(4 段固定链,实线钉死) / MLIR(虚线框,方言可组合,
右侧竖一条结构信息剩余量渐变条)。底部对齐 ch01 三段下降链。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "渐进式下降与维持高层语义是一对:层级可以写死,也可以由方言组合而成"
SUBTITLE = "Open64 与 Clang 的层级在设计时就固定(论文原话 in a rigid way);MLIR 让不同方言的算子在任意层级共存"

COL_W = 210
GAP = 60
PAD = 44
TOP = 130
STEP_H = 52
STEP_GAP = 14

OPEN64 = ["Very High WHIRL", "High WHIRL", "Mid WHIRL", "Low WHIRL", "Very Low WHIRL"]
CLANG = ["AST", "LLVM IR", "SelectionDAG", "MachineInstr", "MCInst"]
MLIR = ["高层方言\n(如 linalg)", "混合下降中间态\n(多方言共存)", "……", "低层方言\n(如 llvm)"]

n_steps = max(len(OPEN64), len(CLANG), len(MLIR))
w = PAD * 2 + COL_W * 3 + GAP * 2 + 60
h = TOP + n_steps * (STEP_H + STEP_GAP) + 170

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
         '<linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">'
         '<stop offset="0%" stop-color="#1d4ed8"/><stop offset="72%" stop-color="#93c5fd"/>'
         '<stop offset="100%" stop-color="#e2e8f0"/></linearGradient></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="16.5" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{w/2}" y="{PAD+22}" text-anchor="middle" font-family="sans-serif" font-size="12" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

cols = [
    (PAD, "Open64 WHIRL", OPEN64, False, "5 个层级(论文自述)"),
    (PAD + COL_W + GAP, "Clang", CLANG, False, "AST -> LLVM IR -> SelectionDAG -> MachineInstr -> MCInst(逐字列出,不给计数)"),
    (PAD + 2*(COL_W + GAP), "MLIR", MLIR, True, "层级由方言组合而成;不同方言算子可在任意层级共存"),
]

for cx0, name, steps, dashed, note in cols:
    cx = cx0 + COL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#1e293b">{esc(name)}</text>')
    n = len(steps)
    # 居中对齐:若该列层级数少于 n_steps,整体居中留白
    offset = (n_steps - n) * (STEP_H + STEP_GAP) / 2
    for i, label in enumerate(steps):
        y = TOP + offset + i * (STEP_H + STEP_GAP)
        stroke_style = 'stroke-dasharray="5,4"' if dashed else ''
        fill = "#eff6ff" if not dashed else "#f8fafc"
        stroke = "#1e3a8a" if not dashed else "#334155"
        L.append(f'<rect x="{cx0}" y="{y}" width="{COL_W}" height="{STEP_H}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.6" {stroke_style}/>')
        lines = label.split("\n")
        ly0 = y + STEP_H/2 - (len(lines)-1)*8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{ly0+k*16}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="12" fill="#0f172a">{esc(line)}</text>')
        if i < n - 1:
            y2 = y + STEP_H
            L.append(f'<line x1="{cx}" y1="{y2}" x2="{cx}" y2="{y2+STEP_GAP-2}" '
                      f'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
    note_y = TOP + n_steps * (STEP_H + STEP_GAP) + 10
    # 注释换行(按列宽粗略断行)
    max_chars = 22
    words = note
    wrapped = []
    cur = ""
    for ch in words:
        cur += ch
        if len(cur) >= max_chars and ch in " ,;)-":
            wrapped.append(cur); cur = ""
    if cur:
        wrapped.append(cur)
    for k, line in enumerate(wrapped):
        L.append(f'<text x="{cx}" y="{note_y+k*15}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="#64748b">{esc(line.strip())}</text>')

# 右侧结构信息剩余量渐变条(与 MLIR 列对齐)
bar_x = PAD + 2*(COL_W+GAP) + COL_W + 20
bar_top = TOP
bar_h = n_steps * (STEP_H + STEP_GAP) - STEP_GAP
L.append(f'<rect x="{bar_x}" y="{bar_top}" width="18" height="{bar_h}" rx="6" fill="url(#grad)"/>')
L.append(f'<text x="{bar_x+9}" y="{bar_top-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#334155">{esc("结构")}</text>')
L.append(f'<text x="{bar_x+9}" y="{bar_top+bar_h+16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#334155">{esc("剩余")}</text>')

# 底部:丢结构注记 + ch01 对齐
note2_y = TOP + n_steps*(STEP_H+STEP_GAP) + 66
L.append(f'<rect x="{PAD}" y="{note2_y-20}" width="{w-2*PAD}" height="34" rx="6" '
         f'fill="#fef3c7" stroke="#b45309" stroke-width="1.2"/>')
L.append(f'<text x="{w/2}" y="{note2_y+2}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#78350f">'
         f'{esc("丢结构是有意识的,只发生在结构不再需要匹配底层执行模型的地方")}</text>')

note3_y = note2_y + 46
L.append(f'<rect x="{PAD}" y="{note3_y-20}" width="{w-2*PAD}" height="34" rx="6" '
         f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>')
L.append(f'<text x="{w/2}" y="{note3_y+2}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#3730a3">'
         f'{esc("本书三段下降链 ttir -> ttadapter -> npubin,是这一原则的一次具体实例化")}</text>')

foot_y = h - 12
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc("依据:arXiv:2002.11054 §2")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-progressive-lowering.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
