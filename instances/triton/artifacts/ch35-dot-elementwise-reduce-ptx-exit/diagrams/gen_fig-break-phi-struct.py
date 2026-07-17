#!/usr/bin/env python3
"""fig-break-phi-struct (before-after 模板)
LLVMIRBreakPhiStruct:1 个 struct phi{i32,i32}(2 incoming)拆成 2 个标量 phi +
4 个 extractvalue + 2 个 insertvalue,replaceAllUsesWith 换用户。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANEL_W, PAD, TOP = 380, 46, 108
GAP = 140
w = PAD * 2 + PANEL_W * 2 + GAP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 1">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>']

L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("BreakPhiStruct:拆箱换优化——struct phi 对 LLVM 优化不友好")}</text>')
L.append(f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("示例:{i32,i32} struct phi(2 incoming = preheader、latch),make_llir 出口收尾一步")}</text>')

px_before = PAD
px_after = PAD + PANEL_W + GAP

L.append(f'<text x="{px_before+PANEL_W/2}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">{esc("处理前")}</text>')
L.append(f'<text x="{px_after+PANEL_W/2}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">{esc("处理后")}</text>')

# ---------- 处理前面板 ----------
BOX_H = 84
by = TOP
L.append(f'<rect x="{px_before}" y="{by}" width="{PANEL_W}" height="{BOX_H}" rx="10" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.8"/>')
L.append(f'<text x="{px_before+PANEL_W/2}" y="{by+26}" text-anchor="middle" font-family="monospace" '
          f'font-size="14" font-weight="bold" fill="#7f1d1d">{esc("1 个 struct phi {i32,i32}")}</text>')
L.append(f'<text x="{px_before+PANEL_W/2}" y="{by+48}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#7f1d1d">{esc("incoming: preheader, latch(2 条边)")}</text>')
L.append(f'<text x="{px_before+PANEL_W/2}" y="{by+68}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#7f1d1d">{esc("下游用户直接消费这个 struct 值")}</text>')

before_bottom = by + BOX_H
L.append(f'<text x="{px_before+PANEL_W/2}" y="{before_bottom+30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#991b1b">'
          f'{esc("LLVM 优化器不擅长拆箱看里面每件货")}</text>')

# ---------- 处理后面板:2 个标量 phi ----------
SCALAR_W, SCALAR_H, SGAP = PANEL_W, 58, 16
ay = TOP
for i in range(2):
    y = ay + i * (SCALAR_H + SGAP)
    L.append(f'<rect x="{px_after}" y="{y}" width="{SCALAR_W}" height="{SCALAR_H}" rx="8" '
              'fill="#dcfce7" stroke="#059669" stroke-width="1.6"/>')
    L.append(f'<text x="{px_after+16}" y="{y+24}" font-family="monospace" font-size="13" '
              f'font-weight="bold" fill="#065f46">{esc(f"phi{i} : i32")}</text>')
    L.append(f'<text x="{px_after+16}" y="{y+44}" font-family="sans-serif" font-size="11" '
              f'fill="#065f46">{esc("incoming 数 = 2(preheader, latch)——与原 struct phi 一致")}</text>')

scalar_bottom = ay + 2 * SCALAR_H + SGAP

# extractvalue / insertvalue 计数条
count_y = scalar_bottom + 20
count_h = 56
L.append(f'<rect x="{px_after}" y="{count_y}" width="{SCALAR_W}" height="{count_h}" rx="8" '
          'fill="#e0e7ff" stroke="#4338ca" stroke-width="1.4"/>')
L.append(f'<text x="{px_after+16}" y="{count_y+22}" font-family="monospace" font-size="12.5" '
          f'font-weight="bold" fill="#3730a3">{esc("extractvalue x 4 = numScalarEl(2) x numOperands(2)")}</text>')
L.append(f'<text x="{px_after+16}" y="{count_y+42}" font-family="monospace" font-size="12.5" '
          f'font-weight="bold" fill="#3730a3">{esc("insertvalue x 2 = numScalarEl(2)  →  重组回 struct")}</text>')

after_bottom = count_y + count_h
L.append(f'<text x="{px_after+PANEL_W/2}" y="{after_bottom+30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#065f46">'
          f'{esc("replaceAllUsesWith 换成新 struct,SROA/寄存器分配能逐件优化")}</text>')

# 中间大箭头
mid_y = (before_bottom + TOP) / 2 if False else TOP + max(BOX_H, scalar_bottom - TOP) / 2
arrow_y = TOP + BOX_H / 2
L.append(f'<line x1="{px_before+PANEL_W+10}" y1="{arrow_y}" x2="{px_after-10}" y2="{arrow_y}" '
          'stroke="#334155" stroke-width="2.4" marker-end="url(#a)"/>')
L.append(f'<text x="{(px_before+PANEL_W+px_after)/2}" y="{arrow_y-12}" text-anchor="middle" '
          f'font-family="monospace" font-size="10.5" font-weight="bold" fill="#334155">'
          f'{esc("processPhiStruct")}</text>')

foot_y = max(before_bottom + 50, after_bottom + 50) + 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" fill="#334155">'
          f'{esc("语义等价:insertvalue 逐元素重组的新 struct 与原 struct phi 每个 element 相等;")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12.5" fill="#334155">'
          f'{esc("这是 make_llir 出口收尾——此后 TTGIR → LLVM(→ PTX by ptxas)五级阶梯走完。")}</text>')

h = foot_y + 40
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L.insert(2, f'<rect width="{w}" height="{h}" fill="white"/>')
L.append('</svg>')

out = Path(__file__).with_name("fig-break-phi-struct.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
