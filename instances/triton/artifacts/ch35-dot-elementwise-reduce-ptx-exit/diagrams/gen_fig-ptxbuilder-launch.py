#!/usr/bin/env python3
"""fig-ptxbuilder-launch (flow 模板)
所有 NVIDIA 硬件指令(mma/cvt/shfl/wgmma)都经同一条脊柱落成 LLVM::InlineAsmOp:
newOperand 登记操作数 -> create(ptxAsm) 登记模板 -> dump()/getConstraints() 并行拼串 -> launch() 出口。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

MAIN = [
    ("newOperand(SSA 值, 约束)", "约束如 =r/=f/=h/r/h"),
    ("create(ptxAsm)", "登记指令模板"),
]
PARALLEL = [
    ("dump()", "$0,$1… 占位拼成 asm_string"),
    ("getConstraints()", "拼成 constraints 参数"),
]
TAIL = ("launch()", "生成一条 LLVM::InlineAsmOp")

BOX_W, BOX_H, HGAP = 230, 70, 40
PAD, TOP = 40, 190
n_main = len(MAIN)
n_par = len(PARALLEL)

w = PAD * 2 + (n_main + n_par + 1) * BOX_W + (n_main + n_par) * HGAP
h = TOP + BOX_H + 220

x_cursor = PAD
X_MAIN = []
for _ in MAIN:
    X_MAIN.append(x_cursor)
    x_cursor += BOX_W + HGAP
PAR_X0 = x_cursor
X_PAR = [PAR_X0, PAR_X0]  # 两个并行框同 x,不同 y(上下叠放)
x_cursor += BOX_W + HGAP
X_TAIL = x_cursor
w = X_TAIL + BOX_W + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("所有硬件指令共享一条出口脊柱:PTXBuilder 拼成一条 inline asm")}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("mma/cvt/shfl/wgmma 走的是同一条 newOperand → create → dump/getConstraints → launch")}</text>']

main_y = TOP
for i, (title, sub) in enumerate(MAIN):
    x = X_MAIN[i]
    L.append(f'<rect x="{x}" y="{main_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{main_y+28}" text-anchor="middle" font-family="monospace" '
              f'font-size="13" font-weight="bold" fill="#0c4a6e">{esc(title)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{main_y+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#0369a1">{esc(sub)}</text>')
for i in range(len(MAIN) - 1):
    x1 = X_MAIN[i] + BOX_W
    x2 = X_MAIN[i+1]
    ay = main_y + BOX_H / 2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" '
              'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# main 最后一格 -> 并行两框(上下)
PAR_GAP = 26
par_total_h = n_par * BOX_H + (n_par - 1) * PAR_GAP
par_y0 = main_y + BOX_H / 2 - par_total_h / 2
src_x = X_MAIN[-1] + BOX_W
src_y = main_y + BOX_H / 2
for i, (title, sub) in enumerate(PARALLEL):
    x = X_PAR[i]
    y = par_y0 + i * (BOX_H + PAR_GAP)
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+28}" text-anchor="middle" font-family="monospace" '
              f'font-size="13" font-weight="bold" fill="#92400e">{esc(title)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#92400e">{esc(sub)}</text>')
    L.append(f'<line x1="{src_x}" y1="{src_y}" x2="{x}" y2="{y+BOX_H/2}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# 并行 -> launch
tail_y = main_y
L.append(f'<rect x="{X_TAIL}" y="{tail_y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>')
L.append(f'<text x="{X_TAIL+BOX_W/2}" y="{tail_y+28}" text-anchor="middle" font-family="monospace" '
          f'font-size="13" font-weight="bold" fill="#5b21b6">{esc(TAIL[0])}</text>')
L.append(f'<text x="{X_TAIL+BOX_W/2}" y="{tail_y+48}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#5b21b6">{esc(TAIL[1])}</text>')
dst_x = X_TAIL
dst_y = tail_y + BOX_H / 2
for i in range(n_par):
    x = X_PAR[i] + BOX_W
    y = par_y0 + i * (BOX_H + PAR_GAP) + BOX_H / 2
    L.append(f'<line x1="{x}" y1="{y}" x2="{dst_x}" y2="{dst_y}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# 侧注:副作用位(挂在 launch 上方)
note_y = tail_y - 70
note_w = BOX_W + 40
note_x = X_TAIL - 20
L.append(f'<rect x="{note_x}" y="{note_y}" width="{note_w}" height="46" rx="8" '
          'fill="#f1f5f9" stroke="#64748b" stroke-width="1.2"/>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+19}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc("同时传入的副作用位:")}</text>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+37}" text-anchor="middle" font-family="monospace" '
          f'font-size="11" font-weight="bold" fill="#334155">{esc("hasSideEffect / isAlignStack")}</text>')
L.append(f'<line x1="{note_x+note_w/2}" y1="{note_y+46}" x2="{X_TAIL+BOX_W/2}" y2="{tail_y}" '
          'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')

# 出口 op 说明条(须避开并行框堆叠底部,取两者最大值)
par_bottom = par_y0 + par_total_h
out_y = max(tail_y + BOX_H, par_bottom) + 40
out_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{out_y}" width="{out_w}" height="46" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+16}" y="{out_y+28}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("出口 op:")} '
          f'<tspan font-family="monospace" font-weight="bold">LLVM::InlineAsmOp</tspan> '
          f'{esc("(AD_ATT dialect)")}</text>')

foot_y = out_y + 46 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" fill="#334155">'
          f'{esc("Triton 不发 LLVM intrinsic 而发 inline asm:PTX 表达力更全,约束串精确锁定寄存器分配。")}</text>')

h = foot_y + 30
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L[2] = f'<rect width="{w}" height="{h}" fill="white"/>'
L.append('</svg>')

out = Path(__file__).with_name("fig-ptxbuilder-launch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
