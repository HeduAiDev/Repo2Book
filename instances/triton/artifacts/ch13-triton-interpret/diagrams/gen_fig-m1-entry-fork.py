#!/usr/bin/env python3
"""fig-m1-entry-fork: flow 模板。TRITON_INTERPRET=1 在装饰器层分叉——
同一份用户核代码，环境变量决定返回 InterpretedFunction（本章）还是 JITFunction（正常路径）。
全坐标由常量/循环计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 760
PAD = 40
ENTRY_Y = 40
ENTRY_W, ENTRY_H = 420, 46
DIAMOND_Y = 130
DIAMOND_W, DIAMOND_H = 460, 74
BRANCH_Y = 260
BOX_W, BOX_H = 320, 96
GAP_X = 40
NOTE_Y = 400
NOTE_W, NOTE_H = 640, 56
H = NOTE_Y + NOTE_H + PAD

cx = W / 2
entry_x = cx - ENTRY_W / 2
diamond_x = cx - DIAMOND_W / 2
left_x = cx - GAP_X / 2 - BOX_W
right_x = cx + GAP_X / 2
note_x = cx - NOTE_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# title
L.append(f'<text x="{cx}" y="24" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#0f172a">'
          f'{esc("装饰器层分叉：同一份 @triton.jit 核代码，两条执行路线")}</text>')

# entry box
L.append(f'<rect x="{entry_x}" y="{ENTRY_Y}" width="{ENTRY_W}" height="{ENTRY_H}" rx="8" '
          'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
L.append(f'<text x="{cx}" y="{ENTRY_Y+ENTRY_H/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" fill="#0f172a">'
          f'{esc("@triton.jit def kernel(...): ...  /  k[grid](...)")}</text>')

# arrow entry -> diamond
L.append(f'<line x1="{cx}" y1="{ENTRY_Y+ENTRY_H}" x2="{cx}" y2="{DIAMOND_Y}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# diamond (decision) — draw as a rotated-look polygon
dcx, dcy = cx, DIAMOND_Y + DIAMOND_H / 2
pts = f'{dcx},{DIAMOND_Y} {dcx+DIAMOND_W/2},{dcy} {dcx},{DIAMOND_Y+DIAMOND_H} {dcx-DIAMOND_W/2},{dcy}'
L.append(f'<polygon points="{pts}" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
decision_label = "os.getenv('TRITON_INTERPRET','0')=='1' ?"
L.append(f'<text x="{dcx}" y="{dcy-4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#78350f">'
          f'{esc(decision_label)}</text>')
L.append(f'<text x="{dcx}" y="{dcy+14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#92400e">{esc("判据锚点 jit.py:L834")}</text>')

diamond_bottom = DIAMOND_Y + DIAMOND_H

# arrow diamond -> left (Yes)
lx_c = left_x + BOX_W / 2
L.append(f'<line x1="{dcx-DIAMOND_W/4}" y1="{dcy+DIAMOND_H/4}" x2="{lx_c}" y2="{BRANCH_Y}" '
          'stroke="#059669" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(dcx-DIAMOND_W/4+lx_c)/2-10}" y="{(dcy+DIAMOND_H/4+BRANCH_Y)/2-6}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#059669">{esc("命中 (=1)")}</text>')

# arrow diamond -> right (No)
rx_c = right_x + BOX_W / 2
L.append(f'<line x1="{dcx+DIAMOND_W/4}" y1="{dcy+DIAMOND_H/4}" x2="{rx_c}" y2="{BRANCH_Y}" '
          'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(dcx+DIAMOND_W/4+rx_c)/2+10}" y="{(dcy+DIAMOND_H/4+BRANCH_Y)/2-6}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#64748b">{esc("未命中 (=0/未设)")}</text>')

# left box: InterpretedFunction (highlighted — this chapter)
L.append(f'<rect x="{left_x}" y="{BRANCH_Y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#ecfdf5" stroke="#059669" stroke-width="2.5"/>')
L.append(f'<text x="{lx_c}" y="{BRANCH_Y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#065f46">{esc("InterpretedFunction")}</text>')
L.append(f'<text x="{lx_c}" y="{BRANCH_Y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#065f46">{esc("CPU 替身串行执行（本章）")}</text>')
L.append(f'<text x="{lx_c}" y="{BRANCH_Y+64}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#047857">{esc("锚点 jit.py:L835-838")}</text>')
L.append(f'<text x="{lx_c}" y="{BRANCH_Y+82}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#047857">{esc("实测 type(k).__name__")}</text>')

# right box: JITFunction (grey — see ch11)
L.append(f'<rect x="{right_x}" y="{BRANCH_Y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')
L.append(f'<text x="{rx_c}" y="{BRANCH_Y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#334155">{esc("JITFunction")}</text>')
L.append(f'<text x="{rx_c}" y="{BRANCH_Y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#334155">{esc("GPU 正常路径（见第十一章）")}</text>')
L.append(f'<text x="{rx_c}" y="{BRANCH_Y+64}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#475569">{esc("锚点 jit.py:L840-849")}</text>')

# bottom note: measured result
L.append(f'<rect x="{note_x}" y="{NOTE_Y}" width="{NOTE_W}" height="{NOTE_H}" rx="8" '
          'fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>')
L.append(f'<text x="{cx}" y="{NOTE_Y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e40af">'
          f'{esc("实测（TRITON_INTERPRET=1，host 无 GPU 跑通）")}</text>')
note_label = "type(k[grid](...)).__name__ == 'InterpretedFunction'  —— 用户零改动，路线彻底分开"
L.append(f'<text x="{cx}" y="{NOTE_Y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#1e3a8a">'
          f'{esc(note_label)}</text>')

# connect boxes down to note (both branches converge conceptually)
L.append(f'<line x1="{lx_c}" y1="{BRANCH_Y+BOX_H}" x2="{lx_c}" y2="{NOTE_Y}" '
          'stroke="#059669" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-entry-fork.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
