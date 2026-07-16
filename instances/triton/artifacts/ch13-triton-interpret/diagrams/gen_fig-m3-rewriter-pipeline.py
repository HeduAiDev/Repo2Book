#!/usr/bin/env python3
"""fig-m3-rewriter-pipeline: flow 模板。FunctionRewriter 从『源码文本』到
『改写后可调用函数』的流水线——5 个主干步骤 + 2 条侧注（行号对齐/缓存）。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STEPS = [
    ("取源码", "inspect.getsourcelines(self.fn)", "interpreter.py:L1143"),
    ("定位 def / dedent / 变换", "_find_def -> _prepare_source -> _transform_ast", "interpreter.py:L1154-L1156"),
    ("compile+exec 落点", "_compile_and_exec 进 fn.__globals__", "interpreter.py:L1187-1195"),
]
SIDE_NOTES = [
    ("行号对齐目的", "报错/断点指回用户源码真实行号", 1),
    ("缓存", "rewritten_fn 类级去重（同函数只改写一次）", 2),
]

BOX_W, BOX_H, VGAP, PAD, TOP = 460, 66, 44, 50, 90
w = PAD * 2 + BOX_W + 300
h = TOP + len(STEPS) * (BOX_H + VGAP) + PAD + 30
cx = PAD + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{cx}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("FunctionRewriter：从『你的源码文本』到『改写后可调用函数』")}</text>']

# entry pill
entry_y = TOP - 34
L.append(f'<rect x="{cx-120}" y="{entry_y}" width="240" height="30" rx="15" '
          'fill="#22c55e" stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<text x="{cx}" y="{entry_y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="white">{esc("用户 @triton.jit 核源码")}</text>')
L.append(f'<line x1="{cx}" y1="{entry_y+30}" x2="{cx}" y2="{TOP}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

box_x = PAD
step_y = []
for i, (title, detail, anchor) in enumerate(STEPS):
    y = TOP + i * (BOX_H + VGAP)
    step_y.append(y)
    L.append(f'<rect x="{box_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              'fill="#eff6ff" stroke="#3b82f6" stroke-width="2"/>')
    L.append(f'<text x="{box_x+18}" y="{y+24}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#1e40af">{esc(f"{i+1}. {title}")}</text>')
    L.append(f'<text x="{box_x+18}" y="{y+44}" font-family="monospace" font-size="11" '
              f'fill="#334155">{esc(detail)}</text>')
    L.append(f'<text x="{box_x+18}" y="{y+60}" font-family="sans-serif" font-size="10" '
              f'fill="#64748b">{esc(anchor)}</text>')
    if i < len(STEPS) - 1:
        L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                  'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

# exit pill
last_y = step_y[-1] + BOX_H
exit_y = last_y + VGAP - 4
L.append(f'<line x1="{cx}" y1="{last_y}" x2="{cx}" y2="{exit_y}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{cx-140}" y="{exit_y}" width="280" height="30" rx="15" '
          'fill="#f97316" stroke="#c2410c" stroke-width="1.5"/>')
L.append(f'<text x="{cx}" y="{exit_y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="white">'
          f'{esc("改写后可调用函数（进 InterpretedFunction）")}</text>')

# side notes attached with dashed callout lines to specific steps
note_x = box_x + BOX_W + 50
note_w = 220
for title, detail, step_idx in SIDE_NOTES:
    ny = step_y[step_idx] + BOX_H / 2 - 24
    L.append(f'<line x1="{box_x+BOX_W}" y1="{step_y[step_idx]+BOX_H/2}" '
              f'x2="{note_x}" y2="{ny+24}" stroke="#7c3aed" stroke-width="1.3" '
              'stroke-dasharray="4,3" marker-end="url(#b)"/>')
    note_h = 48
    L.append(f'<rect x="{note_x}" y="{ny}" width="{note_w}" height="{note_h}" rx="7" '
              'fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.3"/>')
    L.append(f'<text x="{note_x+10}" y="{ny+18}" font-family="sans-serif" font-size="11.5" '
              f'font-weight="bold" fill="#5b21b6">{esc(title)}</text>')
    L.append(f'<text x="{note_x+10}" y="{ny+34}" font-family="sans-serif" font-size="10" '
              f'fill="#4c1d95">{esc(detail[:24])}</text>')
    if len(detail) > 24:
        L.append(f'<text x="{note_x+10}" y="{ny+46}" font-family="sans-serif" font-size="10" '
                  f'fill="#4c1d95">{esc(detail[24:])}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-rewriter-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
