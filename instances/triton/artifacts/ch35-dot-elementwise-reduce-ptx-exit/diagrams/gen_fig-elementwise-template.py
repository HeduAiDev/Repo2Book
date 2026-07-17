#!/usr/bin/env python3
"""fig-elementwise-template (flow 模板,4 阶段横链,子类钩子高亮)
所有逐元素 op 共用一条 CRTP 流水线:拆包入口 -> 算法钩子(子类填) -> 重排+去重 -> 重打包出口。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STAGES = [
    ("拆包入口", ["unpackLLElements", "unpackI32"], False),
    ("算法钩子(子类填)", ["createDestOps", "(如 FpToFpOpConversion)"], True),
    ("重排+去重", ["reorderValues", "maybeDeduplicate"], False),
    ("重打包出口", ["packI32", "packLLElements", "replaceOp"], False),
]

BOX_W, BOX_H, HGAP = 250, 110, 50
PAD, TOP = 40, 150
n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
h = TOP + BOX_H + 150

X = [PAD + i * (BOX_W + HGAP) for i in range(n)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("所有逐元素 op 共用一条 CRTP 流水线,差别只在中间那一格")}</text>',
     f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("基类只搭『拆-算-拼』脚手架;fp8 cvt 和普通 fma 走的是同一条链,只有算法钩子不同")}</text>']

for i, (title, lines, hot) in enumerate(STAGES):
    x, y = X[i], TOP
    fill = "#fef3c7" if hot else "#e0f2fe"
    stroke = "#d97706" if hot else "#0369a1"
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if hot else 1.5}"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{stroke}">{esc(title)}</text>')
    n_lines = len(lines)
    y0 = y + BOX_H/2 - (n_lines-1)*9 + 6
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*18}" text-anchor="middle" font-family="monospace" '
                  f'font-size="12.5" fill="#0f172a">{esc(ln)}</text>')

for i in range(n - 1):
    x1 = X[i] + BOX_W
    x2 = X[i+1]
    ay = TOP + BOX_H / 2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" '
              'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# 高亮钩子的顶部注记(不遮挡其它节点)
hot_i = next(i for i, s in enumerate(STAGES) if s[2])
note_x = X[hot_i]
note_w = BOX_W
note_y = TOP - 66
L.append(f'<rect x="{note_x}" y="{note_y}" width="{note_w}" height="46" rx="8" '
          'fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+19}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#92400e">{esc("同一条链的唯一差异点")}</text>')
L.append(f'<text x="{note_x+note_w/2}" y="{note_y+37}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#92400e">{esc("fp8 cvt / 普通 fma 都填这一格")}</text>')
L.append(f'<line x1="{note_x+note_w/2}" y1="{note_y+46}" x2="{note_x+note_w/2}" y2="{TOP}" '
          'stroke="#d97706" stroke-width="1.4" marker-end="url(#a)"/>')

foot_y = TOP + BOX_H + 46
foot_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{foot_w}" height="60" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y+24}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("四段各自的真实调用顺序(源码锚点):")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+44}" font-family="monospace" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("unpackLLElements→unpackI32 | createDestOps | reorderValues→maybeDeduplicate | packI32→packLLElements→replaceOp")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-elementwise-template.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
