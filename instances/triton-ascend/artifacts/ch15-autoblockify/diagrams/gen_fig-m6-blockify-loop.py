#!/usr/bin/env python3
"""before-after 模板:m6 无法整体批处理的 scf.if 折成一条 scf.for blockify 循环。
左panel=折叠前单实例;右panel=折叠后循环体(逐 iv extract_slice)。
底部标注两档 blockId 的循环上界(0->5, 尾块5->1)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LEFT = ("折叠前(kernel2:受 program_id y 条件控制)",
        [("scf.if", "auto_blockify.mlir:L129", False),
         ("单点 store(谓词判断)", None, False)])
RIGHT = ("折叠后:blockify 循环",
         [("scf.for iv = 0 .. 上界", "{auto_blockify_loop} 标签,mlir:L108", True),
          ("tensor.extract mask[iv] : tensor<5xi1>", "mlir:L109", False),
          ("extract_slice 第 iv 行 → tensor<8>", "mlir:L111", False),
          ("scf.if(谓词判定)", None, False),
          ("单点 store", None, False)])

BOUND_ROWS = [
    ("物理块 blockId=0", "min(max(6-0,0),5) = 5", "5 次迭代,覆盖 0,1,2,3,4"),
    ("尾块 blockId=5", "min(max(6-5,0),5) = 1", "1 次迭代,覆盖 5"),
]

BOX_W, BOX_H, VGAP = 300, 46, 22
PANEL_GAP = 100
PAD, TOP = 40, 92
n_left, n_right = len(LEFT[1]), len(RIGHT[1])
lx = PAD
rx = PAD + BOX_W + PANEL_GAP
w = rx + BOX_W + PAD
body_h = max(n_left, n_right) * (BOX_H + VGAP)
table_top = TOP + body_h + 50
h = table_top + len(BOUND_ROWS) * 30 + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

for px, (title, steps) in ((lx, LEFT), (rx, RIGHT)):
    cx = px + BOX_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, (step, ref, hl) in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        L.append(f'<rect x="{px}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1}"/>')
        ty = y + BOX_H / 2 + (5 if not ref else -2)
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.3" fill="#0f172a">{esc(step)}</text>')
        if ref:
            L.append(f'<text x="{cx}" y="{y+BOX_H-8}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="9.8" fill="#64748b">{esc(ref)}</text>')
        if i < len(steps) - 1:
            y1, y2 = y + BOX_H, y + BOX_H + VGAP
            L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-3}" '
                      'stroke="#64748b" stroke-width="1.4" marker-end="url(#a)"/>')

midy = TOP + (max(n_left, n_right) * (BOX_H + VGAP) - VGAP) / 2 - 30
L.append(f'<line x1="{lx+BOX_W+10}" y1="{midy}" x2="{rx-10}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(lx+BOX_W+rx)/2}" y="{midy-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
          f'fill="#b45309">{esc("无法整体批处理 → 折成循环")}</text>')

# bound table
tw = w - 2 * PAD
col_x = [PAD, PAD + tw * 0.30, PAD + tw * 0.62]
L.append(f'<text x="{PAD}" y="{table_top-16}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("循环上界 = min(max(blockNum-blockId,0),size)  (Utils.cpp:L137-145)")}</text>')
headers = ["物理块", "上界手算(blockNum=6,size=5)", "迭代次数/覆盖"]
for cx0, htext in zip(col_x, headers):
    L.append(f'<text x="{cx0}" y="{table_top}" font-family="sans-serif" font-size="11.5" '
              f'font-weight="bold" fill="#475569">{esc(htext)}</text>')
for i, (a, b, c) in enumerate(BOUND_ROWS):
    ry = table_top + 24 + i * 28
    L.append(f'<text x="{col_x[0]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(a)}</text>')
    L.append(f'<text x="{col_x[1]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(b)}</text>')
    L.append(f'<text x="{col_x[2]}" y="{ry}" font-family="sans-serif" font-size="12" '
              f'fill="#0f172a">{esc(c)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-m6-blockify-loop.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f"wrote {out}")
