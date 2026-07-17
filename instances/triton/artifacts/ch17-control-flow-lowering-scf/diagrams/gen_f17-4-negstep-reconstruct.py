#!/usr/bin/env python3
"""f17-4-negstep-reconstruct: range(10,0,-1) 下降为『0..9 正序 scf.for + 体首
iv=ub-j+lb 反算』,逐次还原用户的递减序列 10..1。state-table 模板,行=每次迭代,
中间以省略行压缩(仅展示 j=0,1,2 与 j=9,与 spec.numbers 逐条对应),
表尾给出完整 reconstructed_seq 与 match=True。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "range(10, 0, -1) 的负步长翻转与体首反算"
SUBTITLE = "MLIR scf.for 只能正着数:边界翻转为 0..10 step 1,体首 iv = ub - j + lb 还原用户递减序列"
COLS = ["scf.for 计数器 j", "iv = ub - j + lb(反算)", "用户看到的 k"]
ROWS = [
    ("0", "10", "10"),
    ("1", "9", "9"),
    ("2", "8", "8"),
    ("…", "…", "…"),
    ("9", "1", "1"),
]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 40, 260, 46, 40, 108, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 74

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (jv, iv, kv) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    is_ellipsis = (jv == "…")
    row_fill = "#f8fafc" if not is_ellipsis else "white"
    for j, val in enumerate((jv, iv, kv)):
        cx = col_x[j]
        if not is_ellipsis:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{row_fill}" stroke="#cbd5e1" stroke-width="1"/>')
        fs = 20 if is_ellipsis else 14
        fw = 'font-weight="bold" ' if j == 0 and not is_ellipsis else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" fill="#0f172a" {fw}>{esc(val)}</text>')

foot_y0 = TOP + HEADER_H + len(ROWS) * ROW_H + 26
L.append(f'<text x="{PAD}" y="{foot_y0}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#15803d">'
          f'{esc("reconstructed_seq = [10,9,8,7,6,5,4,3,2,1]  match = True(与 python range(10,0,-1) 逐项相等)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y0+22}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("IR 体首反算:%9 = arith.subi %2(ub=10), %arg1(j)   %10 = arith.addi %9, %1(lb=0)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y0+42}" font-family="sans-serif" font-size="10.5" '
          f'fill="#94a3b8">'
          f'{esc("Triton v3.2.0 headless 实测：ir.K4_for_negstep(arith.subi=1, arith.addi=3)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-4-negstep-reconstruct.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
