#!/usr/bin/env python3
"""state-table 模板：semantic.cast 按 (src, dst) dtype 选支，每支发一个真 IR op；
最后一行 bf16<->非fp32 高亮为"两跳"（借道 fp32，发 2 个 op）。数据取自
traces/cast_ir.txt（追踪期 ASTSource.make_ir，任何 pass 之前）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "semantic.cast 大 dispatch —— 每支发一个真 IR op，bf16 借道两跳"
SUBTITLE = "追踪期 IR（ASTSource.make_ir，任何 pass 之前）；数据来自 traces/cast_ir.txt"
COLS = ["cast 表达式", "src → dst", "IR op", "op 数"]
ROWS = [
    ("x.to(tl.float16)", "fp32 → fp16", "arith.truncf", "1", "normal"),
    ("a.to(tl.float32)", "fp16 → fp32", "arith.extf", "1", "normal"),
    ("x.to(tl.int32)", "fp32 → int32", "arith.fptosi", "1", "normal"),
    ("s.to(tl.float32)", "int32 → fp32", "arith.sitofp", "1", "normal"),
    ("x.to(tl.bfloat16)", "fp16 → bf16", "arith.extf + arith.truncf", "2", "hot"),
]
COLOR = {"normal": ("#f1f5f9", "#475569"), "hot": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W2, COL_W3, COL_W4, ROW_H, HEADER_H, TOP, PAD = 200, 190, 280, 90, 56, 36, 118, 34
COL_W = [190, 190, 280, 90]
w = PAD * 2 + LABEL_W + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 40
col_x = []
cx = PAD + LABEL_W
for cw in COL_W:
    col_x.append(cx)
    cx += cw
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
L.append(f'<rect x="{PAD}" y="{TOP}" width="{LABEL_W-8}" height="{HEADER_H-6}" rx="3" '
          'fill="#334155"/>')
L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">'
          f'{esc("表达式")}</text>')
for j, name in enumerate(COLS[1:]):
    x = col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(cw-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

# 行
for i, (expr, srcdst, op, n, status) in enumerate(ROWS):
    ry = row_y[i]
    fill, stroke = COLOR[status]
    lw = 2 if status == "hot" else 1
    L.append(f'<rect x="{PAD}" y="{ry+3}" width="{LABEL_W-8}" height="{ROW_H-6}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
    L.append(f'<text x="{PAD+16}" y="{ry+ROW_H/2+5}" '
              f'font-family="sans-serif" font-size="12.5" fill="#1e293b">{esc(expr)}</text>')
    cells = [srcdst, op, n]
    for j, val in enumerate(cells):
        x = col_x[j]
        cw = COL_W[j]
        L.append(f'<rect x="{x}" y="{ry+3}" width="{cw-8}" height="{ROW_H-6}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>')
        weight = 'font-weight="bold" ' if (status == "hot" and j >= 1) else ''
        L.append(f'<text x="{x+(cw-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" {weight}'
                  f'fill="{"#b91c1c" if status=="hot" else "#1e293b"}">{esc(val)}</text>')

# 图例
ly = h - 30
legend = [("#f1f5f9", "#475569", "一进一出，1 个 IR op"),
          ("#fee2e2", "#b91c1c", "无直达指令，借道 fp32 两跳，2 个 IR op")]
lx = PAD
for fill, stroke, label in legend:
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+22}" y="{ly+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + 12 * len(label) + 34

L.append('</svg>')
out = Path(__file__).with_name("fig-cast-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
