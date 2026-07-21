#!/usr/bin/env python3
"""fig-ch09-iteration-domain-derivation — m14 隐式迭代域。
state-table 变体:行=5 个迭代维,列=三个操作数(O/I/K)各轴的索引表达式 + 反解出的上界。
非纯恒等的轴(I 的 w+kw)整格灰显并不画来源箭头,借此点出「w 只有一个来源」这条非平凡性质。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "隐式迭代域:上界从操作数形状反解,IR 里一个字都没写"
SUBTITLE = "论文形状 O:1x988x64 / I:1x990x32 / K:3x32x64——5 个迭代维,5 条边界,0 个手写循环上界"

# (维名, 类型, O轴文本, O轴是否恒等来源, I轴文本, I轴是否恒等来源, K轴文本, K轴是否恒等来源, 上界)
ROWS = [
    ("n",  "parallel",  "O.0 (n)",        True,  "I.0 (n)",         True,  "—",              None,  "1"),
    ("w",  "parallel",  "O.1 (w)",        True,  "I.1 (w+kw)",      False, "—",              None,  "988"),
    ("f",  "parallel",  "O.2 (f)",        True,  "—",               None,  "K.2 (f)",         True,  "64"),
    ("kw", "reduction", "—",              None,  "I.1 (w+kw)",      False, "K.0 (kw)",        True,  "3"),
    ("c",  "reduction", "—",              None,  "I.2 (c)",         True,  "K.1 (c)",         True,  "32"),
]
TOTAL_LABEL = "合计迭代点"
TOTAL_VALUE = "1 x 988 x 64 x 3 x 32 = 6 070 272"

COL_HEADERS = ["迭代维", "类型", "O 的轴", "I 的轴", "K 的轴", "反解出的上界"]
LABEL_W, TYPE_W, OPD_W, BOUND_W = 78, 100, 190, 170
HEADER_H, ROW_H, TOP, PAD = 40, 46, 108, 40
GAP = 6

col_w = [LABEL_W, TYPE_W, OPD_W, OPD_W, OPD_W, BOUND_W]
col_x = []
x = PAD
for cw in col_w:
    col_x.append(x)
    x += cw
w = x + PAD
n_rows = len(ROWS)
h = TOP + HEADER_H + n_rows * ROW_H + ROW_H + 90  # +1 合计行 + 图例/脚注

REDUCTION_BG = "#fef3c7"
PARALLEL_BG = "#eff6ff"
IDENT_COLOR = "#1e3a8a"
NONIDENT_COLOR = "#94a3b8"
HEAD_FILL = "#1e40af"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

# header row
for j, htext in enumerate(COL_HEADERS):
    cx = col_x[j]
    cw = col_w[j]
    L.append(f'<rect x="{cx}" y="{TOP}" width="{cw-GAP}" height="{HEADER_H}" rx="4" '
              f'fill="{HEAD_FILL}"/>')
    L.append(f'<text x="{cx+(cw-GAP)/2}" y="{TOP+HEADER_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="white">{esc(htext)}</text>')

# data rows
for i, (dim, kind, o_txt, o_ok, i_txt, i_ok, k_txt, k_ok, bound) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    bg = REDUCTION_BG if kind == "reduction" else PARALLEL_BG
    L.append(f'<rect x="{PAD}" y="{ry}" width="{w-2*PAD}" height="{ROW_H-GAP}" rx="4" '
              f'fill="{bg}" stroke="#cbd5e1"/>')
    # 维名
    L.append(f'<text x="{col_x[0]+col_w[0]/2-GAP/2}" y="{ry+ROW_H/2-2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="16" font-weight="bold" '
              f'fill="#0f172a">{esc(dim)}</text>')
    # 类型
    type_color = "#b45309" if kind == "reduction" else "#1d4ed8"
    L.append(f'<text x="{col_x[1]+col_w[1]/2-GAP/2}" y="{ry+ROW_H/2-2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{type_color}">{esc(kind)}</text>')
    # 每行只有一个「反解来源」:三个操作数列里最后一个恒等来源(true)
    src_col_idx = None
    for ci, ok in enumerate([o_ok, i_ok, k_ok]):
        if ok:
            src_col_idx = ci
    for ci, (txt, ok) in enumerate([(o_txt, o_ok), (i_txt, i_ok), (k_txt, k_ok)]):
        cxx = col_x[2 + ci]
        cw2 = col_w[2 + ci]
        color = "#94a3b8" if ok is None else (IDENT_COLOR if ok else NONIDENT_COLOR)
        weight = 'font-weight="bold" ' if ok else ''
        suffix = "  ▸" if (ok and ci == src_col_idx) else ""  # 来源标记箭头,嵌在本格内,不跨列画线
        L.append(f'<text x="{cxx+(cw2-GAP)/2}" y="{ry+ROW_H/2-2}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" {weight}'
                  f'fill="{color}">{esc(txt)}{suffix}</text>')
        if ok is False:
            L.append(f'<text x="{cxx+(cw2-GAP)/2}" y="{ry+ROW_H/2+14}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" fill="#94a3b8">'
                      f'{esc("(非纯恒等,不可作边界来源)")}</text>')
    bx = col_x[5]
    bw = col_w[5]
    L.append(f'<text x="{bx+(bw-GAP)/2}" y="{ry+ROW_H/2-2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#0f172a">{esc(bound)}</text>')

# total row
ty = TOP + HEADER_H + n_rows * ROW_H
L.append(f'<rect x="{PAD}" y="{ty}" width="{w-2*PAD}" height="{ROW_H-GAP+6}" rx="4" '
          f'fill="#e2e8f0" stroke="#64748b" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+16}" y="{ty+ROW_H/2+2}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc(TOTAL_LABEL)}</text>')
L.append(f'<text x="{w-PAD-16}" y="{ty+ROW_H/2+2}" text-anchor="end" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#0f172a">{esc(TOTAL_VALUE)}</text>')

# legend
ly = ty + ROW_H + 30
L.append(f'<rect x="{PAD}" y="{ly-13}" width="14" height="14" fill="{PARALLEL_BG}" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{ly-2}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("parallel 维")}</text>')
L.append(f'<rect x="{PAD+130}" y="{ly-13}" width="14" height="14" fill="{REDUCTION_BG}" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+150}" y="{ly-2}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("reduction 维(kw、c——多个迭代点堆进同一输出格)")}</text>')
L.append(f'<text x="{PAD}" y="{ly+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("深色粗体 = 纯恒等索引(可作边界来源);浅灰 = 复合索引(如 w+kw),给不出该维的边界")}</text>')
foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("n/f/c 各有 2 个一致来源,只有 w 依赖唯一来源 O.1——数据:本章参考实现实测")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-iteration-domain-derivation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
