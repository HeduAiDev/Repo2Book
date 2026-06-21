#!/usr/bin/env python3
"""ch03 figure: O0-O3 optimization levels and the compilation/cudagraph defaults they apply."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# Columns: level | compilation mode | cudagraph_mode | fuse_allreduce_rms | flashinfer_autotune
headers = ["优化级", "compilation mode", "cudagraph_mode", "fuse_allreduce_rms", "flashinfer_autotune"]
rows = [
    ("O0", "NONE", "NONE", "False", "False"),
    ("O1", "VLLM_COMPILE", "PIECEWISE", "False", "True"),
    ("O2 (默认)", "VLLM_COMPILE", "FULL_AND_PIECEWISE", "谓词(TP>1)", "True"),
    ("O3 (= O2)", "VLLM_COMPILE", "FULL_AND_PIECEWISE", "谓词(TP>1)", "True"),
]

colw = [150, 200, 250, 220, 230]
x0, y0 = 30, 70
rowh = 50
W = x0 * 2 + sum(colw)
H = y0 + rowh * (len(rows) + 1) + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{x0}" y="40" font-size="20" font-weight="bold" fill="#0f172a">'
         f'O0–O3：优化级与落地的 compilation / cudagraph 默认</text>')

# header row
cx = x0
for j, htxt in enumerate(headers):
    L.append(f'<rect x="{cx}" y="{y0}" width="{colw[j]}" height="{rowh}" '
             f'fill="#1e293b"/>')
    L.append(f'<text x="{cx + colw[j] // 2}" y="{y0 + rowh // 2 + 5}" '
             f'text-anchor="middle" font-size="14" font-weight="bold" '
             f'fill="white">{esc(htxt)}</text>')
    cx += colw[j]

# data rows
level_colors = ["#fee2e2", "#fef3c7", "#dcfce7", "#dbeafe"]
for i, row in enumerate(rows):
    ry = y0 + (i + 1) * rowh
    cx = x0
    for j, cell in enumerate(row):
        bg = level_colors[i] if j == 0 else ("#f8fafc" if i % 2 == 0 else "white")
        L.append(f'<rect x="{cx}" y="{ry}" width="{colw[j]}" height="{rowh}" '
                 f'fill="{bg}" stroke="#cbd5e1" stroke-width="1"/>')
        weight = "bold" if j == 0 else "normal"
        fam = "sans-serif" if j == 0 else "monospace"
        fs = 14 if j == 0 else 13
        L.append(f'<text x="{cx + colw[j] // 2}" y="{ry + rowh // 2 + 5}" '
                 f'text-anchor="middle" font-size="{fs}" font-weight="{weight}" '
                 f'font-family="{fam}" fill="#0f172a">{esc(cell)}</text>')
        cx += colw[j]

# arrow indicating monotone knob (startup vs perf)
ny = y0 + (len(rows) + 1) * rowh + 24
L.append(f'<text x="{x0}" y="{ny}" font-size="13" fill="#64748b">'
         f'级别升高 →　编译/cudagraph 越激进　→　启动越慢、稳态性能越好</text>')
L.append(f'<text x="{x0}" y="{ny + 24}" font-size="13" font-weight="bold" fill="#b91c1c">'
         f'优先级：用户显式设置　&gt;　enforce_eager / 环境变量　&gt;　优化级预设</text>')
L.append('</svg>')

with open("optimization-levels.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote optimization-levels.svg")
