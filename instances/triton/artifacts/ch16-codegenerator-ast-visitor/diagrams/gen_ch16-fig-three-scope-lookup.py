#!/usr/bin/env python3
"""state-table 模板(仿 example-softmax-trace.py):name_lookup 三级作用域查找表。
列 = [名字, ①local, ②global(守卫), ③builtin, 解析结果];行 = 6 个名字。
守卫列按内容语义上色:放行=绿, 拒绝=红, 命中=蓝, 缺/—=灰。
改造点:COLS、ROWS(6 x 5 文本矩阵)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "name_lookup:local -> global(守卫) -> builtin 三级线性查找"
SUBTITLE = "查找层数固定为 3(code_generator.py:L307);6 个名字里 1 个被 global 守卫拒绝(Triton v3.2.0 headless 实测)"

COLS = ["名字", "①local", "②global(守卫)", "③builtin", "解析结果"]
ROWS = [
    ["x", "命中\ntensor", "—", "—", "kernel 参数句柄"],
    ["BLOCK", "命中\nconstexpr", "—", "—", "constexpr(1024)"],
    ["tl", "缺", "放行\n(module)", "—", "module"],
    ["MAX_FUSED", "缺", "放行\n(constexpr 全局)", "—", "编译期常量"],
    ["range", "缺", "缺", "命中", "Triton 版 range"],
    ["LOOKUP_TABLE", "缺", "拒绝\n(非 constexpr)", "—", "raise NameError"],
]


def cell_color(col_idx, text):
    if col_idx == 2:  # 守卫列语义色
        if "放行" in text:
            return ("#ecfdf5", "#047857")
        if "拒绝" in text:
            return ("#fee2e2", "#b91c1c")
    if "命中" in text:
        return ("#eff6ff", "#1d4ed8")
    return None


NAME_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 150, 190, 58, 38, 96, 30
w = PAD * 2 + NAME_W + COL_W * (len(COLS) - 1)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 30
col_x = [PAD] + [PAD + NAME_W + i * COL_W for i in range(len(COLS) - 1)]
col_w = [NAME_W] + [COL_W] * (len(COLS) - 1)
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x, cw = col_x[j], col_w[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(cw-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    for j, cell in enumerate(row):
        x, cw = col_x[j], col_w[j]
        lines = cell.split("\n")
        color = cell_color(j, cell) if j > 0 else None
        if color:
            fill, stroke = color
            L.append(f'<rect x="{x}" y="{ry+4}" width="{cw-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
            text_fill = stroke
        else:
            text_fill = "#374151" if j > 0 else "#0f172a"
        anchor = "start" if j == 0 else "middle"
        tx = x + 12 if j == 0 else x + (cw - 8) / 2
        weight = 'font-weight="bold" ' if (j == 0 or color) else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{tx}" y="{y0+k*16}" text-anchor="{anchor}" '
                      f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                      f'{weight}>{esc(line)}</text>')
    if i < len(ROWS) - 1:
        L.append(f'<line x1="{PAD}" y1="{ry+ROW_H}" x2="{PAD+NAME_W+COL_W*(len(COLS)-1)-8}" '
                  f'y2="{ry+ROW_H}" stroke="#e2e8f0" stroke-width="1"/>')

foot_y = h - PAD + 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("绿=守卫放行;红=守卫拒绝(普通全局变量不许读,只放行 constexpr/module/JITFunction/@builtin/dtype);蓝=命中;灰=缺,继续查下一层")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-three-scope-lookup.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
