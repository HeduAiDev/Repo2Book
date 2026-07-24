#!/usr/bin/env python3
"""before-after 模板改造(两候选对比,非优化前后):910_95 与 A2_A3 两个实现
结构同构,四处差异全部高亮(不是单行高亮)——因为整张图的论点就是「哪几处不同」。
改造点:PANELS 每行 (label, content),ROWS 定义要对比的维度。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROWS = ["--target 取法", "regbased 分叉", "sync_solver 挂载", "独有开关"]
PANELS = [
    ("910_95 分支\n(compiler.py:L298-L499)", [
        "metadata['target'].arch\n(get_common,L263-266,L310)",
        "固定 --enable-hivm-compile=true",
        "只挂 --enable-hivm-graph-\nsync-solver=<v>",
        "（无 A2_A3 专属开关）",
    ]),
    ("A2_A3 分支\n(compiler.py:L502-L696)", [
        "NPUUtils().get_arch()\n(L518-519)",
        "探测二选一：--reg-based=true\n或 --enable-hivm-compile=true\n(L512-515,L646-649)",
        "多挂一条 --enable-hivm-\ncross-core-gss=<v>(L560-565)",
        "enable_ubuf_saving / enable_preload /\ntile_mix_vector_loop / tile_mix_cube_loop\n(L528-538,L607-615)",
    ]),
]
VGAP, PANEL_W, PAD, TOP = 22, 460, 40, 96
PANEL_GAP = 190   # 两面板间距——须能装下中间的维度标签胶囊(宽 160)不重叠
LABEL_W = 160
ROW_H = [56, 62, 62, 68]  # 每行独立高度,按内容行数留够空间
w = PAD * 2 + PANEL_W * 2 + PANEL_GAP
h = TOP + sum(h + VGAP for h in ROW_H) + PAD + 10

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("910_95 与 A2_A3 两候选：骨架同构，四处差异")}</text>']

# 累计每行 y 起点
row_y = []
y_acc = TOP
for rh in ROW_H:
    row_y.append(y_acc)
    y_acc += rh + VGAP

for p, (title, contents) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + PANEL_GAP)
    cx = px + PANEL_W / 2
    title_lines = title.split("\n")
    for k, tl in enumerate(title_lines):
        L.append(f'<text x="{cx}" y="{TOP-40+k*18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(tl)}</text>')
    for i, content in enumerate(contents):
        y = row_y[i]
        rh = ROW_H[i]
        # 全部高亮(琥珀色):四行皆为差异点,整张图论点就是"这里不同"
        L.append(f'<rect x="{px}" y="{y}" width="{PANEL_W}" height="{rh}" rx="8" '
                  'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
        lines = content.split("\n")
        n = len(lines)
        y0 = y + rh / 2 - (n - 1) * 8
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="monospace" font-size="11" fill="#78350f">{esc(line)}</text>')

# 两面板间隙居中的维度标签胶囊(标本行对比的是哪个维度)
gap_cx = PAD + PANEL_W + PANEL_GAP / 2
for i, row in enumerate(ROWS):
    y = row_y[i]
    rh = ROW_H[i]
    label_x = gap_cx - LABEL_W / 2
    L.append(f'<rect x="{label_x}" y="{y+rh/2-14}" width="{LABEL_W}" height="28" rx="6" '
              'fill="#e0e7ff" stroke="#4f46e5"/>')
    L.append(f'<text x="{gap_cx}" y="{y+rh/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="#3730a3">{esc(row)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch28-branch-divergence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
