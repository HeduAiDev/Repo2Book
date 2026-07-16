#!/usr/bin/env python3
"""flow 模板：combine_fn 如何被"再编译"成 reduce_op 的 region body（而非在 Python 里被调用求值）。
5 个线性步骤，每步标注源码行号（均来自 explainer fig-combine-fn-to-region.numbers）。
第 4 步（AST 再编译，非 Python 求值）用橙色高亮，是本图论点的落点。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "combine_fn 是被再编译进 IR region，不是被 Python 调用"
SUBTITLE = "用户 combine_fn 经 make_combine_region → call_JitFunction → fn.parse()+CodeGenerator.visit 被编译成 reduce_op 的 region body"

STEPS = [
    {"label": ["①  core.reduce()", "建 make_combine_region", "闭包"], "loc": "core.py:L2049", "kind": "normal"},
    {"label": ["②  prototype 双入协议", "function_type(in, in×2)", "(2 入 1 出)"], "loc": "core.py:L2051", "kind": "normal"},
    {"label": ["③  call_JitFunction", "(_sum_combine, [a,b])", "a,b 是 tl.tensor"], "loc": "core.py:L2058", "kind": "normal"},
    {"label": ["④  fn.parse() +", "CodeGenerator.visit", "= AST 再编译，非 Python 调用"], "loc": "code_generator.py:L1075", "kind": "key"},
    {"label": ["⑤  create_reduce_ret", "写回 region body", "(结果句柄)"], "loc": "core.py:L2063", "kind": "normal"},
]

COLOR = {"normal": ("#dbeafe", "#1e40af"), "key": ("#ffedd5", "#c2410c")}

BOX_W, BOX_H, GAP, PAD, TOP = 240, 118, 46, 34, 108
n = len(STEPS)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = TOP + BOX_H + 120
box_x = [PAD + i * (BOX_W + GAP) for i in range(n)]
box_y = TOP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

sub_lines = [SUBTITLE[:60], SUBTITLE[60:]]
for i, line in enumerate(sub_lines):
    L.append(f'<text x="{PAD}" y="{PAD+20+i*16}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')

# arrows between boxes (drawn first so boxes sit on top visually is not needed, but keep simple: after boxes)
for i, step in enumerate(STEPS):
    x = box_x[i]
    fill, stroke = COLOR[step["kind"]]
    sw = "3" if step["kind"] == "key" else "1.5"
    L.append(f'<rect x="{x}" y="{box_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    lines = step["label"]
    n_lines = len(lines)
    y0 = box_y + BOX_H / 2 - (n_lines - 1) * 10 + 5
    for k, line in enumerate(lines):
        fs = "12.5" if k == 0 else "11"
        fw = 'font-weight="bold" ' if k == 0 else ''
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*20}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" {fw}'
                  f'fill="{stroke}">{esc(line)}</text>')
    # line-number tag below box
    L.append(f'<text x="{x+BOX_W/2}" y="{box_y+BOX_H+22}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" fill="#475569">{esc(step["loc"])}</text>')

for i in range(n - 1):
    x1 = box_x[i] + BOX_W
    x2 = box_x[i + 1]
    ay = box_y + BOX_H / 2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

legend_y = h - 56
L.append(f'<rect x="{PAD}" y="{legend_y-14}" width="16" height="16" rx="3" fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y-2}" font-family="sans-serif" font-size="12" fill="#374151">构造 / 写回步骤</text>')
L.append(f'<rect x="{PAD+180}" y="{legend_y-14}" width="16" height="16" rx="3" fill="#ffedd5" stroke="#c2410c" stroke-width="2"/>')
L.append(f'<text x="{PAD+202}" y="{legend_y-2}" font-family="sans-serif" font-size="12" fill="#374151">关键步：combine_fn 体在此被编译成 IR，从未被 Python 调用求值</text>')

foot_y = h - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">这解释了下一条规则：combine_fn 里只能写 tl.* 可追踪操作，因为它走的是编译路径，不是 Python 求值路径。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-combine-fn-to-region.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
