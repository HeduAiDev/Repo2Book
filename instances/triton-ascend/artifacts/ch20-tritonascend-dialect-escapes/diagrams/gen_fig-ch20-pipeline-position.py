#!/usr/bin/env python3
"""fig-ch20-pipeline-position: flow 模板。
三条逃生舱（HIVM/HFusion/LLVM）在 ttir_to_linalg 管线里紧邻挂载
（compiler.py:L148-L150），全排在主链收官 add_triton_to_linalg（L157）之前；
呼应 ch10 §10.1 那 18 趟里的三趟。风格与 ch10 fig-ch10-m1-pipeline 一致。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


TITLE = "三条逃生舱在 ttir_to_linalg 管线中的位置（compiler.py:L142-L157）"
SUBTITLE = "hivm → hfusion → llvm 紧邻挂载，全排在主链收官 add_triton_to_linalg 之前；三舱合计仅 5 个 pattern"

# 每个阶段：(标题, [详情行...], 配色 kind)
STAGES = [
    ("前置 pass\n（annotation / unstructure 等）",
     ["L142-L147", "去向：ascend.* 等（发射方言 op）", "回指 ch19 等"],
     "pre"),
    ("① HIVM 舱\nTritonToHIVM",
     ["L148 add_triton_to_hivm", "驱动器：applyPartialConversion", "1 个 pattern → hivm（双核同步）"],
     "hivm"),
    ("② HFusion 舱\nTritonToHFusion",
     ["L149 add_triton_to_hfusion", "驱动器：applyPatternsAndFoldGreedily（贪婪）", "3 个 pattern → hfusion（融合算子）"],
     "hfusion"),
    ("③ LLVM 舱\nTritonToLLVM",
     ["L150 add_triton_to_llvm", "驱动器：applyPartialConversion", "1 个 pattern → LLVM（内联汇编）"],
     "llvm"),
    ("收官：主链\nTritonToLinalg",
     ["L157 add_triton_to_linalg", "驱动器：（大量 pattern，非本章范围）", "去向：linalg / memref（结构化部分）"],
     "final"),
]

COLOR = {
    "pre":     ("#f1f5f9", "#94a3b8", "#334155"),
    "hivm":    ("#dbeafe", "#2563eb", "#1e3a8a"),
    "hfusion": ("#ede9fe", "#7c3aed", "#4c1d95"),
    "llvm":    ("#ccfbf1", "#0d9488", "#134e4a"),
    "final":   ("#fef3c7", "#d97706", "#78350f"),
}

BOX_W, BOX_H = 232, 118
HGAP = 30
PAD, TOP = 40, 100

n = len(STAGES)
row_w = n * BOX_W + (n - 1) * HGAP
w = PAD * 2 + row_w
h = TOP + BOX_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

xs_boxes = [PAD + i * (BOX_W + HGAP) for i in range(n)]

for i, (title, details, kind) in enumerate(STAGES):
    x = xs_boxes[i]
    fill, stroke, text_fill = COLOR[kind]
    sw = 3 if kind in ("hivm", "hfusion", "llvm", "final") else 1.5
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    # 标题（可能多行）
    title_lines = title.split("\n")
    ty = TOP + 22
    for k, line in enumerate(title_lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{ty+k*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{text_fill}">{esc(line)}</text>')
    # 分隔线
    sep_y = ty + len(title_lines) * 17 - 6
    L.append(f'<line x1="{x+14}" y1="{sep_y}" x2="{x+BOX_W-14}" y2="{sep_y}" '
              f'stroke="{stroke}" stroke-width="1" stroke-dasharray="3,2"/>')
    # 详情行
    dy = sep_y + 18
    for line in details:
        L.append(f'<text x="{x+14}" y="{dy}" font-family="sans-serif" font-size="10.5" '
                  f'fill="{text_fill}">{esc(line)}</text>')
        dy += 16

    # 箭头到下一个 box
    if i < n - 1:
        y_mid = TOP + BOX_H / 2
        x2s = xs_boxes[i + 1]
        L.append(f'<line x1="{x+BOX_W}" y1="{y_mid}" x2="{x2s}" y2="{y_mid}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 底部框：主链收官分界线标注（L148-150 均 < L157）
brace_y = TOP + BOX_H + 30
L.append(f'<line x1="{xs_boxes[1]}" y1="{brace_y}" x2="{xs_boxes[3]+BOX_W}" y2="{brace_y}" '
          'stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<line x1="{xs_boxes[1]}" y1="{brace_y}" x2="{xs_boxes[1]}" y2="{brace_y-8}" stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<line x1="{xs_boxes[3]+BOX_W}" y1="{brace_y}" x2="{xs_boxes[3]+BOX_W}" y2="{brace_y-8}" stroke="#94a3b8" stroke-width="1.5"/>')
brace_cx = (xs_boxes[1] + xs_boxes[3] + BOX_W) / 2
L.append(f'<text x="{brace_cx}" y="{brace_y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#475569">{esc("三舱窄侧道：挂载行 L148-L150，全部小于主链收官行 L157")}</text>')

# 图例
legend_y = brace_y + 44
legend_items = [
    ("hivm", "HIVM 舱：applyPartialConversion，1 pattern"),
    ("hfusion", "HFusion 舱：applyPatternsAndFoldGreedily，3 pattern"),
    ("llvm", "LLVM 舱：applyPartialConversion，1 pattern"),
]
lx = PAD
for kind, label in legend_items:
    fill, stroke, _ = COLOR[kind]
    L.append(f'<rect x="{lx}" y="{legend_y-13}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    tw = sum((13 if ord(c) > 0x2E80 else 6.5) for c in label)
    L.append(f'<text x="{lx+22}" y="{legend_y}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + tw + 30

# 底部数字小结
foot_y = legend_y + 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#0f172a" font-weight="bold">'
          f'{esc("三舱 pattern 合计 = 5（HFusion 3 + HIVM 1 + LLVM 1），相较主链大量结构化 pattern 是极小子集")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch20-pipeline-position.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size={w}x{h}")
