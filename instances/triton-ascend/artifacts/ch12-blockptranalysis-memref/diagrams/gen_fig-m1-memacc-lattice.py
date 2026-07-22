#!/usr/bin/env python3
"""fig-m1-memacc-lattice: MemAccType 三态全序格（state-machine 模板改写）。
主链横排三态 Undefined(0) < StrucMemAcc(1) < UnstrucMemAcc(2)，边标 merge=max 语义；
下方叠一个『一处污染全链』的 merge 示例。全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "MemAccType 三态全序格（BlockPtrAnalysis.h:L46 enum class MemAccVal）"
SUBTITLE = "merge 取大（上确界）：一条地址链上任一处非结构化，都会把整链染成 UnstrucMemAcc"

CHAIN = [("Undefined", "0"), ("StrucMemAcc", "1"), ("UnstrucMemAcc", "2")]
CHAIN_COLOR = [("#f1f5f9", "#64748b", "#334155"),
               ("#dcfce7", "#16a34a", "#14532d"),
               ("#fee2e2", "#dc2626", "#7f1d1d")]

BOX_W, BOX_H, HGAP, PAD, TOP = 190, 62, 90, 40, 108
n = len(CHAIN)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
chain_x = [PAD + i * (BOX_W + HGAP) for i in range(n)]

EX_BOX_W, EX_BOX_H, EX_GAP = 220, 56, 60
rule_y = TOP + BOX_H + 44
ex_y = rule_y + 40 + 46
res_y = ex_y + EX_BOX_H + 50
h = res_y + EX_BOX_H + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 主链三态
for i, (name, val) in enumerate(CHAIN):
    x = chain_x[i]
    fill, stroke, text_fill = CHAIN_COLOR[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{text_fill}">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+46}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="{text_fill}">值 = {esc(val)}</text>')
    if i < n - 1:
        ay = TOP + BOX_H / 2
        L.append(f'<line x1="{x+BOX_W}" y1="{ay}" x2="{chain_x[i+1]}" y2="{ay}" '
                  'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x+BOX_W+chain_x[i+1])/2}" y="{ay-10}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">更不结构化</text>')

# merge 规则条
L.append(f'<rect x="{PAD}" y="{rule_y}" width="{w-2*PAD}" height="40" rx="8" '
          'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{rule_y+25}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#3730a3">'
          f'{esc("merge 规则：value = max(this, other)  —  H:L66-L68")}</text>')

# 示例：一处污染全链
L.append(f'<text x="{PAD}" y="{ex_y-14}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#0f172a">示例：地址链上两处分析结果 merge</text>')

ex_x0 = PAD
ex_x1 = PAD + EX_BOX_W + EX_GAP

L.append(f'<rect x="{ex_x0}" y="{ex_y}" width="{EX_BOX_W}" height="{EX_BOX_H}" rx="8" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{ex_x0+EX_BOX_W/2}" y="{ex_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#14532d">操作数 A</text>')
L.append(f'<text x="{ex_x0+EX_BOX_W/2}" y="{ex_y+44}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#14532d">StrucMemAcc（1）</text>')

L.append(f'<rect x="{ex_x1}" y="{ex_y}" width="{EX_BOX_W}" height="{EX_BOX_H}" rx="8" '
          'fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>')
L.append(f'<text x="{ex_x1+EX_BOX_W/2}" y="{ex_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#7f1d1d">操作数 B（间接 load 派生）</text>')
L.append(f'<text x="{ex_x1+EX_BOX_W/2}" y="{ex_y+44}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#7f1d1d">UnstrucMemAcc（2）</text>')

mid_x = (ex_x0 + EX_BOX_W / 2 + ex_x1 + EX_BOX_W / 2) / 2
L.append(f'<line x1="{ex_x0+EX_BOX_W/2}" y1="{ex_y+EX_BOX_H}" x2="{mid_x}" y2="{res_y-10}" '
          'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<line x1="{ex_x1+EX_BOX_W/2}" y1="{ex_y+EX_BOX_H}" x2="{mid_x}" y2="{res_y-10}" '
          'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{mid_x}" y="{ex_y+EX_BOX_H+16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#94a3b8">无因果·仅汇入 merge</text>')

RES_W = 300
L.append(f'<rect x="{mid_x-RES_W/2}" y="{res_y}" width="{RES_W}" height="{EX_BOX_H}" rx="8" '
          'fill="#fee2e2" stroke="#dc2626" stroke-width="2.5"/>')
L.append(f'<text x="{mid_x}" y="{res_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#7f1d1d">merge 结果 = max(1,2)</text>')
L.append(f'<text x="{mid_x}" y="{res_y+44}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#7f1d1d">UnstrucMemAcc（2）—— 整链染色</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-memacc-lattice.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
