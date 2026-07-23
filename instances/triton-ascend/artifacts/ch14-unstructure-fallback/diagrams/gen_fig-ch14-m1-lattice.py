#!/usr/bin/env python3
"""fig-ch14-m1-lattice — state-machine 模板改写为『四态全序链』:
unstructured(0) ⊑ structured(1) ⊑ scalarlike(2) ⊑ scalar(3)。
四格按 enum 声明序左→右排开,⊑ 箭头连接。
`isStructured(dim)` 坍缩谓词的色带**逐态独立标注,不合并成一个跨态框**:
- structured / scalar 两态各自标『isStructured(dim)=true → 不建循环』(绿)。
- scalarlike **单独一带**,写清楚它不直接等价:`PtrOffsetInfo::isStructured(dim)` 的实现是
  `this->scalarLike || structured[dim]==structured || structured[dim]==scalar`——
  `this->scalarLike` 是整个张量独立传播的布尔标志,不是本维取值为 scalarlike 本身;
  这一维恰为 scalarlike,并不必然让 isStructured(dim) 为真(取决于整张量的 scalarLike)。
- unstructured 单独一带,标『false → 建 scf.for』。
数据取自 explainer m1.figure_specs.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

STATES = [
    ("unstructured", 0, "#fecaca", "#dc2626", "偏移不可仿射描述"),
    ("structured", 1, "#bbf7d0", "#16a34a", "(offset,size,stride) 网格"),
    ("scalarlike", 2, "#bbf7d0", "#16a34a", "全维同值(广播源)"),
    ("scalar", 3, "#bbf7d0", "#16a34a", "size==1 的特例"),
]

BOX_W, BOX_H, HGAP, PAD, TOP = 220, 92, 70, 50, 118
w = PAD * 2 + len(STATES) * BOX_W + (len(STATES) - 1) * HGAP
BAND_H_NORMAL = 44
BAND_H_SPECIAL = 66
h = TOP + BOX_H + 44 + BAND_H_SPECIAL + 40 + 24 + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="38" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17" font-weight="bold" fill="#0f172a">'
     f'{esc("四态全序链:unstructured ⊑ structured ⊑ scalarlike ⊑ scalar")}</text>',
     f'<text x="{w/2}" y="62" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12.5" fill="#475569">'
     f'{esc("每个张量每一维取其一;偏序值按 enum 声明序 0<1<2<3,combineInfo 的 std::min 依赖此序")}</text>']

xs_ = []
for i, (name, val, fill, stroke, desc) in enumerate(STATES):
    x = PAD + i * (BOX_W + HGAP)
    xs_.append(x)
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+30}" text-anchor="middle" font-family="monospace" '
              f'font-size="16" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+52}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(f"枚举值 = {val}")}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+72}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#475569">{esc(desc)}</text>')

for i in range(len(STATES) - 1):
    x1, x2 = xs_[i] + BOX_W, xs_[i + 1]
    ay = TOP + BOX_H / 2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" '
              'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+x2)/2}" y="{ay-10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#334155">{esc("⊑")}</text>')
    L.append(f'<text x="{(x1+x2)/2}" y="{ay+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">{esc("更结构化")}</text>')

# isStructured(dim) 坍缩谓词——逐态独立一带,不合并成跨态框
band_y = TOP + BOX_H + 44
ENCROACH = 20  # 每侧向相邻 gap 借一点宽度,三带互不重叠(gap=70,20+20<70)

# unstructured:false → 建 scf.for(独立一带,红)
b0x0 = xs_[0] - ENCROACH
b0x1 = xs_[0] + BOX_W + ENCROACH
L.append(f'<rect x="{b0x0}" y="{band_y}" width="{b0x1-b0x0}" height="{BAND_H_NORMAL}" rx="8" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="6,3"/>')
L.append(f'<text x="{(b0x0+b0x1)/2}" y="{band_y+27}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#991b1b">'
          f'{esc("false → 建 scf.for,逐元素循环该维")}</text>')

# structured:isStructured(dim)=true → 不建循环(独立一带,绿)
b1x0 = xs_[1] - ENCROACH
b1x1 = xs_[1] + BOX_W + ENCROACH
L.append(f'<rect x="{b1x0}" y="{band_y}" width="{b1x1-b1x0}" height="{BAND_H_NORMAL}" rx="8" '
          'fill="#ecfdf5" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="6,3"/>')
L.append(f'<text x="{(b1x0+b1x1)/2}" y="{band_y+27}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#166534">'
          f'{esc("isStructured(dim)=true → 不建循环")}</text>')

# scalarlike:单独一带——不与 structured/scalar 共用同一个『isStructured(dim)=true』结论
b2x0 = xs_[2] - ENCROACH
b2x1 = xs_[2] + BOX_W + ENCROACH
L.append(f'<rect x="{b2x0}" y="{band_y}" width="{b2x1-b2x0}" height="{BAND_H_SPECIAL}" rx="8" '
          'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5" stroke-dasharray="3,3"/>')
L.append(f'<text x="{(b2x0+b2x1)/2}" y="{band_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#854d0e">'
          f'{esc("isStructured(dim) ≠ 本维 scalarlike")}</text>')
L.append(f'<text x="{(b2x0+b2x1)/2}" y="{band_y+42}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#854d0e">'
          f'{esc("取决于整张量 scalarLike 标志,")}</text>')
L.append(f'<text x="{(b2x0+b2x1)/2}" y="{band_y+58}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#854d0e">'
          f'{esc("与本维取值无关,见 §14.5")}</text>')

# scalar:isStructured(dim)=true → 不建循环(独立一带,绿)
b3x0 = xs_[3] - ENCROACH
b3x1 = xs_[3] + BOX_W + ENCROACH
L.append(f'<rect x="{b3x0}" y="{band_y}" width="{b3x1-b3x0}" height="{BAND_H_NORMAL}" rx="8" '
          'fill="#ecfdf5" stroke="#16a34a" stroke-width="1.5" stroke-dasharray="6,3"/>')
L.append(f'<text x="{(b3x0+b3x1)/2}" y="{band_y+27}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#166534">'
          f'{esc("isStructured(dim)=true → 不建循环")}</text>')

# 连接线:每个态框底部到自己那一带的顶部
for i, x in enumerate(xs_):
    cx = x + BOX_W / 2
    L.append(f'<line x1="{cx}" y1="{TOP+BOX_H}" x2="{cx}" y2="{band_y}" '
              'stroke="#94a3b8" stroke-width="1" stroke-dasharray="3,3"/>')

foot_y = band_y + BAND_H_SPECIAL + 40
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("概念范畴:ScalarLike ⊆ Structured;Unstructured = Structured 的补;scalar 是 size==1 的 scalarlike 特例")}</text>')
L.append(f'<text x="{w/2}" y="{foot_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#b45309">'
          f'{esc("PtrOffsetInfo::isStructured(dim) = this->scalarLike || structured[dim]∈{structured,scalar}——见 §14.5/§14.8")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-m1-lattice.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
