#!/usr/bin/env python3
"""before-after 模板(堆叠柱状图):每个 token 的『接受质量 min(p,q)』+『残差质量 (1-beta)p'』
堆叠后精确重构目标 p(x) —— 分布保持定理的可视化证明。虚线标出 p(x) 目标高度,
柱顶数字=堆叠总和,柱下方标注蒙特卡洛经验频率做交叉验证。数字来自 traces/dist_preserving.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TOKENS = ["A", "B", "C", "D"]
ACCEPT = {"A": 0.4, "B": 0.2, "C": 0.1, "D": 0.1}          # min(p,q)
RESID  = {"A": 0.1, "B": 0.1, "C": 0.0, "D": 0.0}          # (1-beta)p'
TARGET = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}          # p(x)
MCFREQ = {"A": 0.499, "B": 0.301, "C": 0.1, "D": 0.1}

BAR_W, BAR_MAXH, GAP_X, PAD, TOP = 92, 220, 46, 44, 130
w = PAD * 2 + len(TOKENS) * (BAR_W + GAP_X)
h = TOP + BAR_MAXH + 150
SCALE = BAR_MAXH / 0.5
baseline = TOP + BAR_MAXH

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-24}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">分布保持:接受质量 + 残差质量 精确重构目标 p(x)</text>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">min(p,q) + (1-beta)p\'(x) = p(x),对每个 token 恒成立(beta=0.8)——与 draft 分布 q 无关</text>']

for i, tok in enumerate(TOKENS):
    bx = PAD + i * (BAR_W + GAP_X)
    acc, res, tgt, mc = ACCEPT[tok], RESID[tok], TARGET[tok], MCFREQ[tok]
    acc_h = acc * SCALE
    res_h = res * SCALE
    # target dashed reference line
    ty = baseline - tgt * SCALE
    L.append(f'<line x1="{bx-10}" y1="{ty}" x2="{bx+BAR_W+10}" y2="{ty}" '
              'stroke="#94a3b8" stroke-dasharray="4,3" stroke-width="1.5"/>')
    # bottom segment: accept mass
    L.append(f'<rect x="{bx}" y="{baseline-acc_h}" width="{BAR_W}" height="{max(acc_h,1)}" '
              'fill="#bfdbfe" stroke="#2563eb" stroke-width="1.5"/>')
    # top segment: residual mass
    if res_h > 0:
        L.append(f'<rect x="{bx}" y="{baseline-acc_h-res_h}" width="{BAR_W}" height="{res_h}" '
                  'fill="#bbf7d0" stroke="#15803d" stroke-width="1.5"/>')
    # segment value labels
    L.append(f'<text x="{bx+BAR_W/2}" y="{baseline-acc_h/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#1d4ed8">{acc:.1f}</text>')
    if res_h > 0:
        L.append(f'<text x="{bx+BAR_W/2}" y="{baseline-acc_h-res_h/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#15803d">{res:.1f}</text>')
    # total label above stack
    L.append(f'<text x="{bx+BAR_W/2}" y="{baseline-acc_h-res_h-10}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">={tgt:.1f}</text>')
    # x label
    L.append(f'<text x="{bx+BAR_W/2}" y="{baseline+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">{esc(tok)}</text>')
    # MC freq below
    L.append(f'<text x="{bx+BAR_W/2}" y="{baseline+42}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#64748b">MC={mc:.3f}</text>')

L.append(f'<line x1="{PAD-10}" y1="{baseline}" x2="{PAD+len(TOKENS)*(BAR_W+GAP_X)-GAP_X+10}" '
          f'y2="{baseline}" stroke="#94a3b8" stroke-width="1"/>')

# legend
ly = h - 66
legend = [("#bfdbfe", "#2563eb", "接受质量 min(p,q)"), ("#bbf7d0", "#15803d", "残差质量 (1-beta)p'(x)")]
lx = PAD
for fill, stroke, label in legend:
    L.append(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 210
L.append(f'<line x1="{lx}" y1="{ly+8}" x2="{lx+24}" y2="{ly+8}" stroke="#94a3b8" '
          'stroke-dasharray="4,3" stroke-width="1.5"/>')
L.append(f'<text x="{lx+30}" y="{ly+13}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">目标 p(x) 参考线</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">A:0.4+0.1=0.5;B:0.2+0.1=0.3;C、D:0.1+0=0.1 —— 蒙特卡洛 N=400000 经验频率逼近各自 p(x)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-distribution-preserving.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
