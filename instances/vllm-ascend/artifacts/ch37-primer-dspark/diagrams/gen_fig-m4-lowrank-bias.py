#!/usr/bin/env python3
"""fig-m4-lowrank-bias：before-after 模板。左=朴素 V×V 转移矩阵 O(V^2)；
右=DSpark 低秩分解 W1(V×r)+W2(V×r) -> O(Vr)，中间标 r=256 瓶颈维度。
底部标注偏置只加在 softmax 之前、p_k 仍是合法分布。全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, PAD, TOP = 980, 40, 118
PANEL_W = 400
GAP = 140
H_BIG = 200  # 朴素矩阵方块边长
BAR_H = 200  # 低秩两条矩阵的高度
BAR_W = 46

H = TOP + H_BIG + 200
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">转移偏置的低秩分解：省参数，不是近似 softmax</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" fill="#475569">'
     f'B(v,x&#8242;) 只是加到 softmax 之前的一个分数——分解方式不改变输出仍是合法分布这件事</text>']

# ---- 左面板：朴素 V x V ----
lx0 = PAD
lcx = lx0 + PANEL_W / 2
L.append(f'<text x="{lcx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#334155">朴素实现</text>')
bx = lcx - H_BIG/2
by = TOP
L.append(f'<rect x="{bx}" y="{by}" width="{H_BIG}" height="{H_BIG}" rx="6" '
         f'fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>')
# 网格纹理示意（不代表真实数字，仅示意"矩阵"这一视觉概念，非数据来源）
GRID = 6
for i in range(1, GRID):
    gx = bx + i * H_BIG / GRID
    gy = by + i * H_BIG / GRID
    L.append(f'<line x1="{gx}" y1="{by}" x2="{gx}" y2="{by+H_BIG}" stroke="#fca5a5" stroke-width="0.8"/>')
    L.append(f'<line x1="{bx}" y1="{gy}" x2="{bx+H_BIG}" y2="{gy}" stroke="#fca5a5" stroke-width="0.8"/>')
L.append(f'<text x="{lcx}" y="{by+H_BIG/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="#b91c1c">B(v, x&#8242;)</text>')
L.append(f'<text x="{lcx}" y="{by+H_BIG+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#b91c1c">V × V 转移矩阵</text>')
L.append(f'<text x="{lcx}" y="{by+H_BIG+46}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#b91c1c">存算 O(V²)</text>')

# ---- 右面板：低秩分解 W1(V x r) x W2(V x r)^T ----
rx0 = lx0 + PANEL_W + GAP
rcx = rx0 + PANEL_W / 2
L.append(f'<text x="{rcx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#15803d">DSpark 低秩分解</text>')

w1x = rx0 + 30
w2x = rx0 + PANEL_W - 30 - BAR_W
by2 = TOP
L.append(f'<rect x="{w1x}" y="{by2}" width="{BAR_W}" height="{BAR_H}" rx="4" '
         f'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<rect x="{w2x}" y="{by2}" width="{BAR_W}" height="{BAR_H}" rx="4" '
         f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{w1x+BAR_W/2}" y="{by2+BAR_H+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#15803d">W1</text>')
L.append(f'<text x="{w1x+BAR_W/2}" y="{by2+BAR_H+34}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#15803d">markov_w1</text>')
L.append(f'<text x="{w1x+BAR_W/2}" y="{by2+BAR_H+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#15803d">V × r</text>')
L.append(f'<text x="{w2x+BAR_W/2}" y="{by2+BAR_H+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#1d4ed8">W2</text>')
L.append(f'<text x="{w2x+BAR_W/2}" y="{by2+BAR_H+34}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#1d4ed8">markov_w2</text>')
L.append(f'<text x="{w2x+BAR_W/2}" y="{by2+BAR_H+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#1d4ed8">V × r</text>')

# r 瓶颈标注（居中，箭头指两条窄边）——瓶颈维度并入连线标签，避免与面板标题同高相撞
mid_x = (w1x + BAR_W + w2x) / 2
L.append(f'<line x1="{w1x+BAR_W+4}" y1="{by2+BAR_H/2}" x2="{w2x-4}" y2="{by2+BAR_H/2}" '
         f'stroke="#d97706" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#ao)"/>')
L.append(f'<text x="{mid_x}" y="{by2+BAR_H/2-16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#d97706">瓶颈 r = 256</text>')
L.append(f'<text x="{mid_x}" y="{by2+BAR_H/2+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#d97706">e = W1[x&#8242;]（取行，r 维）</text>')

# 调用标注（embed/bias）
L.append(f'<text x="{rcx}" y="{by2+BAR_H+70}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#334155">embed(x&#8242;) 取 W1 一行 → bias(e) 与 W2 相乘投回词表</text>')
L.append(f'<text x="{rcx}" y="{by2+BAR_H+90}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#15803d">B(v,x&#8242;) = (W2 · W1[x&#8242;]^T)_v ，存算 O(Vr)</text>')

# ---- 中间大箭头：朴素 -> 低秩 ----
mid_y = by + H_BIG / 2
ax1 = bx + H_BIG + 10
ax2 = w1x - 8
L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2}" y2="{mid_y}" stroke="#d97706" '
         f'stroke-width="2.5" marker-end="url(#ao)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{mid_y-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#d97706">秩 ≤ r 分解</text>')

# ---- 底部：softmax 仍合法分布 ----
foot_y = by2 + BAR_H + 130
L.append(f'<rect x="{PAD}" y="{foot_y-30}" width="{W-2*PAD}" height="80" rx="8" '
         f'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.4"/>')
L.append(f'<text x="{W/2}" y="{foot_y-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#1d4ed8">'
         f'p_k = softmax(U_k + B_k) 仍是合法分布</text>')
L.append(f'<text x="{W/2}" y="{foot_y+14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#1d4ed8">'
         f'→ 验证器以 min(1, q_k/p_k) 保留草稿：目标 q_k 在分子、草稿 p_k 在分母，零改动</text>')
L.append(f'<text x="{W/2}" y="{foot_y+36}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#1d4ed8">低秩只是省了存 B 的方式，不是对 softmax 本身做近似</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-lowrank-bias.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
