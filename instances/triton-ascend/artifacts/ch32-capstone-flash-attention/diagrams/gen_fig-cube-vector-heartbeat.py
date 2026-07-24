#!/usr/bin/env python3
"""swimlane 模板改写:2 泳道(Cube 核 / Vector 核),6 个有序步骤按所属核落位,
按执行顺序连箭头;末步回环到首步表示内循环下一个 BLOCK_N。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "内循环一次迭代 = Cube→Vector→Cube 三段心跳"
SUBTITLE = "_attn_fwd_inner 每处理一个 K/V 块(BLOCK_N 列)走一遍;两次 tl.dot 落 Cube,softmax 全落 Vector"

LANE_CUBE, LANE_VECTOR = "Cube 核（脉动阵列 · 矩阵乘）", "Vector 核（逐元素 / 规约）"
STEPS = [  # (lane, title, code, loc)
    (LANE_CUBE,   "① QK^T",       "tl.dot(q, trans_k)",              "L90"),
    (LANE_VECTOR, "② 减 max 稳定化", "tl.max(qk,1); qk−m_ij",           "L95-100"),
    (LANE_VECTOR, "③ softmax 权重", "p = tl.math.exp(qk)",             "L103"),
    (LANE_VECTOR, "④ 分母求和",     "l_ij = tl.sum(p,1)",              "L113"),
    (LANE_VECTOR, "⑤ 重标定",       "alpha = tl.math.exp(m_i−m_ij)",   "L115"),
    (LANE_CUBE,   "⑥ PV",          "tl.dot(p_cast, v[, acc])",        "L112,L120"),
]
LOOP_LABEL = "for start_n in range(lo,hi,BLOCK_N) — 下一个 K/V 块,回到①"
LOOP_LOC = "L84"

BOX_W, BOX_H, HGAP = 190, 68, 46
PAD, LANE_LABEL_W = 40, 210
LANE_CUBE_Y, LANE_VECTOR_Y = 150, 300
LANE_H = 110
TOP = 100

n = len(STEPS)
w = PAD * 2 + LANE_LABEL_W + n * BOX_W + (n - 1) * HGAP
h = 470

x0 = PAD + LANE_LABEL_W
xs_pos = [x0 + i * (BOX_W + HGAP) for i in range(n)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 泳道底色带 + 标签
for lane_name, y in [(LANE_CUBE, LANE_CUBE_Y), (LANE_VECTOR, LANE_VECTOR_Y)]:
    fill = "#eff6ff" if lane_name == LANE_CUBE else "#ecfdf5"
    L.append(f'<rect x="{PAD}" y="{y-LANE_H/2}" width="{w-2*PAD}" height="{LANE_H}" '
              f'fill="{fill}" opacity="0.6"/>')
    tcol = "#1d4ed8" if lane_name == LANE_CUBE else "#047857"
    L.append(f'<text x="{PAD+12}" y="{y+5}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{tcol}">{esc(lane_name)}</text>')

box_centers = []
for i, (lane, title, code, loc) in enumerate(STEPS):
    y_mid = LANE_CUBE_Y if lane == LANE_CUBE else LANE_VECTOR_Y
    x = xs_pos[i]
    y = y_mid - BOX_H / 2
    fill = "#3b82f6" if lane == LANE_CUBE else "#10b981"
    stroke = "#1d4ed8" if lane == LANE_CUBE else "#047857"
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="white">{esc(title)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+38}" text-anchor="middle" font-family="monospace" '
              f'font-size="10.5" fill="white">{esc(code)}</text>')
    badge_w = max(44, int(len(loc) * 6.6) + 18)
    L.append(f'<rect x="{x+BOX_W/2-badge_w/2}" y="{y+BOX_H-20}" width="{badge_w}" height="16" rx="8" '
              f'fill="white" opacity="0.95"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H-8}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" font-weight="bold" fill="{stroke}">{esc(loc)}</text>')
    box_centers.append((x, y, y_mid))

# 顺序箭头(考虑跨泳道时用折线经中点)
for i in range(n - 1):
    x1, y1, ymid1 = box_centers[i]
    x2, y2, ymid2 = box_centers[i + 1]
    sx, sy = x1 + BOX_W, ymid1
    ex, ey = x2, ymid2
    if ymid1 == ymid2:
        L.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" '
                  f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    else:
        midx = (sx + ex) / 2
        L.append(f'<path d="M{sx},{sy} L{midx},{sy} L{midx},{ey} L{ex},{ey}" '
                  f'fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# 回环箭头:末步 -> 首步(下方绕行)
lx0, ly0, lymid0 = box_centers[0]
lx1, ly1, lymid1 = box_centers[-1]
loop_y = LANE_VECTOR_Y + LANE_H / 2 + 55
start_x, start_y = lx1 + BOX_W / 2, ly1 + BOX_H
end_x, end_y = lx0 + BOX_W / 2, ly0 + BOX_H
L.append(f'<path d="M{start_x},{start_y} L{start_x},{loop_y} L{end_x},{loop_y} L{end_x},{end_y}" '
          f'fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#b)"/>')
L.append(f'<text x="{(start_x+end_x)/2}" y="{loop_y+20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#7c3aed">{esc(LOOP_LABEL)}</text>')
loop_badge_w = max(44, int(len(LOOP_LOC) * 6.6) + 18)
L.append(f'<rect x="{(start_x+end_x)/2-loop_badge_w/2}" y="{loop_y+28}" width="{loop_badge_w}" height="16" rx="8" '
          f'fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.2"/>')
L.append(f'<text x="{(start_x+end_x)/2}" y="{loop_y+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" font-weight="bold" '
          f'fill="#7c3aed">{esc(LOOP_LOC)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-cube-vector-heartbeat.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
