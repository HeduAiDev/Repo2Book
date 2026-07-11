#!/usr/bin/env python3
"""重绘自 arXiv:2604.12989 Figure 2(Illustration of one DDTree decoding round)。
布局对齐原图(ddtree_raw_0.svg / ddtree_raw_1.svg,论文原生 SVG 已下载核对):
(a) 一次块扩散前向从 previous bonus 根出发,构建深度不一的候选树(budget 向高概率分支倾斜);
(b) 验证沿树走两步匹配(绿),在第一个不匹配处产生 next bonus(橙,虚线)。
两个子图共享同一棵树拓扑,仅高亮不同。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "DDTree 一轮解码(重绘自 arXiv:2604.12989 Fig.2)"
SUBTITLE = "(a) 块扩散一次前向构建深度不一的候选树  (b) 验证沿树走两步匹配(绿),首个不匹配处产生 next bonus(橙)"

GRAY_FILL, GRAY_STROKE = "#f1f5f9", "#64748b"
BLUE_FILL, BLUE_STROKE = "#dbeafe", "#1d4ed8"
GREEN_FILL, GREEN_STROKE = "#dcfce7", "#16a34a"
ORANGE_FILL, ORANGE_STROKE = "#fed7aa", "#c2410c"

R = 22  # node radius

def panel(ox, oy, highlight):
    """在偏移 (ox,oy) 处画一棵树;highlight=True 时画验证走过的绿色路径 + next bonus。
    返回该 panel 使用的宽度、高度(供外部布局)。"""
    elems = []
    # local coordinates (before offset)
    input_box = (0, 130, 90, 46)  # x,y,w,h
    root = (150, 153)
    A = (300, 90)     # depth1 top
    B = (300, 230)    # depth1 bottom
    C1 = (430, 40)    # depth2 upper (child of A)
    C2 = (430, 140)   # depth2 middle (child of A)
    D1 = (430, 280)   # depth2 (child of B)
    E1 = (560, 10)    # depth3 (child of C1)
    E2 = (560, 70)    # depth3 (child of C1)
    E3 = (560, 170)   # depth3 (child of C2)
    NEXT_BONUS = (700, 220)  # 仅 panel (b) 使用

    def L(x1, y1, x2, y2, color, w_=2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        elems.append(f'<line x1="{x1+ox}" y1="{y1+oy}" x2="{x2+ox}" y2="{y2+oy}" '
                      f'stroke="{color}" stroke-width="{w_}"{d}/>')

    def node(cx, cy, fill, stroke, sw=2):
        elems.append(f'<circle cx="{cx+ox}" cy="{cy+oy}" r="{R}" fill="{fill}" '
                      f'stroke="{stroke}" stroke-width="{sw}"/>')

    # depth gridlines (3 dashed verticals at depth1/2/3 x-positions)
    for dx in (A[0], C1[0], E1[0]):
        elems.append(f'<line x1="{dx+ox}" y1="{-10+oy}" x2="{dx+ox}" y2="{320+oy}" '
                      'stroke="#e2e8f0" stroke-width="1.4"/>')

    # input box -> root
    ix, iy, iw, ih = input_box
    elems.append(f'<rect x="{ix+ox}" y="{iy+oy}" width="{iw}" height="{ih}" rx="8" '
                  f'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.4"/>')
    L(ix+iw, iy+ih/2, root[0]-R, root[1], "#94a3b8", 2)

    # gray tree edges (unhighlighted skeleton, always drawn light gray first)
    edges = [(root, A), (root, B), (A, C1), (A, C2), (B, D1), (C1, E1), (C1, E2), (C2, E3)]
    for (p, q) in edges:
        L(p[0], p[1], q[0], q[1], "#cbd5e1", 2.4)

    if highlight:
        # highlighted verification path: root -> A -> C2(green,粗箭头,带箭头标记表示行走方向)
        import math
        def shrink(p, q, r):
            dx, dy = q[0]-p[0], q[1]-p[1]
            dist = math.hypot(dx, dy)
            return (q[0]-dx/dist*r, q[1]-dy/dist*r)
        a_end = shrink(root, A, R+2)
        elems.append(f'<line x1="{root[0]+ox}" y1="{root[1]+oy}" x2="{a_end[0]+ox}" y2="{a_end[1]+oy}" '
                      f'stroke="{GREEN_STROKE}" stroke-width="5" marker-end="url(#gArrow)"/>')
        c2_end = shrink(A, C2, R+2)
        elems.append(f'<line x1="{A[0]+ox}" y1="{A[1]+oy}" x2="{c2_end[0]+ox}" y2="{c2_end[1]+oy}" '
                      f'stroke="{GREEN_STROKE}" stroke-width="5" marker-end="url(#gArrow)"/>')
        # dashed orange from C2 to next bonus(首个不匹配处,箭头指向新 bonus)
        nb_end = shrink(C2, NEXT_BONUS, R+2)
        elems.append(f'<line x1="{C2[0]+ox}" y1="{C2[1]+oy}" x2="{nb_end[0]+ox}" y2="{nb_end[1]+oy}" '
                      f'stroke="#c2410c" stroke-width="3" stroke-dasharray="8,5" marker-end="url(#oArrow)"/>')

    # nodes (draw after edges so circles sit on top)
    node(root[0], root[1], BLUE_FILL, BLUE_STROKE, 3)
    elems.append(f'<text x="{root[0]+ox}" y="{root[1]+oy+R+18}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                  f'fill="{BLUE_STROKE}">previous</text>')
    elems.append(f'<text x="{root[0]+ox}" y="{root[1]+oy+R+32}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                  f'fill="{BLUE_STROKE}">bonus</text>')

    plain_nodes = [B, C1, D1, E1, E2, E3]
    green_nodes = [A, C2] if highlight else []
    for n in [A, B, C1, C2, D1, E1, E2, E3]:
        if n in green_nodes:
            node(n[0], n[1], GREEN_FILL, GREEN_STROKE, 3)
        else:
            node(n[0], n[1], GRAY_FILL, GRAY_STROKE, 2)

    if highlight:
        node(NEXT_BONUS[0], NEXT_BONUS[1], ORANGE_FILL, ORANGE_STROKE, 3)
        elems.append(f'<text x="{NEXT_BONUS[0]+ox}" y="{NEXT_BONUS[1]+oy+R+18}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                      f'fill="{ORANGE_STROKE}">next</text>')
        elems.append(f'<text x="{NEXT_BONUS[0]+ox}" y="{NEXT_BONUS[1]+oy+R+32}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                      f'fill="{ORANGE_STROKE}">bonus</text>')

    panel_w = NEXT_BONUS[0] + R + 20 if highlight else E1[0] + R + 20
    panel_h = 320
    return elems, panel_w, panel_h

PANEL_A_OX, PANEL_A_OY = 40, 150
elems_a, pw_a, ph_a = panel(PANEL_A_OX, PANEL_A_OY, highlight=False)

PANEL_B_OX = PANEL_A_OX + pw_a + 60
elems_b, pw_b, ph_b = panel(PANEL_B_OX, PANEL_A_OY, highlight=True)

w = PANEL_B_OX + pw_b + 40
h = PANEL_A_OY + ph_a + 90

L_all = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
         '<defs>'
         '<marker id="gArrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="7" '
         'markerHeight="6" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#16a34a"/></marker>'
         '<marker id="oArrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="7" '
         'markerHeight="6" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#c2410c"/></marker>'
         '</defs>',
         f'<rect width="{w}" height="{h}" fill="white"/>',
         f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
         f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>',
         f'<text x="{PANEL_A_OX+pw_a/2}" y="{PANEL_A_OY-24}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">(a) 建树</text>',
         f'<text x="{PANEL_B_OX+pw_b/2 - 40}" y="{PANEL_A_OY-24}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#0f172a">(b) 验证行走</text>']
L_all.extend(elems_a)
L_all.extend(elems_b)

foot_y = PANEL_A_OY + ph_a + 40
L_all.append(f'<text x="40" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">(a) 树深度不一:budget 向高概率分支倾斜(上支展到深度 3,下支只展到深度 2)——对应本章「best-first 堆」构树过程。</text>')
L_all.append(f'<text x="40" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">(b) 验证沿树走两步(绿色路径,2 个投机节点匹配),第三步目标模型选中的 token 不在树的子节点中——产生 next bonus,树验证仍只需 target 一次前向。</text>')

L_all.append('</svg>')
out = Path(__file__).with_name("paper-fig-ddtree-2.svg")
out.write_text('\n'.join(L_all), encoding="utf-8")
print(f"wrote {out}")
