#!/usr/bin/env python3
"""fig31-3-weight-absorption-arch —— 权重吸收「架构图」(结构图,不是数值相等图)。

论点(一图一论点):上投影 W^UK 在「吸收前」长在 key 路径上——每个历史 token 都要
被它重放大出 full key 再和 query 打分;「吸收后」W^UK 被折进 query 侧、与 W^UQ 合成
静态 W̃,于是 query 直接落到潜空间、和缓存的 c^KV 握手,历史 key 永不物化。橙色 = 被
折叠的上投影(左边在 key 路径、右边搬到了 query 侧),蓝色 = 唯一入缓存的潜向量 c^KV。
输出侧对称(W^UV 折进输出投影 W^O)作底部一行带过。

结构图无数值见证——不引 trace 数字(代数恒等由结合律即证)。几何全部由常量计算。
用法: python3 gen_fig31-3-weight-absorption-arch.py → 同目录 svg。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cw(c):
    o = ord(c)
    if o == 0x20:
        return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if c.isascii() and c.isalnum():
        return 0.58
    return 0.5


def tw(s, size):
    return size * sum(cw(c) for c in s)


# ---------------- palette (语义色) ----------------
C_BOX_F, C_BOX_S, C_T, C_SUB = "#f1f5f9", "#475569", "#0f172a", "#64748b"
C_AB_F, C_AB_S, C_AB_T = "#fef3c7", "#d97706", "#b45309"   # amber = 被折叠的上投影 W^UK / W̃
C_CA_F, C_CA_S, C_CA_T = "#dbeafe", "#2563eb", "#1d4ed8"   # blue  = 唯一入缓存的潜向量 c^KV
C_SC_F, C_SC_S, C_SC_T = "#ede9fe", "#7c3aed", "#6d28d9"   # purple= 打分节点
C_ARR, C_TRANS = "#64748b", "#ea580c"
C_BAD, C_GOOD = "#b91c1c", "#15803d"

L = []


def rect(x, y, w, h, fill, stroke, sw=1.5, rx=9):
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size, fill, anchor="middle", weight=None, style=None):
    wt = f' font-weight="{weight}"' if weight else ''
    st = f' font-style="{style}"' if style else ''
    L.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
             f'font-size="{size}"{wt}{st} fill="{fill}">{esc(s)}</text>')


def arrow(p1, p2, color=C_ARR, sw=2.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{color}" stroke-width="{sw}"{d} marker-end="url(#a)"/>')


def node(x, y, w, h, title, sub, fill, stroke, tcolor):
    """圆角框 + 居中主标题(+可选副标题),两行竖向堆叠 16px 间距(不触发 tag-on-title)。"""
    rect(x, y, w, h, fill, stroke, sw=1.8)
    if sub:
        text(x + w / 2, y + h / 2 - 4, title, 13, tcolor, weight="bold")
        text(x + w / 2, y + h / 2 + 13, sub, 10.5, C_SUB)
    else:
        text(x + w / 2, y + h / 2 + 5, title, 13, tcolor, weight="bold")


# ---------------- geometry (零魔数) ----------------
BH = 50
WIN, WMID, WSC = 124, 128, 138
COLGAP, ROWGAP = 70, 60
PANEL_PAD = 22
ANNOT_H = 26                       # 面板内底部注释行(左面板红字)
AX = 0
BX = WIN + COLGAP
CX = WIN + COLGAP + WMID + COLGAP
PANEL_CONTENT_W = CX + WSC
PANEL_W = PANEL_CONTENT_W + PANEL_PAD * 2
YQ = 0
YK = BH + ROWGAP
SCORE_CY = (YQ + YK) / 2 + BH / 2
PANEL_CONTENT_H = YK + BH + ANNOT_H
PANEL_H = PANEL_CONTENT_H + PANEL_PAD * 2

TITLE_H = 40
LEGEND_H = 30
PTITLE_H = 30                      # 面板标题胶囊条
PTITLE_GAP = 14
TRANS_GAP = 104                    # 两面板之间:放橙色过渡箭头 + 说明
OUTER_PAD = 40
STRIP_GAP = 30
STRIP_H = 86

# 底部两行 takeaway(宽度驱动之一)
STRIP_L1 = "吸收前 → 吸收后:把静态的上投影 W^UK 从 key 路径折进 query 侧,合成 W̃ = (W^UK)^⊤ W^UQ——它是常量,这是可折的全部前提"
STRIP_L1B = "工程上不物化 W̃,让向量顺次穿过 W^UQ、(W^UK)^⊤ 两个瘦因子(见正文严谨框)"
STRIP_L2 = "于是缓存里只留潜向量 c^KV、历史 key 永不物化;输出侧对称——∑ w_j v_j = W^UV(∑ w_j c_j),W^UV 折进输出投影 W^O"
TRANS_LABEL = "把 W^UK 折进 query"

# 面板起点
top_content = TITLE_H + LEGEND_H
PTITLE_Y = top_content
panel_top = PTITLE_Y + PTITLE_H + PTITLE_GAP
PLx = OUTER_PAD
PRx = OUTER_PAD + PANEL_W + TRANS_GAP

# 画布宽度:取「两面板总宽」与「底部 takeaway 行宽」的较大者
_panels_right = PRx + PANEL_W + OUTER_PAD
_strip_right = OUTER_PAD + 16 + max(tw(STRIP_L1, 12.5), tw(STRIP_L1B, 12.5), tw(STRIP_L2, 12.5)) + 16 + OUTER_PAD
W = max(_panels_right, _strip_right)

strip_top = panel_top + PANEL_H + STRIP_GAP
H = strip_top + STRIP_H + OUTER_PAD

# ---------------- SVG ----------------
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}">')
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
         'markerHeight="4.5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
         '<marker id="t" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" markerHeight="4.5" '
         f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_TRANS}"/></marker></defs>')
L.append(f'<rect width="{W:.1f}" height="{H:.1f}" fill="white"/>')

# 标题
text(W / 2, 26, "权重吸收:把静态的上投影 W^UK 提前折进 query,历史 key 永不物化", 16, C_T, weight="bold")

# 图例(3 种语义色 → 必画图例)
legend = [(C_AB_S, "被折叠的上投影 W^UK / W̃"), (C_CA_S, "唯一入缓存的潜向量 c^KV"),
          (C_SC_S, "打分节点(内积)")]
lx = OUTER_PAD
ly = TITLE_H + 12
for color, lab in legend:
    rect(lx, ly - 11, 14, 14, color, color, sw=0, rx=3)
    text(lx + 20, ly, lab, 11.5, C_T, anchor="start")
    lx += 20 + tw(lab, 11.5) + 30

# ---------------- 一个面板 ----------------
def panel(px, title, tbar_color, before):
    # 面板标题胶囊条
    rect(px, PTITLE_Y, PANEL_W, PTITLE_H, tbar_color, tbar_color, sw=0, rx=8)
    text(px + PANEL_W / 2, PTITLE_Y + PTITLE_H / 2 + 5, title, 13.5, "white", weight="bold")
    # 面板容器(浅底,分组 before/after;过渡箭头端点附着其左右边)
    rect(px, panel_top, PANEL_W, PANEL_H, "#fbfcfe", "#e2e8f0", sw=1.2, rx=12)
    ox, oy = px + PANEL_PAD, panel_top + PANEL_PAD
    ax, bx, cx = ox + AX, ox + BX, ox + CX
    yq, yk = oy + YQ, oy + YK
    cyq, cyk = yq + BH / 2, yk + BH / 2
    score_top = oy + SCORE_CY - BH / 2
    score_cy = oy + SCORE_CY
    # 输入列:c^Q(query 潜向量) / c^KV(缓存)
    node(ax, yq, WIN, BH, "c^Q", "query 潜向量", C_BOX_F, C_BOX_S, C_T)
    node(ax, yk, WIN, BH, "c^KV", "缓存·唯一落盘", C_CA_F, C_CA_S, C_CA_T)
    if before:
        # ── 吸收前:query 走 W^UQ;key 路径上还挂着 W^UK,逐 token 物化 full key ──
        arrow((ax + WIN, cyq), (bx, cyq))
        text((ax + WIN + bx) / 2, cyq - 9, "W^UQ", 12, C_T, weight="bold")
        node(bx, yq, WMID, BH, "q", "打分用 query", C_BOX_F, C_BOX_S, C_T)
        node(bx, yk, WMID, BH, "k", "full key", C_AB_F, C_AB_S, C_AB_T)
        arrow((ax + WIN, cyk), (bx, cyk), color=C_AB_S)
        text((ax + WIN + bx) / 2, cyk - 9, "W^UK", 12, C_AB_T, weight="bold")
        node(cx, score_top, WSC, BH, "打分", "⟨q, k⟩", C_SC_F, C_SC_S, C_SC_T)
        arrow((bx + WMID, cyq), (cx, score_cy - BH * 0.28))
        arrow((bx + WMID, cyk), (cx, score_cy + BH * 0.28), color=C_AB_S)
        text(ax + WIN / 2, yk + BH + 17, "每个历史 token 都要重放大出 full key", 10.5, C_BAD)
    else:
        # ── 吸收后:W^UK 折进 query 侧成 W̃,key 路径消失,缓存直接握手 ──
        node(bx, yq, WMID, BH, "q̃", "落在潜空间", C_AB_F, C_AB_S, C_AB_T)
        arrow((ax + WIN, cyq), (bx, cyq), color=C_AB_S)          # 覆盖:query 侧现在走 W̃
        text((ax + WIN + bx) / 2, cyq - 9, "W̃", 13, C_AB_T, weight="bold")
        node(cx, score_top, WSC, BH, "潜空间打分", "⟨q̃, c^KV⟩", C_SC_F, C_SC_S, C_SC_T)
        arrow((bx + WMID, cyq), (cx, score_cy - BH * 0.28), color=C_AB_S)
        arrow((ax + WIN, cyk), (cx, score_cy + BH * 0.28), color=C_CA_S)   # 缓存直接握手
        text(bx + WMID / 2, cyk - 9, "直接内积", 11, C_CA_T, weight="bold")
        text(ax + WIN / 2 + 30, yk + BH + 17, "key 永不物化,省掉逐 token 重放大", 10.5, C_GOOD)


panel(PLx, "吸收前:key 逐 token 物化", C_BAD, before=True)
panel(PRx, "吸收后:W^UK 折进 query", C_GOOD, before=False)

# 面板间橙色过渡箭头(端点附着两面板容器左右边)
mid_y = panel_top + PANEL_H / 2
arrow((PLx + PANEL_W, mid_y), (PRx, mid_y), color=C_TRANS, sw=2.6)
L[-1] = L[-1].replace('marker-end="url(#a)"', 'marker-end="url(#t)"')
text((PLx + PANEL_W + PRx) / 2, mid_y - 12, TRANS_LABEL, 11.5, C_TRANS, weight="bold")

# 底部 takeaway 条
rect(OUTER_PAD, strip_top, W - 2 * OUTER_PAD, STRIP_H, "#fffbeb", C_AB_S, sw=1.4, rx=10)
text(OUTER_PAD + 16, strip_top + 24, STRIP_L1, 12.5, "#78350f", anchor="start", weight="bold")
text(OUTER_PAD + 16, strip_top + 46, STRIP_L1B, 12.5, "#78350f", anchor="start")
text(OUTER_PAD + 16, strip_top + 68, STRIP_L2, 12.5, "#78350f", anchor="start")

L.append('</svg>')
out = Path(__file__).with_name("fig31-3-weight-absorption-arch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W:.0f}x{H:.0f})")
