#!/usr/bin/env python3
"""paper-fig-3: 重绘自 arXiv:2405.04434 Fig.3——MHA / GQA / MQA / MLA 四种注意力
机制的 K、V 缓存结构对比(原图已抓到:https://arxiv.org/html/2405.04434v5/x4.png)。
信息结构对齐原图(逐头独立→分组共享→全体共享→压缩成潜向量按投影现算),配色套本书
语言(与本章 fig31-5 KV cache 账单表同色系:MHA 灰 / GQA 琥珀 / MQA 红 / MLA 蓝),
文字译中。原图为节省宽度画了 8 头,这里保留 6 头(每组仍是 2 头共享,结构关系不变)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

N_HEADS = 6
BOX_W, BOX_H = 26, 62
HEAD_GAP = 14
STEP = BOX_W + HEAD_GAP

COLORS = {"MHA": "#94a3b8", "GQA": "#fbbf24", "MQA": "#fca5a5", "MLA": "#1d4ed8"}
Q_FILL, Q_STROKE = "#f1f5f9", "#1e293b"
INK = "#0f172a"
SUB = "#64748b"

PANELS = [
    {"key": "MHA", "title": "多头注意力(MHA)",
     "groups": [[i] for i in range(N_HEADS)],
     "note1": "每个头各存一份完整 K、V", "note2": "缓存量 = O(头数)"},
    {"key": "GQA", "title": "分组查询注意力(GQA)",
     "groups": [[0, 1], [2, 3], [4, 5]],
     "note1": "每组头共享一份 K、V", "note2": "缓存量 = O(组数)"},
    {"key": "MQA", "title": "多查询注意力(MQA)",
     "groups": [list(range(N_HEADS))],
     "note1": "全部头共享同一份 K、V", "note2": "缓存量 = O(1),但表达力受限"},
]
MLA_TITLE = "多头潜在注意力(MLA)"
MLA_NOTE1 = "只缓存 1 份压缩潜向量"
MLA_NOTE2 = "推理时按投影现算各头 K、V"

# ---- 面板内间距(逐头/分组三态面板通用) ----
PPAD = 22          # 面板左右内边距
PANEL_W = N_HEADS * BOX_W + (N_HEADS - 1) * HEAD_GAP + 2 * PPAD
PANEL_GAP = 36      # 面板之间(含分隔虚线)的间距

# ---- MLA 面板专属:个体 K/V 列 + 投影箭头 + 压缩潜向量 ----
GAP_TO_ARROW = 22
ARROW_W = 108
LATENT_W = 26
MLA_PANEL_W = PPAD + (N_HEADS * BOX_W + (N_HEADS - 1) * HEAD_GAP) + GAP_TO_ARROW + ARROW_W + LATENT_W + PPAD

LEFT_LABEL_W = 78
PAD = 30

# ---- 纵向布局 ----
TOP_PAD = 22
TITLE_Y = TOP_PAD + 16
SUBTITLE_Y = TITLE_Y + 22
PANEL_TITLE_Y = SUBTITLE_Y + 42
ACCENT_Y = PANEL_TITLE_Y + 10
ROW_TOP = ACCENT_Y + 22
V_Y = ROW_TOP
GAP_VK = 8
K_Y = V_Y + BOX_H + GAP_VK
GAP_KQ = 64
Q_Y = K_Y + BOX_H + GAP_KQ
NOTE1_Y = Q_Y + BOX_H + 26
NOTE2_Y = NOTE1_Y + 18
PANEL_BOTTOM = NOTE2_Y + 8
SUMMARY_TOP = PANEL_BOTTOM + 22
SUMMARY_H = 92
BOTTOM_PAD = 22

H = SUMMARY_TOP + SUMMARY_H + BOTTOM_PAD

L = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 0 0">']  # 占位,最后回填 viewBox
DEFS = ['<defs>',
        '<marker id="arrow" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker>']
for key, color in COLORS.items():
    DEFS.append(f'<pattern id="hatch-{key}" width="7" height="7" patternUnits="userSpaceOnUse" '
                f'patternTransform="rotate(45)"><rect width="7" height="7" fill="{color}"/>'
                f'<line x1="0" y1="0" x2="0" y2="7" stroke="#ffffff" stroke-width="3"/></pattern>')
DEFS.append('<pattern id="hatch-legend" width="7" height="7" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><rect width="7" height="7" fill="#475569"/>'
            '<line x1="0" y1="0" x2="0" y2="7" stroke="#ffffff" stroke-width="3"/></pattern>')
DEFS.append('</defs>')

BODY = []


def box(x, y, w, h, fill, stroke, sw=1.6):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6" ' \
           f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def dotted(x1, y1, x2, y2):
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" ' \
           f'stroke="{INK}" stroke-width="1.3" stroke-dasharray="2,3"/>'


# ---- 顶部标题/副标题/图例 ----
BODY.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
            f'font-weight="bold" fill="{INK}">{esc("重绘参考图:四种注意力机制的 K、V 缓存结构对比")}</text>')
BODY.append(f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">{esc("斜纹色块 = 推理时需要缓存;MLA 把逐头 K、V 压缩成一份潜向量,按需投影现算,不逐头单独缓存")}</text>')

# 图例(右上角)
legend_w = 20
LEGEND_X = None  # 占位,下面算出总宽后再放置(见文末回填)

# ---- 逐面板绘制,cursor 为运行中的 x 坐标 ----
cursor = PAD + LEFT_LABEL_W

# 行标签(只在第一个面板左侧画一次)
row_label_cx = PAD + LEFT_LABEL_W - 14
for label, y in [("值(V)", V_Y + BOX_H / 2), ("键(K)", K_Y + BOX_H / 2), ("查询(Q)", Q_Y + BOX_H / 2)]:
    BODY.append(f'<text x="{row_label_cx:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="13" font-weight="bold" '
                f'fill="{INK}">{esc(label)}</text>')

panel_edges = []  # 记录每面板左右边界,便于画分隔虚线

for pi, panel in enumerate(PANELS):
    color = COLORS[panel["key"]]
    px0 = cursor
    px1 = px0 + PANEL_W
    panel_edges.append((px0, px1))
    cx = px0 + PANEL_W / 2

    BODY.append(f'<text x="{cx:.1f}" y="{PANEL_TITLE_Y}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="14.5" font-weight="bold" '
                f'fill="{INK}">{esc(panel["title"])}</text>')
    # 强调色条,呼应本章账单表同色系
    accent_w = N_HEADS * BOX_W + (N_HEADS - 1) * HEAD_GAP
    accent_x = cx - accent_w / 2
    BODY.append(f'<rect x="{accent_x:.1f}" y="{ACCENT_Y}" width="{accent_w:.1f}" height="4" '
                f'rx="2" fill="{color}"/>')

    heads_x0 = px0 + PPAD
    for gi, group in enumerate(panel["groups"]):
        xs_group = [heads_x0 + h_i * STEP for h_i in group]
        gx0, gx1 = xs_group[0], xs_group[-1] + BOX_W
        gcx = (gx0 + gx1) / 2
        # 共享/独立的 K、V(始终 hatch = 缓存)
        BODY.append(box(gcx - BOX_W / 2, V_Y, BOX_W, BOX_H, f'url(#hatch-{panel["key"]})', "#1e293b"))
        BODY.append(box(gcx - BOX_W / 2, K_Y, BOX_W, BOX_H, f'url(#hatch-{panel["key"]})', "#1e293b"))
        for h_i, hx in zip(group, xs_group):
            qx = hx
            BODY.append(box(qx, Q_Y, BOX_W, BOX_H, Q_FILL, Q_STROKE))
            BODY.append(dotted(qx + BOX_W / 2, Q_Y, gcx, K_Y + BOX_H))

    BODY.append(f'<text x="{cx:.1f}" y="{NOTE1_Y}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="12" fill="{INK}">{esc(panel["note1"])}</text>')
    BODY.append(f'<text x="{cx:.1f}" y="{NOTE2_Y}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="11" fill="{SUB}">{esc(panel["note2"])}</text>')

    cursor = px1 + PANEL_GAP

# ---- MLA 面板(个体 K/V 不 hatch + 压缩潜向量 hatch + 投影箭头) ----
mla_px0 = cursor
mla_px1 = mla_px0 + MLA_PANEL_W
panel_edges.append((mla_px0, mla_px1))
heads_w = N_HEADS * BOX_W + (N_HEADS - 1) * HEAD_GAP
heads_x0 = mla_px0 + PPAD
mla_cx_for_title = mla_px0 + PPAD + heads_w / 2 + (GAP_TO_ARROW + ARROW_W + LATENT_W) / 2

BODY.append(f'<text x="{mla_px0 + MLA_PANEL_W / 2:.1f}" y="{PANEL_TITLE_Y}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14.5" font-weight="bold" '
            f'fill="{INK}">{esc(MLA_TITLE)}</text>')
BODY.append(f'<rect x="{heads_x0:.1f}" y="{ACCENT_Y}" width="{heads_w:.1f}" height="4" rx="2" '
            f'fill="{COLORS["MLA"]}"/>')

mla_light = "#dbeafe"
mla_stroke = "#1d4ed8"
last_col_right = heads_x0 + heads_w
for h_i in range(N_HEADS):
    hx = heads_x0 + h_i * STEP
    BODY.append(box(hx, V_Y, BOX_W, BOX_H, mla_light, mla_stroke))
    BODY.append(box(hx, K_Y, BOX_W, BOX_H, mla_light, mla_stroke))
    BODY.append(box(hx, Q_Y, BOX_W, BOX_H, Q_FILL, Q_STROKE))
    BODY.append(dotted(hx + BOX_W / 2, Q_Y, hx + BOX_W / 2, K_Y + BOX_H))

latent_x = last_col_right + GAP_TO_ARROW + ARROW_W
latent_y0, latent_y1 = V_Y, K_Y + BOX_H
BODY.append(box(latent_x, latent_y0, LATENT_W, latent_y1 - latent_y0, 'url(#hatch-MLA)', "#1e293b"))
# 扇形虚线:潜向量 <-> 各头 K/V 区块两角
BODY.append(dotted(last_col_right, V_Y, latent_x, latent_y0))
BODY.append(dotted(last_col_right, K_Y + BOX_H, latent_x, latent_y1))
# 投影箭头(潜向量 -> 各头列),居中于 V/K 两行之间;两端分别贴住潜向量框左边与
# 各头列区块右边(元素边缘取值,不悬空)。
arrow_y = (V_Y + K_Y + BOX_H) / 2
arrow_x0 = latent_x
arrow_x1 = last_col_right
BODY.append(f'<line x1="{arrow_x0:.1f}" y1="{arrow_y:.1f}" x2="{arrow_x1:.1f}" y2="{arrow_y:.1f}" '
            f'stroke="{INK}" stroke-width="5" marker-end="url(#arrow)"/>')
BODY.append(f'<text x="{(arrow_x0 + arrow_x1) / 2:.1f}" y="{arrow_y - 10:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11.5" font-style="italic" '
            f'fill="{INK}">{esc("投影还原")}</text>')
BODY.append(f'<text x="{latent_x + LATENT_W / 2:.1f}" y="{latent_y1 + 20:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11.5" fill="{INK}">{esc("压缩潜向量")}</text>')
BODY.append(f'<text x="{latent_x + LATENT_W / 2:.1f}" y="{latent_y1 + 36:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11.5" fill="{INK}">{esc("KV")}</text>')

mla_cx = mla_px0 + MLA_PANEL_W / 2
BODY.append(f'<text x="{mla_cx:.1f}" y="{NOTE1_Y}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" fill="{INK}">{esc(MLA_NOTE1)}</text>')
BODY.append(f'<text x="{mla_cx:.1f}" y="{NOTE2_Y}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" fill="{SUB}">{esc(MLA_NOTE2)}</text>')

cursor = mla_px1

W = cursor + PAD

# ---- 图例(现在已知总宽,放右上角) ----
legend_x = W - PAD - 190
BODY.insert(0, box(legend_x, TITLE_Y - 15, 18, 14, 'url(#hatch-legend)', "#1e293b", 1.2))
BODY.insert(1, f'<text x="{legend_x + 24:.1f}" y="{TITLE_Y - 3:.1f}" font-family="sans-serif" '
               f'font-size="12" fill="{INK}">{esc("斜纹 = 推理时缓存")}</text>')

# ---- 面板分隔虚线(画在最外层,先加入 BODY 前段亦可,这里直接 append 不影响遮挡关系,均为细线) ----
for i in range(len(panel_edges) - 1):
    div_x = (panel_edges[i][1] + panel_edges[i + 1][0]) / 2
    BODY.append(f'<line x1="{div_x:.1f}" y1="{PANEL_TITLE_Y - 22}" x2="{div_x:.1f}" y2="{PANEL_BOTTOM}" '
                f'stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4"/>')

# ---- 底部综合结论条 ----
BODY.append(f'<rect x="{PAD}" y="{SUMMARY_TOP}" width="{W - 2 * PAD:.1f}" height="{SUMMARY_H}" rx="10" '
            'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
BODY.append(f'<text x="{W / 2:.1f}" y="{SUMMARY_TOP + 32:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#92400e">'
            f'{esc("共享粒度:逐头独立 → 逐组共享 → 全体共享 → 压缩成 1 份潜向量,缓存量随之从 O(头数) 降到 O(1)")}</text>')
BODY.append(f'<text x="{W / 2:.1f}" y="{SUMMARY_TOP + 60:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#92400e">'
            f'{esc("MLA 用「投影现算」换回损失的表达力——缓存量与 MQA 同量级,但每个头看到的 K、V 并不相同")}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">']
L.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')
L += DEFS
L += BODY
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  W={W:.0f} H={H:.0f} ratio={W/H:.2f}")
