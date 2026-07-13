#!/usr/bin/env python3
"""fig-eplb-placement-grid: rebalance_experts 的最终输出——「卡 x 槽 -> 逻辑专家 id」
放置表。复用 paper-fig-1 的网格视觉语言(格子=物理槽位、格内数字=逻辑专家 id)。

数据来源 = explainer/traces/worked_example_trace.txt(MECH constraint-local-exchange
一节的逐卡结果),逐格核对:
  card0: old [3,4,1,2,1] -> final [3,4,1,7,5]  kept(in place)={3,4,1} new={7,5} D2D=2
  card1: old [0,5,6,7,5] -> final [0,4,6,3,2]  kept(in place)={6,0} new={4,3,2} D2D=3
被复制的专家 3(卡0槽0 / 卡1槽3)与专家 4(卡0槽1 / 卡1槽1)两份副本各分居两卡。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---- 数据(worked_example_trace.txt,逐格核对) ----
SLOTS = 5
CARDS = [
    {"name": "卡 0", "slots": [3, 4, 1, 7, 5], "kept": {3, 4, 1}, "d2d": 2},
    {"name": "卡 1", "slots": [0, 4, 6, 3, 2], "kept": {6, 0}, "d2d": 3},
]
# 被复制专家的两份副本坐标 (card_idx, slot_idx)
REPLICATED = [
    {"expert": 3, "a": (0, 0), "b": (1, 3), "color": "#7c3aed", "fill": "#ede9fe"},
    {"expert": 4, "a": (0, 1), "b": (1, 1), "color": "#0891b2", "fill": "#cffafe"},
]

# ---- 颜色 ----
INK = "#0f172a"
SUB = "#64748b"
KEPT_FILL, KEPT_STROKE = "#dcfce7", "#16a34a"       # 原位保留(绿)
NEW_FILL, NEW_STROKE = "#ffedd5", "#ea580c"         # 跨卡新到 / D2D 拷贝(橙)

# ---- 尺寸常量 ----
CELL_W, CELL_H = 56, 46
CELL_GAP = 12
PAD = 30
LABEL_W = 72          # 左侧行标签("卡 0"/"卡 1")宽
RIGHT_W = 150         # 右侧 D2D 注记宽
ROW_GAP = 62          # 两行之间留白(走副本连线)
HEADER_H = 26         # 列头("槽 0".."槽 4")高

TITLE_Y = PAD
SUB1_Y = TITLE_Y + 22
SUB2_Y = SUB1_Y + 19

GRID_LEFT = PAD + LABEL_W
GRID_TOP = SUB2_Y + 26 + HEADER_H     # 顶行格子上沿

COL_STEP = CELL_W + CELL_GAP


def col_x(c):
    return GRID_LEFT + c * COL_STEP


def row_y(r):
    return GRID_TOP + r * (CELL_H + ROW_GAP)


GRID_RIGHT = col_x(SLOTS - 1) + CELL_W
BOTTOM_ROW_Y = row_y(1)
GAP_TOP = row_y(0) + CELL_H
GAP_BOT = BOTTOM_ROW_Y

LEGEND_Y = BOTTOM_ROW_Y + CELL_H + 44
CAPTION_Y = LEGEND_Y + 34

W = GRID_RIGHT + RIGHT_W + PAD
H = CAPTION_Y + 46

DEFS = ['<defs></defs>']
BODY = []

# ---- 标题 ----
BODY.append(f'<text x="{PAD}" y="{TITLE_Y+4}" font-family="sans-serif" font-size="17" '
            f'font-weight="bold" fill="{INK}">'
            f'{esc("rebalance_experts 的最终输出:一张「卡 x 槽 -> 逻辑专家 id」放置表")}</text>')
BODY.append(f'<text x="{PAD}" y="{SUB1_Y+4}" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">'
            f'{esc("本例 2 卡 x 5 槽;格内数字 = 该槽装的逻辑专家 id。绿 = 原位保留(免搬),橙 = 跨卡新到(触发 D2D 拷贝)")}'
            '</text>')
BODY.append(f'<text x="{PAD}" y="{SUB2_Y+4}" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">'
            f'{esc("虚线 = 同一被复制专家的两份副本——始终分居两卡(不共卡)")}'
            '</text>')

# ---- 列头(槽 0 .. 槽 4) ----
for c in range(SLOTS):
    cx = col_x(c) + CELL_W / 2
    BODY.append(f'<text x="{cx:.1f}" y="{GRID_TOP-10:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                f'fill="{SUB}">{esc(f"槽 {c}")}</text>')

# ---- 行标签 + 格子 ----
for r, card in enumerate(CARDS):
    ry = row_y(r)
    # 行标签
    BODY.append(f'<text x="{GRID_LEFT-16:.1f}" y="{ry+CELL_H/2+5:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="14.5" font-weight="bold" '
                f'fill="{INK}">{esc(card["name"])}</text>')
    for c, eid in enumerate(card["slots"]):
        cx = col_x(c)
        is_kept = eid in card["kept"]
        fill = KEPT_FILL if is_kept else NEW_FILL
        stroke = KEPT_STROKE if is_kept else NEW_STROKE
        BODY.append(f'<rect x="{cx:.1f}" y="{ry:.1f}" width="{CELL_W}" height="{CELL_H}" '
                    f'rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        BODY.append(f'<text x="{cx+CELL_W/2:.1f}" y="{ry+CELL_H/2+7:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="20" font-weight="bold" '
                    f'fill="{INK}">{eid}</text>')
    # 右侧 D2D 注记
    ann_x = GRID_RIGHT + 20
    new_cnt = SLOTS - len(card["kept"])
    d2d_label = "D2D 拷贝 {}".format(card["d2d"])
    new_label = "(跨卡新到 {} 格)".format(new_cnt)
    BODY.append(f'<text x="{ann_x:.1f}" y="{ry+CELL_H/2-3:.1f}" '
                f'font-family="sans-serif" font-size="13" font-weight="bold" '
                f'fill="{NEW_STROKE}">{esc(d2d_label)}</text>')
    BODY.append(f'<text x="{ann_x:.1f}" y="{ry+CELL_H/2+15:.1f}" '
                f'font-family="sans-serif" font-size="11.5" '
                f'fill="{SUB}">{esc(new_label)}</text>')

# ---- 副本连线(在两行之间的留白里走正交虚线) ----
lanes = [0.40, 0.64]   # 两条子泳道,避免两组连线完全重合
for i, rep in enumerate(REPLICATED):
    (ca, sa), (cb, sb) = rep["a"], rep["b"]
    ax = col_x(sa) + CELL_W / 2
    bx = col_x(sb) + CELL_W / 2
    lane_y = GAP_TOP + (GAP_BOT - GAP_TOP) * lanes[i]
    color = rep["color"]
    if sa == sb:
        d = f'M{ax:.1f},{GAP_TOP:.1f} L{bx:.1f},{GAP_BOT:.1f}'
    else:
        d = (f'M{ax:.1f},{GAP_TOP:.1f} L{ax:.1f},{lane_y:.1f} '
             f'L{bx:.1f},{lane_y:.1f} L{bx:.1f},{GAP_BOT:.1f}')
    BODY.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8" '
                f'stroke-dasharray="5,4"/>')
    # 端点小圆点
    for (px, py) in ((ax, GAP_TOP), (bx, GAP_BOT)):
        BODY.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{color}"/>')
    # 泳道标签
    mid_x = (ax + bx) / 2
    label = f"专家 {rep['expert']} 两份副本"
    lw = cjk_text_width(label, 11.5)
    BODY.append(f'<rect x="{mid_x-lw/2-6:.1f}" y="{lane_y-9:.1f}" width="{lw+12:.1f}" '
                f'height="17" rx="8" fill="{rep["fill"]}" stroke="{color}" stroke-width="1"/>')
    BODY.append(f'<text x="{mid_x:.1f}" y="{lane_y+3.5:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                f'fill="{color}">{esc(label)}</text>')

# ---- 图例 ----
LEGEND = [
    (KEPT_FILL, KEPT_STROKE, "原位保留(免搬,共 5 格)"),
    (NEW_FILL, NEW_STROKE, "跨卡新到 = D2D 拷贝(共 5 格 = 2+3)"),
]
lx = GRID_LEFT
for key_fill, key_stroke, label in LEGEND:
    BODY.append(f'<rect x="{lx:.1f}" y="{LEGEND_Y-13:.1f}" width="16" height="16" '
                f'rx="3" fill="{key_fill}" stroke="{key_stroke}" stroke-width="1.6"/>')
    BODY.append(f'<text x="{lx+22:.1f}" y="{LEGEND_Y:.1f}" font-family="sans-serif" '
                f'font-size="12" fill="{INK}">{esc(label)}</text>')
    lx += 22 + cjk_text_width(label, 12) + 34

# ---- 图注条(给结论) ----
BODY.append(f'<rect x="{PAD}" y="{CAPTION_Y-16:.1f}" width="{W-2*PAD:.1f}" height="40" '
            f'rx="9" fill="#eef2ff" stroke="#6366f1" stroke-width="1.6"/>')
BODY.append(f'<text x="{W/2:.1f}" y="{CAPTION_Y+9:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12.5" fill="#3730a3">'
            f'{esc("放置表是本章真正的产物:5 格原位免搬、5 格跨卡新到只拷贝 2+3 份;专家 3、4 的副本各分居两卡以摊薄热度")}'
            '</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">']
L.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')
L += DEFS
L += BODY
L.append('</svg>')

out = Path(__file__).with_name("fig-eplb-placement-grid.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  W={W:.0f} H={H:.0f} ratio={W/H:.2f}")
