#!/usr/bin/env python3
"""paper-fig-3: 重绘自 arXiv:2606.19348 Fig.3——CSA 核心架构(原图已抓到:
https://arxiv.org/html/2606.19348v1/x3.png)。信息结构对齐原图:KV token 隐状态
经 token 级压缩器产出压缩 KV 条目;同一份隐状态另走一条压缩器产出压缩索引器 key,
连同由 query 派生的索引器 query 一起送进 Lightning Indexer 内部的多查询注意力算出
索引得分;索引得分 + 压缩 KV 条目一起喂给 Top-k 选择器选出选中的压缩 KV 条目,
与滑窗未压缩 KV 条目拼接后,连同 query 一起送入共享 KV 的多查询注意力——三步
(压缩 → top-k 稀疏 → 与滑窗合并送入核注意力)首尾相连。配色套本章既有 CSA/DSA
强调色(CSA 绿 #059669 / 索引器紫 #7c3aed,承 fig36-1 谱系图配色),文字译中,
非逐字复刻原图像素,provenance=原论文本身。全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


INK, SUB = "#0f172a", "#64748b"
CSA, CSA_DARK = "#059669", "#065f46"
IDX, IDX_DARK = "#7c3aed", "#5b21b6"
NEUTRAL_FILL, NEUTRAL_STROKE = "#f1f5f9", "#475569"
KVCELL_FILL, KVCELL_STROKE = "#a7f3d0", "#047857"
QCELL_FILL, QCELL_STROKE = "#ddd6fe", "#6d28d9"
ARROW = "#64748b"

W = 1360
PAD = 40
TITLE_Y = 34
SUBTITLE_Y = 56

# ---- 纵向站位(自底向上,数值越大越靠画布下方) ----
BOTTOM_Y = 760      # KV / query 隐状态行
COMPRESSOR_Y = 660  # 两个 token 级压缩器的底边
ENTRIES_Y = 560     # 压缩 KV 条目 / 压缩索引器 key / 索引器 query / 滑窗条目 —— 同一高度
MQA_Y = 460         # Lightning Indexer 内部多查询注意力
SCORES_Y = 360      # 索引得分(小柱状图)/ Top-k 选择器 —— 同一高度
SELECTED_Y = 258    # 选中的压缩 KV 条目
CONCAT_Y = 156      # 拼接
TOP_Y = 60          # 共享 KV 的多查询注意力(顶框)
TOP_H = 46

H = BOTTOM_Y + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ARROW}"/></marker>'
    '<marker id="ac" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{CSA}"/></marker>'
    '<marker id="ai" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{IDX}"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="{INK}">CSA:压缩到 1/m → Lightning Indexer 打分 top-k 稀疏 → 与滑窗 KV 合并送入核注意力</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" '
          f'fill="{SUB}">三步首尾相连:①绿色主路径压缩 KV ②紫色 Lightning Indexer 打分选块 ③选中的压缩块与滑窗 KV 拼接后共享 KV MQA</text>')


def cell_row(x0, y, n, cell_w, cell_h, gap, fill, stroke, label=None, label_dy=-10, font_size=12.5):
    for i in range(n):
        x = x0 + i * (cell_w + gap)
        L.append(f'<rect x="{x:.1f}" y="{y}" width="{cell_w}" height="{cell_h}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    span = n * (cell_w + gap) - gap
    if label:
        cx = x0 + span / 2
        L.append(f'<text x="{cx:.1f}" y="{y + label_dy}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{font_size}" font-weight="bold" fill="{INK}">{esc(label)}</text>')
    return x0, x0 + span


def v_arrow(x, y1, y2, color=ARROW, marker="a", width=2):
    L.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y2}" '
              f'stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>')


def dogleg(x1, y1, x2, jog_y, color=ARROW, marker="a", width=2):
    L.append(f'<path d="M {x1:.1f} {y1} L {x1:.1f} {jog_y} L {x2:.1f} {jog_y} L {x2:.1f} {y1 - (y1 - jog_y)}" '
              f'fill="none" stroke="{color}" stroke-width="{width}"/>')


def trapezoid(cx, y_bot, w_top, w_bot, h, fill, stroke, text_lines, font_size=12):
    x_bot_l, x_bot_r = cx - w_bot / 2, cx + w_bot / 2
    x_top_l, x_top_r = cx - w_top / 2, cx + w_top / 2
    y_top = y_bot - h
    L.append(f'<path d="M {x_bot_l:.1f} {y_bot} L {x_bot_r:.1f} {y_bot} L {x_top_r:.1f} {y_top} '
              f'L {x_top_l:.1f} {y_top} Z" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(text_lines)
    for i, line in enumerate(text_lines):
        ly = y_top + h / 2 - (n - 1) * 8 + i * 15 + 5
        L.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{font_size}" fill="white" font-weight="bold">{esc(line)}</text>')
    return y_top


# ================= 底部两路输入 =================
KV_CELL, KV_GAP = 26, 3
N_KV = 14
main_cx = 400
kv_x0 = main_cx - (N_KV * (KV_CELL + KV_GAP) - KV_GAP) / 2
kv_l, kv_r = cell_row(kv_x0, BOTTOM_Y, N_KV, KV_CELL, 30, KV_GAP, NEUTRAL_FILL, NEUTRAL_STROKE,
                       label="KV token 隐状态 (Hidden States of KV Tokens)")

Q_CELL, Q_GAP = 26, 3
N_Q = 4
query_cx = 1230
q_x0 = query_cx - (N_Q * (Q_CELL + Q_GAP) - Q_GAP) / 2
cell_row(q_x0, BOTTOM_Y, N_Q, Q_CELL, 30, Q_GAP, QCELL_FILL, QCELL_STROKE, label="query token 隐状态")

# 滑窗支路取自 KV 隐状态最左侧 4 格(未压缩)
WIN_N = 4
win_src_cx = kv_x0 + (WIN_N * (KV_CELL + KV_GAP) - KV_GAP) / 2
# 索引器 key 压缩器取自 KV 隐状态最右侧一段(与主压缩器共用同一份隐状态)
ikeys_src_cx = kv_r - (WIN_N * (KV_CELL + KV_GAP) - KV_GAP) / 2

# ================= 主压缩路径:token 级压缩器 → 压缩 KV 条目 =================
v_arrow(main_cx, BOTTOM_Y, COMPRESSOR_Y)
comp_top = trapezoid(main_cx, COMPRESSOR_Y, 150, 210, 46, CSA, CSA_DARK,
                      ["token 级压缩器", "(Token-Level Compressor)"])
N_COMP = 5
comp_cell = 26
comp_x0 = main_cx - (N_COMP * (comp_cell + KV_GAP) - KV_GAP) / 2
cell_row(comp_x0, ENTRIES_Y, N_COMP, comp_cell, 30, KV_GAP, KVCELL_FILL, KVCELL_STROKE,
         label="压缩 KV 条目")
v_arrow(main_cx, comp_top, ENTRIES_Y + 30)

# ================= 滑窗 KV 条目(旁路,不经压缩) =================
WIN_CELL = 30
win_x0 = 40
win_l, win_r = cell_row(win_x0, ENTRIES_Y, WIN_N, WIN_CELL, 30, KV_GAP, NEUTRAL_FILL, NEUTRAL_STROKE,
                         label="滑窗 KV 条目")
win_cx = (win_l + win_r) / 2
jog_y = ENTRIES_Y + 60
L.append(f'<path d="M {win_src_cx:.1f} {BOTTOM_Y} L {win_src_cx:.1f} {jog_y} '
          f'L {win_cx:.1f} {jog_y} L {win_cx:.1f} {ENTRIES_Y + 30}" '
          f'fill="none" stroke="{ARROW}" stroke-width="2" marker-end="url(#a)"/>')

# ================= Lightning Indexer(虚线框) =================
box_x0, box_x1 = 720, 1180
box_y0, box_y1 = SCORES_Y - 90, COMPRESSOR_Y + 24
L.append(f'<rect x="{box_x0}" y="{box_y0}" width="{box_x1-box_x0}" height="{box_y1-box_y0}" rx="10" '
          f'fill="none" stroke="{IDX}" stroke-width="2" stroke-dasharray="8,5"/>')
L.append(f'<text x="{box_x0}" y="{box_y0-12}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{IDX}">Lightning Indexer(闪电索引器)</text>')

ikeys_cx = 850
iq_cx = 1080

# 压缩索引器 key:自己的 token 级压缩器,输入取自 KV 隐状态(同一份数据的另一支路)
jog_ikeys_y = COMPRESSOR_Y + 50
L.append(f'<path d="M {ikeys_src_cx:.1f} {BOTTOM_Y} L {ikeys_src_cx:.1f} {jog_ikeys_y} '
          f'L {ikeys_cx:.1f} {jog_ikeys_y} L {ikeys_cx:.1f} {COMPRESSOR_Y}" '
          f'fill="none" stroke="{IDX}" stroke-width="2" marker-end="url(#ai)"/>')
ikeys_comp_top = trapezoid(ikeys_cx, COMPRESSOR_Y, 96, 140, 40, IDX, IDX_DARK,
                            ["token 级压缩器"], font_size=11)
N_IKEYS = 4
ikeys_cell = 24
ikeys_x0 = ikeys_cx - (N_IKEYS * (ikeys_cell + KV_GAP) - KV_GAP) / 2
cell_row(ikeys_x0, ENTRIES_Y, N_IKEYS, ikeys_cell, 30, KV_GAP, "#c4b5fd", IDX_DARK, label="压缩索引器 key", font_size=11.5)
v_arrow(ikeys_cx, ikeys_comp_top, ENTRIES_Y + 30, color=IDX, marker="ai")

# 索引器 query:来自 query 隐状态的旁路分支(不压缩)
N_IQ = 3
iq_cell = 24
iq_x0 = iq_cx - (N_IQ * (iq_cell + KV_GAP) - KV_GAP) / 2
cell_row(iq_x0, ENTRIES_Y, N_IQ, iq_cell, 30, KV_GAP, "#c4b5fd", IDX_DARK, label="索引器 query", font_size=11.5)
L.append(f'<path d="M {query_cx:.1f} {BOTTOM_Y} L {query_cx:.1f} {jog_y+40} '
          f'L {iq_cx:.1f} {jog_y+40} L {iq_cx:.1f} {ENTRIES_Y + 30}" '
          f'fill="none" stroke="{IDX}" stroke-width="2" marker-end="url(#ai)"/>')

# Multi-Query Attention(indexer 内部)
mqa_w, mqa_h = 220, 42
mqa_cx = (ikeys_cx + iq_cx) / 2
L.append(f'<rect x="{mqa_cx-mqa_w/2:.1f}" y="{MQA_Y}" width="{mqa_w}" height="{mqa_h}" rx="7" '
          f'fill="{IDX}" stroke="{IDX_DARK}" stroke-width="1.6"/>')
L.append(f'<text x="{mqa_cx:.1f}" y="{MQA_Y+mqa_h/2+5:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="white">多查询注意力 (MQA)</text>')
v_arrow(ikeys_cx, ENTRIES_Y, MQA_Y + mqa_h, color=IDX, marker="ai")
v_arrow(iq_cx, ENTRIES_Y, MQA_Y + mqa_h, color=IDX, marker="ai")

# 索引得分(小柱状图)
bars_cx = mqa_cx
bar_vals = [0.3, 0.9, 0.15, 0.55, 0.7]
bar_w, bar_gap, bar_max_h = 14, 6, 44
n_bars = len(bar_vals)
bars_x0 = bars_cx - (n_bars * (bar_w + bar_gap) - bar_gap) / 2
bars_base = SCORES_Y - 4
for i, v in enumerate(bar_vals):
    bh = v * bar_max_h
    bx = bars_x0 + i * (bar_w + bar_gap)
    L.append(f'<rect x="{bx:.1f}" y="{bars_base-bh:.1f}" width="{bar_w}" height="{bh:.1f}" '
              f'fill="#86efac" stroke="{CSA_DARK}" stroke-width="1"/>')
L.append(f'<text x="{bars_cx:.1f}" y="{SCORES_Y - bar_max_h - 12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="{IDX}">索引得分 (Index Scores)</text>')
v_arrow(mqa_cx, MQA_Y, bars_base - bar_max_h, color=IDX, marker="ai")

# ================= Top-k 选择器(与索引得分同一高度) =================
topk_cx = main_cx
topk_top = trapezoid(topk_cx, SCORES_Y, 130, 190, 40, IDX, IDX_DARK,
                      ["Top-k 选择器"], font_size=12.5)
v_arrow(topk_cx, ENTRIES_Y, SCORES_Y, color=CSA, marker="ac")
# 索引得分 → Top-k 选择器(横向,起点=最左侧柱底部左下角,终点=选择器右下顶点,均为精确边界点)
L.append(f'<line x1="{bars_x0:.1f}" y1="{bars_base:.1f}" '
          f'x2="{topk_cx+95:.1f}" y2="{SCORES_Y}" '
          f'stroke="{IDX}" stroke-width="2" marker-end="url(#ai)"/>')

# ================= 选中的压缩 KV 条目 =================
N_SEL = 2
sel_cell = 30
sel_x0 = topk_cx - (N_SEL * (sel_cell + KV_GAP) - KV_GAP) / 2
cell_row(sel_x0, SELECTED_Y, N_SEL, sel_cell, 30, KV_GAP, KVCELL_FILL, KVCELL_STROKE,
         label="选中的压缩 KV 条目")
v_arrow(topk_cx, topk_top, SELECTED_Y + 30, color=CSA, marker="ac")

# ================= 拼接 =================
CONCAT_W, CONCAT_H = 300, 44
concat_cx = (win_cx + topk_cx) / 2
L.append(f'<rect x="{concat_cx-CONCAT_W/2:.1f}" y="{CONCAT_Y}" width="{CONCAT_W}" height="{CONCAT_H}" rx="8" '
          f'fill="{NEUTRAL_FILL}" stroke="{NEUTRAL_STROKE}" stroke-width="1.8"/>')
L.append(f'<text x="{concat_cx:.1f}" y="{CONCAT_Y+CONCAT_H/2+5:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{INK}">拼接 (Concatenation)</text>')
v_arrow(win_cx, ENTRIES_Y, CONCAT_Y + CONCAT_H)
v_arrow(topk_cx, SELECTED_Y, CONCAT_Y + CONCAT_H, color=CSA, marker="ac")

# ================= query 隐状态直上(旁路) =================
v_arrow(query_cx, BOTTOM_Y, TOP_Y + TOP_H, width=2)

# ================= 顶部:共享 KV 的多查询注意力 =================
TOP_W = W - 2 * PAD
L.append(f'<rect x="{PAD}" y="{TOP_Y}" width="{TOP_W}" height="{TOP_H}" rx="8" '
          f'fill="{CSA}" stroke="{CSA_DARK}" stroke-width="2"/>')
L.append(f'<text x="{W/2:.1f}" y="{TOP_Y+TOP_H/2+6:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="white">共享 KV 的多查询注意力 (Shared Key-Value Multi-Query Attention)</text>')
v_arrow(concat_cx, CONCAT_Y, TOP_Y + TOP_H, color=CSA, marker="ac")

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
