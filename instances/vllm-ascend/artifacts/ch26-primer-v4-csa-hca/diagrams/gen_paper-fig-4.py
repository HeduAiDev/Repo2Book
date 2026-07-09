#!/usr/bin/env python3
"""paper-fig-4: 重绘自 arXiv:2606.19348 Fig.4——HCA 核心架构(原图已抓到:
https://arxiv.org/html/2606.19348v1/x4.png)。信息结构对齐原图:KV token 隐状态
只走一条 token 级压缩器(无 indexer/top-k 分支)产出重压缩 KV 条目,与滑窗未压缩
KV 条目拼接后,连同 query 一起送入共享 KV 的多查询注意力——比 CSA(paper-fig-3)
少了 lightning indexer 与 top-k 选择两个环节,故整图明显更简单。配色套本章既有
HCA 强调色(橙 #d97706),文字译中,非逐字复刻原图像素,provenance=原论文本身。
全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


INK, SUB = "#0f172a", "#64748b"
HCA = "#d97706"          # HCA 强调色(与本章既有 fig36-1/fig36-6 一致)
HCA_DARK = "#92400e"
NEUTRAL_FILL, NEUTRAL_STROKE = "#f1f5f9", "#475569"
CELL_FILL, CELL_STROKE = "#fde68a", "#b45309"
QCELL_FILL, QCELL_STROKE = "#bbf7d0", "#166534"
ARROW = "#64748b"

W = 1000
PAD = 40
TITLE_Y = 34
SUBTITLE_Y = 56

# ---- 纵向站位(自底向上) ----
BOTTOM_Y = 560          # 隐状态行(token 格子)
COMPRESSOR_Y = 448      # token 级压缩器梯形
ENTRIES_Y = 350         # 压缩后的 KV 条目行
CONCAT_Y = 232          # 拼接框
TOP_Y = 120             # 共享 KV 的 MQA 顶框
TOP_H = 46

H = BOTTOM_Y + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ARROW}"/></marker>'
    '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{HCA}"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="{INK}">HCA:每 m\' 个 token 压成 1 条(不重叠)，直接稠密送入核注意力——无 top-k</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" '
          f'fill="{SUB}">对照 paper-fig-3(CSA):HCA 没有 Lightning Indexer / Top-k Selector 这两级——压缩块本来就少，不必再挑</text>')


def cell_row(x0, y, n, cell_w, cell_h, gap, fill, stroke, label=None, label_dy=-10):
    xs_ = []
    for i in range(n):
        x = x0 + i * (cell_w + gap)
        xs_.append(x)
        L.append(f'<rect x="{x:.1f}" y="{y}" width="{cell_w}" height="{cell_h}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    if label:
        cx = x0 + (n * (cell_w + gap) - gap) / 2
        L.append(f'<text x="{cx:.1f}" y="{y + label_dy}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="{INK}">{esc(label)}</text>')
    return x0, x0 + n * (cell_w + gap) - gap


def v_arrow(x, y1, y2, color=ARROW, marker="a", width=2):
    L.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y2}" '
              f'stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>')


# ================= 底部两路输入 =================
KV_CELL, KV_GAP = 26, 3
N_KV = 14
kv_x0 = 100
kv_l, kv_r = cell_row(kv_x0, BOTTOM_Y, N_KV, KV_CELL, 30, KV_GAP, NEUTRAL_FILL, NEUTRAL_STROKE,
                       label="KV token 隐状态 (Hidden States of KV Tokens)")
kv_cx = (kv_l + kv_r) / 2

Q_CELL, Q_GAP = 26, 3
N_Q = 4
q_x0 = W - PAD - (N_Q * (Q_CELL + Q_GAP) - Q_GAP)
q_l, q_r = cell_row(q_x0, BOTTOM_Y, N_Q, Q_CELL, 30, Q_GAP, QCELL_FILL, QCELL_STROKE,
                     label="query token 隐状态")
q_cx = (q_l + q_r) / 2

# 滑窗支路取自 KV 隐状态最左侧 4 格(未压缩,直接向上,不经压缩器)
WIN_N = 4
win_src_cx = kv_x0 + (WIN_N * (KV_CELL + KV_GAP) - KV_GAP) / 2

# ================= 压缩器 =================
TRAP_W_TOP, TRAP_W_BOT, TRAP_H = 150, 210, 46
comp_cx = kv_cx
def trapezoid(cx, y_bot, w_top, w_bot, h, fill, stroke, text_lines):
    x_bot_l, x_bot_r = cx - w_bot / 2, cx + w_bot / 2
    x_top_l, x_top_r = cx - w_top / 2, cx + w_top / 2
    y_top = y_bot - h
    L.append(f'<path d="M {x_bot_l:.1f} {y_bot} L {x_bot_r:.1f} {y_bot} L {x_top_r:.1f} {y_top} '
              f'L {x_top_l:.1f} {y_top} Z" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(text_lines)
    for i, line in enumerate(text_lines):
        ly = y_top + h / 2 - (n - 1) * 8 + i * 16 + 5
        L.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" fill="white" font-weight="bold">{esc(line)}</text>')
    return y_top


v_arrow(comp_cx, BOTTOM_Y, COMPRESSOR_Y)
comp_top = trapezoid(comp_cx, COMPRESSOR_Y, TRAP_W_TOP, TRAP_W_BOT, TRAP_H, HCA, HCA_DARK,
                      ["token 级压缩器", "(Token-Level Compressor)"])

# ================= 滑窗 KV 条目(直接来自隐状态,不经压缩,画在最左侧留足间距) =================
WIN_CELL = 30
win_x0 = 40
_, _ = cell_row(win_x0, ENTRIES_Y, WIN_N, WIN_CELL, 30, KV_GAP, NEUTRAL_FILL, NEUTRAL_STROKE,
                label="滑窗 KV 条目")
win_cx = win_x0 + (WIN_N * (WIN_CELL + KV_GAP) - KV_GAP) / 2
jog_y = ENTRIES_Y + 60
L.append(f'<path d="M {win_src_cx:.1f} {BOTTOM_Y} L {win_src_cx:.1f} {jog_y} '
          f'L {win_cx:.1f} {jog_y} L {win_cx:.1f} {ENTRIES_Y + 30}" '
          f'fill="none" stroke="{ARROW}" stroke-width="2" marker-end="url(#a)"/>')

# ================= 压缩后的重压缩 KV 条目(m' 很大,条目少 → 只画 3 格示意) =================
N_COMP = 3
comp_cell = 30
comp_x0 = comp_cx - (N_COMP * (comp_cell + KV_GAP) - KV_GAP) / 2
_, _ = cell_row(comp_x0, ENTRIES_Y, N_COMP, comp_cell, 30, KV_GAP, CELL_FILL, CELL_STROKE,
                label="重压缩 KV 条目")
v_arrow(comp_cx, comp_top, ENTRIES_Y + 30)

# ================= 拼接(Concatenation) =================
CONCAT_W, CONCAT_H = 260, 44
concat_cx = (win_cx + comp_cx) / 2
concat_x = concat_cx - CONCAT_W / 2
L.append(f'<rect x="{concat_x:.1f}" y="{CONCAT_Y}" width="{CONCAT_W}" height="{CONCAT_H}" rx="8" '
          f'fill="{NEUTRAL_FILL}" stroke="{NEUTRAL_STROKE}" stroke-width="1.8"/>')
L.append(f'<text x="{concat_cx:.1f}" y="{CONCAT_Y + CONCAT_H/2 + 5:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{INK}">拼接 (Concatenation)</text>')
v_arrow(win_cx, ENTRIES_Y, CONCAT_Y + CONCAT_H)
v_arrow(comp_cx, ENTRIES_Y, CONCAT_Y + CONCAT_H, color=HCA, marker="ah")

# ================= query 隐状态直上(旁路) =================
v_arrow(q_cx, BOTTOM_Y, TOP_Y + TOP_H, width=2)

# ================= 顶部:共享 KV 的多查询注意力 =================
TOP_W = W - 2 * PAD
L.append(f'<rect x="{PAD}" y="{TOP_Y}" width="{TOP_W}" height="{TOP_H}" rx="8" '
          f'fill="{HCA}" stroke="{HCA_DARK}" stroke-width="2"/>')
L.append(f'<text x="{W/2:.1f}" y="{TOP_Y + TOP_H/2 + 6:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="white">共享 KV 的多查询注意力 (Shared Key-Value Multi-Query Attention)</text>')
v_arrow(concat_cx, CONCAT_Y, TOP_Y + TOP_H, color=HCA, marker="ah")

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
