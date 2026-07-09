#!/usr/bin/env python3
"""paper-fig-1: 重绘自 arXiv:2502.11089 Fig.2——NSA 整体架构(原图已抓到:
https://arxiv.org/html/2502.11089v2/x2.png)。信息结构对齐原图两个面板:
左panel = k_{:t},v_{:t} 切成连续块后并行喂入压缩/选择/滑窗三支路,各产出紧凑 KV,
与 q_t 分别做注意力后门控求和;右panel = 三支路各自对应的 token×token 注意力模式
(绿=需算,白=可跳过)。配色套本章既有 NSA 配色(压缩蓝/选择紫/滑窗绿),文字译中,
右侧网格为示意因果稀疏形状(非逐字复刻原图像素数据,provenance=原论文本身)。
全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK, SUB = "#0f172a", "#64748b"
BRANCH_COLORS = {"cmp": "#3b82f6", "slc": "#7c3aed", "win": "#059669"}
GREEN_CELL, WHITE_CELL, GRID_STROKE = "#86efac", "#f8fafc", "#cbd5e1"

W = 1440
PAD = 40
TITLE_TOP = 34
SUBTITLE_TOP = 56
CONTENT_TOP = 96
MID_GAP = 40
LEFT_W = 660
RIGHT_X = PAD + LEFT_W + MID_GAP
RIGHT_W = W - RIGHT_X - PAD

# ---------- 左 panel 纵向布局常量 ----------
L_HEAD_Y = CONTENT_TOP + 18
STRIP_Y = L_HEAD_Y + 26
STRIP_H = 34
N_UNITS = 16
UNIT_W = (LEFT_W - 170) / N_UNITS
STRIP_X = PAD + 130
BRANCH_Y = STRIP_Y + STRIP_H + 46
BRANCH_W, BRANCH_H = 190, 58
BRANCH_GAP = (LEFT_W - BRANCH_W * 3) / 2
KV_Y = BRANCH_Y + BRANCH_H + 40
KV_H = 40
Q_Y = KV_Y + KV_H + 44
ATTN_Y = Q_Y + 70
ATTN_H = 50
GATE_Y = ATTN_Y + ATTN_H + 44
GATE_H = 54
LEFT_BOTTOM = GATE_Y + GATE_H

branch_x = [PAD + i * (BRANCH_W + BRANCH_GAP) for i in range(3)]
branch_cx = [x + BRANCH_W / 2 for x in branch_x]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {int(LEFT_BOTTOM + 30)}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{int(LEFT_BOTTOM + 30)}" fill="white"/>')

# ---- 标题区 ----
L.append(f'<text x="{PAD}" y="{TITLE_TOP}" font-family="sans-serif" font-size="18" '
          f'font-weight="bold" fill="{INK}">NSA 整体架构:三支路并行生成紧凑 KV,右侧是各自的注意力模式</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_TOP}" font-family="sans-serif" font-size="12.5" '
          f'fill="{SUB}">同一 query q_t 并行喂入压缩(cmp)/选择(slc)/滑窗(win)三支路;右侧 token×token 网格:绿=需算注意力分数,白=可跳过</text>')

# ================= 左 panel =================
L.append(f'<text x="{PAD}" y="{L_HEAD_Y}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="{INK}">① 三支路流水线:同一 query 并行喂入三支路</text>')

# k_:t, v_:t 标签 + 切块条
L.append(f'<text x="{PAD}" y="{STRIP_Y + STRIP_H/2 + 5}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{INK}">k_:t, v_:t</text>')
N_BLOCKS = 4
UNITS_PER_BLOCK = N_UNITS // N_BLOCKS
for i in range(N_UNITS):
    blk = i // UNITS_PER_BLOCK
    fill = "#93c5fd" if blk % 2 == 0 else "#3b82f6"
    x = STRIP_X + i * UNIT_W
    L.append(f'<rect x="{x:.1f}" y="{STRIP_Y}" width="{UNIT_W-1.5:.1f}" height="{STRIP_H}" '
              f'fill="{fill}" stroke="#1e3a8a" stroke-width="0.6"/>')
for b in range(1, N_BLOCKS):
    x = STRIP_X + b * UNITS_PER_BLOCK * UNIT_W
    L.append(f'<line x1="{x:.1f}" y1="{STRIP_Y-4}" x2="{x:.1f}" y2="{STRIP_Y+STRIP_H+4}" '
              f'stroke="{INK}" stroke-width="2"/>')
strip_cx = STRIP_X + N_UNITS * UNIT_W / 2
L.append(f'<text x="{strip_cx:.1f}" y="{STRIP_Y+STRIP_H+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="{SUB}">切成连续块(Split to Continuous Blocks)</text>')

# 扇形箭头:切块条 → 3 支路
fan_y0 = STRIP_Y + STRIP_H + 24
for cx, key in zip(branch_cx, ["cmp", "slc", "win"]):
    L.append(f'<line x1="{strip_cx:.1f}" y1="{fan_y0}" x2="{cx:.1f}" y2="{BRANCH_Y}" '
              f'stroke="{BRANCH_COLORS[key]}" stroke-width="1.6" marker-end="url(#a)" opacity="0.75"/>')

BRANCH_LABELS = [
    ("cmp", "压缩支路 Compression", "块内均值池化"),
    ("slc", "选择支路 Selection", "Top-n 块 + concat"),
    ("win", "滑窗支路 Sliding", "保留最近 w 个 token"),
]
for (key, name, detail), x in zip(BRANCH_LABELS, branch_x):
    color = BRANCH_COLORS[key]
    L.append(f'<rect x="{x:.1f}" y="{BRANCH_Y}" width="{BRANCH_W}" height="{BRANCH_H}" rx="8" '
              f'fill="{color}" stroke="#1e293b" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BRANCH_W/2:.1f}" y="{BRANCH_Y+24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" font-weight="bold">{esc(name)}</text>')
    L.append(f'<text x="{x+BRANCH_W/2:.1f}" y="{BRANCH_Y+44}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#f1f5f9">{esc(detail)}</text>')

# 箭头:支路 → 紧凑 KV
KV_LABELS = ["K̃^cmp, Ṽ^cmp", "K̃^slc, Ṽ^slc", "K̃^win, Ṽ^win"]
for cx, key, kv in zip(branch_cx, ["cmp", "slc", "win"], KV_LABELS):
    color = BRANCH_COLORS[key]
    L.append(f'<line x1="{cx:.1f}" y1="{BRANCH_Y+BRANCH_H}" x2="{cx:.1f}" y2="{KV_Y}" '
              f'stroke="{color}" stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<rect x="{cx-85:.1f}" y="{KV_Y}" width="170" height="{KV_H}" rx="6" '
              f'fill="{INK}" stroke="{INK}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx:.1f}" y="{KV_Y+KV_H/2+5:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">{esc(kv)}</text>')

# q_t 橙色框(左侧) + 扇形箭头到 3 个 Attention 框
q_box_w, q_box_h = 90, 36
q_x, q_y = PAD, Q_Y
L.append(f'<rect x="{q_x}" y="{q_y}" width="{q_box_w}" height="{q_box_h}" rx="6" '
          f'fill="#f97316" stroke="#c2410c" stroke-width="1.5"/>')
L.append(f'<text x="{q_x+q_box_w/2:.1f}" y="{q_y+q_box_h/2+5:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" fill="white" font-weight="bold">query q_t</text>')

# q_t → 三个 Attn 框:干线总线(bus)式扇出,箭头落在每个框顶边(偏中心 30px,避免与色彩竖箭头重叠)
bus_x0 = q_x + q_box_w / 2
bus_y = ATTN_Y - 16
drop_xs = [cx - 30 for cx in branch_cx]
L.append(f'<line x1="{bus_x0:.1f}" y1="{q_y+q_box_h}" x2="{bus_x0:.1f}" y2="{bus_y}" '
          f'stroke="#f97316" stroke-width="1.5"/>')
L.append(f'<line x1="{bus_x0:.1f}" y1="{bus_y}" x2="{drop_xs[-1]:.1f}" y2="{bus_y}" '
          f'stroke="#f97316" stroke-width="1.5"/>')
for dx in drop_xs:
    L.append(f'<line x1="{dx:.1f}" y1="{bus_y}" x2="{dx:.1f}" y2="{ATTN_Y}" '
              f'stroke="#f97316" stroke-width="1.5" marker-end="url(#a)"/>')

ATTN_LABELS = ["压缩注意力\nCompressed Attn", "选择注意力\nSelected Attn", "滑窗注意力\nSliding Attn"]
for cx, key, label in zip(branch_cx, ["cmp", "slc", "win"], ATTN_LABELS):
    color = BRANCH_COLORS[key]
    # KV 框 → Attn 框(竖直)
    L.append(f'<line x1="{cx:.1f}" y1="{KV_Y+KV_H}" x2="{cx:.1f}" y2="{ATTN_Y}" '
              f'stroke="{color}" stroke-width="1.6" marker-end="url(#ag)" opacity="0.9"/>')
    L.append(f'<rect x="{cx-95:.1f}" y="{ATTN_Y}" width="190" height="{ATTN_H}" rx="7" '
              f'fill="#dcfce7" stroke="#16a34a" stroke-width="1.6"/>')
    line1, line2 = label.split("\n")
    L.append(f'<text x="{cx:.1f}" y="{ATTN_Y+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#166534" font-weight="bold">{esc(line1)}</text>')
    L.append(f'<text x="{cx:.1f}" y="{ATTN_Y+38}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#166534">{esc(line2)}</text>')

# 汇合箭头 → Gated Output
gate_cx = W_ = PAD + LEFT_W / 2
for cx in branch_cx:
    L.append(f'<line x1="{cx:.1f}" y1="{ATTN_Y+ATTN_H}" x2="{gate_cx:.1f}" y2="{GATE_Y}" '
              f'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)" opacity="0.8"/>')
L.append(f'<rect x="{gate_cx-LEFT_W/2:.1f}" y="{GATE_Y}" width="{LEFT_W}" height="{GATE_H}" rx="8" '
          f'fill="{INK}" stroke="{INK}" stroke-width="2"/>')
L.append(f'<text x="{gate_cx:.1f}" y="{GATE_Y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" fill="white" font-weight="bold">Gated Output: o_t* = Σ_c g_t^c · Attn(q_t, K̃^c, Ṽ^c)</text>')
L.append(f'<text x="{gate_cx:.1f}" y="{GATE_Y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#93c5fd">c ∈ {{cmp, slc, win}};门控 g_t^c 由 q_t 现算(§二正文 Eq.(5))</text>')

# ================= 右 panel:三张 token×token 注意力模式网格 =================
R_HEAD_Y = CONTENT_TOP + 18
L.append(f'<text x="{RIGHT_X}" y="{R_HEAD_Y}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="{INK}">② 对应的注意力模式(token×token,绿=需算,白=可跳过)</text>')

# 图例
leg_y = R_HEAD_Y + 22
L.append(f'<rect x="{RIGHT_X}" y="{leg_y-11}" width="16" height="16" fill="{GREEN_CELL}" stroke="{GRID_STROKE}"/>')
L.append(f'<text x="{RIGHT_X+22}" y="{leg_y+1}" font-family="sans-serif" font-size="11" fill="{INK}">需计算注意力分数</text>')
L.append(f'<rect x="{RIGHT_X+150}" y="{leg_y-11}" width="16" height="16" fill="{WHITE_CELL}" stroke="{GRID_STROKE}"/>')
L.append(f'<text x="{RIGHT_X+172}" y="{leg_y+1}" font-family="sans-serif" font-size="11" fill="{INK}">可跳过</text>')
L.append(f'<text x="{RIGHT_X+RIGHT_W}" y="{leg_y+1}" text-anchor="end" font-family="sans-serif" font-size="10.5" '
          f'fill="{SUB}">行=query(由早到晚) 列=key 位置 ↓因果:下面的 query 能看见更多列</text>')

R_COLS, R_ROWS = 26, 6
CELL = 20
GRID_W = R_COLS * CELL

def visible_cols(r):
    # 因果 + 分块粗放:每行代表性 query 能看见的 key 列数,阶梯式随行增长
    return min(R_COLS, 5 + r * 4)

def cmp_green(r, c):
    return c < visible_cols(r)

def slc_green(r, c):
    vis = visible_cols(r)
    if c >= vis:
        return False
    block = c // 4
    last_block = (vis - 1) // 4
    if block == last_block:
        return True  # 最近块总入选(近邻恒常被选)
    # 稀疏、逐行变化的"被选中块"(示意 top-n 选择,非逐字复刻真实权重)
    return (block * 3 + r * 5) % 7 < 2

def win_green(r, c):
    vis = visible_cols(r)
    W_WIN = 7
    return max(0, vis - W_WIN) <= c < vis

MASKS = [
    ("压缩注意力掩码 Compressed Attention Mask", cmp_green, "cmp"),
    ("选择注意力掩码 Selected Attention Mask", slc_green, "slc"),
    ("滑窗注意力掩码 Sliding Attention Mask", win_green, "win"),
]

mask_top = leg_y + 26
MASK_TITLE_H = 20
MASK_GAP = 22
for title, pattern_fn, key in MASKS:
    color = BRANCH_COLORS[key]
    L.append(f'<text x="{RIGHT_X}" y="{mask_top+13}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="{color}">{esc(title)}</text>')
    grid_y = mask_top + MASK_TITLE_H
    for r in range(R_ROWS):
        for c in range(R_COLS):
            fill = GREEN_CELL if pattern_fn(r, c) else WHITE_CELL
            x = RIGHT_X + c * CELL
            y = grid_y + r * CELL
            L.append(f'<rect x="{x}" y="{y}" width="{CELL-1}" height="{CELL-1}" fill="{fill}" '
                      f'stroke="{GRID_STROKE}" stroke-width="0.6"/>')
    L.append(f'<rect x="{RIGHT_X}" y="{grid_y}" width="{GRID_W}" height="{R_ROWS*CELL}" '
              f'fill="none" stroke="{color}" stroke-width="1.6"/>')
    mask_top = grid_y + R_ROWS * CELL + MASK_GAP

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-1.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
