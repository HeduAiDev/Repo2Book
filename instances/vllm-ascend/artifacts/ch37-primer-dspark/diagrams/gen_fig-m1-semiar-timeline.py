#!/usr/bin/env python3
"""fig-m1-semiar-timeline：半自回归时间线（swimlane 变体）。
上道（实线蓝）= 并行骨干一次非因果前向，出块内 N 个位置的基础 logits U_k。
中道（实线绿）= 序列 Markov 头 for 循环，逐位用上一步采样 token 生成低秩偏置修正。
下道（虚线灰）= 置信度头 + 调度器 Algorithm 1，标注"仅论文侧，本 PR 未接入"。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

N = 3  # 块内位置数（与 m6 worked example 一致）
POS = ["k=0（锚点）", "k=1", "k=2"]

W, PAD = 980, 36
LANE_LABEL_W = 190
TIMELINE_X0 = PAD + LANE_LABEL_W
TIMELINE_W = W - TIMELINE_X0 - PAD
COL_W = TIMELINE_W / N
COL_X = [TIMELINE_X0 + i * COL_W for i in range(N)]

TITLE_Y = 34
SUB_Y = 54
BACKBONE_Y = 92
BACKBONE_H = 64
ARROW_GAP = 22
MARKOV_Y = BACKBONE_Y + BACKBONE_H + ARROW_GAP + 18
MARKOV_H = 58
GAP2 = 40
PAPER_Y = MARKOV_Y + MARKOV_H + GAP2
PAPER_H = 58
FOOT_Y = PAPER_Y + PAPER_H + 66
H = FOOT_Y + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

L.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">半自回归：一次并行骨干前向 + N 步序列 Markov 修正</text>')
L.append(f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">重活干一次（骨干），轻活干 N 次（Markov 头）——省下的正是 N 次完整前向 → 1 次</text>')

# ---- 泳道标签（左侧） ----
LANE_DEFS = [
    (BACKBONE_Y, BACKBONE_H, "并行骨干", "#1d4ed8", "已落地", "#dbeafe"),
    (MARKOV_Y, MARKOV_H, "序列 Markov 头", "#15803d", "已落地", "#dcfce7"),
    (PAPER_Y, PAPER_H, "置信度头+调度器", "#94a3b8", "仅论文侧", "#f1f5f9"),
]
for y, h, name, color, tag, bg in LANE_DEFS:
    L.append(f'<rect x="{PAD}" y="{y}" width="{LANE_LABEL_W-14}" height="{h}" rx="6" '
             f'fill="{bg}" stroke="{color}" stroke-width="1.4"/>')
    L.append(f'<text x="{PAD+10}" y="{y+22}" font-family="sans-serif" font-size="13" '
             f'font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<text x="{PAD+10}" y="{y+40}" font-family="sans-serif" font-size="11" '
             f'fill="{color}">{esc(tag)}</text>')

# ---- 位置刻度（顶部，跨 backbone 和 markov 两道） ----
for i, p in enumerate(POS):
    cx = COL_X[i] + COL_W / 2
    L.append(f'<text x="{cx}" y="{BACKBONE_Y-10}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" fill="#64748b">{esc(p)}</text>')

# ---- 并行骨干：一个横跨 N 个位置的整块（非因果，一次前向） ----
bb_x0, bb_x1 = COL_X[0]+6, COL_X[-1]+COL_W-6
L.append(f'<rect x="{bb_x0}" y="{BACKBONE_Y}" width="{bb_x1-bb_x0}" height="{BACKBONE_H}" rx="8" '
         f'fill="#eff6ff" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{(bb_x0+bb_x1)/2}" y="{BACKBONE_Y+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="#1d4ed8">'
         f'_run_model：1 次非因果前向（N 个 query 位置互相可见）</text>')
L.append(f'<text x="{(bb_x0+bb_x1)/2}" y="{BACKBONE_Y+46}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11.5" fill="#1d4ed8">'
         f'→ 一次性出 U_0, U_1, U_2（块内每个位置的基础 logits）</text>')

# ---- 箭头：backbone → 每个位置对应的 markov 步（下方分叉） ----
for i in range(N):
    cx = COL_X[i] + COL_W / 2
    L.append(f'<line x1="{cx}" y1="{BACKBONE_Y+BACKBONE_H}" x2="{cx}" y2="{MARKOV_Y-8}" '
             f'stroke="#1d4ed8" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#a)"/>')
    L.append(f'<text x="{cx+6}" y="{BACKBONE_Y+BACKBONE_H+16}" font-family="sans-serif" '
             f'font-size="10.5" fill="#1d4ed8">U_{i}</text>')

# ---- 序列 Markov 头：N 个小方块，逐位串联（prev token 传递） ----
MARKOV_MARGIN = 22
for i in range(N):
    x = COL_X[i] + MARKOV_MARGIN
    bw = COL_W - 2 * MARKOV_MARGIN
    L.append(f'<rect x="{x}" y="{MARKOV_Y}" width="{bw}" height="{MARKOV_H}" rx="8" '
             f'fill="#f0fdf4" stroke="#15803d" stroke-width="2"/>')
    L.append(f'<text x="{x+bw/2}" y="{MARKOV_Y+20}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="#15803d">i={i}: logits=U_{i}+B_{i}(prev)</text>')
    L.append(f'<text x="{x+bw/2}" y="{MARKOV_Y+38}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#15803d">→ sample draft_{i}（O(Vr) 偏置 + O(r) 嵌入）</text>')
    if i < N - 1:
        x1 = COL_X[i] + COL_W - MARKOV_MARGIN
        x2 = COL_X[i+1] + MARKOV_MARGIN
        y = MARKOV_Y + MARKOV_H / 2
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#15803d" '
                 f'stroke-width="1.5" marker-end="url(#ag)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-6}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="9" fill="#15803d">prev</text>')

# ---- 论文侧：置信度头 + 调度器（虚线框，不接实际数据流） ----
pb_x0, pb_x1 = COL_X[0]+6, COL_X[-1]+COL_W-6
L.append(f'<rect x="{pb_x0}" y="{PAPER_Y}" width="{pb_x1-pb_x0}" height="{PAPER_H}" rx="8" '
         f'fill="#f8fafc" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{(pb_x0+pb_x1)/2}" y="{PAPER_Y+24}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="#64748b">'
         f'c_k=sigmoid(...)（置信度）→ Algorithm 1 贪心早停调度</text>')
L.append(f'<text x="{(pb_x0+pb_x1)/2}" y="{PAPER_Y+44}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#94a3b8">'
         f'confidence_head 权重被 load_weights 显式跳过；调度器无对应实现代码</text>')

# 虚线连接（示意"论文设想接在 markov 头之后"，但当前无数据流）
mid = COL_X[len(POS)//2] + COL_W/2
L.append(f'<line x1="{mid}" y1="{MARKOV_Y+MARKOV_H}" x2="{mid}" y2="{PAPER_Y-6}" '
         f'stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="2,3"/>')
L.append(f'<text x="{mid+8}" y="{(MARKOV_Y+MARKOV_H+PAPER_Y)/2}" font-family="sans-serif" '
         f'font-size="10" fill="#94a3b8">（无实际连线）</text>')

L.append(f'<text x="{PAD}" y="{FOOT_Y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">本 PR #46995 @f5a8d73 已落地：骨干 + Markov 头（实线泳道）。'
         f'置信度头与硬件感知调度器仅存在于论文侧（虚线泳道，前瞻，尚未合入本书 pin 的 v0.21.0）。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-semiar-timeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
