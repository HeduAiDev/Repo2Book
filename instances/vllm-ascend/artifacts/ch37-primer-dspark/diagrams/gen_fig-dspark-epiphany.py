#!/usr/bin/env python3
"""fig-dspark-epiphany：本章顿悟图（头图，§2.5 五步法）。
一图只打一拳：块内依赖不必重跑整座骨干——纯序列起草要 N 次完整 Transformer 前向，
DSpark 把重活塌成 1 次并行前向 + N 片 O(Vr) 低秩偏置补回块内依赖。
视觉主轴 = 落差：左半 N 个又大又厚的骨干块串成链（每块画成层栈，视觉上「重」），
右半 1 个同样厚的骨干块 + N 片极薄的偏置小片（厚薄落差 = O(完整前向) vs O(Vr)）。
量化落差直接量在图上（24 次乘加 vs 3 次完整层栈前向，玩具 V=4/N=3/r=2）。
全坐标由循环/常量计算，零手写魔数。数字全部来自 figure-requests numbers（带溯源）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

N = 3  # 块内位置数（与 m1/m6 一致）

W, PAD = 980, 40
PANEL_W = 360
GAP = 120                       # 两面板之间给中央大箭头
LX = PAD                        # 左面板左沿
RX = PAD + PANEL_W + GAP        # 右面板左沿
LCX = LX + PANEL_W / 2
RCX = RX + PANEL_W / 2

BW = 240                        # 骨干大块宽
BH = 64                         # 骨干大块高（厚）
VG = 30                         # 左侧串行块之间的间隙（放串行箭头）
BY0 = 128                       # 块起始 y
SH = 16                         # 偏置薄片高（薄）
SG = 8                          # 薄片间隙

TITLE_Y = 32
SUB_Y = 54
PANEL_TITLE_Y = 100

left_bottom = BY0 + N * BH + (N - 1) * VG
FOOT_Y = left_bottom + 44
H = FOOT_Y + 66

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# ---- 标题（打一拳：落差本身） ----
L.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="#0f172a">重活 N 次 → 1 次：块内依赖不必重跑整座骨干</text>')
L.append(f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">纯序列起草每个 token 都要一次完整 Transformer 前向；'
         f'DSpark 一次并行前向出整块，块内依赖只补 N 片 O(Vr) 低秩偏置</text>')


def heavy_block(cx, y, fill, stroke, sub_fill, title, sub):
    """一个『又大又厚』的骨干块：画成层栈（3 条内分隔线）以示『一整层解码器栈』的重。"""
    x = cx - BW / 2
    out = [f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="7" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>']
    for k in range(1, 3):  # 内部层栈纹理
        ly = y + k * BH / 3
        out.append(f'<line x1="{x}" y1="{ly}" x2="{x+BW}" y2="{ly}" '
                   f'stroke="{sub_fill}" stroke-width="1"/>')
    out.append(f'<text x="{cx}" y="{y+BH/2-3}" text-anchor="middle" font-family="sans-serif" '
               f'font-size="12.5" font-weight="bold" fill="{stroke}">{esc(title)}</text>')
    out.append(f'<text x="{cx}" y="{y+BH/2+15}" text-anchor="middle" font-family="sans-serif" '
               f'font-size="11" fill="{stroke}">{esc(sub)}</text>')
    return out

# ================= 左面板：朴素纯序列起草（N 次串行完整前向） =================
L.append(f'<text x="{LCX}" y="{PANEL_TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#b91c1c">朴素：纯序列起草（EAGLE / MTP）</text>')
for i in range(N):
    y = BY0 + i * (BH + VG)
    L += heavy_block(LCX, y, "#fecaca", "#b91c1c", "#fca5a5",
                     f"完整 Transformer 前向 #{i}",
                     "一整层解码器栈")
    if i < N - 1:
        y1 = y + BH
        y2 = y + BH + VG - 4
        L.append(f'<line x1="{LCX}" y1="{y1}" x2="{LCX}" y2="{y2}" stroke="#b91c1c" '
                 f'stroke-width="1.8" marker-end="url(#a)"/>')
        L.append(f'<text x="{LCX+8}" y="{(y1+y2)/2+4}" font-family="sans-serif" '
                 f'font-size="10.5" fill="#b91c1c">等上一个草稿 token</text>')
L.append(f'<text x="{LCX}" y="{left_bottom+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#b91c1c">N 次串行完整骨干前向</text>')

# ================= 右面板：DSpark（1 次并行前向 + N 片低秩偏置） =================
L.append(f'<text x="{RCX}" y="{PANEL_TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#15803d">DSpark：1 次并行前向 + N 片低秩偏置</text>')
# 1 个同样厚的骨干块
L += heavy_block(RCX, BY0, "#bbf7d0", "#15803d", "#86efac",
                 "1 次非因果并行前向",
                 "一次出 U_0,U_1,U_2（块内互见）")
# N 片极薄偏置小片（厚薄落差 = 视觉主轴）
sliver_top = BY0 + BH + 26
sx = RCX - BW / 2
for i in range(N):
    sy = sliver_top + i * (SH + SG)
    L.append(f'<rect x="{sx}" y="{sy}" width="{BW}" height="{SH}" rx="3" '
             f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.4"/>')
    L.append(f'<text x="{sx+8}" y="{sy+SH/2+4}" font-family="sans-serif" '
             f'font-size="10" fill="#1d4ed8">第 {i} 位：O(Vr) 低秩偏置修正</text>')
# 从骨干块下沿指向薄片组的箭头
L.append(f'<line x1="{RCX}" y1="{BY0+BH}" x2="{RCX}" y2="{sliver_top-3}" '
         f'stroke="#15803d" stroke-width="1.6" marker-end="url(#a)"/>')
sliver_bottom = sliver_top + N * (SH + SG) - SG
L.append(f'<text x="{RCX}" y="{sliver_bottom+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#1d4ed8">'
         f'块内依赖 = N 片 softmax 前的低秩偏置（markov_embed O(r) + markov_bias O(Vr)，r=256）</text>')

# ================= 中央大落差箭头：重活 N 次 → 1 次 =================
arr_y = BY0 + BH / 2
ax1 = LX + PANEL_W + 8
ax2 = RX - 8
L.append(f'<line x1="{ax1}" y1="{arr_y}" x2="{ax2}" y2="{arr_y}" stroke="#d97706" '
         f'stroke-width="3.5" marker-end="url(#ao)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#d97706">重活 N 次</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y+24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#d97706">→ 1 次</text>')

# ================= 底部量化落差横幅 =================
box_y = FOOT_Y - 8
L.append(f'<rect x="{PAD}" y="{box_y}" width="{W-2*PAD}" height="52" rx="8" '
         f'fill="#fffbeb" stroke="#d97706" stroke-width="1.4"/>')
L.append(f'<text x="{W/2}" y="{box_y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#92400e">'
         f'玩具尺度 V=4 · N=3 · r=2：右侧三步偏置修正共 24 次乘加，抵掉左侧 3 次完整层栈前向</text>')
L.append(f'<text x="{W/2}" y="{box_y+42}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#b45309">偏置片薄 vs 骨干块厚 ≈ O(Vr) vs 完整前向——省下的正是那 N-1 次重活</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-dspark-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
