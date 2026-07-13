#!/usr/bin/env python3
"""顿悟图(before-after)：EAGLE-3 vs DFlash 的条件注入位置落差。
那一下：同一份 5 层 target 特征压成的 H_t，EAGLE-3 只在输入层递一次、随 draft 深度逐层稀释；
DFlash 把它作持久 K/V 条目注入 draft 每一层——条件全深度等强，接受长度才随层数增长。
视觉主轴=落差：左侧信号自底向上逐层变淡（稀释楔形+缩短的强度条），右侧每层各一条等强注入。
数字全部来自 figure-requests 条目 numbers（Table 9 / Appendix A.3）。零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# ---- 视觉语言 ----
BLUE, BLUE_S = "#bfe3f0", "#3a8fb0"       # 融合条件特征 H_t
LAYER_F, LAYER_S = "#f1f5f9", "#334155"    # draft 层
TRACK = "#e2e8f0"                          # 强度条底槽
WIN_F, WIN_S = "#dcfce7", "#16a34a"        # 赢面(DFlash)统计卡
LOSE_F, LOSE_S = "#f1f5f9", "#94a3b8"      # 对照(EAGLE)统计卡
INK, GREY = "#0f172a", "#64748b"

# ---- 画布 ----
PAD = 40
PANEL_W = 480
MID = 200                                  # 两面板间落差箭头区
w = PAD + PANEL_W + MID + PANEL_W + PAD     # = 1240
h = 780

LEFT_X = PAD
RIGHT_X = PAD + PANEL_W + MID

# ---- 栈几何(两面板共用) ----
N = 5
LH, LG = 50, 20
STACK_TOP = 150
stack_h = N * LH + (N - 1) * LG            # 330
STACK_BOT = STACK_TOP + stack_h            # 480
STACK_CY = (STACK_TOP + STACK_BOT) / 2     # 315

LAYER_LX_OFF = 210                         # 层框相对面板左缘的偏移
LAYER_W = 210
TRACK_X_OFF = 108                          # 强度条相对层框左缘
TRACK_W = 92

def layer_y(r):                            # r=0 顶(L5) .. 4 底(L1)
    return STACK_TOP + r * (LH + LG)

# 左侧强度分数(纯视觉稀释梯度，无数值标签)；右侧全满
LEFT_FRAC = [0.18, 0.34, 0.52, 0.74, 1.0]  # 行 0..4(顶->底)
RIGHT_FRAC = [1.0] * N

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#3a8fb0"/></marker>'
         '<marker id="big" viewBox="0 0 10 8" refX="8" refY="4" markerWidth="7" '
         'markerHeight="6" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#0f766e"/></marker>'
         '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# ---- 标题 ----
L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" font-weight="bold" fill="{INK}">'
         f'{esc("同一份条件特征 H_t，注入位置决定能不能「全深度等强」")}</text>')
L.append(f'<text x="{w/2}" y="60" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" fill="{GREY}">'
         f'{esc("5 层 target 特征压成 1 条 H_t（W_c ∈ ℝ^(D×5D)）——差别只在把它接到 draft 的哪一层")}</text>')

def draw_layers(px, frac):
    """画 5 层栈+每层强度条(按 frac 填蓝)。返回层框左缘 x 与轨道左缘 x。"""
    lx = px + LAYER_LX_OFF
    tx = lx + TRACK_X_OFF
    for r in range(N):
        y = layer_y(r)
        L.append(f'<rect x="{lx}" y="{y}" width="{LAYER_W}" height="{LH}" rx="8" '
                 f'fill="{LAYER_F}" stroke="{LAYER_S}" stroke-width="1.4"/>')
        L.append(f'<text x="{lx+14}" y="{y+LH/2+5}" font-family="sans-serif" '
                 f'font-size="12.5" font-weight="bold" fill="{INK}">Draft L{N-r}</text>')
        # 强度条底槽
        ty = y + 12
        th = LH - 24
        L.append(f'<rect x="{tx}" y="{ty}" width="{TRACK_W}" height="{th}" rx="4" '
                 f'fill="{TRACK}" stroke="#cbd5e1" stroke-width="1"/>')
        fw = max(6, TRACK_W * frac[r])
        L.append(f'<rect x="{tx}" y="{ty}" width="{fw:.1f}" height="{th}" rx="4" '
                 f'fill="{BLUE}" stroke="{BLUE_S}" stroke-width="1.2"/>')
    return lx, tx

# ================= 左面板：EAGLE-3 门口递一次 =================
L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{STACK_TOP-28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14.5" font-weight="bold" fill="{INK}">'
         f'{esc("EAGLE-3：门口递一次，逐层稀释")}</text>')
llx, ltx = draw_layers(LEFT_X, LEFT_FRAC)

# 稀释楔形(栈左侧，底宽顶尖)——信号往上越来越淡
wedge_x = llx - 46
wtop_w, wbot_w = 6, 40
wtop_y, wbot_y = STACK_TOP, STACK_BOT
L.append(f'<polygon points="{wedge_x},{wbot_y} {wedge_x+wbot_w},{wbot_y} '
         f'{wedge_x+wtop_w},{wtop_y}" fill="{BLUE}" stroke="{BLUE_S}" '
         f'stroke-width="1.2" opacity="0.55"/>')
L.append(f'<text x="{wedge_x+18}" y="{STACK_TOP-8}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" fill="{BLUE_S}">越淡</text>')

# H_t 盒(栈下方居中) + 唯一一条注入到底层(L1)
htw, hth = 210, 46
htcx = llx + LAYER_W/2
htx = htcx - htw/2
hty = STACK_BOT + 40
L.append(f'<rect x="{htx}" y="{hty}" width="{htw}" height="{hth}" rx="8" '
         f'fill="{BLUE}" stroke="{BLUE_S}" stroke-width="1.8"/>')
L.append(f'<text x="{htcx}" y="{hty+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="{INK}">融合特征 H_t</text>')
L.append(f'<text x="{htcx}" y="{hty+35}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#1e5c73">只接输入层这一处</text>')
# 注入箭头：H_t 顶 -> 底层(L1)底边
bot_y = layer_y(N-1) + LH
L.append(f'<line x1="{htcx}" y1="{hty}" x2="{htcx}" y2="{bot_y+2}" '
         f'stroke="{BLUE_S}" stroke-width="3" marker-end="url(#a)"/>')
# 稀释注解
L.append(f'<text x="{LEFT_X+PANEL_W/2}" y="{hty+hth+26}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="{GREY}">'
         f'{esc("target 特征随 draft 深度越来越淡（§4.1）")}</text>')

# ================= 右面板：DFlash 每层坐席 =================
L.append(f'<text x="{RIGHT_X+PANEL_W/2}" y="{STACK_TOP-28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14.5" font-weight="bold" fill="{INK}">'
         f'{esc("DFlash：每层各一条 K/V 坐席，等强")}</text>')
rlx, rtx = draw_layers(RIGHT_X, RIGHT_FRAC)

# H_t 盒(栈左侧居中) + 5 条等强注入
r_htw, r_hth = 150, 62
r_htx = RIGHT_X + 8
r_hty = STACK_CY - r_hth/2
L.append(f'<rect x="{r_htx}" y="{r_hty}" width="{r_htw}" height="{r_hth}" rx="8" '
         f'fill="{BLUE}" stroke="{BLUE_S}" stroke-width="1.8"/>')
L.append(f'<text x="{r_htx+r_htw/2}" y="{r_hty+22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="{INK}">融合特征 H_t</text>')
L.append(f'<text x="{r_htx+r_htw/2}" y="{r_hty+40}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#1e5c73">作持久 K/V 条目</text>')
L.append(f'<text x="{r_htx+r_htw/2}" y="{r_hty+55}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#1e5c73">被所有层共享</text>')
# 5 条等粗等强注入：从 H_t 右缘扇入每层左缘(箭头贴层框边，不穿层内文字)
src_x = r_htx + r_htw
for r in range(N):
    y = layer_y(r) + LH/2
    L.append(f'<line x1="{src_x}" y1="{STACK_CY}" x2="{rlx-4}" y2="{y}" '
             f'stroke="{BLUE_S}" stroke-width="2.4" opacity="0.9" marker-end="url(#a)"/>')
# 公式注解
L.append(f'<text x="{RIGHT_X+PANEL_W/2}" y="{STACK_BOT+40}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="{GREY}">'
         f'{esc("每层 i：K_i=[W_i^K H_t; W_i^K H_d]（Appendix A.3）")}</text>')

# ================= 中间落差箭头 =================
arr_y = STACK_CY
ax1 = LEFT_X + PANEL_W + 18
ax2 = RIGHT_X - 18
L.append(f'<line x1="{ax1}" y1="{arr_y}" x2="{ax2-4}" y2="{arr_y}" '
         f'stroke="#0f766e" stroke-width="4" marker-end="url(#big)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y-40}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#0f766e">'
         f'{esc("只换注入位置")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y-22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#0f766e">'
         f'{esc("输入层一点")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y-8}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#0f766e">'
         f'{esc("→ 每层一条")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y+22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="{GREY}">'
         f'{esc("条件全深度等强，")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arr_y+37}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="{GREY}">'
         f'{esc("接受长度随层数增长")}</text>')

# ================= 底部统计卡(量化落差) =================
CARD_Y = 600
CARD_H = 58
CARD_W = 300
# 左：Input 注入(对照)
lc_x = LEFT_X + PANEL_W/2 - CARD_W/2
L.append(f'<rect x="{lc_x}" y="{CARD_Y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
         f'fill="{LOSE_F}" stroke="{LOSE_S}" stroke-width="1.6"/>')
L.append(f'<text x="{lc_x+CARD_W/2}" y="{CARD_Y+22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{INK}">Input 注入（消融对照）</text>')
L.append(f'<text x="{lc_x+CARD_W/2}" y="{CARD_Y+44}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{LOSE_S}">τ 3.5　　η 2.9×</text>')
# 右：KV 注入(赢面)
rc_x = RIGHT_X + PANEL_W/2 - CARD_W/2
L.append(f'<rect x="{rc_x}" y="{CARD_Y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
         f'fill="{WIN_F}" stroke="{WIN_S}" stroke-width="2"/>')
L.append(f'<text x="{rc_x+CARD_W/2}" y="{CARD_Y+22}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{INK}">KV 注入（DFlash）</text>')
L.append(f'<text x="{rc_x+CARD_W/2}" y="{CARD_Y+44}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{WIN_S}">τ 4.2　　η 3.3×</text>')
# 卡间小升幅箭头
mc_y = CARD_Y + CARD_H/2
L.append(f'<line x1="{lc_x+CARD_W+6}" y1="{mc_y}" x2="{rc_x-8}" y2="{mc_y}" '
         f'stroke="#0f766e" stroke-width="2.5" marker-end="url(#big)"/>')

# ================= 图例 + 脚注 =================
LEG_Y = 690
L.append(f'<rect x="{PAD}" y="{LEG_Y-16}" width="18" height="18" rx="3" '
         f'fill="{BLUE}" stroke="{BLUE_S}" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+26}" y="{LEG_Y-2}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">{esc("融合条件特征 H_t（蓝条越短=该层可见条件越弱）")}</text>')

foot_y = 730
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="{GREY}">'
         f'{esc("τ/η：arXiv:2602.06036 Table 9，块扩散起草下 Input→KV 注入消融，GSM8K，5 层草稿器 block 8（论文自报，未独立复现）。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="10.5" '
         f'fill="{GREY}">'
         f'{esc("H_t=RMSNorm(W_c[H^(l1);…;H^(l5)])，W_c ∈ ℝ^(D×5D)（Appendix A.3）；同一份 H_t 被所有 draft 层共享。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-eagle-vs-dflash-injection.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
