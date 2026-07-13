#!/usr/bin/env python3
"""fig-eplb-epiphany — 顿悟头图（spec docs/.../2026-07-12-primer-redesign-design.md §2.5 五步法）。
一图只锚一拳：搬永远够不着地板（最优 90 > 87.5，差 2.5），复制热专家买来「可分性」，恰好贴地 87.5/87.5。
视觉主轴 = 那道够不着的缝被复制抹平（左联柱顶悬在地板线上方 vs 右联两柱齐贴地板线）。
数字全部来自 figure-requests 条目 numbers + worked_example_trace.txt（带溯源），零即兴：
  87.5 = 总热度175/2卡（地板/box_weights）；90、85 = 纯搬运枚举（正文二节严谨框）；
  60/55/10 = 折叠热度；30+30 = trace round1；27.5+27.5 = trace round2；par 1.543->1.0 = trace METRIC。
全部坐标由常量/循环计算，文本全过 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

# ---- 画布与数值坐标系（地板附近放大轴）----
W, H = 1200, 650
AX_X = 96                      # 纵轴位置
VMIN, VMAX = 80.0, 94.0        # 放大到地板附近，让 2.5 的缝可见
PLOT_TOP, PLOT_BOT = 150, 508  # 绘图区上下沿（value 84..94 上部，80 为基线）
FLOOR = 87.5

def yv(v):  # value -> y（值越大越靠上）
    return PLOT_TOP + (VMAX - v) / (VMAX - VMIN) * (PLOT_BOT - PLOT_TOP)

# 颜色语义
C_FLOOR = "#4f46e5"   # 地板线（下界）
C_HOT   = "#ef4444"   # 瓶颈/够不着（红）
C_COLD  = "#94a3b8"   # 非瓶颈的另一卡（灰）
C_TIE   = "#22c55e"   # 贴地/达成（绿）
C_INK   = "#0f172a"
C_SUB   = "#475569"

BW = 84  # 柱宽

# 左联两柱（只许搬）：card A=90（瓶颈，红），card B=85（灰）
LEFT = [("A 卡", 90, "60 + 10×3", C_HOT), ("B 卡", 85, "55 + 10×3", C_COLD)]
cxLA, cxLB = 232, 384
left_cx = [cxLA, cxLB]
# 右联两柱（复制两份后）：两卡各 87.5（贴地，绿）
cxR0, cxR1 = 838, 988
right_cx = [cxR0, cxR1]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
         '<marker id="down" viewBox="0 0 6 10" refX="3" refY="9" markerWidth="5" markerHeight="7" '
         'orient="auto"><path d="M0,0 L3,10 L6,0 Z" fill="' + C_HOT + '"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# ---- 顶部一拳标题 ----
L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-family="sans-serif" '
         f'font-size="21" font-weight="bold" fill="{C_INK}">搬不平的，复制能摊平</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" fill="{C_SUB}">'
         f'纯搬运最优也差 2.5 够不着地板 —— 复制热专家买来「可分性」，恰好贴地 87.5</text>')

# ---- 纵轴（放大地板附近）----
L.append(f'<line x1="{AX_X}" y1="{PLOT_TOP-8}" x2="{AX_X}" y2="{PLOT_BOT}" stroke="{C_SUB}" stroke-width="1.5"/>')
for tick in [85, 90]:
    ty = yv(tick)
    L.append(f'<line x1="{AX_X-5}" y1="{ty}" x2="{AX_X}" y2="{ty}" stroke="{C_SUB}" stroke-width="1.5"/>')
    L.append(f'<text x="{AX_X-9}" y="{ty+4}" text-anchor="end" font-family="sans-serif" '
             f'font-size="12" fill="{C_SUB}">{tick}</text>')
# 基线标注（诚实的放大轴）
L.append(f'<text x="{AX_X-9}" y="{PLOT_BOT+4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="11" fill="{C_SUB}">80</text>')
# 纵轴题名（旋转，缩短到 5 字避免出画布）
axmy = (PLOT_TOP + PLOT_BOT) / 2
L.append(f'<text x="56" y="{axmy}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{C_SUB}" transform="rotate(-90 56 {axmy})">最热卡负载</text>')
# 放大轴的诚实说明（横排，置于轴顶）
L.append(f'<text x="{AX_X+6}" y="{PLOT_TOP-6}" text-anchor="start" font-family="sans-serif" '
         f'font-size="10.5" fill="{C_SUB}">↑ 地板附近放大</text>')

# ---- 地板线（贯穿两联，强调下界）----
fy = yv(FLOOR)
L.append(f'<line x1="{AX_X}" y1="{fy}" x2="{W-40}" y2="{fy}" stroke="{C_FLOOR}" '
         f'stroke-width="2.5" stroke-dasharray="10 5"/>')
# 地板标注放在两联之间的空档（x≈560..690）
gx0, gx1 = 548, 700
L.append(f'<rect x="{gx0}" y="{fy-30}" width="{gx1-gx0}" height="26" rx="6" '
         f'fill="#eef2ff" stroke="{C_FLOOR}" stroke-width="1"/>')
L.append(f'<text x="{(gx0+gx1)/2}" y="{fy-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="{C_FLOOR}">地板 87.5 = 175 ÷ 2 卡</text>')
L.append(f'<text x="{(gx0+gx1)/2}" y="{fy+16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="{C_FLOOR}">任何放置都踩不穿</text>')

# ---- 联标题（下移到各联上方，避开顶部副标题）----
TITLE_Y = 128
L.append(f'<text x="{(cxLA+cxLB)/2}" y="{TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" font-weight="bold" fill="{C_INK}">只许搬（整块不可拆）</text>')
L.append(f'<text x="{(cxR0+cxR1)/2}" y="{TITLE_Y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" font-weight="bold" fill="{C_INK}">复制两份（买可分性）</text>')

# ---- 左联柱 ----
for cx, (name, val, comp, col) in zip(left_cx, LEFT):
    top = yv(val)
    L.append(f'<rect x="{cx-BW/2}" y="{top}" width="{BW}" height="{PLOT_BOT-top}" rx="3" '
             f'fill="{col}" fill-opacity="0.82" stroke="{col}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{top-9}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="16" font-weight="bold" fill="{col}">{val}</text>')
    L.append(f'<text x="{cx}" y="{PLOT_BOT-9}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" fill="white">{esc(comp)}</text>')
    L.append(f'<text x="{cx}" y="{PLOT_BOT+18}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" fill="{C_SUB}">{esc(name)}</text>')

# 左联：那道够不着的缝（floor -> 90 顶）双箭头 + 标注
gap_x = cxLA + BW/2 + 26
L.append(f'<line x1="{gap_x}" y1="{fy}" x2="{gap_x}" y2="{yv(90)}" stroke="{C_HOT}" '
         f'stroke-width="1.8" marker-end="url(#down)"/>')
L.append(f'<line x1="{gap_x}" y1="{yv(90)}" x2="{gap_x}" y2="{fy}" stroke="{C_HOT}" '
         f'stroke-width="1.8" marker-end="url(#down)"/>')
L.append(f'<text x="{gap_x+8}" y="{(fy+yv(90))/2-3}" text-anchor="start" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="{C_HOT}">缝 2.5</text>')
L.append(f'<text x="{gap_x+8}" y="{(fy+yv(90))/2+14}" text-anchor="start" font-family="sans-serif" '
         f'font-size="11" fill="{C_HOT}">压不下去</text>')

# 左联底注：不可拆的原因
L.append(f'<text x="{(cxLA+cxLB)/2}" y="{PLOT_BOT+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{C_INK}">60、55 整块不可拆 → 最优分割只能 90 对 85</text>')

# ---- 右联：复制机制（因果，置于柱上方空档 y≈150..250）----
def cut_glyph(x, y, val, halves, whole_w=52):
    """一个热块 val 被复制成两半 halves=(a,b)，紧凑呈现。返回追加行。"""
    rows = []
    bh = 26
    # 原块（红）
    rows.append(f'<rect x="{x}" y="{y}" width="{whole_w}" height="{bh}" rx="3" '
                f'fill="{C_HOT}" fill-opacity="0.85"/>')
    rows.append(f'<text x="{x+whole_w/2}" y="{y+bh/2+5}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="12.5" font-weight="bold" fill="white">{esc(val)}</text>')
    # 箭头
    ax0 = x + whole_w + 6
    ax1 = ax0 + 34
    rows.append(f'<line x1="{ax0}" y1="{y+bh/2}" x2="{ax1}" y2="{y+bh/2}" stroke="{C_SUB}" '
                f'stroke-width="1.6" marker-end="url(#a)"/>')
    rows.append(f'<text x="{(ax0+ax1)/2}" y="{y-4}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="10.5" fill="{C_SUB}">复制</text>')
    # 两半（绿），中间一道切缝
    hw = 46
    hx = ax1 + 8
    for i, hv in enumerate(halves):
        rows.append(f'<rect x="{hx+i*(hw+6)}" y="{y}" width="{hw}" height="{bh}" rx="3" '
                    f'fill="{C_TIE}" fill-opacity="0.85"/>')
        rows.append(f'<text x="{hx+i*(hw+6)+hw/2}" y="{y+bh/2+5}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="white">{esc(hv)}</text>')
    return rows

cut_x = 770
L.append(f'<rect x="{cut_x-14}" y="150" width="{cxR1+BW/2 - cut_x + 30}" height="82" rx="8" '
         f'fill="#f8fafc" stroke="{C_TIE}" stroke-width="1.2"/>')
L += cut_glyph(cut_x, 162, "60", ("30", "30"))
L += cut_glyph(cut_x, 200, "55", ("27.5", "27.5"))

# ---- 右联柱（两卡各 87.5，柱顶贴地板线）----
for cx in right_cx:
    top = yv(FLOOR)
    L.append(f'<rect x="{cx-BW/2}" y="{top}" width="{BW}" height="{PLOT_BOT-top}" rx="3" '
             f'fill="{C_TIE}" fill-opacity="0.82" stroke="{C_TIE}" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{top-9}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="16" font-weight="bold" fill="#15803d">87.5</text>')
L.append(f'<text x="{cxR0}" y="{PLOT_BOT+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="{C_SUB}">卡 0</text>')
L.append(f'<text x="{cxR1}" y="{PLOT_BOT+18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="{C_SUB}">卡 1</text>')

# 右联：贴地 缝=0 标注（绿勾用 path，不用 emoji）
tie_x = cxR1 + BW/2 + 20
L.append(f'<path d="M{tie_x},{fy-2} l7,8 l13,-16" fill="none" stroke="#15803d" stroke-width="2.6"/>')
L.append(f'<text x="{tie_x+26}" y="{fy-2}" text-anchor="start" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#15803d">贴地 缝 = 0</text>')
L.append(f'<text x="{tie_x+26}" y="{fy+15}" text-anchor="start" font-family="sans-serif" '
         f'font-size="11" fill="#15803d">par 1.543 → 1.0</text>')

# 右联底注
L.append(f'<text x="{(cxR0+cxR1)/2}" y="{PLOT_BOT+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{C_INK}">复制把 60→30+30、55→27.5+27.5，可分了，才够得着地板</text>')

L.append('</svg>')

out = Path(__file__).with_name("fig-eplb-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({W}x{H}, ratio {W/H:.2f})")
