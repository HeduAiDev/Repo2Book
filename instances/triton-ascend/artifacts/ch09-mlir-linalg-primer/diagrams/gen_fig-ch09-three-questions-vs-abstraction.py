#!/usr/bin/env python3
"""fig-ch09-three-questions-vs-abstraction — m23 transformation-oriented IR design。
主轴:抽象层级竖轴(顶=结构化算子层,底=标量+循环+指针层),右侧两条反向渐变条
(通用性/正规性 向下增强,分析与变换可解性 向下衰减);左侧三张卡片(合法性/可施加性/收益)
指向其依附的抽象层级;底部 phase-ordering 反例小插图。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def esc_bold(s):
    # rsvg-convert/Droid Sans Fallback 在 font-weight="bold" 下把「量」(U+91CF)
    # 误渲染成实心方块(逐次重渲复现,与字号/字体族无关)；用 tspan 把该字降回
    # normal 权重规避——此字体的中文本就不随 bold 变粗,视觉零回归。
    return esc(s).replace('量', '<tspan font-weight="normal">量</tspan>')

TITLE = "三问依附在哪一层抽象,是可以设计的"
SUBTITLE = "结构化算子层:合法性与可施加性由算子性质与结构直接导出(legal by design);越往标量+循环层走,越通用也越难分析"

PAD = 44
W = 1500
H = 900

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
         '<marker id="ax" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
         'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
         '<linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">'
         '<stop offset="0%" stop-color="#e2e8f0"/><stop offset="100%" stop-color="#1d4ed8"/></linearGradient>'
         '<linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">'
         '<stop offset="0%" stop-color="#15803d"/><stop offset="100%" stop-color="#e2e8f0"/></linearGradient>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

AXIS_TOP = 110
AXIS_BOT = 560
AXIS_X = 760

# ---- 主轴:抽象层级 ----
L.append(f'<line x1="{AXIS_X}" y1="{AXIS_TOP}" x2="{AXIS_X}" y2="{AXIS_BOT}" '
         f'stroke="#0f172a" stroke-width="2.4" marker-end="url(#a)"/>')
L.append(f'<rect x="{AXIS_X-150}" y="{AXIS_TOP-34}" width="300" height="30" rx="6" '
         f'fill="#1d4ed8"/>')
L.append(f'<text x="{AXIS_X}" y="{AXIS_TOP-13}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="white">{esc("结构化算子层")}</text>')
L.append(f'<rect x="{AXIS_X-150}" y="{AXIS_BOT+6}" width="300" height="30" rx="6" '
         f'fill="#334155"/>')
L.append(f'<text x="{AXIS_X}" y="{AXIS_BOT+27}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="white">{esc_bold("标量 + 循环 + 指针层")}</text>')

# ---- 右侧两条反向渐变条 ----
bar_w = 22
bar1_x = AXIS_X + 260
bar2_x = bar1_x + 150
L.append(f'<rect x="{bar1_x}" y="{AXIS_TOP}" width="{bar_w}" height="{AXIS_BOT-AXIS_TOP}" rx="6" '
         f'fill="url(#g1)"/>')
L.append(f'<text x="{bar1_x+bar_w/2}" y="{AXIS_TOP-28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">{esc("通用性")}</text>')
L.append(f'<text x="{bar1_x+bar_w/2}" y="{AXIS_TOP-14}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">{esc("/正规性")}</text>')
L.append(f'<text x="{bar1_x+bar_w/2}" y="{AXIS_BOT+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" fill="#64748b">{esc("(向下增强)")}</text>')

L.append(f'<rect x="{bar2_x}" y="{AXIS_TOP}" width="{bar_w}" height="{AXIS_BOT-AXIS_TOP}" rx="6" '
         f'fill="url(#g2)"/>')
L.append(f'<text x="{bar2_x+bar_w/2}" y="{AXIS_TOP-28}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#15803d">{esc("分析/变换")}</text>')
L.append(f'<text x="{bar2_x+bar_w/2}" y="{AXIS_TOP-14}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#15803d">{esc("可解性")}</text>')
L.append(f'<text x="{bar2_x+bar_w/2}" y="{AXIS_BOT+18}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" fill="#64748b">{esc("(向下衰减)")}</text>')

# ---- 左侧三张卡片:合法性/可施加性/收益 ----
cards = [
    ("合法性", "施加后是否改变可观察语义", AXIS_TOP + 30),
    ("可施加性", "找位置多难/变换后 IR 多复杂/丢多少信息/后续是否仍好变换", AXIS_TOP + 190),
    ("收益", "按某度量是否有益,通常靠启发式或性能模型", AXIS_TOP + 350),
]
card_w, card_h = 330, 110
card_x = PAD
for name, desc, cy in cards:
    L.append(f'<rect x="{card_x}" y="{cy}" width="{card_w}" height="{card_h}" rx="9" '
              f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.8"/>')
    L.append(f'<text x="{card_x+16}" y="{cy+28}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#3730a3">{esc(name)}</text>')
    # 换行描述(简单按字数断行)
    max_chars = 20
    words = desc
    wrapped = []
    cur = ""
    for ch in words:
        cur += ch
        if len(cur) >= max_chars and ch in "/、,":
            wrapped.append(cur); cur = ""
    if cur:
        wrapped.append(cur)
    for k, line in enumerate(wrapped[:3]):
        L.append(f'<text x="{card_x+16}" y="{cy+50+k*20}" font-family="sans-serif" font-size="11" '
                  f'fill="#4338ca">{esc(line)}</text>')
    # 箭头指向结构化算子层(顶端)
    arrow_y = AXIS_TOP + 15
    L.append(f'<path d="M {card_x+card_w} {cy+card_h/2} '
              f'C {AXIS_X-260} {cy+card_h/2}, {AXIS_X-260} {arrow_y}, {AXIS_X-152} {arrow_y}" '
              f'fill="none" stroke="#6366f1" stroke-width="1.4" stroke-dasharray="4,3" '
              f'marker-end="url(#a)" opacity="0.7"/>')

L.append(f'<text x="{card_x}" y="{AXIS_TOP+16}" font-family="sans-serif" font-size="11" '
         f'font-weight="bold" fill="#4338ca">{esc("三问都指向同一层:结构化算子层")}</text>')

# ---- 底部 phase-ordering 反例插图 ----
foot_y = AXIS_BOT + 90
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{W-2*PAD}" height="170" rx="10" '
         f'fill="#fef2f2" stroke="#dc2626" stroke-width="1.6"/>')
L.append(f'<text x="{PAD+18}" y="{foot_y+26}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#991b1b">{esc("phase-ordering 反例(为何这三问不能只在下层判断)")}</text>')

step_y = foot_y + 60
box_w, box_h = 240, 60
gap2 = 60
x1 = PAD + 40
x2 = x1 + box_w + gap2
x3 = x2 + box_w + gap2

L.append(f'<rect x="{x1}" y="{step_y}" width="{box_w}" height="{box_h}" rx="8" '
         f'fill="#fff" stroke="#334155" stroke-width="1.4"/>')
L.append(f'<text x="{x1+box_w/2}" y="{step_y+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#0f172a">{esc("两个算子融合")}</text>')
L.append(f'<text x="{x1+box_w/2}" y="{step_y+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#334155">{esc("(循环融合,提升时间局部性)")}</text>')

L.append(f'<line x1="{x1+box_w}" y1="{step_y+box_h/2}" x2="{x2}" y2="{step_y+box_h/2}" '
         f'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

L.append(f'<rect x="{x2}" y="{step_y}" width="{box_w}" height="{box_h}" rx="8" '
         f'fill="#dcfce7" stroke="#15803d" stroke-width="1.6"/>')
L.append(f'<text x="{x2+box_w/2}" y="{step_y+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" font-weight="bold" fill="#14532d">{esc("时间局部性变好")}</text>')
L.append(f'<text x="{x2+box_w/2}" y="{step_y+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#166534">{esc("(收益判断:看似有益)")}</text>')

L.append(f'<line x1="{x2+box_w}" y1="{step_y+box_h/2}" x2="{x3}" y2="{step_y+box_h/2}" '
         f'stroke="#dc2626" stroke-width="1.8" marker-end="url(#ax)"/>')
L.append(f'<text x="{(x2+box_w+x3)/2}" y="{step_y-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10" fill="#dc2626">{esc("但……")}</text>')

L.append(f'<rect x="{x3}" y="{step_y}" width="{box_w}" height="{box_h}" rx="8" '
         f'fill="#fee2e2" stroke="#dc2626" stroke-width="1.8"/>')
L.append(f'<text x="{x3+box_w/2}" y="{step_y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#991b1b">{esc("破坏后续识别")}</text>')
L.append(f'<text x="{x3+box_w/2}" y="{step_y+40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" font-weight="bold" fill="#991b1b">{esc("BLAS-2/BLAS-3 库实现")}</text>')
L.append(f'<text x="{x3+box_w/2}" y="{step_y+56}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10" fill="#991b1b">{esc("的能力")}</text>')

L.append(f'<text x="{PAD+18}" y="{foot_y+150}" font-family="sans-serif" font-size="10.5" '
         f'fill="#7f1d1d">{esc("本章可执行佐证:tiling/padding/向量化/bufferization 四条变换与不变换的数值最大偏差均为 0(本章参考实现实测)")}</text>')

foot2_y = H - 14
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc("依据:arXiv:2202.03293 §3.6、§3.6.1;拼写按论文两处原样(transformations-oriented / transformation-oriented),不代论文统一")}</text>')

L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-three-questions-vs-abstraction.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
