#!/usr/bin/env python3
"""fig-m1-ptrstate-anatomy: PtrState 结构解剖（layout 模板）。
四组槽位横排：source(base 指针) / offset(标量起点) / stateInfo[](逐维 stride/shape/dimIndex)
/ sizes(原始形状)。下方叠一个具体的 2D 行主序块示例把槽位填实。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "PtrState 结构解剖（PtrAnalysis.h:L41-L64 struct StateInfo）"
SUBTITLE = "一份指针分析结果 = base 仓库 + 块内起点 + 逐维跳格规则；addState/mulState 就是合并两张这样的单子"

SLOTS = [
    ("source", "base 指针", "%arg1（!tt.ptr<i8>）", "#93c5fd"),
    ("offset", "标量起点", "rem(%9, 1024)", "#86efac"),
    ("stateInfo[]", "逐维 (stride,shape,dimIndex)", "[(%arg4,64,d0),(1,256,d1)]", "#fcd34d"),
    ("sizes", "原始形状", "[64, 256]", "#f9a8d4"),
]

SLOT_W, SLOT_H, GAP, PAD, TOP = 300, 96, 24, 40, 108
n = len(SLOTS)
w = PAD * 2 + n * SLOT_W + (n - 1) * GAP
h = TOP + SLOT_H + 250

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

slot_x = [PAD + i * (SLOT_W + GAP) for i in range(n)]

# 四个槽位盒
for i, (name, meaning, example, color) in enumerate(SLOTS):
    x = slot_x[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{SLOT_W}" height="{SLOT_H}" rx="10" '
              f'fill="{color}" stroke="#1e3a5f" stroke-width="2"/>')
    L.append(f'<text x="{x+SLOT_W/2}" y="{TOP+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+SLOT_W/2}" y="{TOP+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#334155">{esc(meaning)}</text>')
    L.append(f'<text x="{x+SLOT_W/2}" y="{TOP+74}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#78350f" font-weight="bold">{esc(example)}</text>')

# stateInfo 槽位下方展开三元组结构（放大解释第三个槽位）
detail_y = TOP + SLOT_H + 56
detail_x = slot_x[2]
L.append(f'<line x1="{detail_x+SLOT_W/2}" y1="{TOP+SLOT_H}" x2="{detail_x+SLOT_W/2}" '
          f'y2="{detail_y-8}" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{detail_x+SLOT_W/2}" y="{detail_y+10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#92400e">'
          f'{esc("stateInfo 元素三元组 = (stride, shape, dimIndex)")}</text>')

triple_cols = [("stride", "%arg4", "地址每走一格跳多远"), ("shape", "64", "这一维有多少格"),
               ("dimIndex", "d0", "原张量第几维")]
tc_w, tc_gap = 210, 20
tc_total = len(triple_cols) * tc_w + (len(triple_cols) - 1) * tc_gap
tc_x0 = detail_x + SLOT_W / 2 - tc_total / 2
tc_y = detail_y + 26
for i, (label, val, desc) in enumerate(triple_cols):
    tx = tc_x0 + i * (tc_w + tc_gap)
    L.append(f'<rect x="{tx}" y="{tc_y}" width="{tc_w}" height="64" rx="6" '
              'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>')
    L.append(f'<text x="{tx+tc_w/2}" y="{tc_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#713f12">{esc(label)}={esc(val)}</text>')
    L.append(f'<text x="{tx+tc_w/2}" y="{tc_y+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#78350f">{esc(desc)}</text>')

# 底部小结：一个 PtrState 汇总示例 + shouldLinearize 标志
foot_y = tc_y + 64 + 46
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="54" rx="8" '
          'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+18}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#3730a3">'
          f'{esc("示例终态：source=%arg1, offset=rem(%9,1024), stateInfo=[(%arg4,64,d0),(1,256,d1)], sizes=[64,256]")}</text>')
L.append(f'<text x="{PAD+18}" y="{foot_y+42}" font-family="sans-serif" font-size="10.5" '
          f'fill="#4338ca">{esc("shouldLinearize：bool，默认 false（PtrAnalysis.h:L64）——需要线性化回退时才置 true")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-ptrstate-anatomy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
