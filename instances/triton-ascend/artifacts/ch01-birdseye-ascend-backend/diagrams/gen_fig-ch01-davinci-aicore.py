#!/usr/bin/env python3
"""fig-ch01-davinci-aicore — layout 模板。
达芬奇 AI Core = cube 矩阵核 + vector 向量核，配显式片上内存层级
UB/L1/L0A/L0B/L0C 与片外 GM——数据搬运必须编译器显式写，不像 GPU 隐式 cache。
TCoreType 枚举只编码 CUBE/VECTOR 两类核（另有 2 个组合值）；AddressSpace 枚举
5 个片上层级。1:2 配比标注来源为 ch02 硬件原理篇，非从枚举推出。
全部坐标由循环/常量计算，颜色即语义（cube=紫，vector=青，片上=蓝灰，片外=橙）。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

CORE_W, CORE_H = 260, 90
PAD, TOP = 40, 130
CORE_GAP = 60
MEM_W, MEM_H, MEM_GAP = 130, 64, 16
GM_H = 56

MEM_LEVELS = ["L0A", "L0B", "L0C", "UB", "L1"]  # 5 个片上层级，ascend_ir.cc:L413-L417

w = PAD * 2 + max(CORE_W * 2 + CORE_GAP, len(MEM_LEVELS) * (MEM_W + MEM_GAP) - MEM_GAP)
core_y = TOP
mem_y = core_y + CORE_H + 70
gm_y = mem_y + MEM_H + 60
h = gm_y + GM_H + 140

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("目标硬件：达芬奇 AI Core——双核异构 + 显式片上内存层级")}</text>',
     f'<text x="{w/2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
     f'fill="#78716c">{esc("TCoreType 只编码 CUBE/VECTOR 两类核，另有 2 个组合值 CUBE_OR_VECTOR / CUBE_AND_VECTOR（ascend_ir.cc:L421-L422）")}</text>',
     f'<text x="{w/2}" y="76" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'font-weight="bold" fill="#3730a3">{esc("片上内存层级：AddressSpace 5 个枚举值（ascend_ir.cc:L413-L417）")}</text>']

# ── cube / vector 双核 ──
cube_x = w/2 - CORE_GAP/2 - CORE_W
vec_x = w/2 + CORE_GAP/2
L.append(f'<rect x="{cube_x}" y="{core_y}" width="{CORE_W}" height="{CORE_H}" rx="12" '
         f'fill="#ede9fe" stroke="#7c3aed" stroke-width="2.2"/>')
L.append(f'<text x="{cube_x+CORE_W/2}" y="{core_y+30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="#5b21b6">{esc("Cube 核（矩阵）")}</text>')
L.append(f'<text x="{cube_x+CORE_W/2}" y="{core_y+52}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#5b21b6">{esc("TCoreType.CUBE")}</text>')
L.append(f'<text x="{cube_x+CORE_W/2}" y="{core_y+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#7c3aed">{esc("ascend_ir.cc:L421")}</text>')

L.append(f'<rect x="{vec_x}" y="{core_y}" width="{CORE_W}" height="{CORE_H}" rx="12" '
         f'fill="#cffafe" stroke="#0891b2" stroke-width="2.2"/>')
L.append(f'<text x="{vec_x+CORE_W/2}" y="{core_y+30}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="16" fill="#155e75">{esc("Vector 核（向量）")}</text>')
L.append(f'<text x="{vec_x+CORE_W/2}" y="{core_y+52}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#155e75">{esc("TCoreType.VECTOR")}</text>')
L.append(f'<text x="{vec_x+CORE_W/2}" y="{core_y+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#0891b2">{esc("ascend_ir.cc:L422")}</text>')

# ── 片上内存层级：5 个 ──
mem_total = len(MEM_LEVELS) * (MEM_W + MEM_GAP) - MEM_GAP
mem_x0 = (w - mem_total) / 2
mem_xs = [mem_x0 + i * (MEM_W + MEM_GAP) for i in range(len(MEM_LEVELS))]
for i, name in enumerate(MEM_LEVELS):
    x = mem_xs[i]
    L.append(f'<rect x="{x}" y="{mem_y}" width="{MEM_W}" height="{MEM_H}" rx="8" '
             f'fill="#e0e7ff" stroke="#4338ca" stroke-width="1.6"/>')
    L.append(f'<text x="{x+MEM_W/2}" y="{mem_y+MEM_H/2+5}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13.5" font-weight="bold" fill="#3730a3">{esc(name)}</text>')

# 核心 -> 片上内存 连线（cube 连 L0A/L0B/L0C，vector 连 UB）
def bottom_center(x, y, wid, hgt):
    return x + wid/2, y + hgt

def top_center(x, y, wid):
    return x + wid/2, y

cube_bx, cube_by = bottom_center(cube_x, core_y, CORE_W, CORE_H)
vec_bx, vec_by = bottom_center(vec_x, core_y, CORE_W, CORE_H)
for idx in (0, 1, 2):  # L0A, L0B, L0C
    tx, ty = top_center(mem_xs[idx], mem_y, MEM_W)
    L.append(f'<line x1="{cube_bx}" y1="{cube_by+4}" x2="{tx}" y2="{ty-4}" '
             f'stroke="#7c3aed" stroke-width="1.4" marker-end="url(#a)"/>')
tx, ty = top_center(mem_xs[3], mem_y, MEM_W)  # UB
L.append(f'<line x1="{vec_bx}" y1="{vec_by+4}" x2="{tx}" y2="{ty-4}" '
         f'stroke="#0891b2" stroke-width="1.4" marker-end="url(#a)"/>')

# ── GM（片外 DRAM）──
gm_w = mem_total
gm_x = mem_x0
L.append(f'<rect x="{gm_x}" y="{gm_y}" width="{gm_w}" height="{GM_H}" rx="10" '
         f'fill="#ffedd5" stroke="#c2410c" stroke-width="2"/>')
L.append(f'<text x="{gm_x+gm_w/2}" y="{gm_y+GM_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#9a3412">{esc("GM（Global Memory，片外 DRAM）")}</text>')

# GM <-> 片上：显式搬运双向箭头——L0C（cube 侧代表）与 UB（vector 侧代表，对应正文
# 「GM 搬到 UB 再算、算完搬回」的具体例子）各一条，样式一致；箭头间不放文字，避免与
# 箭头相撞——说明文字统一放到底部注记
def gm_link(mem_idx):
    cx = mem_xs[mem_idx] + MEM_W/2
    L.append(f'<line x1="{cx-14}" y1="{gm_y}" x2="{cx-14}" y2="{mem_y+MEM_H+4}" '
             f'stroke="#c2410c" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<line x1="{cx+14}" y1="{mem_y+MEM_H}" x2="{cx+14}" y2="{gm_y-4}" '
             f'stroke="#c2410c" stroke-width="2" marker-end="url(#a)"/>')

gm_link(2)   # L0C（cube 侧代表）
gm_link(3)   # UB（vector 侧代表，正文 GM↔UB 例子）

# ── 底部注记：搬运图例 + 1:2 配比出处澄清 + 为何只画两条搬运箭头 ──
foot_y = h - 82
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
         f'fill="#9a3412" font-weight="bold">{esc("红色箭头 = 显式搬运（tl.load/tl.store，编译器必须写）")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="12" '
         f'fill="#374151">{esc("L0A/L0B/L0C/UB/L1 五级与 GM 之间的搬运同样全部显式；图中以 L0C（cube 侧）与 UB（vector 侧，即正文例子）为代表各画一条。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+36}" font-family="sans-serif" font-size="12" '
         f'fill="#374151">{esc("cube : vector = 1 : 2 是达芬奇硬件事实，由 ch02《达芬奇硬件》原理篇量化建立——")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+54}" font-family="sans-serif" font-size="12" '
         f'fill="#374151">{esc("不是从上方 TCoreType 枚举推出（枚举只编码类别，不编码数量）。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-davinci-aicore.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
