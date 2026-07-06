#!/usr/bin/env python3
"""layout 模板:GPU 内存层级(SRAM vs HBM)+ 标准注意力三次 N×N 往返物化。
claim: 标准注意力把 S、P 两张 N×N 中间矩阵物化到慢速 HBM 再读回,访存量 Theta(N^2) 主导 wall-clock。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "GPU 内存层级与标准注意力的三趟 N×N 往返"
SUBTITLE = "标准注意力(Algorithm 0):S=QKᵀ 写回 HBM → P=softmax(S) 读回算、再写回 → O=PV 读回算——三步各摸一次 N×N"

PAD, TOP = 40, 92
# 两个内存层级方框
SRAM_W, SRAM_H = 190, 110
HBM_W, HBM_H = 190, 170
MEM_GAP = 90
sram_x, sram_y = PAD, TOP + (HBM_H - SRAM_H) / 2
hbm_x, hbm_y = sram_x + SRAM_W + MEM_GAP, TOP

# 右侧:三步物化流程(S 写回、P 读+写、O 读+写)
STEPS = [
    ("① 算 S=QKᵀ → 写回 HBM", "N×N"),
    ("② 读回 S → 算 P=softmax(S) → 写回 HBM", "N×N"),
    ("③ 读回 P、V → 算 O=PV", "N×N"),
]
flow_x = hbm_x + HBM_W + 110
FLOW_BOX_W, FLOW_BOX_H, FLOW_VGAP = 300, 46, 22

w = flow_x + FLOW_BOX_W + PAD
h = TOP + HBM_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+10}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# SRAM 方框(小、快)
L.append(f'<rect x="{sram_x}" y="{sram_y}" width="{SRAM_W}" height="{SRAM_H}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+30}" text-anchor="middle" '
          'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#1e3a8a">{esc("SRAM(片上书桌)")}</text>')
L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+56}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#1e3a8a">{esc("19 TB/s 带宽")}</text>')
L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+76}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#1e3a8a">{esc("192 KB / SM")}</text>')
L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+96}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#3b82f6">{esc("快 10×,但装不下 N×N")}</text>')

# HBM 方框(大、慢)
L.append(f'<rect x="{hbm_x}" y="{hbm_y}" width="{HBM_W}" height="{HBM_H}" rx="8" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="2.5"/>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+30}" text-anchor="middle" '
          'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#7f1d1d">{esc("HBM(片外仓库)")}</text>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+56}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#7f1d1d">{esc("1.5-2.0 TB/s 带宽")}</text>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+76}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#7f1d1d">{esc("40-80 GB")}</text>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+100}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#b91c1c">{esc("存 S、P 两张 N×N 中间矩阵")}</text>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+120}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#b91c1c">{esc("慢 10×,但装得下")}</text>')
L.append(f'<text x="{hbm_x+HBM_W/2}" y="{hbm_y+144}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#7f1d1d">{esc("访存 Θ(Nd+N²)")}</text>')

# SRAM <-> HBM 往返箭头(红色,标三趟)
mid_y = TOP + HBM_H / 2
L.append(f'<line x1="{sram_x+SRAM_W}" y1="{mid_y-14}" x2="{hbm_x}" y2="{mid_y-14}" '
          'stroke="#b91c1c" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<line x1="{hbm_x}" y1="{mid_y+14}" x2="{sram_x+SRAM_W}" y2="{mid_y+14}" '
          'stroke="#b91c1c" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{sram_x+SRAM_W+MEM_GAP/2}" y="{mid_y-22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#b91c1c">{esc("写 S/P")}</text>')
L.append(f'<text x="{sram_x+SRAM_W+MEM_GAP/2}" y="{mid_y+30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#b91c1c">{esc("读 S/P/V")}</text>')
L.append(f'<text x="{sram_x+SRAM_W+MEM_GAP/2}" y="{mid_y+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#b91c1c">{esc("×3 趟")}</text>')

# 右侧三步流程框(垂直排列,箭头连接)
flow_top = TOP + 10
for i, (label, tag) in enumerate(STEPS):
    y = flow_top + i * (FLOW_BOX_H + FLOW_VGAP)
    L.append(f'<rect x="{flow_x}" y="{y}" width="{FLOW_BOX_W}" height="{FLOW_BOX_H}" rx="6" '
              'fill="#fef3c7" stroke="#b45309" stroke-width="1.8"/>')
    L.append(f'<text x="{flow_x+16}" y="{y+20}" font-family="sans-serif" font-size="12.5" '
              f'fill="#78350f">{esc(label)}</text>')
    L.append(f'<text x="{flow_x+FLOW_BOX_W-16}" y="{y+FLOW_BOX_H-12}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="#b45309">{esc(tag)}</text>')
    if i < len(STEPS) - 1:
        y2 = y + FLOW_BOX_H
        L.append(f'<line x1="{flow_x+FLOW_BOX_W/2}" y1="{y2}" x2="{flow_x+FLOW_BOX_W/2}" '
                  f'y2="{y2+FLOW_VGAP-4}" stroke="#64748b" stroke-width="1.5" marker-end="url(#b)"/>')

# 从 HBM 指向流程框(表示三步都要经过 HBM 往返)
hbm_to_flow_y = TOP + HBM_H / 2
L.append(f'<line x1="{hbm_x+HBM_W}" y1="{hbm_to_flow_y}" x2="{flow_x-10}" y2="{hbm_to_flow_y}" '
          'stroke="#64748b" stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#b)"/>')

foot_y = h - 20
FOOT = "三步各读写一次 N×N(S 写回、P 读+写、O 读回算)——访存量随 N² 增长,是注意力的内存带宽墙。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-1-memory-wall.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
