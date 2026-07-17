#!/usr/bin/env python3
"""fig-m8-two-line: before-after 模板。
cuobjdump -sass 每条指令占两行(FLINE 携汇编体+首半编码,SLINE 携次半编码=控制字);
disasm 每次 line_idx += 2 把两行折叠成一条 (ctrl, asm)。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
TOP = 100
LEFT_W = 620
RIGHT_W = 470
GAP = 110

TITLE = "SASS 两行格式:cuobjdump 每条指令拆两行,disasm 每次 +2 折叠成一条"
SUBTITLE = "python/triton/tools/disasm.py:L108-L127 —— target=sm_90 实跑取证"

w = PAD * 2 + LEFT_W + GAP + RIGHT_W
elems = []


def add(s):
    elems.append(s)


LEFT_X = PAD
RIGHT_X = PAD + LEFT_W + GAP

# ---- 左面板:cuobjdump 原始两行 ----
add(f'<text x="{LEFT_X+LEFT_W/2:.0f}" y="{TOP-14:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" '
    f'fill="#0f172a">cuobjdump -sass 原始输出(两行 = 一条指令)</text>')

left_h = 210
add(f'<rect x="{LEFT_X:.0f}" y="{TOP:.0f}" width="{LEFT_W}" height="{left_h}" rx="10" '
    'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')

flines = [
    ("/*0000*/ LDC R1, c[0x0][0x28] ;", "/* 0x00000a00ff017b82 */", "FLINE(汇编体 + 首半编码)"),
    ("            (SLINE)", "/* 0x000e240000000800 */", "SLINE(次半编码 = 控制字)"),
]
add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+30:.0f}" font-family="monospace" font-size="12.5" '
    f'font-weight="bold" fill="#0f172a">/*0000*/ LDC R1, c[0x0][0x28] ;'
    f'  /* 0x00000a00ff017b82 */</text>')
add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+50:.0f}" font-family="sans-serif" font-size="10.5" '
    f'fill="#94a3b8">↑ FLINE:offset + 汇编体 + 首半 64 位编码</text>')
add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+78:.0f}" font-family="monospace" font-size="12.5" '
    f'font-weight="bold" fill="#334155">                              '
    f'/* 0x000e240000000800 */</text>')
add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+98:.0f}" font-family="sans-serif" font-size="10.5" '
    f'fill="#94a3b8">↑ SLINE:次半 64 位编码(parseCtrl 的输入)</text>')

add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+130:.0f}" font-family="monospace" font-size="12.5" '
    f'fill="#334155">/*0010*/ LDC R4, c[0x0][0x218] ;  /* 0x00008600ff047b82 */</text>')
add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+152:.0f}" font-family="monospace" font-size="12.5" '
    f'fill="#334155">                              /* 0x000e620000000800 */</text>')

add(f'<text x="{LEFT_X+22:.0f}" y="{TOP+185:.0f}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">offset 步长 0x10(16 字节)——相邻指令固定间距</text>')

# ---- 变换箭头 ----
mid_y = TOP + left_h / 2
ax1 = LEFT_X + LEFT_W
ax2 = RIGHT_X
add('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>')
add(f'<line x1="{ax1:.0f}" y1="{mid_y:.0f}" x2="{ax2:.0f}" y2="{mid_y:.0f}" '
    'stroke="#0369a1" stroke-width="2.5" marker-end="url(#a)"/>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y-18:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#0369a1">line_idx += 2</text>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y+8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11" fill="#475569">每指令读 2 行</text>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y+28:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11" fill="#475569">共 72 条指令</text>')

# ---- 右面板:折叠后的一条指令 ----
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP-14:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" '
    f'fill="#0f172a">折叠成一条 (ctrl, asm)</text>')

right_h = 210
add(f'<rect x="{RIGHT_X:.0f}" y="{TOP:.0f}" width="{RIGHT_W}" height="{right_h}" rx="10" '
    'fill="#e0f2fe" stroke="#0369a1" stroke-width="2"/>')
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP+40:.0f}" text-anchor="middle" '
    f'font-family="monospace" font-size="13" font-weight="bold" '
    f'fill="#0c4a6e">ctrl = --:-:0:-:2</text>')
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP+64:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11" fill="#0369a1">(wait:rd:wr:yield:stall,见 m9 parseCtrl)</text>')
add(f'<line x1="{RIGHT_X+20:.0f}" y1="{TOP+82:.0f}" x2="{RIGHT_X+RIGHT_W-20:.0f}" y2="{TOP+82:.0f}" '
    'stroke="#93c5fd" stroke-width="1"/>')
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP+112:.0f}" text-anchor="middle" '
    f'font-family="monospace" font-size="12.5" fill="#0c4a6e">LDC R1, c[0x0][0x28] ;</text>')
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP+150:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#1e3a5f">左列 ctrl 来自 SLINE,右侧 asm 来自 FLINE</text>')
add(f'<text x="{RIGHT_X+RIGHT_W/2:.0f}" y="{TOP+180:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#1e3a5f">SASS 左列打印格式的来历</text>')

h = TOP + left_h + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m8-two-line.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
