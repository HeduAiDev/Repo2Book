#!/usr/bin/env python3
"""fig-m3-cubin-embed: before-after 模板。
Python 手里的 cubin 字节经 binascii.hexlify 逐字节内嵌进 compile.c 的 C 数组,
配 cuModuleLoadData(内存直读)+ cuLaunchKernel(一键启动),构成脱离 Python 的自包含 C。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
TOP = 96
PANEL_W = 470
PANEL_GAP = 140

TITLE = "cubin 内嵌:binascii.hexlify 把 GPU 机器码逐字节抄进 C 数组"
SUBTITLE = "python/triton/tools/compile.py:L116-L155 —— 一个 .c 编进 .so 后不再需要 Python 或旁挂 .cubin"

w = PAD * 2 + PANEL_W * 2 + PANEL_GAP
elems = []


def add(s):
    elems.append(s)


LEFT_X = PAD
RIGHT_X = PAD + PANEL_W + PANEL_GAP

# ---- 左面板:Python 手里的 cubin ----
add(f'<text x="{LEFT_X+PANEL_W/2:.0f}" y="{TOP-14:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" '
    f'fill="#0f172a">Python 手里的 cubin(GPU 机器码)</text>')

left_h = 150
add(f'<rect x="{LEFT_X:.0f}" y="{TOP:.0f}" width="{PANEL_W}" height="{left_h}" rx="10" '
    'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')
add(f'<text x="{LEFT_X+24:.0f}" y="{TOP+30:.0f}" font-family="monospace" font-size="13" '
    f'font-weight="bold" fill="#0f172a">cubin bytes: 5648 字节</text>')
add(f'<text x="{LEFT_X+24:.0f}" y="{TOP+58:.0f}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">数组头 6 字节(ELF 魔数,cubin 本质是 ELF):</text>')
add(f'<text x="{LEFT_X+24:.0f}" y="{TOP+82:.0f}" font-family="monospace" font-size="13" '
    f'fill="#334155">0x7f 0x45 0x4c 0x46 0x02 0x01</text>')
add(f'<text x="{LEFT_X+24:.0f}" y="{TOP+82+16:.0f}" font-family="sans-serif" font-size="10.5" '
    f'fill="#94a3b8">(0x7f\'E\'\'L\'\'F\' + 类别 + 端序)</text>')
add(f'<text x="{LEFT_X+24:.0f}" y="{TOP+128:.0f}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">triton.compile(...).kernel —— 编译产出的原始字节串</text>')

# ---- 变换箭头 + 标签 ----
mid_y = TOP + left_h / 2
ax1 = LEFT_X + PANEL_W
ax2 = RIGHT_X
add('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')
add(f'<line x1="{ax1:.0f}" y1="{mid_y:.0f}" x2="{ax2:.0f}" y2="{mid_y:.0f}" '
    'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y-14:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#b45309">binascii.hexlify</text>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y+22:.0f}" text-anchor="middle" '
    f'font-family="monospace" font-size="11.5" fill="#78350f">1 字节 → 2 hex 字符</text>')
add(f'<text x="{(ax1+ax2)/2:.0f}" y="{mid_y+40:.0f}" text-anchor="middle" '
    f'font-family="monospace" font-size="12" font-weight="bold" '
    f'fill="#b45309">5648 → 11296</text>')

# ---- 右面板:填好的 compile.c ----
add(f'<text x="{RIGHT_X+PANEL_W/2:.0f}" y="{TOP-14:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" '
    f'fill="#0f172a">填好的 compile.c(自包含 C 源)</text>')

right_h = 150
add(f'<rect x="{RIGHT_X:.0f}" y="{TOP:.0f}" width="{PANEL_W}" height="{right_h}" rx="10" '
    'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')

code_lines = [
    ("unsigned char CUBIN_NAME[11296] = {", "#78350f", True),
    ("  0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01, ...", "#92400e", False),
    ("};", "#78350f", True),
    ("cuModuleLoadData(&mod, CUBIN_NAME);  // 内存直读,无需 .cubin 文件", "#166534", False),
    ("cuLaunchKernel(func, 1,1,1, 4*32,1,1, 0, stream, args, NULL);", "#1d4ed8", False),
]
cy0 = TOP + 26
for i, (line, color, bold) in enumerate(code_lines):
    y = cy0 + i * 24
    bold_attr = 'font-weight="bold" ' if bold else ''
    add(f'<text x="{RIGHT_X+20:.0f}" y="{y:.0f}" font-family="monospace" font-size="12" '
        f'{bold_attr}fill="{color}">{esc(line)}</text>')

h = TOP + left_h + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m3-cubin-embed.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
