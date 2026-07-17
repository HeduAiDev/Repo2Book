#!/usr/bin/env python3
"""state-table 模板:NvidiaMmaEncoding 用一个整数 versionMajor 分派三代 Tensor Core。
行=versionMajor 取值,列=架构/判定谓词/本章 matmul 实测；实发行(versionMajor=2)高亮。
全坐标计算,零手写魔数,宽度按最长文本估算留够。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "NvidiaMmaEncoding:一个整数 versionMajor 分派三代 Tensor Core"
SUBTITLE = "谓词见 lib/Dialect/TritonGPU/IR/Dialect.cpp:L1858-L1866；本章 64x64 fp16 matmul 实发 versionMajor=2"

COLS = ["versionMajor", "架构", "判定谓词 (Dialect.cpp)", "getThreadsPerWarp"]
ROWS = [
    ["1", "Volta", "isVolta(): versionMajor==1", "[4, 8]"],
    ["2 (本章实发)", "Ampere / Turing", "isAmpere(): versionMajor==2", "[8, 4]"],
    ["3", "Hopper", "isHopper(): versionMajor==3", "[8, 4]"],
]
HIGHLIGHT_ROW = 1
EXTRA = [("实发 mma", "versionMajor=2, versionMinor=0"), ("instrShape", "[16, 8] (mma.16816)")]

PAD, TOP = 46, 116
LABEL_W = 0
col_texts_w = []
for j, col in enumerate(COLS):
    w = text_w(col, 12) * 1.15 + 36   # 表头 bold,按 1.15 倍加权 + 更宽留边
    for row in ROWS:
        w = max(w, text_w(row[j], 13) + 30)
    col_texts_w.append(w)
COL_W = [max(w, 130) for w in col_texts_w]
ROW_H = 44
HEADER_H = 38

col_x = [PAD]
for w in COL_W[:-1]:
    col_x.append(col_x[-1] + w)
table_w = sum(COL_W)
W = int(PAD * 2 + table_w)

extra_w = max(text_w(f"{k}: {v}", 13) for k, v in EXTRA)
W = int(max(W, PAD * 2 + extra_w))
title_w = text_w(TITLE, 17)
subtitle_w = text_w(SUBTITLE, 12)
W = int(max(W, PAD + max(title_w, subtitle_w) + PAD))

H = TOP + HEADER_H + len(ROWS) * ROW_H + 30 + len(EXTRA) * 24 + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
for j, col in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W[j]-4}" height="{HEADER_H}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.2"/>')
    L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="white">{esc(col)}</text>')

# 数据行
for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    is_hl = (i == HIGHLIGHT_ROW)
    row_fill = "#fef3c7" if is_hl else ("#f8fafc" if i % 2 == 0 else "white")
    row_stroke = "#b45309" if is_hl else "#cbd5e1"
    for j, col in enumerate(COLS):
        x = col_x[j]
        L.append(f'<rect x="{x}" y="{ry}" width="{COL_W[j]-4}" height="{ROW_H-4}" '
                  f'fill="{row_fill}" stroke="{row_stroke}" stroke-width="{2 if is_hl else 1}"/>')
        color = "#92400e" if is_hl else "#334155"
        weight = 'font-weight="bold" ' if (is_hl and j == 0) else ''
        L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{ry+(ROW_H-4)/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" {weight}'
                  f'fill="{color}">{esc(row[j])}</text>')

# 实发数据卡片
extra_y0 = TOP + HEADER_H + len(ROWS) * ROW_H + 26
L.append(f'<text x="{PAD}" y="{extra_y0}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#0f172a">{esc("本章 traces/matmul.json 实测:")}</text>')
for i, (k, v) in enumerate(EXTRA):
    y = extra_y0 + 22 + i * 22
    L.append(f'<text x="{PAD}" y="{y}" font-family="sans-serif" font-size="12" '
              f'fill="#475569">{esc(k)}: </text>')
    L.append(f'<text x="{PAD+text_w(k+": ", 12)}" y="{y}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#0f172a">{esc(v)}</text>')

foot_y = H - 16
CAPTION = ("一个整数字段 versionMajor 分派整代 Tensor Core:谓词 isVolta/isAmpere/isHopper 据它选布局分支;"
           "instrShape 携带 MMA 指令的 M x N。为什么布局长这样=硬件 MMA 输出寄存器排布反推,深化留后续章节。")
cap_w = text_w(CAPTION, 12)
if cap_w + PAD * 2 > W:
    mid = len(CAPTION) // 2
    split = CAPTION.rfind(";", 0, mid + 20)
    if split == -1:
        split = mid
    lines = [CAPTION[:split+1], CAPTION[split+1:]]
else:
    lines = [CAPTION]
for i, line in enumerate(lines):
    L.append(f'<text x="{PAD}" y="{foot_y - (len(lines)-1-i)*18}" font-family="sans-serif" '
              f'font-size="12" fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-nvidia-mma-version.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
