#!/usr/bin/env python3
"""layout 模板:DotOperandEncoding 的 kWidth 与 getSizePerThreadForOperand。
顶部:kWidth = 32/bitwidth 的算式卡片。下方两个小网格:a(opIdx=0,M x K)每线程尺寸[2,4],
b(opIdx=1,K x N)每线程尺寸[4,1],高亮沿 K 方向的连续 4 个元素(kWidth=2 的来历)。
全坐标计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    cjk = sum(1 for ch in s if ord(ch) > 0x2e7f)
    other = len(s) - cjk
    return cjk * size * 1.0 + other * size * 0.56

TITLE = "DotOperandEncoding:kWidth 由 dtype 定死,不容自选"
SUBTITLE = "Ampere fp16 matmul 操作数,每线程沿收缩维 K 恰好攥住一次 mma.16816 搬运量(Triton v3.2.0 实测)"

FORMULA = "kWidth = 32 / bitwidth = 32 / 16 = 2"
CELL = 42

# a: opIdx=0, (M,K) 每线程 = [2,4] -> 画 2 行(M) x 4 列(K)
A_ROWS, A_COLS = 2, 4
# b: opIdx=1, (K,N) 每线程 = [4,1] -> 画 4 行(K) x 1 列(N)
B_ROWS, B_COLS = 4, 1

PAD, TOP = 46, 150
GRID_GAP = 140

a_w, a_h = A_COLS * CELL, A_ROWS * CELL
b_w, b_h = B_COLS * CELL, B_ROWS * CELL

a_x0 = PAD
b_x0 = a_x0 + a_w + GRID_GAP

CAPTION_LINES = [
    "a(opIdx=0,M x K)每线程 [M=2,K=4]:沿收缩维 K 的 4 个 fp16 连续排布,恰好凑满一次 ldmatrix/mma.16816 搬运;",
    "b(opIdx=1,K x N)每线程 [K=4,N=1],同样沿 K 连续 4 个。kWidth 由 dtype 位宽定死,mma 布局深化留后续章节。",
]
NUM_CARDS = [("kWidth (fp16)", "2"), ("bitwidth", "16"),
             ("a 每线程 (M,K)", "[2, 4]"), ("b 每线程 (K,N)", "[4, 1]"),
             ("a 沿 K 连续元素", "4")]

card_w_est = max(text_w(l, 11) for l, _ in NUM_CARDS) + 20
W = int(max(PAD * 2 + text_w(TITLE, 17),
            PAD * 2 + text_w(SUBTITLE, 12),
            b_x0 + b_w + 260,
            PAD + max(text_w(s, 12) for s in CAPTION_LINES) + PAD,
            PAD + len(NUM_CARDS) * (card_w_est + 14)))

grid_bottom = TOP + max(a_h, b_h)
cards_y = grid_bottom + 70
foot_y0 = cards_y + 70

H = int(foot_y0 + len(CAPTION_LINES) * 18 + 24)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 算式卡片
formula_w = text_w(FORMULA, 15) + 40
L.append(f'<rect x="{PAD}" y="72" width="{formula_w}" height="40" rx="8" '
          'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+formula_w/2}" y="97" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#4338ca">{esc(FORMULA)}</text>')

# a 网格(M x K),高亮沿 K(列方向)4 个连续
L.append(f'<text x="{a_x0}" y="{TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1d4ed8">{esc("a  opIdx=0  (M, K)")}</text>')
for r in range(A_ROWS):
    for c in range(A_COLS):
        x = a_x0 + c * CELL
        y = TOP + r * CELL
        fill = "#bfdbfe" if r == 0 else "#dbeafe"
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" rx="3" '
                  f'fill="{fill}" stroke="#1d4ed8" stroke-width="1.3"/>')
L.append(f'<text x="{a_x0-8}" y="{TOP+CELL*A_ROWS+16}" font-family="sans-serif" font-size="11" '
          f'fill="#1d4ed8">{esc("M=2 (行)")}</text>')
L.append(f'<text x="{a_x0}" y="{TOP+CELL*A_ROWS+34}" font-family="sans-serif" font-size="11" '
          f'fill="#1d4ed8">{esc("K=4 (列,沿收缩维连续)")}</text>')
# K 方向括线(第一行下方标出 4 连续)
brace_y = TOP - 2
L.append(f'<line x1="{a_x0}" y1="{brace_y}" x2="{a_x0+A_COLS*CELL-3}" y2="{brace_y}" '
          'stroke="#1d4ed8" stroke-width="1.2"/>')

# b 网格(K x N)
L.append(f'<text x="{b_x0}" y="{TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#15803d">{esc("b  opIdx=1  (K, N)")}</text>')
for r in range(B_ROWS):
    for c in range(B_COLS):
        x = b_x0 + c * CELL
        y = TOP + r * CELL
        fill = "#bbf7d0" if c == 0 else "#dcfce7"
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" rx="3" '
                  f'fill="{fill}" stroke="#15803d" stroke-width="1.3"/>')
L.append(f'<text x="{b_x0}" y="{TOP+CELL*B_ROWS+16}" font-family="sans-serif" font-size="11" '
          f'fill="#15803d">{esc("K=4 (行,沿收缩维连续)")}</text>')
L.append(f'<text x="{b_x0}" y="{TOP+CELL*B_ROWS+34}" font-family="sans-serif" font-size="11" '
          f'fill="#15803d">{esc("N=1 (列)")}</text>')

# 数字卡片
card_w = (W - 2 * PAD) / len(NUM_CARDS)
for i, (label, val) in enumerate(NUM_CARDS):
    cx = PAD + i * card_w + card_w / 2
    L.append(f'<rect x="{PAD+i*card_w+4}" y="{cards_y}" width="{card_w-8}" height="50" rx="6" '
              f'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
    L.append(f'<text x="{cx}" y="{cards_y+19}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{esc(label)}</text>')
    L.append(f'<text x="{cx}" y="{cards_y+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(val)}</text>')

for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-dot-operand-sizes.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
