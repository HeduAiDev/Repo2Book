#!/usr/bin/env python3
"""fig-dot-operand-encoding (layout 模板)
A/B 换成 DotOperandEncodingAttr{opIdx, parent, kWidth}——opIdx 区分左右操作数,
parent 都指向刚造的 mma 编码,kWidth 取回溯链最低位宽(f16 上溯到 f16 时=16)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD, TOP = 50, 150
w = 980
h = 640

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#6366f1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="44" font-family="sans-serif" font-size="18.5" '
          f'font-weight="bold" fill="#0f172a">{esc("DotOperandEncodingAttr:A/B 共享 parent,靠 opIdx 分左右")}</text>')
L.append(f'<text x="{PAD}" y="68" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("kWidth 不取 A/B 自身位宽,取回溯 shape-preserving 一元链后的最低位宽(computeOrigBitWidth,AccelerateMatmul.cpp:L173-L212)")}</text>')

CARD_W, CARD_H = 300, 190
A_X = PAD
B_X = PAD + CARD_W + 260
CARD_Y = TOP

def card(x, y, name, opidx, opidx_line, kwidth_line):
    L.append(f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="10" '
              'fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
    L.append(f'<text x="{x+CARD_W/2}" y="{y+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#312e81">{esc(name)}</text>')
    fields = [
        ("opIdx", str(opidx), opidx_line),
        ("parent", "mma 编码(newRetType)", None),
        ("kWidth", "16", kwidth_line),
    ]
    fy = y + 56
    for label, val, anchor in fields:
        L.append(f'<text x="{x+18}" y="{fy}" font-family="sans-serif" font-size="12.5" '
                  f'font-weight="bold" fill="#3730a3">{esc(label)}</text>')
        L.append(f'<text x="{x+CARD_W-18}" y="{fy}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="12.5" fill="#1e1b4b">{esc(val)}</text>')
        if anchor:
            L.append(f'<text x="{x+18}" y="{fy+16}" font-family="sans-serif" font-size="10" '
                      f'fill="#818cf8">{esc(anchor)}</text>')
            fy += 40
        else:
            fy += 24

card(A_X, CARD_Y, "A 操作数编码", 0, "AccelerateMatmul.cpp:L326-L328", "L193-L212 computeOrigBitWidth")
card(B_X, CARD_Y, "B 操作数编码", 1, "AccelerateMatmul.cpp:L335-L337", "L193-L212 computeOrigBitWidth")

# parent 共同指向中间 mma 编码框(下方居中)
MMA_W, MMA_H = 420, 70
mma_x = (A_X + CARD_W + B_X) / 2 - MMA_W / 2
mma_y = CARD_Y + CARD_H + 70
L.append(f'<rect x="{mma_x}" y="{mma_y}" width="{MMA_W}" height="{MMA_H}" rx="8" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{mma_x+MMA_W/2}" y="{mma_y+28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#78350f">{esc("NvidiaMmaEncodingAttr")}</text>')
L.append(f'<text x="{mma_x+MMA_W/2}" y="{mma_y+50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#92400e">{esc("newRetType 的 mma 编码(本章第①张图的重写结果)")}</text>')

for x0 in (A_X + CARD_W / 2, B_X + CARD_W / 2):
    L.append(f'<line x1="{x0}" y1="{CARD_Y+CARD_H}" x2="{mma_x+MMA_W/2}" y2="{mma_y}" '
              'stroke="#6366f1" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(A_X+CARD_W/2+mma_x+MMA_W/2)/2-30}" y="{(CARD_Y+CARD_H+mma_y)/2}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="11" '
          f'fill="#4f46e5">{esc("parent")}</text>')
L.append(f'<text x="{(B_X+CARD_W/2+mma_x+MMA_W/2)/2+30}" y="{(CARD_Y+CARD_H+mma_y)/2}" '
          f'text-anchor="middle" font-family="sans-serif" font-size="11" '
          f'fill="#4f46e5">{esc("parent")}</text>')

foot_y = mma_y + MMA_H + 46
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="90" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{foot_y+24}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("kWidth 语义:回溯 A/B 的 shape-preserving 一元链(如 cast),取链上最低位宽——")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+44}" font-family="sans-serif" font-size="12.5" '
          f'fill="#0f172a">{esc("本例 f16 上溯仍是 f16,kWidth=16;upcast 场景(load f16 算 f32)按更低精度粒度分发才对齐 MMA 每线程装载。")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+68}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc("opIdx=0/1 只区分左右操作数,不影响 parent/kWidth 的取值方式——两者共享同一套推导逻辑。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-dot-operand-encoding.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
