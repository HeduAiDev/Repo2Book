#!/usr/bin/env python3
"""fig-ch09-tiling-fixed-point — m16 tiling 是不动点变换。
before/after 对照:左边一个大 linalg.conv_1d_nwc_wcf,右边 scf.for 包住同名同色的小方块。
重绘自 arXiv:2202.03293 Fig.4(语义示意,不照抄整页 pseudo-IR)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

OPNAME = "linalg.conv_1d_nwc_wcf"
CAPTION = "重绘自 arXiv:2202.03293 Fig.4:tiling 引入 scf.for 与 extract_slice/insert_slice,循环体里仍是同一个 linalg.conv_1d_nwc_wcf"

PAD = 40
W_LEFT, W_RIGHT = 330, 430
GAP = 90
TOP = 100
BOX_H = 100
w = PAD * 2 + W_LEFT + GAP + W_RIGHT
h = TOP + BOX_H + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc("tiling 是不动点变换:切前切后,循环体里的算子没变")}</text>')

# ---------------- left: before ----------------
lx = PAD
cx_l = lx + W_LEFT / 2
L.append(f'<text x="{cx_l}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#334155">{esc("切之前")}</text>')
L.append(f'<rect x="{lx}" y="{TOP}" width="{W_LEFT}" height="{BOX_H}" rx="8" '
         f'fill="#3b82f6" stroke="#1e3a8a" stroke-width="2.2"/>')
L.append(f'<text x="{cx_l}" y="{TOP+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="white">{esc(OPNAME)}</text>')
# 三条进出边标注
edges_l = [("ins: I 1x8x2", -1), ("ins: K 3x2x3", -1), ("outs: O 1x6x3", 1)]
ey = TOP - 38
L.append(f'<text x="{cx_l}" y="{ey}" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
         f'fill="#475569">{esc("ins(I:1x8x2, K:3x2x3)  outs(O:1x6x3)")}</text>')
L.append(f'<line x1="{cx_l}" y1="{TOP+BOX_H}" x2="{cx_l}" y2="{TOP+BOX_H+34}" '
         f'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
L.append(f'<text x="{cx_l}" y="{TOP+BOX_H+52}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#334155">{esc("一次算完整块,只有 1 种局部输出类型:1x6x3")}</text>')

# ---------------- right: after ----------------
rx = lx + W_LEFT + GAP
cx_r = rx + W_RIGHT / 2
L.append(f'<text x="{cx_r}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#334155">{esc("切之后(tile 宽 4)")}</text>')
# scf.for 外框
loop_pad = 26
L.append(f'<rect x="{rx}" y="{TOP-4}" width="{W_RIGHT}" height="{BOX_H+8}" rx="10" '
         f'fill="#f8fafc" stroke="#0f172a" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{rx+10}" y="{TOP+16}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#0f172a">{esc("scf.for w = 0 to 6 step 4")}</text>')
inner_w, inner_h = W_RIGHT - 2*loop_pad, BOX_H - 40
ix = rx + loop_pad
iy = TOP + 24
L.append(f'<rect x="{ix}" y="{iy}" width="{inner_w}" height="{inner_h}" rx="8" '
         f'fill="#3b82f6" stroke="#1e3a8a" stroke-width="2.2"/>')
L.append(f'<text x="{ix+inner_w/2}" y="{iy+inner_h/2+5}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" font-weight="bold" '
         f'fill="white">{esc(OPNAME)}</text>')
L.append(f'<text x="{cx_r}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" font-size="11" '
         f'fill="#475569">{esc("extract_slice(I,K) 局部形状随迭代变  /  insert_slice(O)")}</text>')

L.append(f'<line x1="{cx_r}" y1="{TOP+BOX_H+4}" x2="{cx_r}" y2="{TOP+BOX_H+34}" '
         f'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
L.append(f'<text x="{cx_r}" y="{TOP+BOX_H+52}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#334155">{esc("2 次迭代,2 种局部输出类型:")}</text>')
L.append(f'<text x="{cx_r}" y="{TOP+BOX_H+70}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#b45309">{esc("满块 1x4x3  与  边界块 1x2x3")}</text>')
L.append(f'<text x="{cx_r}" y="{TOP+BOX_H+90}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("没有哪个静态类型对两次迭代都合法")}</text>')

# 中间大箭头
mid_y = TOP + BOX_H / 2
L.append(f'<line x1="{lx+W_LEFT+10}" y1="{mid_y}" x2="{rx-10}" y2="{mid_y}" '
         f'stroke="#b45309" stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{lx+W_LEFT+GAP/2}" y="{mid_y-12}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
         f'fill="#b45309">{esc("tiling")}</text>')

# 底部:切前切后共同不变量
foot1_y = h - 74
L.append(f'<rect x="{PAD}" y="{foot1_y-24}" width="{w-2*PAD}" height="36" rx="6" '
         f'fill="#ecfdf5" stroke="#15803d" stroke-width="1.3"/>')
L.append(f'<text x="{w/2}" y="{foot1_y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#14532d">'
         f'{esc("迭代器类型两侧相同:parallel x 3 + reduction x 2;与不切时结果的最大偏差 = 0 —— 变的只是操作数形状,算子本身不变")}</text>')

foot2_y = h - 20
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc(CAPTION)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-tiling-fixed-point.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
