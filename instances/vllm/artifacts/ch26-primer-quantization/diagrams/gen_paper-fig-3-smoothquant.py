#!/usr/bin/env python3
"""paper-fig-3-smoothquant: 重绘自 arXiv:2211.10438 Figure 3 —— per-tensor / per-token /
per-channel 三种量化粒度的定义图解。
(a) per-tensor：X 全矩阵共用一个标量 scale Δ_X^[1]；W 全矩阵共用一个标量 Δ_W^[1]。
(b) per-token + per-channel：X 按 token 维 T 每行一个 scale（形状 [T×1]，贴在 X 左侧的
    竖直条带上）；W 按输出通道维 C_o 每列一个 scale（形状 [1×C_o]，贴在 W 顶部的水平
    条带上）；输入通道维 C_i 不能拆——矩阵乘法的收缩维必须保持整体一致。
这是定义性图解，本身不含数值——scale 条带用渐变色示意"逐行/逐列各自一把尺子"。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def esc_bold(s):
    """转义并在粗体文本里把"量"字拆到 font-weight=normal 的 tspan——
    这套渲染管线(rsvg-convert)的粗体 CJK 回退字体缺"量"字形,粗体直出会变豆腐块。"""
    return '<tspan font-weight="normal">量</tspan>'.join(esc(p) for p in s.split('量'))


TITLE = "量化粒度定义：per-tensor 共用一把尺子，per-token / per-channel 各自一把尺子"
SUBTITLE = "重绘自 arXiv:2211.10438 Figure 3"

X_W, X_H = 130, 86     # X: T(行) x C_i(列)
W_W, W_H = 130, 86     # W: C_i(行) x C_o(列)
BAR = 14               # scale 条带宽度
GREEN, GREEN_STROKE = "#bbf7d0", "#16a34a"
TAN, TAN_STROKE = "#fde3b8", "#c2820f"
DASH_RED = "#dc2626"

PAD = 40
COL_GAP = 110          # X 右边到 W 左边的水平间距（容纳 * 与两侧行标签，互不相碰）
RIGHT_MARGIN = 210     # W 右侧给 per-channel 侧注留白
x_x = PAD + 34
w_x = x_x + X_W + COL_GAP
star_x = x_x + X_W + 22
w = w_x + W_W + RIGHT_MARGIN

TOP = 108
ROW_A_Y = TOP + 46          # 第一行矩阵顶部
ROW_A_CAP_Y = ROW_A_Y + X_H + 34 + 8 + 30   # (a) 结论行
ROW_B_Y = ROW_A_CAP_Y + 60  # 第二行矩阵顶部（留够上方 per-token/per-channel 侧注空间）
ROW_B_CAP_Y = ROW_B_Y + X_H + 46
FOOT_Y = ROW_B_CAP_Y + 30
h = FOOT_Y + 18 + 26


def esc_defs():
    return (
        '<defs>'
        '<linearGradient id="gradX" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#bbf7d0"/><stop offset="1" stop-color="#15803d"/>'
        '</linearGradient>'
        '<linearGradient id="gradW" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#fde3b8"/><stop offset="1" stop-color="#c2820f"/>'
        '</linearGradient>'
        '</defs>'
    )


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', esc_defs(),
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+2}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# ================= row (a): per-tensor =================
y = ROW_A_Y
L.append(f'<rect x="{x_x}" y="{y}" width="{X_W}" height="{X_H}" fill="{GREEN}" stroke="{GREEN_STROKE}" stroke-width="1.3"/>')
L.append(f'<text x="{x_x+X_W/2}" y="{y-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">C_i</text>')
L.append(f'<text x="{x_x-10}" y="{y+X_H/2+4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">T</text>')
L.append(f'<text x="{x_x+X_W/2}" y="{y+X_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#166534">X</text>')
sw_x, sw_y = x_x - 26, y - 26
L.append(f'<rect x="{sw_x}" y="{sw_y}" width="14" height="14" fill="{GREEN_STROKE}" stroke="#0f172a" stroke-width="0.8"/>')
L.append(f'<text x="{sw_x+18}" y="{sw_y+11}" font-family="sans-serif" font-size="10" '
         f'fill="#334155">&#916;_X^[1]</text>')
bx0, by0 = x_x - 34, y - 34
L.append(f'<rect x="{bx0}" y="{by0}" width="{X_W+34+8}" height="{X_H+34+8}" fill="none" '
         f'stroke="{DASH_RED}" stroke-width="1.3" stroke-dasharray="5,3"/>')
L.append(f'<text x="{bx0}" y="{by0-6}" font-family="sans-serif" font-size="10.5" '
         f'fill="{DASH_RED}">per-tensor quant.</text>')

L.append(f'<text x="{star_x}" y="{y+X_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="#0f172a">*</text>')

L.append(f'<rect x="{w_x}" y="{y}" width="{W_W}" height="{W_H}" fill="{TAN}" stroke="{TAN_STROKE}" stroke-width="1.3"/>')
L.append(f'<text x="{w_x+W_W/2}" y="{y-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">C_o</text>')
L.append(f'<text x="{w_x-10}" y="{y+W_H/2+4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">C_i</text>')
L.append(f'<text x="{w_x+W_W/2}" y="{y+W_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#92400e">W</text>')
sw_x2, sw_y2 = w_x - 26, y - 26
L.append(f'<rect x="{sw_x2}" y="{sw_y2}" width="14" height="14" fill="{TAN_STROKE}" stroke="#0f172a" stroke-width="0.8"/>')
L.append(f'<text x="{sw_x2+18}" y="{sw_y2+11}" font-family="sans-serif" font-size="10" '
         f'fill="#334155">&#916;_W^[1]</text>')
bx1, by1 = w_x - 34, y - 34
L.append(f'<rect x="{bx1}" y="{by1}" width="{W_W+34+8}" height="{W_H+34+8}" fill="none" '
         f'stroke="{DASH_RED}" stroke-width="1.3" stroke-dasharray="5,3"/>')
L.append(f'<text x="{bx1}" y="{by1-6}" font-family="sans-serif" font-size="10.5" '
         f'fill="{DASH_RED}">per-tensor quant.</text>')

L.append(f'<text x="{PAD}" y="{ROW_A_CAP_Y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">(a) per-tensor {esc_bold("量化：全矩阵共用一个标量 scale")}</text>')

# ================= row (b): per-token + per-channel =================
y2 = ROW_B_Y
L.append(f'<rect x="{x_x}" y="{y2}" width="{X_W}" height="{X_H}" fill="{GREEN}" stroke="{GREEN_STROKE}" stroke-width="1.3"/>')
L.append(f'<text x="{x_x+X_W/2}" y="{y2-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">C_i</text>')
L.append(f'<text x="{x_x+X_W/2}" y="{y2+X_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#166534">X</text>')
bar_x = x_x - BAR - 4
L.append(f'<rect x="{bar_x}" y="{y2}" width="{BAR}" height="{X_H}" fill="url(#gradX)" '
         f'stroke="#0f172a" stroke-width="0.8"/>')
L.append(f'<text x="{bar_x-6}" y="{y2+X_H/2+4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="10.5" fill="#0f172a">T</text>')
tb0x, tb0y = bar_x - 6, y2 - 6
L.append(f'<rect x="{tb0x}" y="{tb0y}" width="{BAR+12}" height="{X_H+12}" fill="none" '
         f'stroke="{DASH_RED}" stroke-width="1.3" stroke-dasharray="5,3"/>')
L.append(f'<text x="{x_x-44}" y="{y2+X_H+26}" font-family="sans-serif" font-size="10.5" '
         f'fill="{DASH_RED}">per-token quant.</text>')
L.append(f'<text x="{x_x-44}" y="{y2-16}" font-family="sans-serif" font-size="9.5" '
         f'fill="#334155">&#916;_X^[T&#215;1]</text>')

L.append(f'<text x="{star_x}" y="{y2+X_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="#0f172a">*</text>')

L.append(f'<rect x="{w_x}" y="{y2}" width="{W_W}" height="{W_H}" fill="{TAN}" stroke="{TAN_STROKE}" stroke-width="1.3"/>')
L.append(f'<text x="{w_x-10}" y="{y2+W_H/2+4}" text-anchor="end" font-family="sans-serif" '
         f'font-size="11" fill="#0f172a">C_i</text>')
L.append(f'<text x="{w_x+W_W/2}" y="{y2+W_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#92400e">W</text>')
bar_y = y2 - BAR - 4
L.append(f'<rect x="{w_x}" y="{bar_y}" width="{W_W}" height="{BAR}" fill="url(#gradW)" '
         f'stroke="#0f172a" stroke-width="0.8"/>')
cb0x, cb0y = w_x - 6, bar_y - 6
L.append(f'<rect x="{cb0x}" y="{cb0y}" width="{W_W+12}" height="{BAR+12}" fill="none" '
         f'stroke="{DASH_RED}" stroke-width="1.3" stroke-dasharray="5,3"/>')
L.append(f'<text x="{w_x}" y="{bar_y-14}" font-family="sans-serif" font-size="10.5" '
         f'fill="{DASH_RED}">per-channel quant.</text>')
L.append(f'<text x="{w_x+W_W+10}" y="{y2+W_H/2-4}" font-family="sans-serif" font-size="9.5" '
         f'fill="#334155">&#916;_W^[1&#215;C_o]</text>')
L.append(f'<text x="{w_x+W_W+10}" y="{y2+W_H/2+9}" font-family="sans-serif" font-size="9.5" '
         f'fill="#334155">（C_o 逐通道各一把尺）</text>')

L.append(f'<text x="{PAD}" y="{ROW_B_CAP_Y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">(b) per-token + per-channel：X 逐 token（T）、'
         f'W 逐输出通道（C_o）各配一把尺子</text>')

FOOT_LINES = [
    "输入通道维 C_i 是矩阵乘法的收缩维，两种粒度都不能沿它拆——",
    "只有 T（token）和 C_o（输出通道）这两个「外维」能各自配一把尺子。",
]
for i, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{FOOT_Y+i*18}" font-family="sans-serif" font-size="11.5" '
             f'fill="#64748b">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-3-smoothquant.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
