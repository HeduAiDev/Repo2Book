#!/usr/bin/env python3
"""fig31-4-decoupled-rope: before-after——左(错误路线)直接对 k^C 加 RoPE,中间矩阵
M(delta) 随相对位置变(-0.5378 -> 0.6912),破坏吸收;右(解耦)位置只走独立 q_pe/k_pe,
c_kv 主体 W~ 保持静态(-0.5378 恒定)从而可吸收。数字全部来自 traces/rope.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PAD, TOP, PANEL_W = 40, 96, 360
BOX_W, BOX_H, VGAP = 320, 50, 26
w = PAD * 2 + PANEL_W * 2 + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 640">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="640" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("为何需要解耦 RoPE:旋转矩阵夹在中间,权重吸收就此失效")}</text>']

def panel_title(px, cx, title, color):
    L.append(f'<rect x="{px}" y="{TOP-46}" width="{PANEL_W}" height="30" rx="6" fill="{color}"/>')
    L.append(f'<text x="{cx}" y="{TOP-25}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="white">{esc(title)}</text>')

def step_box(cx, y, text, hl=False, bad=False):
    fill = "#fee2e2" if bad else ("#dcfce7" if hl else "#e2e8f0")
    stroke = "#b91c1c" if bad else ("#15803d" if hl else "#64748b")
    L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{2.2 if (hl or bad) else 1}"/>')
    color = "#991b1b" if bad else ("#15803d" if hl else "#0f172a")
    weight = 'font-weight="bold" ' if (hl or bad) else ''
    lines = text.split("\n")
    y0 = y + BOX_H/2 - (len(lines)-1)*8 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{y0+k*16}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12.5" {weight}fill="{color}">{esc(ln)}</text>')

def arrow(cx, y1, y2):
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#64748b" '
             'stroke-width="1.5" marker-end="url(#a)"/>')

# LEFT panel: direct RoPE on k^C (bad)
px1 = PAD
cx1 = px1 + PANEL_W/2
panel_title(px1, cx1, "直接对 k^C 加 RoPE(破坏吸收)", "#b91c1c")
left_steps = [
    "q_t^T k_j^rope = h_t^T(W^Q)^T R_(j-t) W_UK c_j",
    "旋转矩阵 R_(j-t) 夹在 (W^Q)^T 与 W_UK 之间",
    "M(δ)随δ变:M(0)[0,0]=-0.5378\nM(3)[0,0]=0.6912",
]
y = TOP
for i, s in enumerate(left_steps):
    step_box(cx1, y, s, bad=(i == len(left_steps)-1))
    if i < len(left_steps)-1:
        arrow(cx1, y+BOX_H, y+BOX_H+VGAP-4)
    y += BOX_H + VGAP

# RIGHT panel: decoupled RoPE (good)
px2 = PAD + PANEL_W + 90
cx2 = px2 + PANEL_W/2
panel_title(px2, cx2, "解耦 RoPE(保住吸收)", "#15803d")
right_steps = [
    "位置只走独立 q_pe/k_pe(d_h_r=2 维)",
    "c_kv 主体不加 RoPE,保持位置无关",
    "W̃=(W_UK)^T W_UQ 静态:W̃[0,0]=-0.5378 恒定",
]
y = TOP
for i, s in enumerate(right_steps):
    step_box(cx2, y, s, hl=(i == len(right_steps)-1))
    if i < len(right_steps)-1:
        arrow(cx2, y+BOX_H, y+BOX_H+VGAP-4)
    y += BOX_H + VGAP

# middle connector
midy = TOP + (len(left_steps)*(BOX_H+VGAP) - VGAP)/2
L.append(f'<text x="{(px1+PANEL_W+px2)/2}" y="{midy-14}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#0f172a" font-weight="bold">{esc("对比")}</text>')
L.append(f'<line x1="{px1+PANEL_W+8}" y1="{midy}" x2="{px2-8}" y2="{midy}" '
         'stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#a)"/>')

# bottom: middle-matrix growth strip (delta 0..3, values from trace) + e2e callout
strip_top = y + 30
L.append(f'<text x="{PAD}" y="{strip_top}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">{esc("中间矩阵 M(δ)[0,0] 随相对位置 δ 的变化(左路线,若真的这样做)")}</text>')
deltas = [(0, -0.5378), (1, -0.4659), (2, 0.125), (3, 0.6912)]
bar_top = strip_top + 20
bar_h_scale = 70
lo = min(v for _, v in deltas)
span = max(v for _, v in deltas) - lo
bx = PAD
bar_w = 150
for d, v in deltas:
    frac = (v - lo) / span
    bh = 10 + frac * bar_h_scale
    by = bar_top + bar_h_scale - bh
    color = "#b91c1c" if d > 0 else "#15803d"
    L.append(f'<rect x="{bx}" y="{by}" width="{bar_w}" height="{bh}" rx="4" '
             f'fill="{color}" fill-opacity="0.75" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{bx+bar_w/2}" y="{by-8}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{color}">{v}</text>')
    L.append(f'<text x="{bx+bar_w/2}" y="{bar_top+bar_h_scale+20}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#64748b">{esc(f"δ={d}")}</text>')
    bx += bar_w + 30

call_top = bar_top + bar_h_scale + 46
call_w = w - 2*PAD
L.append(f'<rect x="{PAD}" y="{call_top}" width="{call_w}" height="56" rx="10" '
         'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{PAD+call_w/2}" y="{call_top+34}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#92400e">'
         f'{esc("端到端验证:解耦 RoPE 下 decode 增量计算 vs prefill 一次性计算,3 步最大绝对差 = 0.0")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig31-4-decoupled-rope.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
