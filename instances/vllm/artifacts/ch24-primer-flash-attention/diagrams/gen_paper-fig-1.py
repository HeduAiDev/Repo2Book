#!/usr/bin/env python3
"""论文精髓图重绘:arXiv:2205.14135 Fig.1 —
左:FlashAttention 用 tiling 避免把 N×N 注意力矩阵物化到 HBM(外层红箭头遍历 K,V 块、
内层蓝箭头遍历 Q 块,搬进 SRAM 算);右:GPT-2 上相对 PyTorch 标准实现的 7.6× 加速柱状图。
忠实重绘自 ar5iv 抓到的原图(assets/x1.png),信息结构对齐,配色套本书视觉语言,文字译中。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

FIG_ID = "paper-fig-1"
TITLE = "重绘自 arXiv:2205.14135 Fig.1"
N_BLOCKS = 6

# ---------- 版式常量 ----------
PAD, TOP = 40, 108
BLK, GAP_S = 34, 3           # 单个绿色块的宽/高、块间距
COPY, DASH = 34, 4           # 橙色 Copy 方块边长
PANEL_GAP = 70

# 左面板(tiling 图)几何
row_len = N_BLOCKS * BLK + (N_BLOCKS - 1) * GAP_S
q_x = PAD + 70
q_y = TOP + 60
k_x = q_x + COPY + 90
k_y = TOP
v_x = k_x + row_len + COPY + 90
v_y = q_y
center_x = k_x + row_len / 2
out_y = q_y + row_len + 70

left_w = (v_x + BLK) - PAD + 30
left_h = out_y + BLK + 70 - TOP + PAD

# 右面板(柱状图)几何
chart_x0 = PAD + left_w + PANEL_GAP
chart_w, chart_h = 320, 300
chart_y0 = TOP + 100
axis_top_ms = 18.0  # y 轴顶部对应的 ms 值(略高于 17,留白)

w = chart_x0 + chart_w + 170 + PAD
h = TOP + max(left_h, chart_h + 120) + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="red" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
          '<marker id="blue" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
          '<marker id="gray" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="purple" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
SUBTITLE = "FlashAttention 用分块(tiling)避免把 N×N 注意力矩阵物化到 HBM,换来 GPT-2 上 7.6× 实测加速"
L.append(f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')


def row_blocks(xs_start, y, horizontal=True, count=N_BLOCKS):
    out = []
    for i in range(count):
        if horizontal:
            bx, by = xs_start + i * (BLK + GAP_S), y
        else:
            bx, by = xs_start, y + i * (BLK + GAP_S)
        out.append((bx, by))
    return out


# ---------------- 左面板:tiling 示意图 ----------------
L.append(f'<text x="{k_x + row_len/2}" y="{k_y-34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#dc2626">{esc("外层循环(Outer Loop)")}</text>')
L.append(f'<line x1="{k_x}" y1="{k_y-20}" x2="{k_x+row_len}" y2="{k_y-20}" '
          'stroke="#dc2626" stroke-width="2.4" marker-end="url(#red)"/>')
L.append(f'<text x="{k_x-14}" y="{k_y+BLK/2+4}" text-anchor="end" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("Kᵀ: d×N")}</text>')
for bx, by in row_blocks(k_x, k_y):
    L.append(f'<rect x="{bx}" y="{by}" width="{BLK}" height="{BLK}" rx="3" '
              'fill="#86efac" stroke="#166534" stroke-width="1.6"/>')

# Q 列(左,蓝色内层循环)
L.append(f'<text x="{q_x+BLK/2}" y="{q_y-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("Q: N×d")}</text>')
for bx, by in row_blocks(q_x, q_y, horizontal=False):
    L.append(f'<rect x="{bx}" y="{by}" width="{BLK}" height="{BLK}" rx="3" '
              'fill="#86efac" stroke="#166534" stroke-width="1.6"/>')
L.append(f'<line x1="{q_x-22}" y1="{q_y}" x2="{q_x-22}" y2="{q_y+row_len}" '
          'stroke="#2563eb" stroke-width="2.4" marker-end="url(#blue)"/>')
L.append(f'<text x="{q_x-32}" y="{q_y+row_len/2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#2563eb" '
          f'transform="rotate(-90 {q_x-32} {q_y+row_len/2})">{esc("内层循环(Inner Loop)")}</text>')

# V 列(右,红色外层循环)
L.append(f'<text x="{v_x+BLK/2}" y="{v_y-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("V: N×d")}</text>')
for bx, by in row_blocks(v_x, v_y, horizontal=False):
    L.append(f'<rect x="{bx}" y="{by}" width="{BLK}" height="{BLK}" rx="3" '
              'fill="#86efac" stroke="#166534" stroke-width="1.6"/>')
L.append(f'<line x1="{v_x+BLK+22}" y1="{v_y}" x2="{v_x+BLK+22}" y2="{v_y+row_len}" '
          'stroke="#dc2626" stroke-width="2.4" marker-end="url(#red)"/>')
L.append(f'<text x="{v_x+BLK+32}" y="{v_y+row_len/2}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#dc2626" '
          f'transform="rotate(-90 {v_x+BLK+32} {v_y+row_len/2})">{esc("外层循环(Outer Loop)")}</text>')

# 中间:QKᵀ 虚线框(不物化)+ 三个橙色 Copy 方块 + 紫色 compute
box_x, box_y = q_x + BLK + 40, q_y - 4
box_w, box_h = (v_x - 20) - box_x, row_len + 8
L.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="6" '
          'fill="none" stroke="#334155" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{box_x+box_w/2}" y="{box_y-8}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#334155">{esc("QKᵀ: N×N（虚框=不写回 HBM）")}</text>')

copy_k = (k_x + row_len/2 - COPY/2, k_y + BLK + 24)
copy_q = (q_x + BLK + 8, q_y + row_len/2 - COPY/2)
copy_v = (v_x - COPY - 8, v_y + row_len/2 - COPY/2)
compute = (box_x + box_w/2 - COPY/2, box_y + box_h/2 - COPY/2)

for (cx, cy), label in [(copy_k, "Copy"), (copy_q, "Copy"), (copy_v, "Copy")]:
    L.append(f'<rect x="{cx}" y="{cy}" width="{COPY}" height="{COPY}" rx="3" '
              'fill="#fdba74" stroke="#c2410c" stroke-width="1.6"/>')
    L.append(f'<text x="{cx+COPY/2}" y="{cy+COPY/2+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9.5" fill="#7c2d12">{esc(label)}</text>')
L.append(f'<rect x="{compute[0]}" y="{compute[1]}" width="{COPY}" height="{COPY}" rx="3" '
          'fill="#fdba74" stroke="#c2410c" stroke-width="1.6"/>')
L.append(f'<text x="{compute[0]+COPY/2}" y="{compute[1]+COPY/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="9.5" fill="#7c2d12">{esc("Copy")}</text>')

# K -> copy_k(黑色实线),Q -> copy_q,V -> copy_v
L.append(f'<line x1="{k_x+row_len/2}" y1="{k_y+BLK}" x2="{copy_k[0]+COPY/2}" y2="{copy_k[1]}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#gray)"/>')
L.append(f'<line x1="{q_x+BLK}" y1="{q_y+row_len/2}" x2="{copy_q[0]}" y2="{copy_q[1]+COPY/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#gray)"/>')
L.append(f'<line x1="{v_x}" y1="{v_y+row_len/2}" x2="{copy_v[0]+COPY}" y2="{copy_v[1]+COPY/2}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#gray)"/>')
# copy_* -> compute(紫色虚线,"在 SRAM 算";端点落在 compute 方框边缘,不穿过框内文字)
compute_edges = {
    "top": (compute[0] + COPY / 2, compute[1]),
    "left": (compute[0], compute[1] + COPY / 2),
    "right": (compute[0] + COPY, compute[1] + COPY / 2),
}
for (cx, cy), edge_key in ((copy_k, "top"), (copy_q, "left"), (copy_v, "right")):
    ex, ey = compute_edges[edge_key]
    L.append(f'<line x1="{cx+COPY/2}" y1="{cy+COPY/2}" x2="{ex}" y2="{ey}" '
              'stroke="#7c3aed" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#purple)"/>')
L.append(f'<text x="{compute[0]+COPY/2+10}" y="{compute[1]+COPY+16}" text-anchor="start" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#7c3aed">{esc("在 SRAM 上算完这块")}</text>')

# 输出行(底部)
out_row_x = k_x
L.append(f'<line x1="{compute[0]+COPY/2}" y1="{compute[1]+COPY}" x2="{compute[0]+COPY/2}" y2="{out_y-14}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#gray)"/>')
L.append(f'<text x="{compute[0]+COPY/2+8}" y="{(compute[1]+COPY+out_y)/2}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("写回 HBM")}</text>')
L.append(f'<text x="{out_row_x-14}" y="{out_y+BLK/2+4}" text-anchor="end" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc("sm(QKᵀ)V: N×d")}</text>')
for bx, by in row_blocks(out_row_x, out_y):
    L.append(f'<rect x="{bx}" y="{by}" width="{BLK}" height="{BLK}" rx="3" '
              'fill="#86efac" stroke="#166534" stroke-width="1.6"/>')
L.append(f'<line x1="{out_row_x}" y1="{out_y+BLK+20}" x2="{out_row_x+row_len}" y2="{out_y+BLK+20}" '
          'stroke="#2563eb" stroke-width="2.4" marker-end="url(#blue)"/>')
L.append(f'<text x="{out_row_x+row_len/2}" y="{out_y+BLK+38}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#2563eb">{esc("内层循环(Inner Loop)")}</text>')
L.append(f'<text x="{center_x}" y="{out_y+BLK+64}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">{esc("FlashAttention")}</text>')

# ---------------- 右面板:GPT-2 耗时柱状图 ----------------
L.append(f'<text x="{chart_x0+chart_w/2}" y="{chart_y0-76}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" '
          f'fill="#0f172a">{esc("Attention on GPT-2")}</text>')

axis_x, axis_y_bottom = chart_x0 + 50, chart_y0 + chart_h
axis_y_top = chart_y0
L.append(f'<line x1="{axis_x}" y1="{axis_y_top}" x2="{axis_x}" y2="{axis_y_bottom}" '
          'stroke="#0f172a" stroke-width="1.6"/>')
L.append(f'<line x1="{axis_x}" y1="{axis_y_bottom}" x2="{axis_x+220}" y2="{axis_y_bottom}" '
          'stroke="#0f172a" stroke-width="1.6"/>')
for tick_ms in (0, 5, 10, 15):
    ty = axis_y_bottom - tick_ms / axis_top_ms * chart_h
    L.append(f'<line x1="{axis_x-5}" y1="{ty}" x2="{axis_x}" y2="{ty}" stroke="#0f172a" stroke-width="1.2"/>')
    L.append(f'<text x="{axis_x-10}" y="{ty+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{tick_ms}</text>')
L.append(f'<text x="{axis_x-38}" y="{(axis_y_top+axis_y_bottom)/2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#334155" '
          f'transform="rotate(-90 {axis_x-38} {(axis_y_top+axis_y_bottom)/2})">{esc("Time (ms)")}</text>')

# PyTorch 堆叠柱(自底向上,读图近似值,ms):Matmul 2.2 / Mask 4.4 / Softmax 3.6 / Dropout 4.6 / Matmul 2.2
PT_SEGMENTS = [("Matmul", 2.2, "#93c5fd"), ("Mask", 4.4, "#60a5fa"), ("Softmax", 3.6, "#3b82f6"),
               ("Dropout", 4.6, "#2563eb"), ("Matmul", 2.2, "#1d4ed8")]
bar_w = 70
pt_x = axis_x + 40
cum = 0.0
for name, val, color in PT_SEGMENTS:
    y_bot = axis_y_bottom - cum / axis_top_ms * chart_h
    y_top = axis_y_bottom - (cum + val) / axis_top_ms * chart_h
    L.append(f'<rect x="{pt_x}" y="{y_top}" width="{bar_w}" height="{y_bot-y_top}" '
              f'fill="{color}" stroke="#1e3a5f" stroke-width="1.2"/>')
    L.append(f'<text x="{pt_x+bar_w+8}" y="{(y_top+y_bot)/2+4}" font-family="sans-serif" '
              f'font-size="10" fill="#1e3a5f">{esc(name)}</text>')
    cum += val
L.append(f'<text x="{pt_x+bar_w/2}" y="{axis_y_bottom+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#0f172a">{esc("PyTorch")}</text>')

fa_x = pt_x + bar_w + 90
fa_val = 2.2
fa_y_top = axis_y_bottom - fa_val / axis_top_ms * chart_h
L.append(f'<rect x="{fa_x}" y="{fa_y_top}" width="{bar_w}" height="{axis_y_bottom-fa_y_top}" '
          'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.2"/>')
L.append(f'<text x="{fa_x+bar_w+8}" y="{fa_y_top-4}" font-family="sans-serif" '
          f'font-size="10" fill="#1e3a5f">{esc("Fused Kernel")}</text>')
L.append(f'<text x="{fa_x+bar_w/2}" y="{axis_y_bottom+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="#0f172a">{esc("FlashAttention")}</text>')

# 7.6x 加速标注
arrow_y = fa_y_top - 30
L.append(f'<line x1="{pt_x+bar_w/2}" y1="{axis_y_top-14}" x2="{fa_x+bar_w/2}" y2="{axis_y_top-14}" '
          'stroke="#c2410c" stroke-width="1.8" marker-end="url(#red)"/>')
L.append(f'<text x="{(pt_x+fa_x+bar_w)/2}" y="{axis_y_top-20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#c2410c">{esc("7.6× 加速(论文原文数字)")}</text>')

foot_y = h - 16
FOOT = ("柱高为读图近似值(总时长约 17ms → 约 2.2ms);7.6× 为论文正文给出的精确加速倍数,"
        "非本图逐段推导。")
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name(f"{FIG_ID}.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, canvas {w}x{h}")
