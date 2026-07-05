#!/usr/bin/env python3
"""fig27-3: 量化粒度谱 —— per-tensor / per-channel / per-group(MXFP) 在同一权重矩阵上的 scale 覆盖。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def rect(x, y, w, h, fill, stroke, sw=1.0):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 42, "一个量化方案 = 选一种 scale 粒度", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "同一权重矩阵 [out=6, in=8] 上，scale 怎么铺，决定精度与显存的权衡", 14, "middle", "#64748b")

ROWS, COLS = 6, 8
cell = 34
GROUP = 4  # per-group 每 4 个输入通道共享一个 scale

# scale 配色（同一 scale 的格子同色）
palette = ["#fca5a5", "#fdba74", "#fcd34d", "#86efac", "#7dd3fc", "#c4b5fd", "#f9a8d4", "#a5b4fc"]
# per-group 第二组用同色系的深一档——表达「同一行(per-channel)再按输入组细分」=> [out, in//g] 二维网格
palette2 = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#0ea5e9", "#8b5cf6", "#ec4899", "#6366f1"]


def draw_matrix(ox, oy, mode, title, sub):
    gw = COLS * cell
    text(ox + gw / 2, oy - 38, title, 16, "middle", "#0f172a", "bold")
    text(ox + gw / 2, oy - 16, sub, 11.5, "middle", "#475569", mono=True)
    for r in range(ROWS):
        for c in range(COLS):
            if mode == "tensor":
                col = "#93c5fd"
            elif mode == "channel":
                col = palette[r % len(palette)]
            else:  # group: per-(out 行 × in 组) → 二维网格 [out, in//g]，每行仍各自分组
                col = palette[r % len(palette)] if (c // GROUP) == 0 else palette2[r % len(palette2)]
            rect(ox + c * cell, oy + r * cell, cell, cell, col, "#ffffff", 1.2)
    # 外框
    rect(ox, oy, gw, ROWS * cell, "none", "#334155", 1.8)
    # 轴标
    text(ox - 10, oy + ROWS * cell / 2, "out", 12, "end", "#64748b")
    text(ox + gw / 2, oy + ROWS * cell + 22, "in →", 12, "middle", "#64748b")


gap = 80
m1x = 90
m2x = m1x + COLS * cell + gap + 70
m3x = m2x + COLS * cell + gap + 70
my = 180

draw_matrix(m1x, my, "tensor", "per-tensor", "全矩阵 1 个 scale")
draw_matrix(m2x, my, "channel", "per-channel", "每行(out) 1 个 scale")
draw_matrix(m3x, my, "group", "per-group", "每(out 行 × in 组) 1 个 scale → [out, in//g]")

# 组分隔虚线（per-group）
gx = m3x
for g in range(1, COLS // GROUP):
    xx = gx + g * GROUP * cell
    L.append(f'<line x1="{xx}" y1="{my}" x2="{xx}" y2="{my + ROWS * cell}" stroke="#1e293b" stroke-width="2.4"/>')

# 底部说明卡
cards = [
    (m1x, "#eff6ff", "#3b82f6", ["最省、最糙", "整张矩阵共享 1 个标量", "几乎不用于权重"]),
    (m2x, "#ecfdf5", "#10b981", ["W8A8_DYNAMIC 用这档", "weight_scale 形状 [out, 1]", "每输出通道独立缩放"]),
    (m3x, "#f5f3ff", "#7c3aed", ["W4A8 用 weight_scale_second [out, in//g]", "MXFP 极端：每组 1 个 e8m0 共享指数", "= microscaling 微缩放，护精度最强"]),
]
cy = my + ROWS * cell + 56
for cxx, fill, stroke, lines in cards:
    cw = COLS * cell + 40
    bx = cxx - 20
    L.append(f'<rect x="{bx}" y="{cy}" width="{cw}" height="118" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    for i, ln in enumerate(lines):
        text(bx + 16, cy + 32 + i * 28, "• " + ln, 12, "start", "#334155")

# 粒度箭头谱
ay = cy + 150
L.append(f'<line x1="{m1x}" y1="{ay}" x2="{m3x + COLS * cell}" y2="{ay}" stroke="#94a3b8" stroke-width="2"/>')
text(m1x, ay + 22, "粗 / 省显存", 13, "start", "#64748b")
text(m3x + COLS * cell, ay + 22, "细 / 护精度", 13, "end", "#64748b")
text(W / 2, ay + 22, "粒度越细，scale 越多、显存略增，但量化误差越小", 12.5, "middle", "#475569")

L.append('</svg>')
open("fig27-3-granularity.svg", "w").write('\n'.join(L))
print("wrote fig27-3-granularity.svg")
