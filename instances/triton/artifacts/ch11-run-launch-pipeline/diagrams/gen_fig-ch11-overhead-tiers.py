#!/usr/bin/env python3
"""state-table 模板(定制为对数柱状对比 + 交叉线示意):
左:三档 launch 代价对数横轴柱状对比(冷编/热编/稳态命中)。
右:GPU 计算时间(斜线)与稳态命中固定开销(水平线)交叉——发射受限区 vs 算力受限区。"""
import math
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "launch 三档开销：冷编 / 热编 / 稳态命中跨约 6 个数量级"
SUBTITLE = "同一 add_kernel（N=1024, BLOCK_SIZE=256）；对数横轴，条长按 log10(耗时/微秒) 计"

BARS = [
    ("首次冷编（含 ptxas/LLVM 预热，一次性）", 1951.11 * 1000, "#b91c1c", "1951.11 ms", "≈443591.2×"),
    ("热进程真编（慢路径 compile）", 98.379 * 1000, "#d97706", "98.379 ms", "≈22366.8×"),
    ("稳态命中（快路径，run 侧 Python 记账）", 4.398, "#15803d", "4.398 μs", "1×（基准）"),
]

PAD, TOP = 44, 118
CHART_W = 640
BAR_H, BAR_GAP = 46, 34
LABEL_W = 40
AXIS_MIN_LOG, AXIS_MAX_LOG = 0, 7  # us: 10^0=1us .. 10^7=10s

def bar_len(us):
    lg = math.log10(us)
    lg = max(AXIS_MIN_LOG, min(AXIS_MAX_LOG, lg))
    return (lg - AXIS_MIN_LOG) / (AXIS_MAX_LOG - AXIS_MIN_LOG) * CHART_W

n = len(BARS)
chart_h = TOP + n * (BAR_H + BAR_GAP) + 40

RIGHT_W = 480
RIGHT_RH = 300
w = PAD * 3 + CHART_W + RIGHT_W
# 预算高度:取(左侧柱状轴脚, 右侧交叉图脚)两者较大者,再加三行图注
axis_y_est = TOP + n * (BAR_H + BAR_GAP) + 6
right_bottom_est = TOP + 10 + RIGHT_RH
foot_y = max(axis_y_est + 66, right_bottom_est + 70)
h = foot_y + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc(SUBTITLE)}</text>')

# 左侧对数柱状图
chart_x0 = PAD
y = TOP
for label, us, color, val_text, ratio_text in BARS:
    blen = bar_len(us)
    L.append(f'<text x="{chart_x0}" y="{y-8}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    L.append(f'<rect x="{chart_x0}" y="{y}" width="{max(blen,4)}" height="{BAR_H}" rx="6" '
              f'fill="{color}" opacity="0.88"/>')
    L.append(f'<text x="{chart_x0+max(blen,4)+10}" y="{y+BAR_H/2-4}" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{color}">{esc(val_text)}</text>')
    L.append(f'<text x="{chart_x0+max(blen,4)+10}" y="{y+BAR_H/2+14}" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(ratio_text + " 于稳态")}</text>')
    y += BAR_H + BAR_GAP

axis_y = y + 6
L.append(f'<line x1="{chart_x0}" y1="{axis_y}" x2="{chart_x0+CHART_W}" y2="{axis_y}" '
          'stroke="#94a3b8" stroke-width="1.2"/>')
for e in range(AXIS_MIN_LOG, AXIS_MAX_LOG + 1, 2):
    tx = chart_x0 + (e - AXIS_MIN_LOG) / (AXIS_MAX_LOG - AXIS_MIN_LOG) * CHART_W
    L.append(f'<line x1="{tx}" y1="{axis_y}" x2="{tx}" y2="{axis_y+5}" stroke="#94a3b8"/>')
    label = {0: "1 μs", 2: "100 μs", 4: "10 ms", 6: "1 s"}.get(e, f"1e{e} μs")
    L.append(f'<text x="{tx}" y="{axis_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{esc(label)}</text>')
L.append(f'<text x="{chart_x0+CHART_W/2}" y="{axis_y+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">{esc("耗时（对数刻度）")}</text>')

# 右侧交叉线示意图(发射受限 vs 算力受限)
rx0 = chart_x0 + CHART_W + 90
rw, rh = RIGHT_W - 70, RIGHT_RH
ry0 = TOP + 10
const_y = ry0 + rh - 46  # 稳态命中常数线
cross_x = rx0 + 0.42 * rw  # GPU 计算斜线与常数线交点

L.append(f'<text x="{rx0}" y="{ry0-24}" font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#0f172a">{esc("规模 vs 耗时：两条曲线的交点")}</text>')

# 两个 regime 全高背景色块(先画,后续坐标轴/曲线/文字叠加在其上)
L.append(f'<rect x="{rx0}" y="{ry0}" width="{cross_x-rx0}" height="{rh}" fill="#dcfce7" opacity="0.45"/>')
L.append(f'<rect x="{cross_x}" y="{ry0}" width="{rx0+rw-cross_x}" height="{rh}" fill="#dbeafe" opacity="0.45"/>')

# 坐标轴
L.append(f'<line x1="{rx0}" y1="{ry0+rh}" x2="{rx0+rw}" y2="{ry0+rh}" stroke="#334155" '
          'stroke-width="1.4" marker-end="url(#a)"/>')
L.append(f'<line x1="{rx0}" y1="{ry0+rh}" x2="{rx0}" y2="{ry0}" stroke="#334155" '
          'stroke-width="1.4" marker-end="url(#a)"/>')
L.append(f'<text x="{rx0+rw/2}" y="{ry0+rh+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("kernel 计算规模（元素数）")}</text>')
L.append(f'<text x="{rx0-14}" y="{ry0-6}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("耗时")}</text>')

L.append(f'<line x1="{rx0}" y1="{const_y}" x2="{rx0+rw}" y2="{const_y}" '
          'stroke="#15803d" stroke-width="2.2"/>')
L.append(f'<text x="{rx0+8}" y="{const_y-8}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="#15803d">{esc("发射固定开销 ≈4.398 μs（常数，与规模无关）")}</text>')

L.append(f'<line x1="{rx0}" y1="{ry0+rh-4}" x2="{rx0+rw}" y2="{ry0+8}" '
          'stroke="#1d4ed8" stroke-width="2.2"/>')
L.append(f'<text x="{rx0+rw-6}" y="{ry0+22}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#1d4ed8">{esc("GPU 计算时间")}</text>')
L.append(f'<text x="{rx0+rw-6}" y="{ry0+38}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#1d4ed8">{esc("（随规模线性增）")}</text>')

L.append(f'<circle cx="{cross_x}" cy="{const_y}" r="5" fill="#7c2d12"/>')
L.append(f'<line x1="{cross_x}" y1="{ry0}" x2="{cross_x}" y2="{ry0+rh}" stroke="#94a3b8" '
          'stroke-width="1" stroke-dasharray="4,3"/>')
L.append(f'<text x="{cross_x}" y="{ry0+rh+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#7c2d12">{esc("交点")}</text>')

L.append(f'<text x="{rx0+(cross_x-rx0)/2}" y="{ry0+rh-32}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#15803d">'
          f'{esc("发射受限区")}</text>')
L.append(f'<text x="{rx0+(cross_x-rx0)/2}" y="{ry0+rh-18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#166534">{esc("（小 kernel）")}</text>')

L.append(f'<text x="{cross_x+(rx0+rw-cross_x)/2}" y="{ry0+rh-32}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#1d4ed8">'
          f'{esc("算力受限区")}</text>')

L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("冷编/命中 ≈443591.2×；热编/命中 ≈22366.8×——性能测量必须先预热才反映稳态成本。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("4.398 μs 为 headless warmup 路径 run 侧记账（未含真 C++ 发射），是真实固定开销的下界；")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+42}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("小 kernel 落在发射受限区时，优化应靠融合 kernel / 增大 grid / CUDA graph，而非堆算力。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-overhead-tiers.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
