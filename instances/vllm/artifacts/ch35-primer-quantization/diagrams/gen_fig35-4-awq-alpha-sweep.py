#!/usr/bin/env python3
"""fig35-4-awq-alpha-sweep: AWQ 用激活幅度定缩放 s=s_X^alpha；损失随 alpha 呈 U 形，
甜点在内部（此例 alpha=0.25 降损 31%），alpha=1 过度缩放反而最差。
折线图：x=alpha, y=层输出重构损失 L(s_X^alpha)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "AWQ 缩放强度 alpha 扫描：损失呈 U 形，甜点 alpha=0.25 降损 30.87%"
SUBTITLE = "s = s_X^alpha；alpha=0 不保护显著权重，alpha=1 过度缩放撑大非显著通道格距"

POINTS = [(0.0, 0.2118), (0.25, 0.1464), (0.5, 0.1517), (0.75, 0.1651), (1.0, 0.2791)]
BEST_IDX = 1

PAD_L, PAD_R, PAD_T, PAD_B = 90, 60, 110, 90
PLOT_W, PLOT_H = 520, 300
w = PAD_L + PLOT_W + PAD_R
h = PAD_T + PLOT_H + PAD_B + 90

xs_vals = [p[0] for p in POINTS]
ys_vals = [p[1] for p in POINTS]
xmin, xmax = 0.0, 1.0
ymin, ymax = min(ys_vals), max(ys_vals)
y_span = ymax - ymin
y_lo = ymin - y_span * 0.18
y_hi = ymax + y_span * 0.18
X_MARGIN = 26  # keep alpha=0 / alpha=1 points off the axis lines

def xpix(x):
    return PAD_L + X_MARGIN + (x - xmin) / (xmax - xmin) * (PLOT_W - 2 * X_MARGIN)

def ypix(y):
    # larger loss -> higher on chart (smaller pixel y); smaller loss -> lower (valley at bottom)
    return PAD_T + (y_hi - y) / (y_hi - y_lo) * PLOT_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD_L-40}" y="46" font-family="sans-serif" font-size="16.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD_L-40}" y="68" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# axes
ax_x0, ax_y0 = xpix(xmin), ypix(y_lo)
ax_x1, ax_y1 = xpix(xmax), ypix(y_hi)
L.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+PLOT_H}" stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<line x1="{PAD_L}" y1="{PAD_T+PLOT_H}" x2="{PAD_L+PLOT_W}" y2="{PAD_T+PLOT_H}" stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<text x="{PAD_L-14}" y="{PAD_T+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">L(alpha)</text>')

# y gridlines/labels at each data value level (min & max)
for yv in (ymin, ymax):
    py = ypix(yv)
    L.append(f'<line x1="{PAD_L}" y1="{py}" x2="{PAD_L+PLOT_W}" y2="{py}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4,3"/>')
    L.append(f'<text x="{PAD_L-14}" y="{py+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#94a3b8">{yv:.4f}</text>')

# line connecting points
pts_px = [(xpix(x), ypix(y)) for x, y in POINTS]
path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts_px)
L.append(f'<path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>')

# points + labels + x labels
for i, ((x, y), (px, py)) in enumerate(zip(POINTS, pts_px)):
    is_best = (i == BEST_IDX)
    color = "#047857" if is_best else ("#b91c1c" if i in (0, len(POINTS) - 1) else "#3b82f6")
    r = 7 if is_best else 5.5
    L.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" stroke="white" stroke-width="1.5"/>')
    label = f"L({x:g})={y:.4f}" + (" (best)" if is_best else "")
    ly = py - 14 if i % 2 == 0 else py + 22
    anchor = "start" if i == 0 else ("end" if i == len(POINTS) - 1 else "middle")
    lx = px + 12 if anchor == "start" else (px - 12 if anchor == "end" else px)
    L.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
              f'font-size="11.5" fill="{color}">{esc(label)}</text>')
    L.append(f'<text x="{px:.1f}" y="{PAD_T+PLOT_H+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">alpha={x:g}</text>')

# legend
legend_y = PAD_T + PLOT_H + 52
L.append(f'<circle cx="{PAD_L+8}" cy="{legend_y-4}" r="6" fill="#047857"/>')
L.append(f'<text x="{PAD_L+22}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#334155">最优（甜点）</text>')
L.append(f'<circle cx="{PAD_L+150}" cy="{legend_y-4}" r="6" fill="#b91c1c"/>')
L.append(f'<text x="{PAD_L+164}" y="{legend_y}" font-family="sans-serif" font-size="12" fill="#334155">两端点（无缩放 / 过度缩放）</text>')

foot_y = legend_y + 30
foot_lines = [
    "reduction = 30.87%（相对 alpha=0 基线 0.2118 降到最优 0.1464）。",
    "RoundErr 实测均值 0.2501，约等于理论上的 0.25 格。",
    "缩放全在离线完成，vllm 只见已折进 scales 的打包 INT4 权重。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD_L-40}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-4-awq-alpha-sweep.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
