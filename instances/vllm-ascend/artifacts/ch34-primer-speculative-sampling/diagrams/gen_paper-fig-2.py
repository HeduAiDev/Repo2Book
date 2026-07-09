#!/usr/bin/env python3
"""论文精髓图重绘 —— arXiv:2211.17192 Figure 2(§3.1,Eq.1 的函数图像)。
E[#生成 token] = (1 - alpha^(gamma+1)) / (1 - alpha) 作为 alpha 的函数,按 gamma 分成一族曲线
(gamma=1/3/5/7/正无穷),外加标准解码基线 y=1。gamma=正无穷时收敛到上界 1/(1-alpha)
(alpha=0.9 处恰为 10,与原图 y 轴上限一致)。本章正文只在 alpha=0.8 处列了 gamma=1/3/5/10
四个离散值,是这条曲线族在 alpha=0.8 这一条竖线上的切片——图上用红点标出正文引用的
(alpha=0.8, gamma=5 -> 3.689) 这一点,呼应正文表格。
公式与坐标全部由代码按闭式计算,非拍照描点。
provenance = 论文原图本身(key_figure 重绘,豁免 explainer/spec.numbers 通道)。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def E(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


GAMMAS = [1, 3, 5, 7]
GAMMA_COLOR = {1: "#2563eb", 3: "#f59e0b", 5: "#16a34a", 7: "#dc2626"}
INF_COLOR = "#7c3aed"
BASELINE_COLOR = "#334155"

A_MIN, A_MAX = 0.5, 0.9
Y_MIN, Y_MAX = 0.0, 10.0

# ---- 画布几何 ----
PAD_L, PAD_R, PAD_T, PAD_B = 74, 34, 100, 60
PLOT_W, PLOT_H = 760, 420
w = PAD_L + PLOT_W + PAD_R
h = PAD_T + PLOT_H + PAD_B + 46


def px(alpha):
    return PAD_L + (alpha - A_MIN) / (A_MAX - A_MIN) * PLOT_W


def py(y):
    return PAD_T + PLOT_H - (y - Y_MIN) / (Y_MAX - Y_MIN) * PLOT_H


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

TITLE = "arXiv:2211.17192 Fig.2 重绘：期望接受 token 数 E[#tokens] 随 α 变化，按 γ 分族"
SUBTITLE = "E[#tokens] = (1 - α^(γ+1)) / (1 - α)；γ→∞ 时收敛到上界 1/(1-α)（α=0.9 处恰为 10）"
L.append(f'<text x="{PAD_L-40}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD_L-40}" y="55" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# ---- 图例(顶部一行) ----
legend_y = 82
legend_items = [("基线(标准解码 y=1)", BASELINE_COLOR, True)]
legend_items += [(f"γ={g}", GAMMA_COLOR[g], False) for g in GAMMAS]
legend_items += [("γ=∞（上界 1/(1-α)）", INF_COLOR, False)]
lx = PAD_L - 40
for label, color, dashed in legend_items:
    if dashed:
        L.append(f'<line x1="{lx}" y1="{legend_y-4}" x2="{lx+22}" y2="{legend_y-4}" '
                  f'stroke="{color}" stroke-width="2" stroke-dasharray="5,3"/>')
    else:
        L.append(f'<line x1="{lx}" y1="{legend_y-4}" x2="{lx+22}" y2="{legend_y-4}" '
                  f'stroke="{color}" stroke-width="3"/>')
    tx = lx + 28
    L.append(f'<text x="{tx}" y="{legend_y}" font-family="sans-serif" font-size="12.5" '
              f'fill="{color}" font-weight="bold">{esc(label)}</text>')
    lx = tx + 7.6 * len(label) + 26

plot_y0 = PAD_T + 20
plot_h2 = PLOT_H - 20
# 重新定义 py 使用调整后的绘图区(留出图例空间)
PLOT_TOP = plot_y0


def py2(y):
    return PLOT_TOP + plot_h2 - (y - Y_MIN) / (Y_MAX - Y_MIN) * plot_h2


# ---- 坐标轴与网格 ----
L.append(f'<rect x="{PAD_L}" y="{PLOT_TOP}" width="{PLOT_W}" height="{plot_h2}" '
          f'fill="#fafafa" stroke="#cbd5e1" stroke-width="1"/>')
for yv in range(0, 11, 2):
    gy = py2(yv)
    L.append(f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L+PLOT_W}" y2="{gy:.1f}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{PAD_L-10}" y="{gy+4:.1f}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11.5" fill="#64748b">{yv}</text>')
L.append(f'<text x="38" y="{PLOT_TOP+plot_h2/2:.1f}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155" transform="rotate(-90 38 {PLOT_TOP+plot_h2/2:.1f})" '
          f'text-anchor="middle">E[#tokens]</text>')

x_ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
for xv in x_ticks:
    gx = px(xv)
    L.append(f'<line x1="{gx:.1f}" y1="{PLOT_TOP}" x2="{gx:.1f}" y2="{PLOT_TOP+plot_h2}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{gx:.1f}" y="{PLOT_TOP+plot_h2+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#64748b">{xv:.2f}</text>')
L.append(f'<text x="{PAD_L+PLOT_W/2:.1f}" y="{PLOT_TOP+plot_h2+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#334155">α（草稿模型接受率）</text>')

# ---- baseline y=1 ----
by1 = py2(1.0)
L.append(f'<line x1="{PAD_L}" y1="{by1:.1f}" x2="{PAD_L+PLOT_W}" y2="{by1:.1f}" '
          f'stroke="{BASELINE_COLOR}" stroke-width="1.8" stroke-dasharray="6,4"/>')

# ---- 曲线:gamma=1,3,5,7(闭式采样) ----
N = 160
for g in GAMMAS:
    pts = []
    for i in range(N + 1):
        a = A_MIN + (A_MAX - A_MIN) * i / N
        yv = E(a, g)
        pts.append(f"{px(a):.1f},{py2(yv):.1f}")
    stroke_w = 3.2 if g == 5 else 2.2
    L.append(f'<polyline points="{" ".join(pts)}" fill="none" '
              f'stroke="{GAMMA_COLOR[g]}" stroke-width="{stroke_w}"/>')

# ---- gamma=infinity:上界 1/(1-alpha) ----
pts_inf = []
for i in range(N + 1):
    a = A_MIN + (A_MAX - A_MIN) * i / N
    yv = 1.0 / (1.0 - a)
    pts_inf.append(f"{px(a):.1f},{py2(min(yv, Y_MAX)):.1f}")
L.append(f'<polyline points="{" ".join(pts_inf)}" fill="none" '
          f'stroke="{INF_COLOR}" stroke-width="2.2" stroke-dasharray="2,3"/>')

# ---- 高亮本章引用点:alpha=0.8, gamma=5 -> 3.689 ----
hx, hy = px(0.8), py2(3.689)
L.append(f'<line x1="{hx:.1f}" y1="{PLOT_TOP+plot_h2}" x2="{hx:.1f}" y2="{hy:.1f}" '
          f'stroke="#b45309" stroke-width="1.3" stroke-dasharray="3,3"/>')
L.append(f'<line x1="{PAD_L}" y1="{hy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" '
          f'stroke="#b45309" stroke-width="1.3" stroke-dasharray="3,3"/>')
L.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="5.5" fill="#fff7ed" stroke="#b45309" stroke-width="2"/>')
label_x = hx + 12
label_y = hy - 12
L.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#b45309">本章引用：α=0.8, γ=5 → 3.689</text>')

foot_y = h - 22
L.append(f'<text x="{PAD_L-40}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">α 越大整条曲线越高、饱和越慢；γ 越大越贴近上界 1/(1-α)，但永远碰不到——'
          f'正文表格只是这条曲线族在 α=0.8 一条竖线上的切片</text>')
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}  aspect={w/h:.2f}")
