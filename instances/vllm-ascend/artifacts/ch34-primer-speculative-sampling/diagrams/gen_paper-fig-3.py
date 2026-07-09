#!/usr/bin/env python3
"""论文精髓图重绘 —— arXiv:2211.17192 Figure 3(§3.3,Theorem 3.8 speedup 公式的最优 gamma)。
speedup(alpha,gamma,c) = (1 - alpha^(gamma+1)) / ((1-alpha)(gamma*c+1))；
对每个 (alpha, c) 在整数 gamma 上暴力搜索最大化 speedup 的 gamma* —— 因 gamma 只能取整数，
最优值曲线是阶梯状。四条曲线对应成本系数 c=0.01/0.02/0.05/0.1(与原图一致)。
本章正文用的 c=0.05, alpha=0.8 -> 最优 gamma=8 这一点在图上用圆点标出，呼应正文表格。
公式与最优化过程全部由代码现算，非拍照描点。
provenance = 论文原图本身(key_figure 重绘，豁免 explainer/spec.numbers 通道)。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def speedup(alpha, gamma, c):
    return (1 - alpha ** (gamma + 1)) / ((1 - alpha) * (gamma * c + 1))


def optimal_gamma(alpha, c, max_gamma=40):
    best_g, best_v = 0, speedup(alpha, 0, c)
    for g in range(1, max_gamma + 1):
        v = speedup(alpha, g, c)
        if v > best_v:
            best_v, best_g = v, g
    return best_g


C_VALUES = [0.01, 0.02, 0.05, 0.1]
C_COLOR = {0.01: "#2563eb", 0.02: "#f59e0b", 0.05: "#16a34a", 0.1: "#dc2626"}

A_MIN, A_MAX = 0.5, 0.9
Y_MIN, Y_MAX = 0.0, 25.0

PAD_L, PAD_R, PAD_T, PAD_B = 74, 34, 100, 60
PLOT_W, PLOT_H = 760, 420
w = PAD_L + PLOT_W + PAD_R
h = PAD_T + PLOT_H + PAD_B + 46


def px(alpha):
    return PAD_L + (alpha - A_MIN) / (A_MAX - A_MIN) * PLOT_W


PLOT_TOP = PAD_T + 20
plot_h2 = PLOT_H - 20


def py(y):
    return PLOT_TOP + plot_h2 - (y - Y_MIN) / (Y_MAX - Y_MIN) * plot_h2


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

TITLE = "arXiv:2211.17192 Fig.3 重绘：最优 γ 随 α 变化，按草稿成本系数 c 分族"
SUBTITLE = "对每个 (α,c)，在整数 γ 上暴力搜索使 speedup=(1-α^(γ+1))/((1-α)(γc+1)) 最大的 γ* —— 阶梯状(γ 只能取整数)"
L.append(f'<text x="{PAD_L-40}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD_L-40}" y="55" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

legend_y = 82
lx = PAD_L - 40
for c in C_VALUES:
    color = C_COLOR[c]
    L.append(f'<line x1="{lx}" y1="{legend_y-4}" x2="{lx+22}" y2="{legend_y-4}" '
              f'stroke="{color}" stroke-width="3"/>')
    tx = lx + 28
    label = f"c={c}"
    L.append(f'<text x="{tx}" y="{legend_y}" font-family="sans-serif" font-size="13" '
              f'fill="{color}" font-weight="bold">{esc(label)}</text>')
    lx = tx + 8.0 * len(label) + 34

L.append(f'<rect x="{PAD_L}" y="{PLOT_TOP}" width="{PLOT_W}" height="{plot_h2}" '
          f'fill="#fafafa" stroke="#cbd5e1" stroke-width="1"/>')
for yv in range(0, 26, 5):
    gy = py(yv)
    L.append(f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{PAD_L+PLOT_W}" y2="{gy:.1f}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{PAD_L-10}" y="{gy+4:.1f}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11.5" fill="#64748b">{yv}</text>')
L.append(f'<text x="30" y="{PLOT_TOP+plot_h2/2:.1f}" font-family="sans-serif" font-size="12.5" '
          f'fill="#334155" transform="rotate(-90 30 {PLOT_TOP+plot_h2/2:.1f})" '
          f'text-anchor="middle">最优 γ*</text>')

x_ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
for xv in x_ticks:
    gx = px(xv)
    L.append(f'<line x1="{gx:.1f}" y1="{PLOT_TOP}" x2="{gx:.1f}" y2="{PLOT_TOP+plot_h2}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{gx:.1f}" y="{PLOT_TOP+plot_h2+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#64748b">{xv:.2f}</text>')
L.append(f'<text x="{PAD_L+PLOT_W/2:.1f}" y="{PLOT_TOP+plot_h2+40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#334155">α（草稿模型接受率）</text>')

N = 400
for c in C_VALUES:
    pts = []
    for i in range(N + 1):
        a = A_MIN + (A_MAX - A_MIN) * i / N
        g = optimal_gamma(a, c)
        pts.append(f"{px(a):.1f},{py(g):.1f}")
    stroke_w = 3.2 if c == 0.05 else 2.2
    L.append(f'<polyline points="{" ".join(pts)}" fill="none" '
              f'stroke="{C_COLOR[c]}" stroke-width="{stroke_w}"/>')

# 高亮本章引用点:c=0.05, alpha=0.8 -> 最优 gamma=8
# 标注框固定放在 alpha~0.63 附近、gamma~21 高度的空白区(该处所有曲线都远低于此),
# 用一条斜向引导线连到实际的点,避免与曲线/其它标注文字重叠。
hi_g = optimal_gamma(0.8, 0.05)
hx, hy = px(0.8), py(hi_g)
box_cx = px(0.635)
label_y = py(21.5)
box_w, box_h = 322, 26
L.append(f'<line x1="{box_cx:.1f}" y1="{label_y+box_h/2:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" '
          f'stroke="#b45309" stroke-width="1.3" stroke-dasharray="3,3"/>')
L.append(f'<line x1="{PAD_L}" y1="{hy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" '
          f'stroke="#b45309" stroke-width="1.3" stroke-dasharray="3,3"/>')
L.append(f'<circle cx="{hx:.1f}" cy="{hy:.1f}" r="5.5" fill="#fff7ed" stroke="#b45309" stroke-width="2"/>')
L.append(f'<rect x="{box_cx-box_w/2:.1f}" y="{label_y-box_h/2:.1f}" width="{box_w}" height="{box_h}" '
          f'rx="5" fill="white" stroke="#b45309" stroke-width="1.2"/>')
L.append(f'<text x="{box_cx:.1f}" y="{label_y+4.5:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#b45309">本章引用：c=0.05, α=0.8 → γ*={hi_g}</text>')

foot_y = h - 22
L.append(f'<text x="{PAD_L-40}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">c 越大，草稿越贵，最优 γ 越早被摁住(c=0.1 时到 α=0.9 也只到约 10)；'
          f'c 越小，可以放心猜得更远(c=0.01 时到 α=0.9 冲到 24)</text>')
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}  aspect={w/h:.2f}  highlight_gamma={hi_g}")
