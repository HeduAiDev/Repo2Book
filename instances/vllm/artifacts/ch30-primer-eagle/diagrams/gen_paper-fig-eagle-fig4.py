#!/usr/bin/env python3
"""paper-fig-eagle-fig4: 论文精髓图重绘(原图非本章自产机制图)。
重绘自 arXiv:2401.15077 Fig.4——token/feature/feature&shifted-token 三种草稿输入
在 Vicuna 7B、MT-bench(温度0)上、随训练 epoch(1-7)演化的准确率与加速比双折线图。
数据 provenance=原论文图本身(illustrator 契约"论文精髓图重绘"节,provenance 豁免;
非本章 explainer 素材通道)——从下载的原图 x4.png 按坐标网格逐点读出(见 illustrator 报告)。
两个面板(Speedup/Acc)x 轴均为 Epoch,ticks 2/4/6;三条折线用与原图一致的配色区分。
全部坐标由循环计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

EPOCHS = [1, 2, 3, 4, 5, 6, 7]
SERIES = [
    ("token", "#f59e0b", [1.37, 1.39, 1.42, 1.43, 1.46, 1.48, 1.48], [0.26, 0.27, 0.28, 0.28, 0.29, 0.30, 0.32]),
    ("feature", "#16a34a", [1.53, 1.65, 1.76, 1.82, 1.82, 1.86, 1.87], [0.52, 0.57, 0.59, 0.61, 0.62, 0.62, 0.63]),
    ("feature&shifted-token", "#3b82f6", [1.95, 2.26, 2.46, 2.53, 2.66, 2.70, 2.77], [0.62, 0.69, 0.73, 0.75, 0.76, 0.77, 0.78]),
]
PANELS = [
    ("Speedup 加速比", 1.2, 2.9, [1.5, 2.0, 2.5], 1),
    ("Acc 准确率", 0.2, 0.85, [0.4, 0.6, 0.8], 2),
]

PAD_L, PAD_T, PANEL_W, PANEL_H, GAP = 78, 150, 460, 320, 90
W = PAD_L * 2 + PANEL_W * 2 + GAP
H = PAD_T + PANEL_H + 90

def px(panel_x0, epoch):
    return panel_x0 + (epoch - 1) / (EPOCHS[-1] - 1) * PANEL_W

def py(y0, y_lo, y_hi, val):
    return y0 + PANEL_H * (1 - (val - y_lo) / (y_hi - y_lo))

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="40" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc("三种草稿输入的准确率与加速比（Vicuna 7B / MT-bench，温度=0）")}</text>',
     f'<text x="{W/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("重绘自 arXiv:2401.15077 Fig.4：feature&shifted-token 全程领先，随训练收敛到准确率≈0.78、加速比≈2.77x")}</text>']

# legend
leg_y = 95
lx = W / 2 - 300
for name, color, _, _ in SERIES:
    L.append(f'<line x1="{lx}" y1="{leg_y}" x2="{lx+26}" y2="{leg_y}" stroke="{color}" stroke-width="3"/>')
    L.append(f'<circle cx="{lx+13}" cy="{leg_y}" r="4" fill="{color}"/>')
    L.append(f'<text x="{lx+34}" y="{leg_y+4}" font-family="sans-serif" font-size="12" '
             f'fill="#0f172a">{esc(name)}</text>')
    lx += 34 + 8 * len(name) + 34

for pi, (title, y_lo, y_hi, gridvals, _) in enumerate(PANELS):
    x0 = PAD_L + pi * (PANEL_W + GAP)
    y0 = PAD_T
    L.append(f'<text x="{x0+PANEL_W/2}" y="{y0-16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    # axes
    L.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+PANEL_H}" stroke="#334155" stroke-width="1.5"/>')
    L.append(f'<line x1="{x0}" y1="{y0+PANEL_H}" x2="{x0+PANEL_W}" y2="{y0+PANEL_H}" stroke="#334155" stroke-width="1.5"/>')
    # y gridlines + labels
    for gv in gridvals:
        gy = py(y0, y_lo, y_hi, gv)
        L.append(f'<line x1="{x0}" y1="{gy}" x2="{x0+PANEL_W}" y2="{gy}" stroke="#e2e8f0" stroke-width="1"/>')
        L.append(f'<text x="{x0-10}" y="{gy+4}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="12" fill="#64748b">{esc(str(gv))}</text>')
    # x ticks (2,4,6)
    for ev in (2, 4, 6):
        gx = px(x0, ev)
        L.append(f'<line x1="{gx}" y1="{y0+PANEL_H}" x2="{gx}" y2="{y0+PANEL_H+6}" stroke="#334155" stroke-width="1.5"/>')
        L.append(f'<text x="{gx}" y="{y0+PANEL_H+22}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" fill="#334155">{esc(str(ev))}</text>')
    L.append(f'<text x="{x0+PANEL_W/2}" y="{y0+PANEL_H+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#64748b">{esc("Epoch")}</text>')
    # series
    for name, color, speed, acc in SERIES:
        vals = speed if pi == 0 else acc
        pts = [(px(x0, e), py(y0, y_lo, y_hi, v)) for e, v in zip(EPOCHS, vals)]
        d = " ".join(f'{"M" if i==0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
        L.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in pts:
            L.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    # endpoint annotation (epoch 7 final values)
    for name, color, speed, acc in SERIES:
        val = speed[-1] if pi == 0 else acc[-1]
        x, y = px(x0, 7), py(y0, y_lo, y_hi, val)
        L.append(f'<text x="{x+10}" y="{y+4}" font-family="sans-serif" font-size="11" '
                  f'font-weight="bold" fill="{color}">{esc(f"{val:.2f}" if pi==1 else f"{val:.2f}x")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-eagle-fig4.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
