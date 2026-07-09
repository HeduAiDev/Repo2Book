#!/usr/bin/env python3
"""论文精髓图重绘:arXiv:2307.08691 Fig.4 —
A100 GPU 上 FlashAttention-2 相对 FlashAttention 与标准 PyTorch 实现的前向+反向速度
(TFLOPs/s)对比柱状图。原图共 4 个子图(有/无因果掩码 × head_dim 64/128,每个子图另有
xformers、FlashAttention-Triton 两条参照柱);本图截取"head_dim=64、含因果掩码"这一子图,
且只保留 PyTorch / FlashAttention / FlashAttention-2 三条主线柱(对应本章正文关心的"比
FA 快多少、比标准实现快多少"),数字取自 ar5iv 抓到的原图(assets/x3.png)柱顶标注,逐一读数。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

FIG_ID = "paper-fig-4"
TITLE = "重绘自 arXiv:2307.08691 Fig.4"
SUBTITLE = "A100、head_dim=64、含因果掩码配置:FlashAttention-2 前向+反向速度(TFLOPs/s)"

SEQLENS = ["512", "1k", "2k", "4k", "8k", "16k"]
PYTORCH = [15, 16, 17, 18, 18, None]          # None = OOM(原图标注)
FLASHATTN = [58, 70, 77, 87, 92, 97]
FLASHATTN2 = [88, 119, 140, 156, 165, 171]
SERIES = [("PyTorch", PYTORCH, "#60a5fa"), ("FlashAttention", FLASHATTN, "#fb923c"),
          ("FlashAttention-2", FLASHATTN2, "#8b5cf6")]

PAD, TOP = 40, 122
CHART_W, CHART_H = 760, 340
AXIS_MAX = 190

w = PAD * 2 + CHART_W + 40
h = TOP + CHART_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

axis_x, axis_y_bottom = PAD + 60, TOP + CHART_H
axis_y_top = TOP
L.append(f'<line x1="{axis_x}" y1="{axis_y_top}" x2="{axis_x}" y2="{axis_y_bottom}" '
         'stroke="#0f172a" stroke-width="1.6"/>')
L.append(f'<line x1="{axis_x}" y1="{axis_y_bottom}" x2="{axis_x+CHART_W-60}" y2="{axis_y_bottom}" '
         'stroke="#0f172a" stroke-width="1.6"/>')
for tick in (0, 50, 100, 150):
    ty = axis_y_bottom - tick / AXIS_MAX * CHART_H
    L.append(f'<line x1="{axis_x-5}" y1="{ty}" x2="{axis_x}" y2="{ty}" stroke="#0f172a" stroke-width="1.2"/>')
    L.append(f'<text x="{axis_x-10}" y="{ty+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{tick}</text>')
# y 轴单位已在副标题里写明(TFLOPs/s),此处只放一个贴轴顶的小标签,不用旋转文字
# (旋转长标签在 100 刻度处与刻度数字易碰撞),避免几何冲突。
L.append(f'<text x="{axis_x-6}" y="{axis_y_top-10}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10.5" fill="#334155">'
          f'{esc("TFLOPs/s")}</text>')

n_group = len(SEQLENS)
n_series = len(SERIES)
group_w = (CHART_W - 60) / n_group
bar_w = 26
bar_gap = 6
group_content_w = n_series * bar_w + (n_series - 1) * bar_gap

for gi, seqlen in enumerate(SEQLENS):
    gx0 = axis_x + gi * group_w + (group_w - group_content_w) / 2
    for si, (name, vals, color) in enumerate(SERIES):
        bx = gx0 + si * (bar_w + bar_gap)
        val = vals[gi]
        if val is None:
            L.append(f'<text x="{bx+bar_w/2}" y="{axis_y_bottom-8}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10" font-weight="bold" '
                      f'fill="#94a3b8" transform="rotate(-90 {bx+bar_w/2} {axis_y_bottom-8})">'
                      f'{esc("OOM")}</text>')
            continue
        by_top = axis_y_bottom - val / AXIS_MAX * CHART_H
        L.append(f'<rect x="{bx}" y="{by_top}" width="{bar_w}" height="{axis_y_bottom-by_top}" '
                  f'fill="{color}" stroke="#1e3a5f" stroke-width="1"/>')
        L.append(f'<text x="{bx+bar_w/2}" y="{by_top-6}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" font-weight="bold" fill="#0f172a">{val}</text>')
    L.append(f'<text x="{gx0+group_content_w/2}" y="{axis_y_bottom+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#334155">{esc(seqlen)}</text>')

L.append(f'<text x="{axis_x+(CHART_W-60)/2}" y="{axis_y_bottom+44}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#334155">{esc("Sequence length")}</text>')

# 图例
legend_y = TOP - 14
legend_x = axis_x + 260
for name, _, color in SERIES:
    L.append(f'<rect x="{legend_x}" y="{legend_y-12}" width="16" height="14" rx="2" fill="{color}" '
              'stroke="#1e3a5f" stroke-width="0.8"/>')
    L.append(f'<text x="{legend_x+22}" y="{legend_y}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(name)}</text>')
    legend_x += 22 + len(name) * 7.6 + 22

# 结论条
concl_y = axis_y_bottom + 68
L.append(f'<rect x="{PAD}" y="{concl_y}" width="{w-2*PAD}" height="40" rx="6" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
CONCL = "本子图内 FA-2/FA ≈ 1.5–1.8×;FA-2/PyTorch 在 8k 达 ≈9.2×——论文正文称全部配置下最高达 10×"
L.append(f'<text x="{w/2}" y="{concl_y+25}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc(CONCL)}</text>')

foot_y = concl_y + 40 + 24
FOOT = ("原图另有 head_dim=128、无因果掩码等 3 个子图及 xformers/FlashAttention-Triton "
        "两条参照柱,本图仅节选与正文相关的一角。")
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT)}</text>')

h = foot_y + 16
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L.append('</svg>')
out = Path(__file__).with_name(f"{FIG_ID}.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, canvas {w}x{h}")
