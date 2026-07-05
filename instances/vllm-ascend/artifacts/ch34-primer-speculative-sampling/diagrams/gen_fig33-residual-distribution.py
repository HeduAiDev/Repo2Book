#!/usr/bin/env python3
"""before-after 模板(柱状双面板):拒绝后从残差分布 p'=norm(max(0,p-q)) 重采样。
左面板=原始缺口 p-q(裁剪到 0);右面板=归一化后的 p'。数字来自 explainer/traces/residual.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TOKENS = ["A", "B", "C", "D"]
GAP_RAW = {"A": 0.1, "B": 0.1, "C": -0.2, "D": 0.0}      # p - q
GAP_CLIP = {"A": 0.1, "B": 0.1, "C": 0.0, "D": 0.0}       # max(0, p-q)
PPRIME = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0}         # normalized

BAR_W, BAR_MAXH, GAP_X, PAD, TOP = 60, 160, 26, 40, 110
PANEL_W = len(TOKENS) * (BAR_W + GAP_X)
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + BAR_MAXH + 150
SCALE = BAR_MAXH / 0.5  # 0.5 是两面板出现的最大幅值(|p-q| 或 p')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-10}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">被拒后从残差分布重采样:欠提议 token 才拿到找补质量</text>',
     f'<text x="{PAD}" y="{PAD+10}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">p=[0.5,0.3,0.1,0.1], q=[0.4,0.2,0.3,0.1] — 归一化常数 Sum max(0,p-q) = 0.2 = 1-beta</text>']

baseline = TOP + BAR_MAXH

def bar_panel(px, title, values, color_pos, color_zero, value_fmt):
    L.append(f'<text x="{px+PANEL_W/2}" y="{TOP-24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#0f172a">{esc(title)}</text>')
    L.append(f'<line x1="{px-8}" y1="{baseline}" x2="{px+PANEL_W}" y2="{baseline}" '
              'stroke="#94a3b8" stroke-width="1"/>')
    for i, tok in enumerate(TOKENS):
        bx = px + i * (BAR_W + GAP_X) + GAP_X / 2
        val = values[tok]
        bh = val * SCALE
        color = color_zero if val == 0 else color_pos
        L.append(f'<rect x="{bx}" y="{baseline-bh}" width="{BAR_W}" height="{max(bh,2)}" rx="4" '
                  f'fill="{color[0]}" stroke="{color[1]}" stroke-width="1.5"/>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{baseline-bh-8 if bh>0 else baseline-10}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="12.5" '
                  f'font-weight="bold" fill="{color[1]}">{value_fmt(val)}</text>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{baseline+20}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="#0f172a">{esc(tok)}</text>')

COLOR_POS = ("#fef3c7", "#b45309")
COLOR_ZERO = ("#f1f5f9", "#94a3b8")

px1 = PAD
bar_panel(px1, "缺口 max(0, p-q)  ——  裁剪掉的负值旁注原始 p-q",
          GAP_CLIP, COLOR_POS, COLOR_ZERO, lambda v: f"{v:.1f}")
# annotate raw negative for C, zero for D underneath the axis
raw_note_y = baseline + 38
for i, tok in enumerate(TOKENS):
    bx = px1 + i * (BAR_W + GAP_X) + GAP_X / 2
    raw = GAP_RAW[tok]
    if raw < 0:
        L.append(f'<text x="{bx+BAR_W/2}" y="{raw_note_y}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#94a3b8">'
                  f'原始 p-q={raw:.1f}(裁剪为 0)</text>')

px2 = PAD + PANEL_W + 90
bar_panel(px2, "归一化残差分布 p' = norm(max(0,p-q))",
          PPRIME, ("#dcfce7", "#15803d"), COLOR_ZERO, lambda v: f"{v:.1f}")

# arrow between panels
midy = TOP + BAR_MAXH / 2
L.append(f'<line x1="{px1+PANEL_W+8}" y1="{midy}" x2="{px2-8}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(px1+PANEL_W+px2)/2}" y="{midy-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#b45309">÷0.2 归一化</text>')

foot_y = h - 36
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">A、B(目标想要更多,q&lt;p)平分残差质量各得 0.5;C、D(q&gt;=p,已被过度提议)分不到任何找补质量</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">蒙特卡洛 N=400000 重采样得 [0.501,0.499,0.0,0.0],与 p\' 吻合</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-residual-distribution.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
