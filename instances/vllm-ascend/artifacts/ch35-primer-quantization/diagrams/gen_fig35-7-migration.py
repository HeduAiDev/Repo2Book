#!/usr/bin/env python3
"""fig35-7-migration — before-after 模板（柱状图）：SmoothQuant 迁移前后激活/权重通道 absmax。
α=0.5 把激活 outlier(8.9) 压到与权重相等的几何均值(1.1549)，乘积无损(残差 0.0)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
                buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

CH_LABELS = ["通道0", "通道1", "通道2", "通道3"]
X_BEFORE = [8.9035, 0.3001, 0.3775, 0.201]
W_BEFORE = [0.1498, 0.3899, 0.4081, 0.3682]
X_AFTER = [1.1549, 0.3421, 0.3925, 0.2721]
W_AFTER = [1.1549, 0.3421, 0.3925, 0.2721]  # 与 X_AFTER 相等（几何均值）

PANEL_W, PAD, TOP = 420, 46, 130
BAR_W, BAR_GAP, GROUP_GAP = 26, 4, 30
CHART_H = 260
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + CHART_H + 130

MAXV = 9.5

def bar_x0(panel_px, i):
    return panel_px + 40 + i * (2 * BAR_W + BAR_GAP + GROUP_GAP)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext("α=0.5 迁移：激活 outlier 从 8.9 压到与权重相等的 1.1549，乘积无损")}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("T=16, Ci=4；输入通道0 约 44× 激活 outlier；Eq.3 迁移残差 = 0.0（代数恒等）")}</text>']


def panel(title, px, x_vals, w_vals, x_color, w_color, legend):
    grp = [f'<text x="{px+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" '
           f'font-family="sans-serif" font-size="14" font-weight="bold" '
           f'fill="#0f172a">{esc(title)}</text>']
    base_y = TOP + CHART_H
    grp.append(f'<line x1="{px+20}" y1="{base_y}" x2="{px+PANEL_W-20}" y2="{base_y}" '
                'stroke="#334155" stroke-width="1.5"/>')
    for i, (xv, wv) in enumerate(zip(x_vals, w_vals)):
        gx = bar_x0(px, i)
        xh = xv / MAXV * (CHART_H - 30)
        wh = wv / MAXV * (CHART_H - 30)
        # 激活柱：数值标签右对齐在柱顶右缘，向左侧(远离另一柱)展开，避免与权重柱标签相撞
        grp.append(f'<rect x="{gx}" y="{base_y-xh}" width="{BAR_W}" height="{xh}" '
                    f'fill="{x_color}" stroke="#1e3a5f" stroke-width="1"/>')
        grp.append(f'<text x="{gx+BAR_W}" y="{base_y-xh-6}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="10" font-weight="bold" '
                    f'fill="#0f172a">{xv:g}</text>')
        # 权重柱：数值标签左对齐在柱顶左缘，向右侧展开
        wx = gx + BAR_W + BAR_GAP
        grp.append(f'<rect x="{wx}" y="{base_y-wh}" width="{BAR_W}" height="{wh}" '
                    f'fill="{w_color}" stroke="#1e3a5f" stroke-width="1"/>')
        grp.append(f'<text x="{wx}" y="{base_y-wh-6}" text-anchor="start" '
                    f'font-family="sans-serif" font-size="10" font-weight="bold" '
                    f'fill="#0f172a">{wv:g}</text>')
        # 通道标签
        grp.append(f'<text x="{gx+BAR_W+BAR_GAP/2}" y="{base_y+18}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="#64748b">{esc(CH_LABELS[i])}</text>')
    # 图例
    ly = base_y + 40
    lx = px + 20
    grp.append(f'<rect x="{lx}" y="{ly}" width="14" height="14" rx="2" fill="{x_color}" stroke="#1e3a5f"/>')
    grp.append(f'<text x="{lx+20}" y="{ly+12}" font-family="sans-serif" font-size="11" '
                f'fill="#334155">{esc(legend[0])}</text>')
    grp.append(f'<rect x="{lx+150}" y="{ly}" width="14" height="14" rx="2" fill="{w_color}" stroke="#1e3a5f"/>')
    grp.append(f'<text x="{lx+170}" y="{ly+12}" font-family="sans-serif" font-size="11" '
                f'fill="#334155">{esc(legend[1])}</text>')
    return grp

L.extend(panel("迁移前：激活 outlier 主导", PAD, X_BEFORE, W_BEFORE,
                "#fca5a5", "#93c5fd", ["激活 X absmax", "权重 W absmax"]))
L.extend(panel("迁移后（α=0.5）：outlier 被摊平", PAD + PANEL_W + 100, X_AFTER, W_AFTER,
                "#86efac", "#93c5fd", ["激活 X̂ absmax", "权重 Ŵ absmax(=X̂)"]))

# 中间箭头 + 关键数字
midy = TOP + CHART_H / 2 - 20
ax1 = PAD + PANEL_W + 14
ax2 = PAD + PANEL_W + 86
L.append(f'<line x1="{ax1}" y1="{midy}" x2="{ax2}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{midy-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#d97706">{esc("迁移")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{midy+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#64748b">{esc("残差=0.0")}</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("激活通道最大值差从 44.2888× 压到 4.2447×；迁移后 X̂ 与 Ŵ 每通道 absmax 逐一相等（几何均值）——s 可离线融进上一层，运行时零开销。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-7-migration.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
