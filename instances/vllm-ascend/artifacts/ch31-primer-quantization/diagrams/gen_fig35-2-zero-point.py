#!/usr/bin/env python3
"""fig35-2-zero-point — before-after 变体：两条数轴（对称 vs 非对称量化网格）上下对比。
数据 a=[0.1,0.55,0.9,0.32]（全正）；对称网格浪费负半轴，非对称网格贴着数据区间。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 已知渲染环境缺陷：字符"量"在 synthetic-bold 下会被错渲成实心方块。粗体文本经 btext() 拆 tspan 规避。
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

DATA = [0.1, 0.55, 0.9, 0.32]
DATA_LABELS = ["0.1", "0.55", "0.9", "0.32"]

# 对称量化：Δ=0.3，8 个码位 -4..3 → dequant 值 -1.2..0.9
SYM_TICKS = [round(-1.2 + 0.3 * i, 2) for i in range(8)]  # -1.2..0.9
SYM_NEAREST = {0.1: 0.0, 0.55: 0.6, 0.9: 0.9, 0.32: 0.3}
SYM_MAX_ERR = 0.1  # 由 0.1 -> 0.0 贡献

# 非对称量化：scale=0.1143, zero_point=-1，8 个码位 dequant = (q+1)*scale, q=0..7
ASYM_SCALE = 0.1143
ASYM_TICKS = [round(ASYM_SCALE * i, 4) for i in range(1, 9)]  # 0.1143..0.9144
ASYM_NEAREST = {0.1: 0.1143, 0.55: 0.5714, 0.9: 0.9143, 0.32: 0.3429}
ASYM_MAX_ERR = 0.0229  # 由 0.32 -> 0.3429 贡献 (|0.32-0.3429|=0.0229)

PAD, W = 50, 900
AXIS_L, AXIS_R = 140, 860
TOP1, TOP2 = 130, 340
PANEL_H = 170
H = TOP2 + PANEL_H + 70

def x_of(v, vmin, vmax):
    return AXIS_L + (v - vmin) / (vmax - vmin) * (AXIS_R - AXIS_L)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext("零点摆对地方：非对称量化把最大误差从 0.1 降到 0.0229")}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("输入 a = [0.1, 0.55, 0.9, 0.32]（全为正，3-bit / 8 档位）")}</text>']


def panel(title, top, ticks, nearest, max_err, vmin, vmax, wasted_range, color_scheme):
    grp = [f'<text x="{AXIS_L}" y="{top-14}" font-family="sans-serif" font-size="14" '
           f'fill="#0f172a">{btext(title)}</text>']
    grp.append(f'<text x="{AXIS_R+10}" y="{top-14}" text-anchor="end" font-family="sans-serif" '
               f'font-size="13" font-weight="bold" fill="#b91c1c">'
               f'{esc(f"max|误差| = {max_err}")}</text>')
    axis_y = top + 60
    # 坐标轴（纯直线，不加箭头——数轴不表示流向，避免与右侧标注重叠）
    grp.append(f'<line x1="{AXIS_L-10}" y1="{axis_y}" x2="{AXIS_R+10}" y2="{axis_y}" '
               'stroke="#334155" stroke-width="1.5"/>')
    # 浪费区间遮罩（对称量化的负半轴）
    if wasted_range:
        wx1 = x_of(wasted_range[0], vmin, vmax)
        wx2 = x_of(wasted_range[1], vmin, vmax)
        grp.append(f'<rect x="{wx1}" y="{axis_y-38}" width="{wx2-wx1}" height="76" '
                   'fill="#fecaca" opacity="0.35"/>')
        grp.append(f'<text x="{(wx1+wx2)/2}" y="{axis_y-44}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="11" fill="#b91c1c">'
                   f'{esc("浪费（数据从不落在此区间）")}</text>')
    # 网格刻度
    for t in ticks:
        tx = x_of(t, vmin, vmax)
        grp.append(f'<line x1="{tx}" y1="{axis_y-8}" x2="{tx}" y2="{axis_y+8}" '
                   'stroke="#94a3b8" stroke-width="1.5"/>')
        grp.append(f'<text x="{tx}" y="{axis_y+24}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="10" fill="#64748b">{t:g}</text>')
    # 数据点 + 误差连线
    for v, lbl in zip(DATA, DATA_LABELS):
        dx = x_of(v, vmin, vmax)
        n = nearest[v]
        nx = x_of(n, vmin, vmax)
        err = abs(v - n)
        is_max = abs(err - max_err) < 1e-6
        col = "#b91c1c" if is_max else "#0f766e"
        grp.append(f'<circle cx="{dx}" cy="{axis_y-30}" r="5" fill="{col}"/>')
        grp.append(f'<text x="{dx}" y="{axis_y-38}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="11" font-weight="bold" '
                   f'fill="{col}">{esc(lbl)}</text>')
        if abs(dx - nx) > 1.0:
            grp.append(f'<line x1="{dx}" y1="{axis_y-25}" x2="{nx}" y2="{axis_y-2}" '
                       f'stroke="{col}" stroke-width="1.3" stroke-dasharray="3,2"/>')
        else:
            grp.append(f'<line x1="{dx}" y1="{axis_y-25}" x2="{dx}" y2="{axis_y-2}" '
                       f'stroke="{col}" stroke-width="1.3" stroke-dasharray="3,2"/>')
    return grp

sym_group = panel("左：对称量化（Δ=0.3，零点钉在 0）", TOP1, SYM_TICKS, SYM_NEAREST,
                   SYM_MAX_ERR, -1.3, 1.0, (-1.3, -0.05), "sym")
asym_group = panel("右：非对称量化（scale=0.1143, zero_point=-1）", TOP2, ASYM_TICKS,
                    ASYM_NEAREST, ASYM_MAX_ERR, 0.0, 1.0, None, "asym")

L.extend(sym_group)
L.extend(asym_group)

foot_y = H - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("红点=贡献最大误差的数据点；左图负半轴 8 档中有 4 档全程未用；右图 8 档全部落在 [0.1,0.9] 数据区间内。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-2-zero-point.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
