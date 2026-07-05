#!/usr/bin/env python3
"""fig35-3-granularity — layout 模板：同一 2 token x 3 通道矩阵 X，三种缩放粒度并排对比。
per-tensor / per-token(行) 硬件可行；per-channel(输入通道/收缩维) 数学最优但 INT8 GEMM 不可行。"""
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

X = [[0.1, 0.2, 10.0], [0.15, 0.1, 8.0]]  # 2 token x 3 channel
N_ROW, N_COL = 2, 3
CELL, GAP = 50, 5
GRID_W = N_COL * (CELL + GAP) - GAP
GRID_H = N_ROW * (CELL + GAP) - GAP
GROUP_COLORS = ["#93c5fd", "#86efac", "#fcd34d"]

PANELS = [
    {
        "title_lines": ["per-tensor", "（硬件可行）"],
        "badge": ("✓", "#047857"),
        "group_of": lambda r, c: 0,
        "extra_w": 0,
        "bottom_lines": ["scale = 0.0787", "一个标量覆盖整矩阵"],
        "right_labels": None,
    },
    {
        "title_lines": ["per-token（按行）", "（硬件可行）"],
        "badge": ("✓", "#047857"),
        "group_of": lambda r, c: r,
        "extra_w": 100,
        "bottom_lines": ["每 token(行) 一个 scale", ""],
        "right_labels": ["scale=0.0787", "scale=0.063"],
    },
    {
        "title_lines": ["per-channel（输入通道）", "（数学最优，硬件不可行）"],
        "badge": ("✗", "#b91c1c"),
        "group_of": lambda r, c: c,
        "extra_w": 0,
        "bottom_lines": ["每输入通道(列) 一个 scale", "——收缩维，GEMM 无法拆"],
        "right_labels": None,
        "col_scale_labels": ["0.0012", "0.0016", "0.0787"],
    },
]

PANEL_W = GRID_W + 40  # 基础面板宽（含左右留白）
PAD = 36
PANEL_GAP = 40
panel_widths = [PANEL_W + p["extra_w"] for p in PANELS]
W = PAD * 2 + sum(panel_widths) + PANEL_GAP * (len(PANELS) - 1)

TOP = 190
PANEL_H = GRID_H + 110
H = TOP + PANEL_H + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{btext("为兼容 INT8 GEMM，缩放只能沿外维——per-channel(输入通道) 数学最优却做不到")}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("X 是 2 token × 3 通道，通道 2 是 outlier（与正文示例数据一致）")}</text>']

x_cursor = PAD
for p_idx, panel in enumerate(PANELS):
    pw = panel_widths[p_idx]
    px = x_cursor
    x_cursor += pw + PANEL_GAP
    cx = px + pw / 2
    gx0 = px + (pw - GRID_W) / 2 - (panel["extra_w"] / 2 if panel["right_labels"] else 0)
    gy0 = TOP

    # 面板标题（两行，居中于本面板宽度内，避免相邻面板文字相撞）
    for li, line in enumerate(panel["title_lines"]):
        L.append(f'<text x="{cx}" y="{TOP-92+li*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#0f172a">{esc(line)}</text>')
    # 可行性徽标
    badge_char, badge_color = panel["badge"]
    L.append(f'<circle cx="{cx}" cy="{TOP-40}" r="15" fill="white" '
              f'stroke="{badge_color}" stroke-width="2.5"/>')
    L.append(f'<text x="{cx}" y="{TOP-34}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="17" font-weight="bold" '
              f'fill="{badge_color}">{esc(badge_char)}</text>')

    # 网格
    for r in range(N_ROW):
        for c in range(N_COL):
            x = gx0 + c * (CELL + GAP)
            y = gy0 + r * (CELL + GAP)
            gid = panel["group_of"](r, c)
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="5" '
                      f'fill="{color}" stroke="#64748b" stroke-width="1.3"/>')
            L.append(f'<text x="{x+CELL/2}" y="{y+CELL/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="#0f172a">'
                      f'{esc(f"{X[r][c]:g}")}</text>')

    # 行右侧 scale 标注（per-token 面板）
    if panel["right_labels"]:
        for r in range(N_ROW):
            y = gy0 + r * (CELL + GAP) + CELL / 2 + 4
            lx = gx0 + GRID_W + 12
            L.append(f'<text x="{lx}" y="{y}" font-family="sans-serif" font-size="12" '
                      f'font-weight="bold" fill="#1e3a5f">{esc(panel["right_labels"][r])}</text>')

    # 列下方 scale 标注（per-channel 面板）
    if panel.get("col_scale_labels"):
        sy0 = gy0 + GRID_H + 22
        for c in range(N_COL):
            x = gx0 + c * (CELL + GAP) + CELL / 2
            L.append(f'<text x="{x}" y="{sy0}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" font-weight="bold" '
                      f'fill="#b91c1c">{esc(panel["col_scale_labels"][c])}</text>')

    # 面板底部说明行
    by = gy0 + GRID_H + (46 if panel.get("col_scale_labels") else 26)
    for li, line in enumerate(panel["bottom_lines"]):
        if not line:
            continue
        weight = 'font-weight="bold" fill="#1e3a5f"' if li == 0 and not panel.get("col_scale_labels") else 'fill="#64748b"'
        L.append(f'<text x="{cx}" y="{by+li*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" {weight}>{esc(line)}</text>')

foot_y = H - 20
foot1 = "per-channel(输入通道) 虽能把 outlier 通道(0.0787)与非 outlier(0.0012/0.0016)分开，但该维是 INT8"
foot2 = "GEMM 的收缩(累加)维，无法在矩阵乘中还原——落地代码只提供 per-tensor/per-channel(输出维) 两种参数形状。"
L.append(f'<text x="{PAD}" y="{foot_y-16}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(foot1)}</text>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(foot2)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig35-3-granularity.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
