#!/usr/bin/env python3
"""paper-fig-awq-2 —— 重绘自 arXiv:2306.00978 (AWQ) Fig.2：
三联对比 —— (a) 朴素 RTN 量化(PPL 43.2)；(b) 按激活挑 1% 显著权重保 FP16
(PPL 13.0，但混合精度硬件效率差)；(c) AWQ 按通道缩放再统一量化(同样 PPL
13.0，格式统一、硬件友好)。informational structure 对齐原图三面板+激活行
+高亮列,配色套用本书视觉语言,矩阵数值为示意(与原图一样是教学用小样例，
非逐位复刻原图像素)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

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

TITLE = "AWQ 的出发点：按激活挑 1% 显著权重能救回精度，但混合精度不利于硬件——AWQ 改用整列缩放"
SUBTITLE = ("重绘自 arXiv:2306.00978 Fig.2（OPT-6.7B，INT3-g128）：中间面板 43.2→13.0 靠给显著通道特殊待遇；"
            "右面板 AWQ 用按通道缩放同样拿到 13.0，且量化后格式统一")

# ---- 共用的示意权重矩阵（3 行 × 4 列，教学小样例，非精确复刻原图像素）----
W_FP16 = [[1.1, -0.4, 2.6, -1.2],
          [-0.7, 1.3, -2.0, 0.6],
          [0.5, -0.9, 1.8, -0.3]]
ACT = [0.3, 0.4, 9.6, 0.5]          # 每通道激活幅值：通道 2 是显著通道
SALIENT_COL = 2
N_ROW, N_COL = 3, 4

Q_RTN = [[1, 0, 3, -1], [-1, 1, -2, 1], [1, -1, 2, 0]]         # (a) 朴素取整，各列一视同仁
Q_AWQ = [[1, 0, 2, -1], [-1, 1, -2, 1], [1, -1, 2, 0]]          # (c) 缩放后再统一量化，格式与(a)同构

CELL, GAP = 42, 4
GRID_W = N_COL * (CELL + GAP) - GAP
GRID_H = N_ROW * (CELL + GAP) - GAP
ACT_H = 30
ARROW_GAP = 22

PANEL_W = GRID_W + 30
PAD, PANEL_GAP = 40, 46
W = PAD * 2 + PANEL_W * 3 + PANEL_GAP * 2

TOP = 168            # 面板网格顶部 y（激活行 + 箭头都在其上方）
PANEL_H = ACT_H + ARROW_GAP + GRID_H
BADGE_Y = TOP + PANEL_H + 34
NOTE_Y1 = BADGE_Y + 48
NOTE_Y2 = NOTE_Y1 + 18
H = NOTE_Y2 + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{btext(TITLE)}</text>']
# 副标题换行（太长，拆两行）
sub1 = "重绘自 arXiv:2306.00978 Fig.2（OPT-6.7B，INT3-g128）："
sub2 = "中间面板 43.2→13.0 靠给显著通道特殊待遇；右面板 AWQ 用按通道缩放同样拿到 13.0，且量化后格式统一"
L.append(f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(sub1)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+34}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(sub2)}</text>')


def draw_grid(px, gy, cells, col_colors=None):
    """画一个 N_ROW x N_COL 网格；col_colors: 每列一个 (fill,stroke)，None 用默认蓝。"""
    for r in range(N_ROW):
        for c in range(N_COL):
            x = px + c * (CELL + GAP)
            y = gy + r * (CELL + GAP)
            if col_colors is not None:
                fill, stroke = col_colors[c]
            else:
                fill, stroke = "#dbeafe", "#1e3a5f"
            L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
            val = cells[r][c]
            txt = f"{val:.1f}" if isinstance(val, float) else f"{val:+d}" if val != 0 else "0"
            L.append(f'<text x="{x+CELL/2}" y="{y+CELL/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="#0f172a">{esc(txt)}</text>')


def draw_activation_row(px, ay, highlight_col=None):
    """画激活幅值行（1 行 N_COL 列的窄条），高亮列标红。"""
    for c in range(N_COL):
        x = px + c * (CELL + GAP)
        is_hot = (highlight_col is not None and c == highlight_col)
        fill = "#fca5a5" if is_hot else "#e2e8f0"
        stroke = "#b91c1c" if is_hot else "#94a3b8"
        L.append(f'<rect x="{x}" y="{ay}" width="{CELL}" height="{ACT_H}" rx="3" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>')
        L.append(f'<text x="{x+CELL/2}" y="{ay+ACT_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" '
                  f'font-weight="{"bold" if is_hot else "normal"}" '
                  f'fill="{"#991b1b" if is_hot else "#475569"}">{esc(f"{ACT[c]:g}")}</text>')


def panel_x(idx):
    return PAD + idx * (PANEL_W + PANEL_GAP)


# ===== 面板 (a)：RTN 朴素量化 =====
px0 = panel_x(0)
gx0 = px0 + (PANEL_W - GRID_W) / 2
L.append(f'<text x="{px0+PANEL_W/2}" y="{TOP-46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(a) RTN 朴素量化")}</text>')
L.append(f'<text x="{px0+PANEL_W/2}" y="{TOP-28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("各列一视同仁地取整")}</text>')
gy_a = TOP + ACT_H + ARROW_GAP
draw_grid(gx0, gy_a, Q_RTN)
L.append(f'<text x="{px0+PANEL_W/2}" y="{BADGE_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#b91c1c">{esc("PPL 43.2")}</text>')

# ===== 面板 (b)：混合精度保护显著权重 =====
px1 = panel_x(1)
gx1 = px1 + (PANEL_W - GRID_W) / 2
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP-46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(b) 混合精度保护显著权重")}</text>')
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP-28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("按激活挑 1% 通道保 FP16")}</text>')
draw_activation_row(gx1, TOP, highlight_col=SALIENT_COL)
# 箭头：从激活行高亮列 指向 下方网格对应列
arr_x = gx1 + SALIENT_COL * (CELL + GAP) + CELL / 2
L.append(f'<line x1="{arr_x}" y1="{TOP+ACT_H+4}" x2="{arr_x}" y2="{TOP+ACT_H+ARROW_GAP-4}" '
          f'stroke="#b91c1c" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{arr_x+10}" y="{TOP+ACT_H+ARROW_GAP/2+4}" font-family="sans-serif" '
          f'font-size="9.5" fill="#b91c1c">{esc("挑显著通道")}</text>')
mix_cells = [[Q_RTN[r][c] if c != SALIENT_COL else W_FP16[r][c] for c in range(N_COL)]
             for r in range(N_ROW)]
mix_colors = [("#fed7aa", "#c2410c") if c == SALIENT_COL else ("#dbeafe", "#1e3a5f")
              for c in range(N_COL)]
draw_grid(gx1, gy_a, mix_cells, col_colors=mix_colors)
L.append(f'<text x="{gx1+SALIENT_COL*(CELL+GAP)+CELL/2}" y="{gy_a+GRID_H+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="9.5" font-weight="bold" '
          f'fill="#c2410c">{esc("FP16 通道")}</text>')
L.append(f'<text x="{px1+PANEL_W/2}" y="{BADGE_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#047857">{esc("PPL 13.0")}</text>')
L.append(f'<text x="{px1+PANEL_W/2}" y="{BADGE_Y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#b91c1c">{esc("混合精度：INT3+FP16 混排，硬件效率差")}</text>')

# ===== 面板 (c)：AWQ 按通道缩放 =====
px2 = panel_x(2)
gx2 = px2 + (PANEL_W - GRID_W) / 2
L.append(f'<text x="{px2+PANEL_W/2}" y="{TOP-46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(c) AWQ：先缩放再量化")}</text>')
L.append(f'<text x="{px2+PANEL_W/2}" y="{TOP-28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("按激活幅值定缩放 s，再统一量化")}</text>')
draw_activation_row(gx2, TOP, highlight_col=SALIENT_COL)
arr_x2 = gx2 + SALIENT_COL * (CELL + GAP) + CELL / 2
L.append(f'<line x1="{arr_x2}" y1="{TOP+ACT_H+4}" x2="{arr_x2}" y2="{TOP+ACT_H+ARROW_GAP-4}" '
          f'stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{arr_x2+10}" y="{TOP+ACT_H+ARROW_GAP/2+4}" font-family="sans-serif" '
          f'font-size="9.5" fill="#1d4ed8">{esc("s=α 网格搜")}</text>')
draw_grid(gx2, gy_a, Q_AWQ)  # 统一蓝色：量化后与(a)同格式，无需混合精度
L.append(f'<text x="{px2+PANEL_W/2}" y="{gy_a+GRID_H+16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" font-weight="bold" fill="#1d4ed8">{esc("统一 INT3 格式")}</text>')
L.append(f'<text x="{px2+PANEL_W/2}" y="{BADGE_Y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="#047857">{esc("PPL 13.0")}</text>')
L.append(f'<text x="{px2+PANEL_W/2}" y="{BADGE_Y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#047857">{esc("量化前整列缩放，量化后格式统一、硬件友好")}</text>')

# ===== 图注（结论）=====
L.append(f'<text x="{PAD}" y="{NOTE_Y1}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("(a)(b)(c) 均取同一份 3×4 示意权重矩阵；(b)(c) 的激活幅值行相同，通道 2 显著(9.6)——")}</text>')
L.append(f'<text x="{PAD}" y="{NOTE_Y2}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("(b) 靠切换精度保住它，(c) 靠缩放把它挪进同一套整数刻度里，两条路线同精度、不同硬件代价。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-awq-2.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
