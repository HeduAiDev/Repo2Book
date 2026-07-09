#!/usr/bin/env python3
"""paper-fig-smoothquant-4 —— 重绘自 arXiv:2211.10438 (SmoothQuant) Fig.4：
OPT-13B 真实层的激活/权重幅值：少数通道激活幅值 >70 且固定出现在同一批通道，
权重原本平坦；SmoothQuant 迁移后激活 outlier 被抹平，权重在这些通道略微
抬高但仍平滑。informational structure 对齐原图四联(Activation 原始/迁移后、
Weight 原始/迁移后 + 顶部 Smooth/迁移难度两条箭头)，原图是 3D 透视图，这里
改画成信息量相同、更易读的 2D 逐通道柱状图，配色套本书视觉语言，通道位置
与数值为示意(与原图一样是教学可视化，同一组通道位置在四张图里保持一致)。"""
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

TITLE = "极少数通道系统性大上百倍、且固定出现：SmoothQuant 把这个难度从激活迁移给权重"
SUBTITLE = "重绘自 arXiv:2211.10438 Fig.4：OPT-13B 某线性层的激活/权重幅值（原图为 3D 透视图，改画为逐通道柱状图，信息结构不变）"

N_CH = 24
OUTLIER_IDX = {5, 14, 20}

ACT_ORIG = [1.2, 1.8, 2.1, 1.5, 2.4, 74.0, 1.9, 2.2, 1.6, 2.0, 1.4, 1.8, 2.3, 1.7,
            71.0, 2.0, 1.5, 1.9, 2.2, 1.6, 76.0, 1.8, 2.1, 1.4]
ACT_SQ = [3.2, 4.1, 3.8, 3.5, 4.4, 8.6, 3.9, 4.2, 3.6, 4.0, 3.4, 3.8, 4.3, 3.7,
          8.1, 4.0, 3.5, 3.9, 4.2, 3.6, 8.8, 3.8, 4.1, 3.4]
W_ORIG = [0.18, 0.22, 0.16, 0.20, 0.19, 0.21, 0.17, 0.23, 0.18, 0.20, 0.16, 0.19,
          0.22, 0.18, 0.21, 0.19, 0.17, 0.20, 0.18, 0.22, 0.19, 0.17, 0.20, 0.18]
W_SQ = [0.25, 0.30, 0.24, 0.28, 0.26, 0.85, 0.27, 0.31, 0.25, 0.29, 0.24, 0.27,
        0.30, 0.26, 0.80, 0.28, 0.24, 0.27, 0.25, 0.30, 0.88, 0.26, 0.29, 0.24]

PANELS = [
    ("Activation（原始）", ACT_ORIG, 80, "#ef4444", "#94a3b8", "难量化", "#b91c1c"),
    ("Activation（SmoothQuant 后）", ACT_SQ, 10, "#22c55e", "#94a3b8", "易量化", "#047857"),
    ("Weight（原始）", W_ORIG, 0.32, "#22c55e", "#94a3b8", "本来就很易量化", "#047857"),
    ("Weight（SmoothQuant 后）", W_SQ, 1.0, "#f59e0b", "#94a3b8", "略难但仍易量化", "#b45309"),
]

PAD, PANEL_GAP = 40, 26
PANEL_W = 232
BAR_W = (PANEL_W - 20) / N_CH
TOP_ARROWS_H = 74
TITLE_H = 60
PTITLE_H = 34
CHART_H = 230
TAG_H = 22
W = PAD * 2 + PANEL_W * 4 + PANEL_GAP * 3
TOP = TITLE_H + TOP_ARROWS_H
CHART_TOP = TOP + PTITLE_H
CHART_BASE = CHART_TOP + CHART_H
NOTE_Y1 = CHART_BASE + TAG_H + 30
NOTE_Y2 = NOTE_Y1 + 18
NOTE_Y3 = NOTE_Y2 + 18
H = NOTE_Y3 + 26

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
     '<marker id="s" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+2}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']


def panel_x(idx):
    return PAD + idx * (PANEL_W + PANEL_GAP)


# ===== 顶部两条箭头：Smooth（面板1→2） + 迁移量化难度（面板1→4）=====
arrow_y1 = TITLE_H + 18
arrow_y2 = TITLE_H + 42
p0c, p1c = panel_x(0) + PANEL_W / 2, panel_x(1) + PANEL_W / 2
p3c = panel_x(3) + PANEL_W / 2
L.append(f'<line x1="{p0c}" y1="{arrow_y2}" x2="{p3c}" y2="{arrow_y2}" '
          f'stroke="#7c3aed" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(p0c+p3c)/2}" y="{arrow_y2-8}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#7c3aed">{esc("迁移量化难度")}</text>')
L.append(f'<line x1="{p0c}" y1="{arrow_y1}" x2="{p1c}" y2="{arrow_y1}" '
          f'stroke="#0369a1" stroke-width="2" marker-end="url(#s)"/>')
L.append(f'<text x="{(p0c+p1c)/2}" y="{arrow_y1-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" font-weight="bold" fill="#0369a1">{esc("平滑（Smooth）")}</text>')


def draw_panel(idx, name, values, y_max, hot_color, base_color, tag, tag_color):
    px = panel_x(idx)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-8}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    # y 轴框 + 基线
    L.append(f'<line x1="{px+10}" y1="{CHART_TOP}" x2="{px+10}" y2="{CHART_BASE}" '
              f'stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<line x1="{px+10}" y1="{CHART_BASE}" x2="{px+PANEL_W-10}" y2="{CHART_BASE}" '
              f'stroke="#334155" stroke-width="1.2"/>')
    L.append(f'<text x="{px+6}" y="{CHART_TOP+4}" text-anchor="end" font-family="sans-serif" '
              f'font-size="9" fill="#94a3b8">{esc(f"{y_max:g}")}</text>')
    for i, v in enumerate(values):
        x = px + 10 + i * BAR_W
        bar_h = min(v / y_max, 1.0) * CHART_H
        is_hot = i in OUTLIER_IDX
        color = hot_color if is_hot else base_color
        L.append(f'<rect x="{x:.2f}" y="{CHART_BASE-bar_h:.2f}" width="{max(BAR_W-1.2,1):.2f}" '
                  f'height="{bar_h:.2f}" fill="{color}"/>')
    L.append(f'<text x="{cx}" y="{CHART_BASE+8+TAG_H-6}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{tag_color}">{esc(tag)}</text>')


for i, p in enumerate(PANELS):
    draw_panel(i, *p)

# ===== 图例（红/橙=显著通道，灰=普通通道）=====
leg_y = CHART_BASE + TAG_H + 6
# 已通过每图下方 tag 结论化说明；此处仅在图注给出通道对应关系

# ===== 图注（结论）=====
L.append(f'<text x="{PAD}" y="{NOTE_Y1}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("四图取同一份 24 通道示意数据、通道位置对齐：3 个红/橙色通道(第 6/15/21 个)在激活原始图里 >70，")}</text>')
L.append(f'<text x="{PAD}" y="{NOTE_Y2}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("在权重原始图里和其余通道一样平坦——outlier 只长在激活上、且固定长在同几个通道，跨 token 稳定出现。")}</text>')
L.append(f'<text x="{PAD}" y="{NOTE_Y3}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("SmoothQuant 迁移后：这几个通道的激活被压回个位数，权重在同样几个通道略微抬高——量化难度被搬了家，两边都变得可控。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-smoothquant-4.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
