#!/usr/bin/env python3
"""paper-fig-awq-3 —— 重绘自 arXiv:2306.00978 (AWQ) Fig.3：
Llama-2-7B/RTX4090 瓶颈三联图。(a) 生成阶段远比 context 阶段慢(饼图)；
(b) 生成阶段是访存瓶颈、算术强度极低，W4A16 把算术强度提到 4，峰值算力
提高 4 倍(roofline)；(c) 权重访存量远大于激活(对数柱状图)。三联信息结构
对齐原图，配色套本书视觉语言，数值取自原图标注(饼图 10ms/310ms、roofline
1/4/165 TFLOPS 折点、柱状图 134/1.7/271/0.2 MB)。"""
import math
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

TITLE = "为什么 AWQ/GPTQ 都选权重-only(W4A16)：生成阶段是访存瓶颈，权重访存量又远大于激活"
SUBTITLE = "重绘自 arXiv:2306.00978 Fig.3（Llama-2-7B / RTX 4090）"

PAD, PANEL_GAP = 40, 56
PANEL_W = 400
TOP = 140
PANEL_H = 340
W = PAD * 2 + PANEL_W * 3 + PANEL_GAP * 2
FOOT_Y1 = TOP + PANEL_H + 66
FOOT_Y2 = FOOT_Y1 + 18
H = FOOT_Y2 + 26

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{btext(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']


def panel_x(idx):
    return PAD + idx * (PANEL_W + PANEL_GAP)


# ============================================================
# (a) 饼图：context 200 token=10ms 远小于 generation 20 token=310ms
# ============================================================
px0 = panel_x(0)
cx0 = px0 + PANEL_W / 2
cy0 = TOP + 170
R = 80
CONTEXT_MS, GEN_MS = 10, 310
TOTAL_MS = CONTEXT_MS + GEN_MS

L.append(f'<text x="{cx0}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(a) 生成阶段远慢于 context 阶段")}</text>')


def pie_path(cx, cy, r, start_deg, end_deg):
    s = math.radians(start_deg - 90)
    e = math.radians(end_deg - 90)
    x1, y1 = cx + r * math.cos(s), cy + r * math.sin(s)
    x2, y2 = cx + r * math.cos(e), cy + r * math.sin(e)
    large = 1 if (end_deg - start_deg) > 180 else 0
    return f'M{cx},{cy} L{x1:.2f},{y1:.2f} A{r},{r} 0 {large} 1 {x2:.2f},{y2:.2f} Z'


gen_deg = 360 * GEN_MS / TOTAL_MS
L.append(f'<path d="{pie_path(cx0, cy0, R, 0, gen_deg)}" fill="#ef4444" stroke="white" stroke-width="2"/>')
L.append(f'<path d="{pie_path(cx0, cy0, R, gen_deg, 360)}" fill="#64748b" stroke="white" stroke-width="2"/>')

# 引出标注：generation 切片（大）
mid_gen = math.radians(gen_deg / 2 - 90)
lx = cx0 + (R + 30) * math.cos(mid_gen)
ly = cy0 + (R + 30) * math.sin(mid_gen)
L.append(f'<text x="{lx}" y="{ly-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#b91c1c">{esc("310 ms")}</text>')
L.append(f'<text x="{lx}" y="{ly+12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("生成 20 token")}</text>')
# 引出标注：context 切片（小），用引线拉到外面避免和小扇形文字重叠（同时留足与上方标题的间距）
mid_ctx = math.radians(gen_deg + (360 - gen_deg) / 2 - 90)
sx = cx0 + R * math.cos(mid_ctx)
sy = cy0 + R * math.sin(mid_ctx)
ex = cx0 + (R + 45) * math.cos(mid_ctx)
ey = cy0 + (R + 45) * math.sin(mid_ctx)
L.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
          f'stroke="#334155" stroke-width="1.2"/>')
L.append(f'<text x="{ex:.1f}" y="{ey-6:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#334155">{esc("10 ms")}</text>')
L.append(f'<text x="{ex:.1f}" y="{ey+10:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#64748b">{esc("context 200 token")}</text>')

L.append(f'<text x="{cx0}" y="{TOP+PANEL_H-20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("对端上交互式生成：单步生成比处理提示词慢得多")}</text>')

# ============================================================
# (b) roofline：算术强度 vs 峰值 TFLOPS，165 处出现折点
# ============================================================
px1 = panel_x(1)
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(b) 生成阶段被访存卡死，算术强度极低")}</text>')

X_MAX, Y_MAX, KNEE = 300, 180, 165
CHART_W, CHART_H = 300, 210
BX0 = px1 + 62
BY0 = TOP + 24
BY_BASE = BY0 + CHART_H


def bx(v):
    return BX0 + v / X_MAX * CHART_W


def by(v):
    return BY_BASE - v / Y_MAX * CHART_H


# 网格 + 轴刻度
for gx in (75, 150, 225, 300):
    L.append(f'<line x1="{bx(gx):.1f}" y1="{BY0}" x2="{bx(gx):.1f}" y2="{BY_BASE}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{bx(gx):.1f}" y="{BY_BASE+16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.5" fill="#64748b">{gx}</text>')
for gy in (36, 72, 108, 144, 180):
    L.append(f'<line x1="{BX0}" y1="{by(gy):.1f}" x2="{BX0+CHART_W}" y2="{by(gy):.1f}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{BX0-8}" y="{by(gy)+3:.1f}" text-anchor="end" '
              f'font-family="sans-serif" font-size="9.5" fill="#64748b">{gy}</text>')
L.append(f'<line x1="{BX0}" y1="{BY_BASE}" x2="{BX0+CHART_W}" y2="{BY_BASE}" stroke="#334155" stroke-width="1.3"/>')
L.append(f'<line x1="{BX0}" y1="{BY0}" x2="{BX0}" y2="{BY_BASE}" stroke="#334155" stroke-width="1.3"/>')
L.append(f'<text x="{BX0+CHART_W/2}" y="{BY_BASE+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("算术强度 (FLOPs/Byte)")}</text>')
L.append(f'<text x="{BX0-42}" y="{BY0-6}" text-anchor="start" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("峰值 TFLOPS")}</text>')

# roofline：0→(165,165) 上升，之后打平到 300
L.append(f'<path d="M{bx(0):.1f},{by(0):.1f} L{bx(KNEE):.1f},{by(KNEE):.1f} '
          f'L{bx(X_MAX):.1f},{by(KNEE):.1f}" fill="none" stroke="#1d4ed8" stroke-width="2.4"/>')
L.append(f'<circle cx="{bx(KNEE):.1f}" cy="{by(KNEE):.1f}" r="4" fill="#1d4ed8"/>')
L.append(f'<text x="{bx(KNEE):.1f}" y="{by(KNEE)-10:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="9.5" fill="#1d4ed8">{esc("165(算力打满)")}</text>')

# 生成阶段两点：算术强度 1(1 TFLOPS,W16A16) 与 4(4 TFLOPS,W4A16)，都紧贴原点——用引线拉出标注
gx1, gy1 = bx(1), by(1)
gx4, gy4 = bx(4), by(4)
L.append(f'<circle cx="{gx1:.1f}" cy="{gy1:.1f}" r="3.6" fill="#b91c1c"/>')
L.append(f'<circle cx="{gx4:.1f}" cy="{gy4:.1f}" r="3.6" fill="#047857"/>')
lbl_x, lbl_y = BX0 + 90, BY0 + 34
L.append(f'<line x1="{gx1:.1f}" y1="{gy1:.1f}" x2="{lbl_x-6}" y2="{lbl_y+40}" '
          f'stroke="#b91c1c" stroke-width="1" stroke-dasharray="2,2"/>')
L.append(f'<line x1="{gx4:.1f}" y1="{gy4:.1f}" x2="{lbl_x-6}" y2="{lbl_y+16}" '
          f'stroke="#047857" stroke-width="1" stroke-dasharray="2,2"/>')
L.append(f'<text x="{lbl_x}" y="{lbl_y}" font-family="sans-serif" font-size="10.5" '
          f'font-weight="bold" fill="#0f172a">{esc("生成阶段（访存瓶颈）")}</text>')
L.append(f'<text x="{lbl_x}" y="{lbl_y+18}" font-family="sans-serif" font-size="10" '
          f'fill="#047857">{esc("W4A16：算术强度 4 → 4 TFLOPS")}</text>')
L.append(f'<text x="{lbl_x}" y="{lbl_y+36}" font-family="sans-serif" font-size="10" '
          f'fill="#b91c1c">{esc("W16A16：算术强度 1 → 1 TFLOPS")}</text>')
L.append(f'<text x="{lbl_x}" y="{lbl_y+58}" font-family="sans-serif" font-size="10" '
          f'fill="#64748b">{esc("同一条访存斜率线上，权重变 4 bit 直接把峰值抬高 4×")}</text>')

L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP+PANEL_H-20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc("context 阶段算术强度 ≥165，已在算力打满的平台段")}</text>')

# ============================================================
# (c) 对数柱状图：权重访存量远大于激活
# ============================================================
px2 = panel_x(2)
L.append(f'<text x="{px2+PANEL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("(c) 权重访存量远大于激活")}</text>')

GROUPS = [("Attention", 134, 1.7, "79×"), ("FFN", 271, 0.2, "1700×")]
LOG_LO, LOG_HI = -2, 3  # 10^-2 .. 10^3
CX0 = px2 + 70
CY_BASE = TOP + 24 + 210
CHART_H2 = 210
CHART_W2 = 300


def logy(v):
    lv = math.log10(v)
    frac = (lv - LOG_LO) / (LOG_HI - LOG_LO)
    return CY_BASE - frac * CHART_H2


for p in range(LOG_LO, LOG_HI + 1):
    yy = logy(10 ** p)
    L.append(f'<line x1="{CX0}" y1="{yy:.1f}" x2="{CX0+CHART_W2}" y2="{yy:.1f}" '
              f'stroke="#e2e8f0" stroke-width="1"/>')
    sup = {-2: "⁻²", -1: "⁻¹", 0: "⁰", 1: "¹", 2: "²", 3: "³"}[p]
    L.append(f'<text x="{CX0-8}" y="{yy+3:.1f}" text-anchor="end" font-family="sans-serif" '
              f'font-size="9.5" fill="#64748b">{esc(f"10{sup}")}</text>')
L.append(f'<line x1="{CX0}" y1="{logy(10**LOG_LO):.1f}" x2="{CX0}" y2="{logy(10**LOG_HI):.1f}" '
          f'stroke="#334155" stroke-width="1.3"/>')
base_y = logy(10 ** LOG_LO)
L.append(f'<line x1="{CX0}" y1="{base_y:.1f}" x2="{CX0+CHART_W2}" y2="{base_y:.1f}" '
          f'stroke="#334155" stroke-width="1.3"/>')
L.append(f'<text x="{CX0-42}" y="{logy(10**LOG_HI)-10:.1f}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">{esc("显存占用(MB)")}</text>')

BAR_W2, BAR_GAP2, GROUP_GAP2 = 34, 8, 70
gx_cursor = CX0 + 30
for name, w_mb, a_mb, ratio in GROUPS:
    wy, ay = logy(w_mb), logy(a_mb)
    L.append(f'<rect x="{gx_cursor}" y="{wy:.1f}" width="{BAR_W2}" height="{base_y-wy:.1f}" '
              f'fill="#991b1b" stroke="#7f1d1d" stroke-width="1"/>')
    L.append(f'<text x="{gx_cursor+BAR_W2/2}" y="{wy-8:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="#7f1d1d">{esc(f"{w_mb:g}")}</text>')
    ax = gx_cursor + BAR_W2 + BAR_GAP2
    L.append(f'<rect x="{ax}" y="{ay:.1f}" width="{BAR_W2}" height="{base_y-ay:.1f}" '
              f'fill="#94a3b8" stroke="#475569" stroke-width="1"/>')
    L.append(f'<text x="{ax+BAR_W2/2}" y="{ay-8:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="#334155">{esc(f"{a_mb:g}")}</text>')
    # 组名 + 倍率
    gcx = gx_cursor + BAR_W2 + BAR_GAP2 / 2
    L.append(f'<text x="{gcx}" y="{base_y+18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(name)}</text>')
    L.append(f'<text x="{gcx}" y="{wy-24:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="#b45309">{esc(ratio)}</text>')
    gx_cursor += 2 * BAR_W2 + BAR_GAP2 + GROUP_GAP2

# 图例
leg_y = base_y + 40
L.append(f'<rect x="{CX0}" y="{leg_y}" width="14" height="14" rx="2" fill="#991b1b" stroke="#7f1d1d"/>')
L.append(f'<text x="{CX0+20}" y="{leg_y+12}" font-family="sans-serif" font-size="10.5" '
          f'fill="#334155">{esc("权重")}</text>')
L.append(f'<rect x="{CX0+70}" y="{leg_y}" width="14" height="14" rx="2" fill="#94a3b8" stroke="#475569"/>')
L.append(f'<text x="{CX0+90}" y="{leg_y+12}" font-family="sans-serif" font-size="10.5" '
          f'fill="#334155">{esc("激活")}</text>')

# ===== 图注 =====
L.append(f'<text x="{PAD}" y="{FOOT_Y1}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("(a)(b) 说明在端上交互式生成里，逐 token 生成远慢于处理提示词、且被访存带宽卡死——W4A16 把峰值算力从 1 抬到 4 TFLOPS；")}</text>')
L.append(f'<text x="{PAD}" y="{FOOT_Y2}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("(c) 进一步给出硬件账本：权重访存量比激活大 79～1700 倍，只压缩权重(权重-only)已经能吃掉绝大部分访存开销。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-awq-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
