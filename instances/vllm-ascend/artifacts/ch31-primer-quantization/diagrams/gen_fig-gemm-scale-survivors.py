#!/usr/bin/env python3
"""fig-gemm-scale-survivors — INT8 GEMM 反量化的结构本质（SmoothQuant Eq.2）。
论点：反量化 scale 只能乘在输出里存活的两个外维（T 行、C_o 列）；收缩维 C_i 一经
求和便抹掉通道身份，激活 per-channel scale 数学最优却拆不回来、插不进去。
玩具维度 T=2、C_i=3（通道 2 为 outlier）；被划掉的 per-channel scale=[0.0012,0.0016,0.0787]。
数字全部来自 figure-requests.json 条目 numbers（带溯源）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

_BOLD_BREAK = {"量"}
def btext(s):
    parts, buf = [], ""
    for ch in s:
        if ch in _BOLD_BREAK:
            if buf:
                parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>'); buf = ""
            parts.append(f'<tspan font-weight="normal">{esc(ch)}</tspan>')
        else:
            buf += ch
    if buf:
        parts.append(f'<tspan font-weight="bold">{esc(buf)}</tspan>')
    return "".join(parts)

W, H = 1140, 620
CELL = 40
MY = 250          # 矩阵竖直中心
GREEN, GREEN_D = "#dcfce7", "#16a34a"
RED, RED_D = "#fee2e2", "#dc2626"
BLUE_D = "#2563eb"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# ---- 标题 ----
L.append(f'<text x="50" y="44" font-family="sans-serif" font-size="23" fill="#1e40af">'
         f'{btext("反量化 scale 只能乘在存活的外维——收缩维 C_i 求和后拆不回来")}</text>')
L.append(f'<text x="50" y="72" font-family="sans-serif" font-size="14" fill="#475569">'
         f'{esc("激活 per-channel scale 数学最优，但通道身份在 GEMM 求和里被抹掉，插不进 INT8 GEMM")}</text>')

def matrix(x, nrow, ncol, cellfn, box_stroke="#64748b", box_fill=None):
    """在竖直中心 MW 处画 nrow×ncol 网格，返回 (left, top, w, h)。cellfn(i,j)->(text,fill,tcolor)."""
    wmat, hmat = ncol * CELL, nrow * CELL
    top = MY - hmat / 2
    if box_fill:
        L.append(f'<rect x="{x-4}" y="{top-4}" width="{wmat+8}" height="{hmat+8}" rx="6" '
                 f'fill="{box_fill}" stroke="{box_stroke}" stroke-width="2"/>')
    for i in range(nrow):
        for j in range(ncol):
            cx, cy = x + j * CELL, top + i * CELL
            txt, fill, tcol = cellfn(i, j)
            L.append(f'<rect x="{cx}" y="{cy}" width="{CELL}" height="{CELL}" '
                     f'fill="{fill}" stroke="#cbd5e1" stroke-width="1"/>')
            L.append(f'<text x="{cx+CELL/2}" y="{cy+CELL/2+5}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="13" fill="{tcol}">{esc(txt)}</text>')
    return x, top, wmat, hmat

def op(x, sym):
    L.append(f'<text x="{x}" y="{MY+7}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="26" fill="#334155">{esc(sym)}</text>')

def label(cx, top, main, sub, color="#334155"):
    L.append(f'<text x="{cx}" y="{top-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" fill="{color}" font-weight="bold">{esc(main)}</text>')
    if sub:
        L.append(f'<text x="{cx}" y="{top-13}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="11.5" fill="#64748b">{esc(sub)}</text>')

# ---------- 布局：Y = diag(ΔX) · X̄ · W̄ · diag(ΔW) ----------
x = 55
op(x, "Y")
L.append(f'<text x="{x}" y="{MY+34}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("T×C_o")}</text>')
x += 34
op(x, "=")
x += 34

# diag(Δ_X)：沿 T 行存活（绿）
def diagX(i, j):
    if i == j: return ("Δ", GREEN, GREEN_D)
    return ("0", "#f8fafc", "#cbd5e1")
lx, top, wmat, hmat = matrix(x, 2, 2, diagX, box_stroke=GREEN_D, box_fill="white")
label(lx+wmat/2, top, "diag(Δ_X)", "沿 T 行 · 存活", GREEN_D)
L.append(f'<text x="{lx+wmat/2}" y="{top+hmat+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{GREEN_D}" font-weight="bold">{esc("scale 存活 ✓")}</text>')
x = lx + wmat + 24
op(x, "·"); x += 24

# X̄^INT8 (T×C_i)，通道 2 = outlier
def barX(i, j):
    if j == 2: return ("·", RED, RED_D)
    return ("·", "#eff6ff", BLUE_D)
lx, top, wmat, hmat = matrix(x, 2, 3, barX, box_stroke="#94a3b8", box_fill="white")
label(lx+wmat/2, top, "X̄  (T×C_i)", "T=2 token", "#334155")
# C_i 列头
ci_names = ["通道0", "通道1", "通道2*"]
for j, nm in enumerate(ci_names):
    col = RED_D if j == 2 else "#64748b"
    L.append(f'<text x="{lx+j*CELL+CELL/2}" y="{top+hmat+16}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" fill="{col}">{esc(nm)}</text>')
xbar_right = lx + wmat
x = xbar_right + 26
op(x, "·"); x += 26

# W̄^INT8 (C_i×C_o)
def barW(i, j):
    if i == 2: return ("·", RED, RED_D)
    return ("·", "#eff6ff", BLUE_D)
lx, top, wmat, hmat = matrix(x, 3, 2, barW, box_stroke="#94a3b8", box_fill="white")
label(lx+wmat/2, top, "W̄  (C_i×C_o)", "C_i=3", "#334155")
xw_left = lx
x = lx + wmat + 24
op(x, "·"); x += 24

# diag(Δ_W)：沿 C_o 列存活（绿）
def diagW(i, j):
    if i == j: return ("Δ", GREEN, GREEN_D)
    return ("0", "#f8fafc", "#cbd5e1")
lx, top, wmat, hmat = matrix(x, 2, 2, diagW, box_stroke=GREEN_D, box_fill="white")
label(lx+wmat/2, top, "diag(Δ_W)", "沿 C_o 列 · 存活", GREEN_D)
L.append(f'<text x="{lx+wmat/2}" y="{top+hmat+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{GREEN_D}" font-weight="bold">{esc("scale 存活 ✓")}</text>')

# ---------- C_i 收缩：∑ 抹掉通道身份（红）----------
mid = (xbar_right + xw_left) / 2
# 单行说明置于所有矩阵标签之上，避免与 W̄ 标题相撞
L.append(f'<text x="{mid}" y="128" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" fill="{RED_D}" font-weight="bold">{esc("沿 C_i 求和 → 通道身份消失")}</text>')
# ∑ 符号落在 X̄ / W̄ 之间的空档
L.append(f'<text x="{mid}" y="{MY-46}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="30" fill="{RED_D}">{esc("∑")}</text>')
# 红括弧连 X̄ 右缘与 W̄ 左缘
L.append(f'<path d="M{xbar_right} {MY-22} Q {mid} {MY-34} {xw_left} {MY-22}" '
         f'fill="none" stroke="{RED_D}" stroke-width="1.6" stroke-dasharray="4 3"/>')

# ---------- 被划掉的 per-channel scale（想插在 C_i 处）----------
py = 460
CELL_S = 60
pw = 3 * CELL_S
px = mid - pw / 2
vals = ["0.0012", "0.0016", "0.0787"]
for j, v in enumerate(vals):
    cx = px + j * CELL_S
    L.append(f'<rect x="{cx}" y="{py}" width="{CELL_S}" height="{CELL-4}" rx="3" '
             f'fill="{RED}" stroke="{RED_D}" stroke-width="1.4"/>')
    L.append(f'<text x="{cx+CELL_S/2}" y="{py+23}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" fill="{RED_D}">{esc(v)}</text>')
# 红叉（细、半透明，压在整条上但不遮数字）
L.append(f'<line x1="{px-6}" y1="{py-6}" x2="{px+pw+6}" y2="{py+CELL-2}" stroke="{RED_D}" '
         f'stroke-width="2.2" stroke-opacity="0.55"/>')
L.append(f'<line x1="{px-6}" y1="{py+CELL-2}" x2="{px+pw+6}" y2="{py-6}" stroke="{RED_D}" '
         f'stroke-width="2.2" stroke-opacity="0.55"/>')
# 虚线箭头：想乘在 C_i 收缩处（指向 ∑ 空档）
L.append(f'<line x1="{mid}" y1="{py-8}" x2="{mid}" y2="{MY+18}" stroke="{RED_D}" stroke-width="1.6" '
         f'stroke-dasharray="5 4" marker-end="url(#a)"/>')
L.append(f'<text x="{mid}" y="{py+CELL+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" fill="{RED_D}" font-weight="bold">'
         f'{esc("想要的激活 per-channel scale = [0.0012, 0.0016, 0.0787]")}</text>')
L.append(f'<text x="{mid}" y="{py+CELL+42}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="{RED_D}">'
         f'{esc("diag(scale_Ci) 想插在 C_i 这里——求和已抹掉通道身份，插不进去 ✗")}</text>')

# ---------- 图注 ----------
L.append(f'<text x="50" y="565" font-family="sans-serif" font-size="12.5" fill="#475569">'
         f'{esc("Y = diag(Δ_X)·(X̄^INT8 W̄^INT8)·diag(Δ_W)，X∈R^{T×C_i}，W∈R^{C_i×C_o}（SmoothQuant §3 Eq.2，arXiv:2211.10438）。")}</text>')
L.append(f'<text x="50" y="588" font-family="sans-serif" font-size="12.5" fill="#475569">'
         f'{esc("绿 = scale 存活的外维（T 行 / C_o 列，可在 GEMM 外还原）；红 = 收缩维 C_i，求和后通道身份消失，per-channel scale 拆不回来。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-gemm-scale-survivors.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
