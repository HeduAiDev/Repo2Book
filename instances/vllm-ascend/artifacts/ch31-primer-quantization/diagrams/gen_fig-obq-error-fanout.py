#!/usr/bin/env python3
"""fig-obq-error-fanout — GPTQ/OBQ 二阶补偿的方向本质（OBQ §3 Eq.2）。
论点：补偿方向 = H_F^{-1} 第 q 列；w_q 量化锁定后，其取整误差按这一列的比例
扇给所有尚未量化的邻居权重。数字全部来自 explainer trace M4（带溯源）。"""
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

W, H = 1000, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#ea580c"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

ORANGE, ORANGE_D = "#fed7aa", "#ea580c"
RED, RED_D = "#fee2e2", "#dc2626"
GREEN_D = "#16a34a"

# ---- 标题 ----
L.append(f'<text x="50" y="44" font-family="sans-serif" font-size="22" fill="#1e40af">'
         f'{btext("二阶补偿的方向 = H⁻¹ 第 q 列：锁定权重的取整误差按这一列扇给邻居")}</text>')
L.append(f'<text x="50" y="72" font-family="sans-serif" font-size="14" fill="#475569">'
         f'{esc("不是逐个取整，而是让整层输出尽量不变——先量化权重的误差有方向地摊派出去")}</text>')

# ---- H^{-1} 第 q 列方向标注 ----
L.append(f'<text x="500" y="140" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="{ORANGE_D}" font-weight="bold">'
         f'{esc("扇出比例来自 H_F⁻¹ 第 1 列（q=1）")}</text>')

# ---- 三个权重盒 ----
BW, BH, BY = 196, 100, 200
cx = {"w0": 205, "w1": 500, "w2": 795}

def box(name, cxv, lines, fill, stroke, tcol):
    x = cxv - BW / 2
    L.append(f'<rect x="{x}" y="{BY}" width="{BW}" height="{BH}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>')
    n = len(lines)
    for i, (txt, sz, col, bold) in enumerate(lines):
        yy = BY + BH / 2 - (n - 1) * 12 + i * 24
        wattr = ' font-weight="bold"' if bold else ''
        body = btext(txt) if bold else esc(txt)
        L.append(f'<text x="{cxv}" y="{yy}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="{sz}"{wattr} fill="{col}">{body}</text>')

# 中间：已量化锁定的 w1（误差源）
box("w1", cx["w1"],
    [("w_1 = -0.812", 15, "#334155", True),
     ("量化 → -0.6693", 14, RED_D, False),
     ("取整误差 e_q = -0.1427", 13.5, RED_D, True)],
    RED, RED_D, RED_D)
# 两侧：接收补偿的邻居
box("w0", cx["w0"],
    [("w_0 = -0.6", 15, "#334155", True),
     ("（尚未量化）", 12.5, "#64748b", False)],
    "#eff6ff", "#2563eb", "#334155")
box("w2", cx["w2"],
    [("w_2 = 0.192", 15, "#334155", True),
     ("（尚未量化）", 12.5, "#64748b", False)],
    "#eff6ff", "#2563eb", "#334155")

# ---- 扇出箭头（从 w1 两侧射向邻居）----
ay = BY + BH / 2
# 左：w1 左缘 → w0 右缘
L.append(f'<line x1="{cx["w1"]-BW/2-4}" y1="{ay}" x2="{cx["w0"]+BW/2+6}" y2="{ay}" '
         f'stroke="{ORANGE_D}" stroke-width="2.6" marker-end="url(#o)"/>')
lmid = (cx["w1"]-BW/2 + cx["w0"]+BW/2) / 2
L.append(f'<text x="{lmid}" y="{ay-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" fill="{ORANGE_D}" font-weight="bold">{esc("δ = -0.0297")}</text>')
# 右：w1 右缘 → w2 左缘
L.append(f'<line x1="{cx["w1"]+BW/2+4}" y1="{ay}" x2="{cx["w2"]-BW/2-6}" y2="{ay}" '
         f'stroke="{ORANGE_D}" stroke-width="2.6" marker-end="url(#o)"/>')
rmid = (cx["w1"]+BW/2 + cx["w2"]-BW/2) / 2
L.append(f'<text x="{rmid}" y="{ay-12}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" fill="{ORANGE_D}" font-weight="bold">{esc("δ = -0.0955")}</text>')

# ---- 补偿后结果（盒下方）----
ry = BY + BH + 30
L.append(f'<text x="{cx["w0"]}" y="{ry}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="{GREEN_D}" font-weight="bold">{esc("补偿后 → -0.6297")}</text>')
L.append(f'<text x="{cx["w1"]}" y="{ry}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" fill="#64748b">{esc("自身 δ = 0（已锁定）")}</text>')
L.append(f'<text x="{cx["w2"]}" y="{ry}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="{GREEN_D}" font-weight="bold">{esc("补偿后 → 0.0965")}</text>')

# ---- δ_F 全向量 + 公式 ----
L.append(f'<text x="500" y="400" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="{ORANGE_D}" font-weight="bold">'
         f'{esc("δ_F = [-0.0297, 0.0, -0.0955]   （→ w0 / 自身 / w2）")}</text>')
L.append(f'<text x="500" y="438" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" fill="#334155">'
         f'{esc("δ_F = -( (w_q - quant(w_q)) / [H_F⁻¹]_qq ) · (H_F⁻¹)_{:,q}")}</text>')
L.append(f'<text x="500" y="460" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#94a3b8">{esc("OBQ §3 Eq.2，arXiv:2210.17323")}</text>')

# ---- 结论图注 ----
L.append(f'<text x="50" y="508" font-family="sans-serif" font-size="13" fill="#475569">'
         f'{esc("w1 锁定到量化网格（-0.6693，取整误差 -0.1427）后，误差不是丢掉，而是按 H_F⁻¹ 第 1 列的比例扇给两个尚未量化的邻居：")}</text>')
L.append(f'<text x="50" y="530" font-family="sans-serif" font-size="13" fill="#475569">'
         f'{esc("w0 被推到 -0.6297、w2 被推到 0.0965，让整层输出尽量不变——方向由 Hessian 决定，不是逐权重就近取整。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-obq-error-fanout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
