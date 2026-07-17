#!/usr/bin/env python3
"""figure: rescale-identity-with-without (before-after 模板改)
claim: 同一 2 块输入:带 rescale(旧 l/acc 乘 alpha=0.367879)得正确 O=[1.462117,1.337835]、
与全矩阵 softmax 逐位相等;漏掉 rescale 直接累加得错误 O=[1.231059,1.0]——alpha 是恒等性的全部。
数据来源: explainer/explainer.json mechanism m04-rescale-identity-alpha
(explainer/traces/tiling_rescale.json)。全坐标计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "rescale 是恒等性的全部:带 alpha vs 漏 alpha"
SUBTITLE = "同一 2 块输入(块 1 ℓ=1.367879, acc=[1.0,0.367879]);块 2 running max 从 1 抬到 2 时,ℓ/acc 是否乘 alpha=0.367879"

PANELS = [
    {
        "title": "带 rescale(正确)",
        "steps": [
            ("块2: alpha=e^{1-2}=0.367879", False),
            ("旧 ℓ·alpha + rowsum(P̃) = 1.871094", True),
            ("旧 acc·alpha + P̃V = [2.735759, 2.503215]", True),
            ("O = acc/ℓ = [1.462117, 1.337835]", False),
        ],
        "verdict": "✓ 与全矩阵 softmax 逐位相等",
        "verdict_ok": True,
    },
    {
        "title": "漏掉 rescale(错误)",
        "steps": [
            ("块2: alpha 未施加", False),
            ("旧 ℓ 直接 + rowsum(P̃) = 2.735759", True),
            ("旧 acc 直接 + P̃V = [3.367879, 2.735759]", True),
            ("O = acc/ℓ = [1.231059, 1.0]", False),
        ],
        "verdict": "✗ 偏离全矩阵 softmax(两分量偏 15.8%/25.3%)",
        "verdict_ok": False,
    },
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 320, 50, 24, 380, 44, 132
w = PAD * 2 + PANEL_W * 2 + 90
n_steps = len(PANELS[0]["steps"])
h = TOP + n_steps * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for p, panel in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    is_ok = panel["verdict_ok"]
    head_fill = "#ecfdf5" if is_ok else "#fef2f2"
    head_stroke = "#047857" if is_ok else "#b91c1c"
    L.append(f'<rect x="{px}" y="{TOP-10}" width="{PANEL_W}" height="34" rx="6" '
              f'fill="{head_fill}" stroke="{head_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{TOP+12}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" '
              f'fill="{head_stroke}">{esc(panel["title"])}</text>')
    step_top = TOP + 44
    for i, (step, hot) in enumerate(panel["steps"]):
        y = step_top + i * (BOX_H + VGAP)
        fill = "#fef3c7" if hot else "#e2e8f0"
        stroke = "#d97706" if hot else "#64748b"
        sw = 2 if hot else 1
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#0f172a">{esc(step)}</text>')
        if i < len(panel["steps"]) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    verdict_y = step_top + n_steps * (BOX_H + VGAP) - VGAP + 34
    v_fill = "#ecfdf5" if is_ok else "#fef2f2"
    v_stroke = "#047857" if is_ok else "#b91c1c"
    L.append(f'<rect x="{px}" y="{verdict_y}" width="{PANEL_W}" height="40" rx="6" '
              f'fill="{v_fill}" stroke="{v_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{verdict_y+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" '
              f'fill="{v_stroke}">{esc(panel["verdict"])}</text>')

midy = TOP + 44 + (n_steps * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+80}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-12}" text-anchor="middle" '
         'font-family="sans-serif" font-size="11" font-weight="bold" '
         f'fill="#d97706">{esc("唯一差异")}</text>')

foot_y = h - 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("黄底=两路径唯一的分岔点(是否对旧 ℓ/acc 乘 alpha);其余步骤完全相同。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc("锚点 python/tutorials/06-fused-attention.py:L62(alpha)、L63(ℓ 乘 alpha)、L65(acc 乘 alpha)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("rescale-identity-with-without.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
