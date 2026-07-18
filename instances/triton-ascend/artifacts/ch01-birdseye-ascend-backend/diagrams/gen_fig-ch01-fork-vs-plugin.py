#!/usr/bin/env python3
"""fig-ch01-fork-vs-plugin — before-after 模板。
左：OOT 插件靠注册表顶替、一行上游源码不改。右：fork 让上游 Triton 3.2.0 整树在内、
昇腾增量原位加在 third_party/ascend/。右侧高亮两处 fork 血统证据。
坐标全部由循环/常量计算，文本全部 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANELS = [
    ("OOT 插件（如 vllm-ascend）", [
        "上游源码：0 行改动",
        "注册表：entry_points 顶替",
        "换编译下降链？做不到",
    ], None),
    ("triton-ascend：fork", [
        "上游 Triton 3.2.0 整树在内",
        "third_party/ascend/ 原位加量",
        "backend/compiler.py 直接\nimport 上游 triton._C.libtriton",
    ], (1, 2)),  # 高亮下标（右侧第 1、2 行是 fork 血统证据）
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 260, 50, 24, 320, 40, 76
n_steps = len(PANELS[0][1])
w = PAD * 2 + PANEL_W * 2 + 90
content_bottom = TOP + n_steps * BOX_H + (n_steps - 1) * VGAP
h = content_bottom + 34 + 2 * 16 + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("fork 而非插件：三支柱之一")}</text>']

for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = hot is not None and i in hot
        lines = step.split("\n")
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2.4 if hl else 1}"/>')
        n = len(lines)
        y0 = y + BOX_H/2 - (n-1)*8 + 4
        for k, ln in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*15}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="12" fill="#0f172a">{esc(ln)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                     'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + (n_steps * (BOX_H + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+80}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+45}" y="{midy-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#b45309">{esc("整树 fork")}</text>')

foot_y = content_bottom + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("血统证据不在 backend/compiler.py 自己的版权头（华为单版权）——")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+16}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("而在 tutorials/01-vector-add.py:L1-L3 的 3 行双版权头(Huawei+Tillet+OpenAI)与 compiler.py:L34 的 import 语句。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-fork-vs-plugin.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
