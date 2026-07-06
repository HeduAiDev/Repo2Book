#!/usr/bin/env python3
"""fig36-4-tree-multiround: flow 模板。树节点上的多轮投机采样——
候选1 被拒后,目标分布调成残差,再试候选2(接受)。对比链式在此止步 0 接受。
数字来自 explainer.json fig36-4 numbers(traces/tree_verify.json + paper.md L528/Table 5)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROUNDS = [
    {
        "title": "轮 1：候选 token 2",
        "lines": ["p(t)=0.15  p̂(t)=0.7", "ratio = min(1,p/p̂) = 0.214", "u = 0.9  →  拒绝"],
        "verdict": "reject",
    },
    {
        "title": "轮 2：候选 token 0",
        "lines": ["p ← norm(max(0,p−p̂))", "ratio = min(1,p/p̂) = 0.992", "u = 0.2  →  接受"],
        "verdict": "accept",
    },
]
BOX_W, BOX_H, HGAP, PAD, TOP = 300, 110, 150, 44, 100
n = len(ROUNDS)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
h = TOP + BOX_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD-8}" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc("树节点上的多轮投机采样：第一个候选被拒不散场")}</text>']

xpos = [PAD + i * (BOX_W + HGAP) for i in range(n)]
for i, (x, r) in enumerate(zip(xpos, ROUNDS)):
    reject = r["verdict"] == "reject"
    fill = "#fee2e2" if reject else "#dcfce7"
    stroke = "#b91c1c" if reject else "#15803d"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(r["title"])}</text>')
    for k, line in enumerate(r["lines"]):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+44+k*20}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" fill="{stroke}">{esc(line)}</text>')
    if i < n - 1:
        y = TOP + BOX_H/2
        L.append(f'<line x1="{x+BOX_W}" y1="{y}" x2="{x+BOX_W+HGAP-4}" y2="{y}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{x+BOX_W+HGAP/2}" y="{y-10}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="#92400e">{esc("余量传下一候选")}</text>')

foot1_y = TOP + BOX_H + 40
foot2_y = foot1_y + 22
L.append(f'<text x="{PAD}" y="{foot1_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("对比：链式验证在轮 1 拒绝即停止，本节点 0 个接受；树式多轮后接住 token 0，本节点 +1 接受。")}</text>')
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("论文 Table 5：树草稿/树验证相比链式，平均接受长度 τ 提升 +0.6~+0.8，加速比提升 +0.3~+0.5。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-4-tree-multiround.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
