#!/usr/bin/env python3
"""fig36-2-uncertainty-branch: state-machine 模板改造为「一态分岔为二」。
根态 f_I 在采样随机性下可走两条转移边;转移边标「超前一步 token」,唯一决定去哪个后继特征。
数字来自 explainer.json fig36-2 numbers(traces/shifted_token.json + paper.md L34)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROOT_LABEL = ["f_I", "‖f_I‖ = 0.659"]
BRANCHES = [
    {
        "edge_label": "超前 token = 0",
        "box": ["branch A：下一特征", "‖f‖ = 1.137，f[0] = 0.587", "LM Head → token 1（conf 0.222）"],
    },
    {
        "edge_label": "超前 token = 2",
        "box": ["branch B：下一特征", "‖f‖ = 0.85，f[0] = −0.38", "LM Head → token 4（conf 0.213）"],
    },
]

ROOT_W, ROOT_H = 220, 60
BOX_W, BOX_H = 300, 76
HGAP, VGAP = 100, 110
PAD, TOP = 50, 90

w = PAD * 2 + BOX_W * 2 + HGAP
h = TOP + ROOT_H + VGAP + BOX_H + 70

root_cx = w / 2
root_x = root_cx - ROOT_W / 2
root_y = TOP

branch_y = TOP + ROOT_H + VGAP
branch_cx = [PAD + BOX_W / 2, PAD + BOX_W + HGAP + BOX_W / 2]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD-14}" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc("特征不确定性：同一 f_I 靠超前 token 消解分岔")}</text>']

# root state
L.append(f'<rect x="{root_x}" y="{root_y}" width="{ROOT_W}" height="{ROOT_H}" rx="30" '
         'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.5"/>')
for i, line in enumerate(ROOT_LABEL):
    fw = 'font-weight="bold" ' if i == 0 else ''
    fs = 15 if i == 0 else 12
    L.append(f'<text x="{root_cx}" y="{root_y+ROOT_H/2-4+i*17}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fs}" {fw}fill="#0c4a6e">{esc(line)}</text>')

root_bottom = root_y + ROOT_H
for cx, br in zip(branch_cx, BRANCHES):
    by = branch_y
    # edge from root bottom edge to top of branch box
    root_edge_x = root_cx + (cx - root_cx) * 0.0  # keep vertical drop point at root center split below
    L.append(f'<line x1="{root_cx}" y1="{root_bottom}" x2="{cx}" y2="{by}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    mx, my = (root_cx + cx) / 2, (root_bottom + by) / 2
    L.append(f'<rect x="{mx-58}" y="{my-13}" width="116" height="22" rx="5" '
              'fill="#fef3c7" stroke="#d97706" stroke-width="1"/>')
    L.append(f'<text x="{mx}" y="{my+2}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="#92400e">{esc(br["edge_label"])}</text>')
    # branch box
    bx = cx - BOX_W / 2
    L.append(f'<rect x="{bx}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              'fill="#fce7f3" stroke="#be185d" stroke-width="1.5"/>')
    for i, line in enumerate(br["box"]):
        fw = 'font-weight="bold" ' if i == 0 else ''
        L.append(f'<text x="{cx}" y="{by+22+i*20}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" {fw}fill="#831843">{esc(line)}</text>')

foot_y = h - 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("结论：feature&shifted-token 把随机分支变成确定映射（EAGLE 第二大观察，加速比 1.9x → 2.8x）")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-2-uncertainty-branch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
