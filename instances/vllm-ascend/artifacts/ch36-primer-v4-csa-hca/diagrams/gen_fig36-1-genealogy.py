#!/usr/bin/env python3
"""flow 模板改造:MLA -> DSA -> CSA/HCA 三代正交演进。
四个阶段横排,每个阶段标注它压缩的是哪一维;CSA/HCA 用括号连成一组,
标出 V4 把前两代的轴叠加。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "MLA -> DSA -> CSA/HCA:三代压缩沿正交轴推进,V4 把前两条线叠加"
SUBTITLE = "每一代压缩的是不同的一维;轴彼此正交,故可以逐代叠加而不冲突"

STAGES = [
    ("MLA", "压 KV 维度到低秩 latent", "轴:每 token 的 KV 维度", "#2563eb", "第31章"),
    ("DSA", "lightning indexer +\ntop-k 稀疏选块", "轴:看哪些 token", "#7c3aed", "第32章"),
    ("CSA", "每 m=4 token 压 1 条\n+ DSA top-k 稀疏", "轴:序列长度 + 复用选块", "#059669", "本章"),
    ("HCA", "每 m'=128 token 压 1 条\n+ 稠密 MQA", "轴:序列长度(更狠)", "#d97706", "本章"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 210, 74, 46, 40, 130
n = len(STAGES)
w = PAD * 2 + BOX_W * n + GAP * (n - 1)
h = TOP + BOX_H + 190

col_x = [PAD + i * (BOX_W + GAP) for i in range(n)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+4}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for i, (name, desc, axis, color, tag) in enumerate(STAGES):
    x = col_x[i]
    # 阶段编号圆
    L.append(f'<circle cx="{x+BOX_W/2}" cy="{TOP-26}" r="13" fill="{color}"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="white">{i+1}</text>')
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{color}" stroke="#1e293b" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="16" fill="white" font-weight="bold">{esc(name)}</text>')
    for k, line in enumerate(desc.split("\n")):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+44+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="#e2e8f0">{esc(line)}</text>')
    # 轴标签
    axis_y = TOP + BOX_H + 26
    L.append(f'<rect x="{x}" y="{axis_y}" width="{BOX_W}" height="34" rx="5" '
              f'fill="#f8fafc" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{axis_y+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="{color}" '
              f'font-weight="bold">{esc(axis)}</text>')
    # 出处标签
    L.append(f'<text x="{x+BOX_W/2}" y="{axis_y+52}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#94a3b8">{esc(tag)}</text>')
    if i < n - 1:
        ax1 = x + BOX_W
        ax2 = col_x[i+1]
        ay = TOP + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                  'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

# CSA/HCA 合流括号:标出 V4 把前两条轴叠加
bx1 = col_x[2]
bx2 = col_x[3] + BOX_W
by = TOP + BOX_H + 78
L.append(f'<path d="M {bx1} {by} L {bx1} {by+10} L {bx2} {by+10} L {bx2} {by}" '
          'fill="none" stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<rect x="{(bx1+bx2)/2-190}" y="{by+18}" width="380" height="52" rx="6" '
          'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
L.append(f'<text x="{(bx1+bx2)/2}" y="{by+38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#b91c1c">V4 合流:CSA = 压序列长 + DSA 稀疏选块</text>')
L.append(f'<text x="{(bx1+bx2)/2}" y="{by+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#b91c1c">HCA = 更狠压序列长 + 稠密 —— 两条老线叠加,不是推倒重来</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">三代各压不同一维(KV维/选哪些/序列长),彼此正交故可叠加;CSA、HCA 都建在 MLA 的低秩 latent 与 DSA 的稀疏机制之上</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-1-genealogy.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
