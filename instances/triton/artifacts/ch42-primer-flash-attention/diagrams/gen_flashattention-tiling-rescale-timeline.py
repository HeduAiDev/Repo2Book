#!/usr/bin/env python3
"""figure: flashattention-tiling-rescale-timeline (tiling/时序模板改)
claim: 一行 Q 对 4 个 K/V 切成 2 块增量更新:第 2 块把 running max 从 1 抬到 2、
三件套 m/l/O 同乘 alpha=0.367879 降标度后累加,末尾一次归一化得 O=[1.462117,1.337835],
与全矩阵 softmax 逐位相等。
数据来源: explainer/explainer.json mechanism m03-attention-online-three-way
(explainer/traces/tiling_rescale.json)。全坐标计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "FlashAttention 分块遍历时序:2 块 K/V 增量更新一行 Q"
SUBTITLE = "外层锁一块 Q 常驻 SRAM;内层每来一块 K/V 就地更新 m/l/acc 三件套,running max 抬高时旧账先乘 alpha 降标度"

Q_BOX = ("Q 常驻 SRAM", "q=[1,0]  scale=1.0")

BLOCKS = [
    {
        "name": "块 1  K0V0",
        "kv": "K=[[1,0],[0,1]]  V=[[1,0],[0,1]]",
        "score": "S=[1, 0]",
        "steps": [
            "rowmax(S)=1",
            "m^(1)=1  (首块,alpha=0 清零初值)",
            "P̃=e^{S-m}=[1.0, 0.367879]",
            "l^(1)=1.367879",
            "acc^(1)=[1.0, 0.367879]  (未归一)",
        ],
        "alpha": None,
    },
    {
        "name": "块 2  K1V1",
        "kv": "K=[[1,1],[2,0]]  V=[[1,1],[2,2]]",
        "score": "S=[1, 2]",
        "steps": [
            "rowmax(S)=2",
            "m^(2)=2  (1 -> 2 抬高)",
            "alpha=e^{1-2}=0.367879  旧 l/acc 先乘 alpha",
            "P̃=e^{S-m}=[0.367879, 1.0]",
            "l^(2)=alpha*l^(1)+rowsum(P̃)=1.871094",
            "acc^(2)=alpha*acc^(1)+P̃V=[2.735759, 2.503215]",
        ],
        "alpha": "0.367879",
    },
]

EPILOGUE = [
    "归一化(仅此一次): O = acc^(2) / l^(2)",
    "O = [1.462117, 1.337835]",
    "全矩阵一次性 softmax: O_full = [1.462117, 1.337835]",
    "逐位相等(exact) ✓",
]

PAD, TOP = 42, 108
Q_W, Q_H = 190, 96
BLOCK_W = 330
GAP = 150
EPI_W = 300
ROW_H = 21
STEP_TOP_PAD = 92

n_steps_max = max(len(b["steps"]) for b in BLOCKS)
block_h = STEP_TOP_PAD + n_steps_max * ROW_H + 20
epi_h = STEP_TOP_PAD - 20 + len(EPILOGUE) * ROW_H + 20

w = PAD * 2 + Q_W + GAP + BLOCK_W * len(BLOCKS) + (len(BLOCKS) - 1) * GAP + GAP + EPI_W
h = TOP + max(block_h, Q_H) + 130

qy = TOP + (max(block_h, Q_H) - Q_H) / 2
qx = PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>',
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>',
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
     'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>',
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# Q box (locked in SRAM)
L.append(f'<rect x="{qx}" y="{qy}" width="{Q_W}" height="{Q_H}" rx="8" '
          'fill="#dbeafe" stroke="#1e40af" stroke-width="2.5"/>')
L.append(f'<text x="{qx+Q_W/2}" y="{qy+28}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#1e3a5f">{esc(Q_BOX[0])}</text>')
L.append(f'<text x="{qx+Q_W/2}" y="{qy+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#334155">{esc(Q_BOX[1])}</text>')
L.append(f'<text x="{qx+Q_W/2}" y="{qy+72}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10" fill="#1e40af">{esc("(整个内层循环反复读,只 load 一次)")}</text>')

block_x = []
cur_x = qx + Q_W + GAP
for i, b in enumerate(BLOCKS):
    block_x.append(cur_x)
    cur_x += BLOCK_W + GAP
epi_x = cur_x

# arrow: Q -> block1, block1 -> block2 (dashed data-flow), Q -> block2 (dashed, "反复读")
L.append(f'<line x1="{qx+Q_W}" y1="{qy+Q_H/2}" x2="{block_x[0]}" y2="{TOP+30}" '
          'stroke="#1e40af" stroke-width="2" marker-end="url(#a)"/>')

for i, b in enumerate(BLOCKS):
    bx = block_x[i]
    by = TOP
    bh = block_h
    L.append(f'<rect x="{bx}" y="{by}" width="{BLOCK_W}" height="{bh}" rx="8" '
              'fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>')
    L.append(f'<rect x="{bx}" y="{by}" width="{BLOCK_W}" height="30" rx="8" '
              'fill="#475569"/>')
    L.append(f'<rect x="{bx}" y="{by+16}" width="{BLOCK_W}" height="14" fill="#475569"/>')
    L.append(f'<text x="{bx+BLOCK_W/2}" y="{by+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(b["name"])}</text>')
    L.append(f'<text x="{bx+14}" y="{by+48}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(b["kv"])}</text>')
    L.append(f'<text x="{bx+14}" y="{by+66}" font-family="sans-serif" font-size="11" '
              f'font-weight="bold" fill="#1e3a5f">{esc(b["score"])}</text>')
    for k, step in enumerate(b["steps"]):
        sy = by + STEP_TOP_PAD + k * ROW_H
        is_alpha_line = "alpha=" in step and b["alpha"]
        if is_alpha_line:
            L.append(f'<rect x="{bx+8}" y="{sy-14}" width="{BLOCK_W-16}" height="19" rx="4" '
                      'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
        fill = "#92400e" if is_alpha_line else "#334155"
        weight = 'font-weight="bold" ' if is_alpha_line else ''
        L.append(f'<text x="{bx+14}" y="{sy}" font-family="sans-serif" font-size="11" '
                  f'{weight}fill="{fill}">{esc(step)}</text>')
    if i < len(BLOCKS) - 1:
        nx = block_x[i + 1]
        midy = by + block_h / 2
        L.append(f'<line x1="{bx+BLOCK_W}" y1="{midy}" x2="{nx}" y2="{midy}" '
                  'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
        arrow_mid_x = (bx + BLOCK_W + nx) / 2
        L.append(f'<text x="{arrow_mid_x}" y="{midy-24}" text-anchor="middle" '
                  'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="#d97706">{esc("alpha=0.367879")}</text>')
        L.append(f'<text x="{arrow_mid_x}" y="{midy-10}" text-anchor="middle" '
                  'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="#d97706">{esc("降标度")}</text>')

# dashed arrow Q -> block2 ("反复读,同一个 q")
by2 = TOP
L.append(f'<path d="M {qx+Q_W} {qy+Q_H-10} Q {(qx+Q_W+block_x[1])/2} {qy+Q_H+60} '
          f'{block_x[1]+30} {by2+block_h+4}" fill="none" stroke="#93c5fd" '
          'stroke-width="1.5" stroke-dasharray="5,4" marker-end="url(#a)"/>')
L.append(f'<text x="{(qx+Q_W+block_x[1])/2-40}" y="{qy+Q_H+78}" text-anchor="middle" '
          'font-family="sans-serif" font-size="10" '
          f'fill="#1e40af">{esc("同一个 q,反复读")}</text>')

# epilogue box
by = TOP
L.append(f'<rect x="{epi_x}" y="{by}" width="{EPI_W}" height="{block_h}" rx="8" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
L.append(f'<rect x="{epi_x}" y="{by}" width="{EPI_W}" height="30" rx="8" fill="#047857"/>')
L.append(f'<rect x="{epi_x}" y="{by+16}" width="{EPI_W}" height="14" fill="#047857"/>')
L.append(f'<text x="{epi_x+EPI_W/2}" y="{by+20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="white" font-weight="bold">{esc("epilogue 归一化")}</text>')
for k, line in enumerate(EPILOGUE):
    sy = by + STEP_TOP_PAD + k * ROW_H + (block_h - epi_h) / 2 - 10
    is_final = "逐位相等" in line
    weight = 'font-weight="bold" ' if is_final else ''
    fill = "#047857" if is_final else "#334155"
    L.append(f'<text x="{epi_x+16}" y="{sy}" font-family="sans-serif" font-size="11" '
              f'{weight}fill="{fill}">{esc(line)}</text>')
L.append(f'<line x1="{block_x[-1]+BLOCK_W}" y1="{by+block_h/2}" x2="{epi_x}" y2="{by+block_h/2}" '
          'stroke="#047857" stroke-width="2" marker-end="url(#a)"/>')

foot_y = TOP + max(block_h, Q_H) + 60
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("黄底=alpha rescale 步骤(恒等性的全部);running 状态 m/l/acc 全程只有 O(块) 大小,从不物化 4 列(推广即 N 列)整行打分。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc("锚点 python/tutorials/06-fused-attention.py:L46-74(内层)、L185-186(归一化)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("flashattention-tiling-rescale-timeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
