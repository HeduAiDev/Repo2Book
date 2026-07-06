#!/usr/bin/env python3
"""fig35-6-three-methods: 同一离群层上，SmoothQuant(W8A8) 与 AWQ(W4) 各自把量化-反量化
往返误差压到同制式 RTN 基线之下。state-table，按制式分两组（W8A8 组 / W4 组）。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "同一 50x 离群层：每种缓解都赢过自己制式下的 RTN 裸量化"
SUBTITLE = "误差按量化制式分组比较（W8A8 组 / W4 weight-only 组），不跨制式并列"

COLS = ["方法", "量化制式", "输出误差 ||Y_hat - Y||", "相对误差 %"]
GROUPS = [
    {
        "name": "W8A8（激活+权重 8-bit per-tensor）",
        "rows": [
            {"vals": ["RTN", "W8A8", "0.8424", "1.32"], "kind": "baseline"},
            {"vals": ["SmoothQuant", "W8A8", "0.2604", "0.41"], "kind": "best"},
        ],
        "reduction": "reduction 69.09%",
    },
    {
        "name": "W4 weight-only（权重 4-bit，激活全精度）",
        "rows": [
            {"vals": ["RTN", "W4-weight-only", "3.5991", "5.63"], "kind": "baseline"},
            {"vals": ["AWQ", "W4-weight-only", "2.2373", "3.50"], "kind": "best"},
        ],
        "reduction": "reduction 37.84%",
    },
]

COL_W = [150, 200, 220, 140]
CALLOUT_W = 190
LABEL_X = 40
ROW_H = 46
HEADER_H = 44
GROUP_TITLE_H = 30
GROUP_GAP = 22
TOP = 108
PAD = 40
FOOT_N = 3

col_x = []
x = LABEL_X
for cw in COL_W:
    col_x.append(x)
    x += cw
table_w = x
w = table_w + CALLOUT_W + PAD * 2

n_rows_total = sum(len(g["rows"]) for g in GROUPS)
table_bottom = (TOP + HEADER_H + len(GROUPS) * GROUP_TITLE_H + n_rows_total * ROW_H
                + len(GROUPS) * GROUP_GAP)
foot_y = table_bottom + 10
h = foot_y + (FOOT_N - 1) * 18 + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    cx = PAD + col_x[j]
    cw = COL_W[j]
    L.append(f'<rect x="{cx}" y="{TOP}" width="{cw-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{cx+(cw-6)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white">{esc(name)}</text>')

cy = TOP + HEADER_H
KIND_COLOR = {
    "baseline": ("#f1f5f9", "#64748b", "#475569"),
    "best": ("#ecfdf5", "#047857", "#047857"),
}
for g in GROUPS:
    cy += GROUP_TITLE_H
    L.append(f'<text x="{PAD}" y="{cy-8}" font-family="sans-serif" font-size="13" '
              f'fill="#0f172a">{esc(g["name"])}</text>')
    group_top = cy
    for row in g["rows"]:
        fill, stroke, text_fill = KIND_COLOR[row["kind"]]
        for j, val in enumerate(row["vals"]):
            cx = PAD + col_x[j]
            cw = COL_W[j]
            L.append(f'<rect x="{cx}" y="{cy+4}" width="{cw-6}" height="{ROW_H-8}" rx="3" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            L.append(f'<text x="{cx+(cw-6)/2}" y="{cy+ROW_H/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" fill="{text_fill}">{esc(val)}</text>')
        cy += ROW_H
    group_mid = (group_top + cy) / 2
    L.append(f'<text x="{PAD+table_w+16}" y="{group_mid+4}" font-family="sans-serif" '
              f'font-size="13" fill="#047857">{esc(g["reduction"])}</text>')
    cy += GROUP_GAP

foot_lines = [
    "灰=RTN 基线（无缓解），绿=对应缓解方法（严格更低）。数字来自同一 50x 激活离群层的往返误差实测。",
    "W4 制式整体误差量级约为 W8A8 的 4-8 倍——压缩率越高（4-bit vs 8-bit），越依赖误差控制手段。",
    "vllm 推理期只消费离线量化好的定点权重/scale，本图三种方法均在离线校准阶段完成。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-6-three-methods.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
