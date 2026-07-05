#!/usr/bin/env python3
"""tensor-flow 模板改造:lightning indexer 打分流水线。
两个 indexer 头各对 query 与前驱 s 做点积 -> ReLU 清零负值 -> 按权重 w 加权求和 = I_{t,s}。
用两条前驱(s=2 强正、s=3 全负)对比展示 ReLU 的『合不来记 0、不倒扣』。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Lightning Indexer 打分:每头点积 → ReLU 清零负相关 → 加权求和"
SUBTITLE = "I_{t,s} = Σ_j w_j·ReLU(q_j^I·k_s^I);H^I=2 头,d^I=3,w=[1.0, 2.0]"

# 两行示例:s=2(强正,拿最高分) / s=3(两头皆负,清零)
ROWS = [
    {"s": "s=2", "dot": [0.0, 1.5], "relu": [0.0, 1.5], "score": "I = 1·0.0 + 2·1.5 = 3.0", "note": "最高分 → top-k 选中", "hi": True},
    {"s": "s=3", "dot": [-2.0, -0.5], "relu": [0.0, 0.0], "score": "I = 1·0.0 + 2·0.0 = 0.0", "note": "两头皆负 → ReLU 清零,不倒扣", "hi": False},
]
STAGE_LABELS = ["头0 q·k", "头1 q·k", "ReLU(头0)", "ReLU(头1)", "加权求和 I_{t,s}"]

PAD, TOP = 40, 100
ROW_H = 130
STAGE_W = 190
STAGE_GAP = 40
n_stage = len(STAGE_LABELS)
w = PAD * 2 + STAGE_W * n_stage + STAGE_GAP * (n_stage - 1) + 90
h = TOP + ROW_H * len(ROWS) + 110

stage_x = [PAD + 90 + i * (STAGE_W + STAGE_GAP) for i in range(n_stage)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 阶段表头
for i, name in enumerate(STAGE_LABELS):
    x = stage_x[i]
    L.append(f'<text x="{x+STAGE_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#374151">{esc(name)}</text>')

for r, row in enumerate(ROWS):
    ry = TOP + 10 + r * ROW_H
    hi = row["hi"]
    hi_fill, hi_stroke = ("#ecfdf5", "#047857") if hi else ("#fee2e2", "#b91c1c")
    # 行标签(前驱 s)
    L.append(f'<text x="{PAD}" y="{ry+50}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(row["s"])}</text>')
    box_h = 40
    values = row["dot"] + row["relu"]
    for c in range(4):
        x = stage_x[c]
        y = ry
        val = values[c]
        is_relu_stage = c >= 2
        neg = val < 0
        fill = "#fee2e2" if neg else ("#dbeafe" if not is_relu_stage else "#dcfce7")
        stroke = "#b91c1c" if neg else ("#1d4ed8" if not is_relu_stage else "#15803d")
        L.append(f'<rect x="{x}" y="{y}" width="{STAGE_W}" height="{box_h}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+STAGE_W/2}" y="{y+box_h/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{stroke}" '
                  f'font-weight="bold">{val:.1f}</text>')
        if c < 3:
            nx = stage_x[c+1]
            L.append(f'<line x1="{x+STAGE_W}" y1="{y+box_h/2}" x2="{nx}" y2="{y+box_h/2}" '
                      'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
    # 加权求和阶段
    sx = stage_x[4]
    sy = ry
    L.append(f'<rect x="{sx}" y="{sy}" width="{STAGE_W}" height="{box_h}" rx="6" '
              f'fill="{hi_fill}" stroke="{hi_stroke}" stroke-width="2.5"/>')
    L.append(f'<text x="{sx+STAGE_W/2}" y="{sy+box_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="{hi_stroke}" '
              f'font-weight="bold">{esc(row["score"])}</text>')
    L.append(f'<line x1="{stage_x[3]+STAGE_W}" y1="{sy+box_h/2}" x2="{sx}" y2="{sy+box_h/2}" '
              'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
    # 注释
    L.append(f'<text x="{sx}" y="{sy+box_h+22}" font-family="sans-serif" font-size="11.5" '
              f'fill="{hi_stroke}">{esc(row["note"])}</text>')

# 落地 callout
callout_y = TOP + 10 + ROW_H * len(ROWS) + 12
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="40" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+25}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">落地每对打分 H^I·d^I=64·128=8192 MAC,仍远小于主注意力单条 KV 的 73728 MAC(约 1/9)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig32-lightning-indexer.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
