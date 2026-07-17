#!/usr/bin/env python3
"""flow 模板：runOnOperation 是全书第一个完整 analysis→transform 闭环。
三阶段线性流：分析(只读) → walk 收集布局 → 改写。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "runOnOperation：全书第一个完整 analysis→transform 闭环"
SUBTITLE = "Coalesce::runOnOperation —— 后续 ch26-35 每个优化 pass 都是这个范式的变体"

STAGES = [
    {
        "label": "阶段1：分析（只读）",
        "code": "ModuleAxisInfoAnalysis\n(moduleOp)",
        "note": "读 contiguity/divisibility\n三元组，不改 IR",
        "color": ("#eff6ff", "#3b82f6"),
    },
    {
        "label": "阶段2：walk 收集布局",
        "code": "setCoalescedEncoding\n→ layoutMap",
        "note": "据分析结果决定\n每个 load/store 的目标布局",
        "color": ("#fef9c3", "#ca8a04"),
    },
    {
        "label": "阶段3：改写",
        "code": "coalesceOp(kv)",
        "note": "convert_layout 夹层\n落地新布局",
        "color": ("#ecfdf5", "#047857"),
    },
]

BOX_W, BOX_H, GAP, PAD, TOP = 340, 130, 90, 40, 108
w = PAD * 2 + BOX_W * len(STAGES) + GAP * (len(STAGES) - 1)
h = TOP + BOX_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1e3a5f"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(len(STAGES))]

for i, st in enumerate(STAGES):
    x = xs_[i]
    fill, stroke = st["color"]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(st["label"])}</text>')
    L.append(f'<rect x="{x+16}" y="{TOP+34}" width="{BOX_W-32}" height="42" rx="4" '
              f'fill="white" stroke="{stroke}" stroke-width="1"/>')
    code_lines = st["code"].split("\n")
    cy0 = TOP + 34 + 21 - (len(code_lines) - 1) * 8
    for k, ln in enumerate(code_lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{cy0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{stroke}">{esc(ln)}</text>')
    note_lines = st["note"].split("\n")
    ny0 = TOP + 90
    for k, ln in enumerate(note_lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{ny0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#334155">{esc(ln)}</text>')
    if i < len(STAGES) - 1:
        ax1 = x + BOX_W
        ax2 = ax1 + GAP
        ay = TOP + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-6}" y2="{ay}" '
                  'stroke="#1e3a5f" stroke-width="2" marker-end="url(#a)"/>')

# bottom: what the analysis reads (contiguity/divisibility) tying stage1->stage2
foot1_y = TOP + BOX_H + 46
L.append(f'<text x="{PAD}" y="{foot1_y}" font-family="sans-serif" font-size="12" '
          f'fill="#374151">{esc("分析读的量：contiguity / divisibility（AxisInfo 三元组，AxisInfo.h:L23-L108）")}</text>')
foot2_y = foot1_y + 26
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("AxisInfo 只读推断出每根轴的静态真相 → Coalesce 据 contiguity 定 order、据 divisibility/contiguity 定向量宽 → coalesceOp 落地改写。")}</text>')
foot3_y = foot2_y + 20
L.append(f'<text x="{PAD}" y="{foot3_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("ch26-35 的每个优化 pass 都是这个范式的变体。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-analysis-transform-loop.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
