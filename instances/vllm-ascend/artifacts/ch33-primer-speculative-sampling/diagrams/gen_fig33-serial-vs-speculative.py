#!/usr/bin/env python3
"""swimlane 模板(时间轴双泳道):标准解码 vs 投机解码——串行目标模型(M_p)调用次数对比。
两条泳道各自是一条沿时间展开的活动条:标准解码泳道逐格都是 M_p;投机解码泳道每组
= gamma(7) 个 M_q 猜测(合并成一个色块标 x7)后接 1 个 M_p 验证。全坐标由循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "标准解码 vs 投机解码:串行目标模型(M_p)调用次数"
SUBTITLE = "复现论文 Figure 1(38 token 仅需 9 次 M_p)与 Figure 5(每次验证前 gamma=7 个 M_q 猜测)"

PAD, TOP, LANE_LABEL_W = 40, 92, 150
BOX_W, BOX_H, GAP = 54, 40, 10
ELLIPSIS_W = 34

# ---- 泳道 A:标准解码——逐格 M_p,无草稿 ----
laneA_blocks = ["M_p #1", "M_p #2", "M_p #3", "M_p #4"]  # 展示样本
laneA_total = "共 38 次串行 M_p 调用(每次仅出 1 个 token)"

# ---- 泳道 B:投机解码——每组[M_q x7][M_p 验证],样本展示 2 组 ----
laneB_groups = [("M_q x7\n(草稿,并行猜测)", "M_p\n验证+纠错"),
                 ("M_q x7\n(草稿,并行猜测)", "M_p\n验证+纠错")]
laneB_total = "共 9 次串行 M_p 调用 → 同样产出 38 个 token"

MQ_W = 96

def row_width(n_boxes, n_groups_wide):
    return n_boxes * (BOX_W + GAP) + ELLIPSIS_W + GAP + BOX_W

wA = LANE_LABEL_W + len(laneA_blocks) * (BOX_W + GAP) + ELLIPSIS_W + GAP + BOX_W + PAD
wB = LANE_LABEL_W + len(laneB_groups) * (MQ_W + GAP + BOX_W + GAP) + ELLIPSIS_W + GAP + MQ_W + GAP + BOX_W + PAD
w = max(wA, wB) + PAD
LANE_H = 74
h = TOP + LANE_H * 2 + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# lane background bands
for i, name in enumerate(["标准解码", "投机解码"]):
    ly = TOP + i * (LANE_H + 40)
    L.append(f'<rect x="{PAD}" y="{ly}" width="{w-2*PAD}" height="{LANE_H}" rx="8" '
              f'fill="#f8fafc" stroke="#cbd5e1"/>')
    L.append(f'<text x="{PAD+14}" y="{ly+LANE_H/2+5}" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')

# ---- Lane A: 标准解码 ----
lyA = TOP
x = PAD + LANE_LABEL_W
by = lyA + (LANE_H - BOX_H) / 2
for i, lab in enumerate(laneA_blocks):
    L.append(f'<rect x="{x}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="6" '
              'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#5b21b6">{esc(lab)}</text>')
    x += BOX_W + GAP
L.append(f'<text x="{x+ELLIPSIS_W/2}" y="{by+BOX_H/2+6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="18" fill="#64748b">...</text>')
x += ELLIPSIS_W + GAP
L.append(f'<rect x="{x}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="6" '
          'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#5b21b6">M_p #38</text>')
L.append(f'<text x="{PAD+LANE_LABEL_W}" y="{lyA+LANE_H+18}" font-family="sans-serif" '
          f'font-size="12.5" fill="#5b21b6" font-weight="bold">{esc(laneA_total)}</text>')

# ---- Lane B: 投机解码 ----
lyB = TOP + LANE_H + 40
by = lyB + (LANE_H - BOX_H) / 2
x = PAD + LANE_LABEL_W
for mq_lab, mp_lab in laneB_groups:
    L.append(f'<rect x="{x}" y="{by}" width="{MQ_W}" height="{BOX_H}" rx="6" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    for k, line in enumerate(mq_lab.split("\n")):
        L.append(f'<text x="{x+MQ_W/2}" y="{by+BOX_H/2-3+k*13}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">{esc(line)}</text>')
    x += MQ_W + GAP
    L.append(f'<line x1="{x-GAP}" y1="{by+BOX_H/2}" x2="{x}" y2="{by+BOX_H/2}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<rect x="{x}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="6" '
              'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
    for k, line in enumerate(mp_lab.split("\n")):
        L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2-3+k*13}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="#5b21b6">{esc(line)}</text>')
    x += BOX_W + GAP
L.append(f'<text x="{x+ELLIPSIS_W/2}" y="{by+BOX_H/2+6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="18" fill="#64748b">...</text>')
x += ELLIPSIS_W + GAP
L.append(f'<rect x="{x}" y="{by}" width="{MQ_W}" height="{BOX_H}" rx="6" '
          'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
L.append(f'<text x="{x+MQ_W/2}" y="{by+BOX_H/2-3}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">M_q x7</text>')
L.append(f'<text x="{x+MQ_W/2}" y="{by+BOX_H/2+10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">(草稿,并行猜测)</text>')
x += MQ_W + GAP
L.append(f'<line x1="{x-GAP}" y1="{by+BOX_H/2}" x2="{x}" y2="{by+BOX_H/2}" '
          'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{x}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="6" '
          'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2-3}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#5b21b6">M_p #9</text>')
L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2+10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#5b21b6">验证+纠错</text>')
L.append(f'<text x="{PAD+LANE_LABEL_W}" y="{lyB+LANE_H+18}" font-family="sans-serif" '
          f'font-size="12.5" fill="#1d4ed8" font-weight="bold">{esc(laneB_total)}</text>')

foot_y = h - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">紫=目标模型 M_p 串行前向(唯一决定墙钟延迟的资源);蓝=草稿模型 M_q 猜测(gamma=7,与前一次 M_p 验证并行摊销)</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-serial-vs-speculative.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
