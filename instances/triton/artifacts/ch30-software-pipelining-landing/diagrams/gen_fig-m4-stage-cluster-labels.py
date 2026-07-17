#!/usr/bin/env python3
"""fig-m4-stage-cluster-labels: state-table — load 打 stage 0、dot 打 stage 2
(=num_stages-1),两拍的错位就是预取提前量 distToUse=2。
数字来自 explainer.json m4.figure_specs[0].numbers:
  0 = MatmulLoopPipeline.cpp:L590-592 load stage=(0-0)x1
  2 = MatmulLoopPipeline.cpp:L568 dot stage=numStages-1=2
  1 = MatmulLoopPipeline.cpp:L565 stagesBetweenLoads=ceil(1,1)=1
  3 = matmul_sm90_ns3.ttgir.mlir:L61 numBuffers=distToUse(2)+1=3
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "CoarseSchedule 打标:load 早两拍,dot 压轴"
SUBTITLE = "fp16 matmul,numStages=3,maxIndirectionLevel=0(两 load 直喂 dot,无间接寻址)"

COLS = ["dot(root use)", "loadA", "loadB"]
ROW_LABELS = ["间接层 indLevel", "stage 公式", "stage", "cluster", "distToUse"]

CELLS = {
    "间接层 indLevel": ["—", "0", "0"],
    "stage 公式":      ["numStages\n−1", "(0−0)×1", "(0−0)×1"],
    "stage":           ["2", "0", "0"],
    "cluster":         ["rootUsersCluster\n(front)", "loadsClusters[0]", "loadsClusters[0]"],
    "distToUse":       ["—", "stage[dot]−stage[loadA]\n= 2−0 = 2", "2−0 = 2"],
}

HIGHLIGHT_ROW = "stage"
STATUS = {"stage": ["final", "prefetch", "prefetch"]}
COLOR = {"final": ("#fef3c7", "#b45309"), "prefetch": ("#dbeafe", "#1d4ed8")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 158, 220, 60, 36, 108, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 36 + 22 * 3 + 20
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

for i in range(len(ROW_LABELS) + 1):
    y = TOP + HEADER_H + i * ROW_H
    L.append(f'<line x1="{PAD+LABEL_W}" y1="{y}" x2="{w-PAD}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

table_bottom = TOP + HEADER_H + ROW_H * len(ROW_LABELS)
foot_y0 = table_bottom + 36
FOOT_LINES = [
    "stagesBetweenLoads = ceil(numStages−2, maxIndLevel+1) = ceil(1,1) = 1;"
    "distToUse=2。",
    "=> numBuffers = distToUse(2)+1(MMAv3) = 3,与最终 IR memdesc<3x...> 一致。",
    "橙=root use 的最终 stage(压轴计算),蓝=load 的早期 stage(提前预取)。",
]
for k, line in enumerate(FOOT_LINES):
    fill = "#64748b" if k == len(FOOT_LINES) - 1 else "#374151"
    L.append(f'<text x="{PAD}" y="{foot_y0+k*22}" font-family="sans-serif" font-size="12.5" '
              f'fill="{fill}">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m4-stage-cluster-labels.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
