#!/usr/bin/env python3
"""fig-m3-ringbuffer-rotation: state-table — 3 槽环形缓冲上,读/写游标以固定 2 拍
相位差旋转,写在前、读在后,互不覆盖。
数字来自 explainer.json m3.figure_specs[0].numbers:
  3  = matmul_sm90_ns3.ttgir.mlir:L61 memdesc<3x...> 首维(numBuffers)
  1  = matmul_sm90_ns3.ttgir.mlir:L83 insertIdx 初值 %c1_i32
  -1 = matmul_sm90_ns3.ttgir.mlir:L83 extractIdx 初值 %c-1_i32
  2  = ringbuffer_trace.json 相位差 = maxStage
数据取自 traces/ringbuffer_trace.json 的 6 轮 iter 演化。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "环形缓冲:3 槽转盘,读写游标相位差恒为 2"
SUBTITLE = "numBuffers=3,insertIdx 初值=1,extractIdx 初值=-1(matmul_sm90_ns3.ttgir.mlir:L61,L83)"

COLS = ["iter 0", "iter 1", "iter 2", "iter 3", "iter 4", "iter 5"]
ROW_LABELS = ["extractIdx", "读槽/取用", "insertIdx", "写槽", "谓词 iv<trip-2", "预取动作"]

CELLS = {
    "extractIdx":       ["0", "1", "2", "0", "1", "2"],
    "读槽/取用":         ["槽0\niter0(预填)", "槽1\niter1(预填)", "槽2\niter2",
                          "槽0\niter3", "槽1\niter4", "槽2\niter5"],
    "insertIdx":        ["2", "0", "1", "2", "0", "1"],
    "写槽":              ["槽2", "槽0", "槽1", "槽2", "槽0", "槽1"],
    "谓词 iv<trip-2":    ["真", "真", "真", "真", "假", "假"],
    "预取动作":          ["iter2", "iter3", "iter4", "iter5", "关闭\n(临尾)", "关闭\n(临尾)"],
}

HIGHLIGHT_ROW = "谓词 iv<trip-2"
STATUS = {"谓词 iv<trip-2": ["on", "on", "on", "on", "off", "off"]}
COLOR = {"on": ("#ecfdf5", "#047857"), "off": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 168, 132, 58, 34, 108, 34
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 76
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
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
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
                      f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

# 行分隔线,便于横向扫读
for i in range(len(ROW_LABELS) + 1):
    y = TOP + HEADER_H + i * ROW_H
    L.append(f'<line x1="{PAD+LABEL_W}" y1="{y}" x2="{w-PAD}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">绿=谓词真(仍在预取),红=谓词假(临尾关闭预取)。'
          f'extractIdx 恒落后 insertIdx 两拍(3 槽内环绕),读写永不撞槽。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m3-ringbuffer-rotation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
