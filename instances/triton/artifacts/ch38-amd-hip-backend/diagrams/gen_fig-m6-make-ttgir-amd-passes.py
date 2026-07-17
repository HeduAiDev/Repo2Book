#!/usr/bin/env python3
"""fig-m6-make-ttgir-amd-passes：make_ttgir 流水线位置跨后端一致——加速矩阵乘
与软件流水这两站，AMD 换专属 pass 且门控条件不同。swimlane 变体：两条水平
泳道（NVIDIA/AMD），共享的站位用竖直参考线对齐，每站下方挂各自的 pass 框。
全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "make_ttgir 同一条流水线，同一个站位，换后端专属 pass"
SUBTITLE = "加速矩阵乘、软件流水——位置不变，AMD 换 pass 名与门控条件"

LANES = ["NVIDIA", "AMD"]
STATIONS = ["① 加速矩阵乘", "② 软件流水（门控）"]

# 每个 lane 在每个 station 下的框内容：(pass 名, 细节行list)
CELLS = {
    ("NVIDIA", 0): ("add_accelerate_matmul(pm)", ["参数个数：0"]),
    ("AMD", 0): ("amd.add_accelerate_matmul(pm, arch, nonkdim, kpack)", ["参数个数：3"]),
    ("NVIDIA", 1): ("add_pipeline(pm, num_stages)", ["门控：capability // 10 >= 8"]),
    ("AMD", 1): ("add_stream_pipelinev2(pm, num_stages)", ["门控：has_matrix_core_feature(arch)"]),
}

PAD, TOP = 40, 130
LANE_H = 130
LANE_GAP = 40
STATION_W = 480
STATION_GAP = 40
LANE_LABEL_W = 90

n_lanes = len(LANES)
n_st = len(STATIONS)
w = PAD * 2 + LANE_LABEL_W + STATION_W * n_st + STATION_GAP * (n_st - 1)
STATION_HEADER_Y = TOP + 40
STATION_HEADER_H = 28
lanes_top = STATION_HEADER_Y + STATION_HEADER_H + 24
h = lanes_top + n_lanes * (LANE_H + LANE_GAP) + 70

station_x = [PAD + LANE_LABEL_W + i * (STATION_W + STATION_GAP) for i in range(n_st)]
lane_y = [lanes_top + i * (LANE_H + LANE_GAP) for i in range(n_lanes)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="{PAD+22}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 站位表头（跨两条泳道的公共坐标参考）
for j, station in enumerate(STATIONS):
    x = station_x[j]
    L.append(f'<rect x="{x}" y="{STATION_HEADER_Y}" width="{STATION_W}" height="{STATION_HEADER_H}" rx="5" '
              'fill="#334155" stroke="#1e293b"/>')
    L.append(f'<text x="{x+STATION_W/2}" y="{STATION_HEADER_Y+STATION_HEADER_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="white">{esc(station)}</text>')
    # 竖直参考线贯穿两条泳道，标示"同一站位"
    line_top = STATION_HEADER_Y + STATION_HEADER_H
    line_bottom = lane_y[-1] + LANE_H
    L.append(f'<line x1="{x+STATION_W/2}" y1="{line_top}" x2="{x+STATION_W/2}" y2="{line_bottom}" '
              'stroke="#94a3b8" stroke-width="1" stroke-dasharray="3,4"/>')

LANE_COLOR = {"NVIDIA": ("#dcfce7", "#15803d"), "AMD": ("#fee2e2", "#dc2626")}

for i, lane in enumerate(LANES):
    y = lane_y[i]
    fill, stroke = LANE_COLOR[lane]
    # 泳道标签
    L.append(f'<rect x="{PAD}" y="{y}" width="{LANE_LABEL_W-10}" height="{LANE_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD+(LANE_LABEL_W-10)/2}" y="{y+LANE_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="{stroke}">{esc(lane)}</text>')
    # 泳道生命线（贯穿全部 station，体现"同一条流水线"）
    line_y = y + LANE_H / 2
    L.append(f'<line x1="{PAD+LANE_LABEL_W}" y1="{line_y}" x2="{w-PAD}" y2="{line_y}" '
              f'stroke="{stroke}" stroke-width="1" stroke-dasharray="2,3" opacity="0.4"/>')
    for j, station in enumerate(STATIONS):
        x = station_x[j]
        pass_name, details = CELLS[(lane, j)]
        box_w = STATION_W - 20
        box_x = x + 10
        L.append(f'<rect x="{box_x}" y="{y+10}" width="{box_w}" height="{LANE_H-20}" rx="8" '
                  f'fill="white" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{box_x+box_w/2}" y="{y+34}" text-anchor="middle" '
                  f'font-family="monospace" font-size="12" font-weight="bold" '
                  f'fill="#0f172a">{esc(pass_name)}</text>')
        for k, d in enumerate(details):
            L.append(f'<text x="{box_x+box_w/2}" y="{y+58+k*18}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" fill="{stroke}">{esc(d)}</text>')

foot_y = h - 46
FOOT_LINES = [
    "third_party/amd/backend/compiler.py:L218（accelerate_matmul）/L222,228（流水门控+pass）；",
    "third_party/nvidia/backend/compiler.py:L227（accelerate_matmul）/L231,239（流水门控+pass）。",
]
for k, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y+k*16}" font-family="sans-serif" font-size="10.8" '
              f'fill="#64748b">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m6-make-ttgir-amd-passes.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
