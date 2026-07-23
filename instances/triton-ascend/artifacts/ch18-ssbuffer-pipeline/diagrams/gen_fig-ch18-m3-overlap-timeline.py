#!/usr/bin/env python3
"""swimlane 模板(改造成 Gantt 时间轴):单缓冲严格串行 vs 双缓冲搬算重叠,
两个纵向堆叠的时间轴面板,同一像素/单位比例尺,直接用总长差可视化加速比。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
LOAD_BG, LOAD_FG = "#dbeafe", "#1e40af"
COMPUTE_BG, COMPUTE_FG = "#dcfce7", "#15803d"
BUF0_MARK, BUF1_MARK = "#1e40af", "#b45309"

TITLE = "double-buffer 让 DMA 搬运与 Vector 计算在时间上重叠"
SUB = "T_load=3、T_compute=2(演示延迟,非实测);frontCnt 恒领先 postCnt 一个身位,写读落不同 buffer"

# (label, start, dur, lane, buf_idx)  lane: 0=Load 1=Compute
SERIAL = [
    ("tile0", 0, 3, 0, 0), ("tile0", 3, 2, 1, 0),
    ("tile1", 5, 3, 0, 1), ("tile1", 8, 2, 1, 1),
    ("tile2", 10, 3, 0, 0), ("tile2", 13, 2, 1, 0),
    ("tile3", 15, 3, 0, 1), ("tile3", 18, 2, 1, 1),
]
SERIAL_TOTAL = 20
DOUBLE = [
    ("tile0", 0, 3, 0, 0),
    ("tile1", 3, 3, 0, 1), ("tile0", 3, 2, 1, 0),
    ("tile2", 6, 3, 0, 0), ("tile1", 6, 2, 1, 1),
    ("tile3", 9, 3, 0, 1), ("tile2", 9, 2, 1, 0),
    ("tile3", 12, 2, 1, 1),
]
DOUBLE_TOTAL = 14

PX_PER_UNIT = 26
LANE_LABEL_W = 118
PAD = 42
RIGHT_MARGIN = 118
LANE_H = 38
LANE_GAP = 10
BOX_H = 30
PANEL_TITLE_H = 24
NOTE_H = 18
AXIS_H = 26
PANEL_GAP = 34

def panel_height(has_note=False):
    return PANEL_TITLE_H + (NOTE_H if has_note else 0) + LANE_H * 2 + LANE_GAP + AXIS_H

max_units = max(SERIAL_TOTAL, DOUBLE_TOTAL)
W = PAD + LANE_LABEL_W + max_units * PX_PER_UNIT + RIGHT_MARGIN
TOP = 116
H = TOP + panel_height(False) + PANEL_GAP + panel_height(True) + 116

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB)}</text>']

LANE_NAMES = ["Load(DMA)", "Compute(Vector)"]


def draw_panel(y0, title, boxes, total_units, note=None):
    x0 = PAD + LANE_LABEL_W
    L.append(f'<text x="{PAD}" y="{y0}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="{INK}">{esc(title)}</text>')
    if note:
        L.append(f'<text x="{PAD}" y="{y0+NOTE_H}" font-family="sans-serif" font-size="11.5" '
                  f'font-weight="bold" fill="{BUF1_MARK}">{esc(note)}</text>')
    lanes_top = y0 + (10 + NOTE_H if note else 10)
    for lane_i, name in enumerate(LANE_NAMES):
        ly = lanes_top + lane_i * (LANE_H + LANE_GAP)
        L.append(f'<text x="{PAD}" y="{ly+LANE_H/2+4}" font-family="sans-serif" '
                  f'font-size="12" fill="{INK}">{esc(name)}</text>')
        L.append(f'<line x1="{x0}" y1="{ly+LANE_H/2}" x2="{x0+total_units*PX_PER_UNIT}" '
                  f'y2="{ly+LANE_H/2}" stroke="#e2e8f0" stroke-width="10"/>')
    for label, start, dur, lane_i, buf_idx in boxes:
        ly = lanes_top + lane_i * (LANE_H + LANE_GAP)
        bx = x0 + start * PX_PER_UNIT
        bw = dur * PX_PER_UNIT
        by = ly + (LANE_H - BOX_H) / 2
        bg, fg = (LOAD_BG, LOAD_FG) if lane_i == 0 else (COMPUTE_BG, COMPUTE_FG)
        mark = BUF1_MARK if buf_idx == 1 else BUF0_MARK
        gap = 2  # 留白避免相邻(甚至相邻紧贴)方块的描边视觉粘连
        L.append(f'<rect x="{bx+gap/2}" y="{by}" width="{bw-gap}" height="{BOX_H}" rx="5" '
                  f'fill="{bg}" stroke="{fg}" stroke-width="1.6"/>')
        L.append(f'<rect x="{bx+gap/2}" y="{by}" width="4" height="{BOX_H}" fill="{mark}"/>')
        short = f"t{label[-1]}"  # tile0 -> t0,仅标 tile 序号;buffer 编号交给左侧色条 + 图例
        L.append(f'<text x="{bx+bw/2}" y="{by+BOX_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{fg}">{esc(short)}</text>')
    axis_y = lanes_top + 2 * LANE_H + LANE_GAP + 16
    L.append(f'<line x1="{x0}" y1="{axis_y}" x2="{x0+total_units*PX_PER_UNIT}" y2="{axis_y}" '
              f'stroke="{GRAY}" stroke-width="1"/>')
    for t in range(0, total_units + 1, 5 if total_units >= 10 else 1):
        tx = x0 + t * PX_PER_UNIT
        L.append(f'<line x1="{tx}" y1="{axis_y}" x2="{tx}" y2="{axis_y+5}" stroke="{GRAY}"/>')
        L.append(f'<text x="{tx}" y="{axis_y+18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10" fill="{GRAY}">{t}</text>')
    L.append(f'<text x="{x0+total_units*PX_PER_UNIT+10}" y="{axis_y+4}" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{INK}">总耗时 {total_units}</text>')


y_serial = TOP
draw_panel(y_serial, "单缓冲:搬(Load)与算(Compute)严格串行", SERIAL, SERIAL_TOTAL)
y_double = y_serial + panel_height(False) + PANEL_GAP
draw_panel(y_double, "双缓冲 N=2:搬第 i+1 块与算第 i 块并行(不同 buffer)", DOUBLE, DOUBLE_TOTAL,
           note="frontCnt − postCnt 稳态差 = 1(写恒领先读一拍,故写读落不同 buffer)")

foot_y = y_double + panel_height(True) + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{INK}">本例(4 tile):20 → 14 个单位,加速 1.43×'
          f'(渐近上界 (3+2)/max(3,2) ≈ 1.67×)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
          f'fill="{GRAY}">条左侧色条=buffer 编号(蓝=buf0,黄=buf1);同一 tile 的 Load/Compute'
          f'色条一致,证明读写同一份 buffer,且与相邻 tile 交替不同</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch18-m3-overlap-timeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
