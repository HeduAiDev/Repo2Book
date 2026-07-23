#!/usr/bin/env python3
"""fig-m1-engines-race: 达芬奇一个核内多条异步引擎各跑各的指令流、互不知情。
横轴=指令时间线,每条泳道一条独立引擎(3 条搬运 + 4 条计算/定点),用两个真实
指令块(MTE2 写 buffer / V 读同一 buffer)演示硬件不检测跨引擎依赖时会撞车。
全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "达芬奇一个核内:多条异步引擎各跑各的,硬件不检测跨引擎依赖"
SUBTITLE = "3 条搬运引擎(MTE1/MTE2/MTE3)+ 4 条计算/定点引擎(M/V/FIX/S)——各自顺序执行、彼此乱序;红色=竞态"

LANES = [
    ("MTE1", "#e0f2fe", "#0369a1"),
    ("MTE2", "#e0f2fe", "#0369a1"),
    ("MTE3", "#e0f2fe", "#0369a1"),
    ("M", "#f3e8ff", "#7e22ce"),
    ("V", "#f3e8ff", "#7e22ce"),
    ("FIX", "#f3e8ff", "#7e22ce"),
    ("S", "#f3e8ff", "#7e22ce"),
]
# 每条泳道自己的指令块 (start, width, label);时间单位为任意刻度
BLOCKS = {
    "MTE1": [(20, 60, "搬运(其它数据)")],
    "MTE2": [(90, 140, "load 写 buffer X")],
    "MTE3": [(340, 60, "store")],
    "M":    [(20, 70, "mmad(其它数据)")],
    "V":    [(190, 110, "vadd 读 buffer X")],
    "FIX":  [(260, 60, "fixpipe")],
    "S":    [(20, 40, "标量运算")],
}
RACE = {"lane_a": "MTE2", "lane_b": "V", "label": "V 在 MTE2 写完前就开读——读到未写完的 X"}

PAD, TOP, LANE_H, LANE_GAP, LABEL_W = 40, 96, 40, 14, 130
TIMELINE_W = 460
w = PAD * 2 + LABEL_W + TIMELINE_W + 20
h = TOP + len(LANES) * (LANE_H + LANE_GAP) + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

track_x0 = PAD + LABEL_W
lane_y = {}
for i, (name, fill, stroke) in enumerate(LANES):
    y = TOP + i * (LANE_H + LANE_GAP)
    lane_y[name] = y
    L.append(f'<rect x="{PAD}" y="{y}" width="{LABEL_W-10}" height="{LANE_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{y+LANE_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">{esc(name)}</text>')
    L.append(f'<line x1="{track_x0}" y1="{y+LANE_H/2}" x2="{track_x0+TIMELINE_W}" y2="{y+LANE_H/2}" '
              'stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3,3"/>')
    for (start, width, label) in BLOCKS.get(name, []):
        bx = track_x0 + start
        is_race = name in (RACE["lane_a"], RACE["lane_b"])
        bfill = "#fecaca" if is_race else "#dbeafe"
        bstroke = "#b91c1c" if is_race else "#3b82f6"
        L.append(f'<rect x="{bx}" y="{y+4}" width="{width}" height="{LANE_H-8}" rx="5" '
                  f'fill="{bfill}" stroke="{bstroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{bx+width/2}" y="{y+LANE_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="#0f172a">{esc(label)}</text>')

# 竞态标注:MTE2 写块结束点 与 V 读块开始点之间的重叠区(用竖直箭头,标注挪到 V 块正下方,避免压穿中间泳道文字)
a_start, a_width, _ = BLOCKS["MTE2"][0]
b_start, b_width, _ = BLOCKS["V"][0]
a_end_x = track_x0 + a_start + a_width
b_start_x = track_x0 + b_start
overlap_x0, overlap_x1 = min(a_end_x, b_start_x), max(a_end_x, b_start_x)
y_a = lane_y["MTE2"] + LANE_H
y_b = lane_y["V"]
L.append(f'<rect x="{overlap_x0-2}" y="{y_a-2}" width="{max(overlap_x1-overlap_x0,4)+4}" '
          f'height="{y_b-y_a+4}" fill="#fee2e2" fill-opacity="0.55" stroke="#dc2626" '
          'stroke-width="1.5" stroke-dasharray="4,3"/>')
mid_x = (overlap_x0 + overlap_x1) / 2
L.append(f'<line x1="{mid_x}" y1="{y_a}" x2="{mid_x}" y2="{y_b}" '
          'stroke="#dc2626" stroke-width="2" marker-end="url(#a)"/>')
callout_y = lane_y["V"] + LANE_H + 20
L.append(f'<text x="{track_x0+b_start+b_width/2}" y="{callout_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#b91c1c">{esc(RACE["label"])}</text>')

# 底部图例 + 数字标注
foot_y = h - 56
L.append(f'<rect x="{PAD}" y="{foot_y-16}" width="16" height="16" fill="#dbeafe" stroke="#3b82f6"/>')
L.append(f'<text x="{PAD+22}" y="{foot_y-3}" font-family="sans-serif" font-size="11" '
          'fill="#334155">正常指令块(无同步下靠运气不撞车)</text>')
L.append(f'<rect x="{PAD+280}" y="{foot_y-16}" width="16" height="16" fill="#fecaca" stroke="#b91c1c"/>')
L.append(f'<text x="{PAD+302}" y="{foot_y-3}" font-family="sans-serif" font-size="11" '
          'fill="#334155">跨引擎读写同一 buffer,无同步则可能撞车</text>')
foot2_y = h - 30
L.append(f'<text x="{PAD}" y="{foot2_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">共 3 条搬运引擎(MTE1/MTE2/MTE3)+ 4 条计算/定点引擎(M/V/FIX/S);'
          f'每对引擎间同步信号位上限 8 个(kTotalEventIdNum)</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m1-engines-race.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
