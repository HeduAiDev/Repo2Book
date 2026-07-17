#!/usr/bin/env python3
"""fig-m3-pass-pipeline-bands: flow 模板。
make_ttgir 的真实 pass 顺序:基线段(17,常开)中间插两处 capability 门控——
cap//10>=8 处的 1-pass 门 + 8-pass 门(合计 9,≥sm80 追加),
cap//10>=9 处的 2-pass 门(fence/tma,≥sm90 追加)。
下方三行按 capability=70/80/90 显示实际点亮的门与总 pass 数。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# 主时间线上的分段:(标签, 段内 pass 数, 类型 baseline|gate8|gate9, 备注行)
SEGMENTS = [
    ("基线①", 2, "base", None),
    ("门 A", 1, "gate8", "f32_dot_tc\nL222 · 第31章"),
    ("基线②", 7, "base", "含 accelerate_matmul\nL227 · 第28章"),
    ("门 B", 8, "gate8", "warp-spec四连(第31章)\n+add_pipeline(第29/30章)"),
    ("基线③", 7, "base", None),
    ("门 C", 2, "gate9", "fence+tma_lowering\nL249-250 · 第30章"),
    ("基线④", 1, "base", None),
]

COL_BASE = "#e2e8f0"
COL_BASE_STROKE = "#64748b"
COL_G8 = "#dcfce7"
COL_G8_STROKE = "#22c55e"
COL_G9 = "#ffedd5"
COL_G9_STROKE = "#f97316"

PAD = 46
TOP = 96
UNIT_W = 30   # 每个 pass 计数单位的宽度基准(段宽 = count * UNIT_W,含最小宽)
MIN_SEG_W = 70
SEG_GAP = 14
BOX_H = 54

n = len(SEGMENTS)


def seg_w(count):
    return max(MIN_SEG_W, count * UNIT_W)


widths = [seg_w(c) for _, c, _, _ in SEGMENTS]
total_seg_w = sum(widths) + SEG_GAP * (n - 1)
w = PAD * 2 + total_seg_w
if w < 1180:
    w = 1180
    # 重新按比例撑开段宽以占满画布
    extra = w - PAD * 2 - SEG_GAP * (n - 1)
    scale = extra / sum(widths)
    widths = [ww * scale for ww in widths]

NOTE_ROWS_MAX = 2
NOTE_ZONE_H = 46
CAP_ROW_TOP = TOP + BOX_H + NOTE_ZONE_H + 46
CAP_ROW_H = 40
CAP_ROW_GAP = 14
LEGEND_Y = CAP_ROW_TOP + 3 * (CAP_ROW_H + CAP_ROW_GAP) + 26
h = LEGEND_Y + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(
    f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("make_ttgir 的 pass 流水线:capability//10 门控三档,按序真实先后")}</text>'
)
L.append(
    f'<text x="{w/2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="12" '
    f'fill="#475569">{esc("third_party/nvidia/backend/compiler.py:L218-L254 · 段宽 ∝ 段内 pass 数")}</text>'
)

# --- 主时间线:按 SEGMENTS 顺序绘制,段宽按 pass 数比例 ---
xs_pos = []
x_cursor = PAD
for wdt in widths:
    xs_pos.append(x_cursor)
    x_cursor += wdt + SEG_GAP

for i, (label, count, kind, note) in enumerate(SEGMENTS):
    x = xs_pos[i]
    wd = widths[i]
    if kind == "base":
        fill, stroke = COL_BASE, COL_BASE_STROKE
    elif kind == "gate8":
        fill, stroke = COL_G8, COL_G8_STROKE
    else:
        fill, stroke = COL_G9, COL_G9_STROKE
    dash = "" if kind == "base" else ' stroke-dasharray="6,4"'
    L.append(f'<rect x="{x}" y="{TOP}" width="{wd}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    cx = x + wd / 2
    L.append(f'<text x="{cx}" y="{TOP+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    L.append(f'<text x="{cx}" y="{TOP+40}" text-anchor="middle" font-family="monospace" '
              f'font-size="12" fill="#334155">{esc(str(count))} pass</text>')
    # 段间箭头(除最后一段外)
    if i < n - 1:
        x1 = x + wd
        x2 = xs_pos[i + 1]
        ymid = TOP + BOX_H / 2
        L.append(f'<line x1="{x1}" y1="{ymid}" x2="{x2}" y2="{ymid}" stroke="#334155" '
                  f'stroke-width="1.8" marker-end="url(#a)"/>')
    # 备注(章节回指),放在段下方
    if note:
        lines = note.split("\n")
        note_y0 = TOP + BOX_H + 22
        for j, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{note_y0+j*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10.5" fill="#475569">{esc(line)}</text>')

# 门控阈值标注(在门 A/B 上方标 cap//10>=8;门 C 上方标 cap//10>=9)
gate_idx8 = [i for i, s in enumerate(SEGMENTS) if s[2] == "gate8"]
g8_x0 = xs_pos[gate_idx8[0]]
g8_x1 = xs_pos[gate_idx8[-1]] + widths[gate_idx8[-1]]
L.append(f'<text x="{(g8_x0+g8_x1)/2}" y="{TOP-10}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" font-weight="bold" fill="#15803d">{esc("cap//10 >= 8 开门(L221,L231)")}</text>')

gate_idx9 = [i for i, s in enumerate(SEGMENTS) if s[2] == "gate9"]
g9_x0 = xs_pos[gate_idx9[0]]
g9_x1 = xs_pos[gate_idx9[-1]] + widths[gate_idx9[-1]]
L.append(f'<text x="{(g9_x0+g9_x1)/2}" y="{TOP-10}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" font-weight="bold" fill="#c2410c">{esc("cap//10 >= 9 开门(L248)")}</text>')

# --- capability 实例行:70 / 80 / 90,显示各门是否点亮 + 总 pass 数 ---
CAPS = [
    ("capability = 70", [False, False], 17),
    ("capability = 80", [True, False], 26),
    ("capability = 90", [True, True], 28),
]

label_w = 170
total_label_reserve = 130
row_x0 = PAD + label_w + 12
row_w = w - PAD - row_x0 - total_label_reserve

for r, (cap_label, gates_on, total) in enumerate(CAPS):
    ry = CAP_ROW_TOP + r * (CAP_ROW_H + CAP_ROW_GAP)
    L.append(f'<text x="{PAD}" y="{ry+CAP_ROW_H/2+5}" font-family="monospace" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(cap_label)}</text>')
    # 该行按与主时间线相同的段宽比例画一条压缩色带,门未点亮的段用灰色斜纹(浅灰+虚线)覆盖表示"跳过"
    for i, (label, count, kind, note) in enumerate(SEGMENTS):
        x = row_x0 + (xs_pos[i] - PAD) * (row_w / total_seg_w)
        wd = widths[i] * (row_w / total_seg_w)
        if kind == "base":
            fill, stroke, on = COL_BASE, COL_BASE_STROKE, True
        elif kind == "gate8":
            on = gates_on[0]
            fill, stroke = (COL_G8, COL_G8_STROKE) if on else ("#f1f5f9", "#cbd5e1")
        else:
            on = gates_on[1]
            fill, stroke = (COL_G9, COL_G9_STROKE) if on else ("#f1f5f9", "#cbd5e1")
        dash = "" if (kind == "base" or on) else ' stroke-dasharray="3,3"'
        L.append(f'<rect x="{x}" y="{ry}" width="{wd}" height="{CAP_ROW_H}" rx="5" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"{dash}/>')
    # 总 pass 数标注(行右侧)
    L.append(f'<text x="{w-PAD}" y="{ry+CAP_ROW_H/2+5}" text-anchor="end" font-family="monospace" '
              f'font-size="13" font-weight="bold" fill="#1d4ed8">{esc(f"= {total} pass")}</text>')

# --- 图例 ---
leg_items = [
    (COL_BASE, COL_BASE_STROKE, "基线(常开,17 道)"),
    (COL_G8, COL_G8_STROKE, "≥sm80 解锁(9 道)"),
    (COL_G9, COL_G9_STROKE, "≥sm90 解锁(2 道)"),
    ("#f1f5f9", "#cbd5e1", "门未点亮(跳过)"),
]
lx = PAD
for fill, stroke, text in leg_items:
    L.append(f'<rect x="{lx}" y="{LEGEND_Y-12}" width="20" height="16" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    L.append(f'<text x="{lx+26}" y="{LEGEND_Y}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(text)}</text>')
    lx += 26 + len(text) * 11.5 * 1.02 + 26

L.append('</svg>')
out_path = Path(__file__).with_name("fig-m3-pass-pipeline-bands.svg")
out_path.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out_path}  ({w}x{h})")
