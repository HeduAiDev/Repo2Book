#!/usr/bin/env python3
"""fig-ch19-td-triple-to-dump: before-after 模板改造。
左面板 = make_range 的 .td 三元组(arguments/results/assemblyFormat)，
右面板 = 对应的一行 TTIR dump，从高亮段各自垂直引出标注框解释含义。
坐标全部由循环/常量算，文本全 esc()，dump 行按等宽字符步进精确定位。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.56) for ch in s)


TITLE = "一行 dump 的长相由 .td 三元组决定 —— 以 make_range 为最小样本"
SUBTITLE = "include/triton/Dialect/Triton/IR/TritonOps.td:L803-L824"

BEFORE_LABEL = ".td 三元组"
AFTER_LABEL = "TTIR dump 里的一行"

BEFORE_LINES = [
    ("arguments", "I32Attr:$start, I32Attr:$end", "无操作数，两个属性"),
    ("results", "TT_IntTensor:$result", "1D i32 张量"),
    ("assemblyFormat", 'attr-dict `:` type($result)', "先打印属性字典，再冒号+结果类型"),
]

# 分段：(text, tag) tag in {None, 'attr', 'type'} 用于配色 + 底部标注定位
DUMP_SEGMENTS = [
    ("%offs_0 = tt.make_range ", None),
    ("{end = 4 : i32, start = 0 : i32}", "attr"),
    (" : ", None),
    ("tensor<4xi32>", "type"),
]
DUMP_LINE = "".join(t for t, _ in DUMP_SEGMENTS)

CALLOUTS = [
    ("attr", "$start / $end 属性 → {start = 0, end = 4}"),
    ("type", "type($result) → tensor<4xi32>（4 来自 BLOCK=4）"),
]

COLOR_ATTR = "#b45309"
COLOR_TYPE = "#1e40af"
TAG_COLOR = {"attr": COLOR_ATTR, "type": COLOR_TYPE, None: "#0f172a"}
TAG_FILL = {"attr": "#fef3c7", "type": "#dbeafe"}
TAG_TEXTCOLOR = {None: "#e2e8f0", "attr": "#fbbf24", "type": "#7dd3fc"}

PAD = 40
TOP = 108
PANEL_GAP = 130
ROW_H = 82  # 每字段三行：字段名 / 值 / 注释
MONO_SIZE = 13
MONO_CHAR_W = MONO_SIZE * 0.62  # 等宽字体单字符步进（ASCII）

# ---- 左面板宽度：取值行、注释行两者最长者
LEFT_W = max(
    max(cjk_w(v, 13) for _, v, _ in BEFORE_LINES),
    max(cjk_w(n, 11.5) for _, _, n in BEFORE_LINES),
) + 32
RIGHT_W = max(len(DUMP_LINE) * MONO_CHAR_W + 40, cjk_w(AFTER_LABEL, 13) + 40, 520)

w = PAD * 2 + LEFT_W + PANEL_GAP + RIGHT_W
panel_h_val = ROW_H * len(BEFORE_LINES)
dump_box_h = 56
CALLOUT_TOP = TOP + dump_box_h + 46
h = CALLOUT_TOP + 46 * len(CALLOUTS) + 50

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{COLOR_ATTR}"/></marker>'
     '<marker id="ty" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{COLOR_TYPE}"/></marker>'
     '</defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="monospace" font-size="12.5" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

# ---- 左面板：.td 三元组（每字段三行：名称 / 值 / 注释）----
lx = PAD
L.append(f'<text x="{lx}" y="{TOP-10}" font-family="sans-serif" font-size="14" '
          f'font-weight="bold" fill="#0f172a">{esc(BEFORE_LABEL)}</text>')
L.append(f'<rect x="{lx}" y="{TOP}" width="{LEFT_W:.0f}" height="{panel_h_val}" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.3"/>')
for i, (field, val, note) in enumerate(BEFORE_LINES):
    ry = TOP + i * ROW_H
    if i > 0:
        L.append(f'<line x1="{lx}" y1="{ry}" x2="{lx+LEFT_W:.0f}" y2="{ry}" '
                  'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{lx+14}" y="{ry+22}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="#1e40af">{esc(field)}</text>')
    L.append(f'<text x="{lx+14}" y="{ry+42}" font-family="monospace" font-size="13" '
              f'fill="#0f172a">{esc(val)}</text>')
    L.append(f'<text x="{lx+14}" y="{ry+62}" font-family="sans-serif" font-size="11.5" '
              f'fill="#94a3b8">{esc(note)}</text>')

# ---- 右面板：dump 行（等宽步进定位，逐段配色）----
rx = PAD + LEFT_W + PANEL_GAP
L.append(f'<text x="{rx:.0f}" y="{TOP-10}" font-family="sans-serif" font-size="14" '
          f'font-weight="bold" fill="#0f172a">{esc(AFTER_LABEL)}</text>')
L.append(f'<rect x="{rx:.0f}" y="{TOP:.0f}" width="{RIGHT_W:.0f}" height="{dump_box_h}" rx="8" '
          'fill="#0f172a"/>')
seg_x = rx + 16
seg_y = TOP + dump_box_h / 2 + 5
# 记录每个 tag 段的中心 x（供下方标注箭头定位）
tag_center_x = {}
for text, tag in DUMP_SEGMENTS:
    seg_w = len(text) * MONO_CHAR_W
    L.append(f'<text x="{seg_x:.1f}" y="{seg_y:.0f}" font-family="monospace" '
              f'font-size="{MONO_SIZE}" fill="{TAG_TEXTCOLOR[tag]}" '
              f'xml:space="preserve">{esc(text)}</text>')
    if tag is not None:
        tag_center_x[tag] = seg_x + seg_w / 2
    seg_x += seg_w

# 中间箭头：before → after（对齐 dump 框纵向中心，避开下方标注区）
mid_y = TOP + dump_box_h / 2
L.append(f'<line x1="{lx+LEFT_W:.0f}" y1="{mid_y:.0f}" x2="{rx-8:.0f}" y2="{mid_y:.0f}" '
          'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(lx+LEFT_W+rx)/2:.0f}" y="{mid_y-10:.0f}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#64748b">assemblyFormat</text>')

# ---- 下方标注：从 dump 里对应高亮段正下方垂直引出，接到各自标注框 ----
cy0 = TOP + dump_box_h
box_left = rx
for i, (tag, text) in enumerate(CALLOUTS):
    cy = CALLOUT_TOP + i * 46
    color = TAG_COLOR[tag]
    marker = "ar" if tag == "attr" else "ty"
    ax = tag_center_x[tag]
    # 箭头起点贴住 dump 高亮段正下方，终点落在标注框上沿、限制在框内范围
    ax_clamped = min(max(ax, box_left + 24), box_left + RIGHT_W - 24)
    L.append(f'<line x1="{ax:.1f}" y1="{cy0:.0f}" x2="{ax_clamped:.1f}" '
              f'y2="{cy-16:.0f}" stroke="{color}" stroke-width="1.6" '
              f'stroke-dasharray="3,3" marker-end="url(#{marker})"/>')
    L.append(f'<rect x="{box_left:.0f}" y="{cy-12:.0f}" width="{RIGHT_W:.0f}" height="30" '
              f'rx="6" fill="{TAG_FILL[tag]}" stroke="{color}" stroke-width="1.3"/>')
    L.append(f'<text x="{box_left+12:.0f}" y="{cy+8:.0f}" font-family="sans-serif" '
              f'font-size="12" fill="{color}">{esc(text)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch19-td-triple-to-dump.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
