#!/usr/bin/env python3
"""fig-ch10-m3-before-after: 分水岭落地——tt.addptr 的 tensor-of-pointers 被
memref.reinterpret_cast 替换（before-after 模板）。左右两栏共享前两步（指针分析结果一致），
第三步分岔并高亮：ttir 侧仍是 4 路各算各的指针，ttadapter 侧铸成 1 条结构化 memref。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "分水岭落地：tt.addptr → memref.reinterpret_cast"
SUBTITLE = "承 m2 还原结果 (offset=8, sizes=[4], strides=[1])；BlockDataParser::rewriteAddPtr 替换 1 处 tt.addptr"

PANELS = [
    ("ttir：tensor-of-pointers", [
        "tt.splat %x_ptr\n: tensor<4x!tt.ptr<f32>>",
        "偏移已还原（m2）\noffset=8, size=4, stride=1",
        "tt.addptr %xptr_t, %off\n: tensor<4x!tt.ptr<f32>>\n（4 路各算各的门牌号）",
    ], 2, "#fee2e2", "#b91c1c"),
    ("ttadapter：结构化 memref", [
        "tt.splat %x_ptr\n: tensor<4x!tt.ptr<f32>>",
        "偏移已还原（m2）\noffset=8, size=4, stride=1",
        "memref.reinterpret_cast %x_ptr\nto offset:[8], sizes:[4], strides:[1]\n: memref<4xf32,\nstrided<[1], offset:8>>",
    ], 2, "#dcfce7", "#15803d"),
]

BOX_W, PANEL_W, PAD, TOP = 260, 340, 40, 96
GUTTER = 210
STEP_H = [56, 56, 96]
VGAP = 24

h_content = sum(STEP_H) + VGAP * (len(STEP_H) - 1)
w = PAD * 2 + PANEL_W * 2 + GUTTER
h = TOP + h_content + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+16}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

panel_x = []
for p in range(2):
    px = PAD + p * (PANEL_W + GUTTER)
    panel_x.append(px)

step_y = [TOP]
for sh in STEP_H[:-1]:
    step_y.append(step_y[-1] + sh + VGAP)

for p, (title, steps, hot, hl_fill, hl_stroke) in enumerate(PANELS):
    px = panel_x[p]
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = step_y[i]
        bh = STEP_H[i]
        hl = (i == hot)
        fill = hl_fill if hl else "#e2e8f0"
        stroke = hl_stroke if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{bh}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if hl else 1.2}"/>')
        lines = step.split("\n")
        n = len(lines)
        y0 = y + bh / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" '
                      f'fill="{stroke if hl else "#0f172a"}" '
                      f'font-weight="{"bold" if hl else "normal"}">{esc(line)}</text>')
        if i < len(steps) - 1:
            y1 = y + bh
            y2 = step_y[i + 1]
            L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-2}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 中间的转换箭头（贯穿 highlight 行）；箭头两端各留出安全边距，不触碰面板框
gap_left = panel_x[0] + PANEL_W / 2 + BOX_W / 2 + 14
gap_right = panel_x[1] + PANEL_W / 2 - BOX_W / 2 - 14
mid_y = step_y[2] + STEP_H[2] / 2
mid_cx = (gap_left + gap_right) / 2
L.append(f'<line x1="{gap_left}" y1="{mid_y}" x2="{gap_right}" y2="{mid_y}" '
         'stroke="#d97706" stroke-width="2.5" marker-end="url(#ah)"/>')
for k, line in enumerate(["BlockDataParser::", "rewriteAddPtr"]):
    L.append(f'<text x="{mid_cx}" y="{mid_y-30+k*16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="#92400e">{esc(line)}</text>')
L.append(f'<text x="{mid_cx}" y="{mid_y+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#92400e">替换 1 处 tt.addptr</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">同一段访存，从『4 路各自解引用的指针』变成『1 条带 (offset,sizes,strides) 的结构化视图』——昇腾达芬奇吃的正是右边这种 memref</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch10-m3-before-after.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
