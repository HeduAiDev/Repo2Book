#!/usr/bin/env python3
"""flow 模板改造:absorbCommon 三出口决策流程图。竖向主线(No 继续)+右侧分支(Yes 提前退出)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
DECISION = "#f59e0b"
DECISION_BG = "#fffbeb"
ACTION = "#334155"
ACTION_BG = "white"
EXIT_BG = "#ecfdf5"
EXIT_STROKE = "#047857"

W, H = 1400, 760
PAD = 40
MAIN_X = 260
SIDE_X = 760
BOX_W = 300

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="18" font-weight="bold" '
     f'fill="{INK}">{esc("absorbCommon:后向传递函数的三条出口")}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="13" fill="{GRAY}">'
     f'{esc("入口→能力硬钉→内存语义(WRITE)→否则遍历 outputs 按位或求并,三出口互斥")}</text>']


def rect_box(cx, cy, text_lines, w=BOX_W, h=None, fill=ACTION_BG, stroke=ACTION,
             bold=False):
    if h is None:
        h = 34 + 18 * (len(text_lines) - 1) + 20
    x, y = cx - w / 2, cy - h / 2
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.6"/>')
    n = len(text_lines)
    y0 = cy - (n - 1) * 9 + 4
    for k, line in enumerate(text_lines):
        wt = 'font-weight="bold" ' if bold and k == 0 else ''
        L.append(f'<text x="{cx}" y="{y0+k*18}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" {wt}fill="{INK}">{esc(line)}</text>')
    return (cx, cy, w, h)


def diamond(cx, cy, text_lines, w=BOX_W + 30, h=76):
    pts = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
    L.append(f'<polygon points="{pts}" fill="{DECISION_BG}" stroke="{DECISION}" '
              'stroke-width="1.8"/>')
    n = len(text_lines)
    y0 = cy - (n - 1) * 8 + 4
    for k, line in enumerate(text_lines):
        L.append(f'<text x="{cx}" y="{y0+k*16}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11.5" fill="#92400e">{esc(line)}</text>')
    return (cx, cy, w, h)


def vline(x, y1, y2):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#334155" '
              'stroke-width="1.6" marker-end="url(#a)"/>')


def label(x, y, text, color="#334155", anchor="middle", bold=True):
    wt = 'font-weight="bold" ' if bold else ''
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
              f'font-size="11.5" {wt}fill="{color}">{esc(text)}</text>')


# entry
y1 = 100
rect_box(MAIN_X, y1, ["入口:node.absorbCommon()"], h=40)
vline(MAIN_X, y1 + 20, y1 + 68)

# decision 1: ability == CUBE_ONLY?
y2 = y1 + 106
diamond(MAIN_X, y2, ["ability ==", "CUBE_ONLY ?", "(DAG.cpp:L341-343)"])
# Yes -> exit right
ex1_y = y2
L.append(f'<line x1="{MAIN_X+ (BOX_W+30)/2}" y1="{y2}" x2="{SIDE_X-140}" y2="{ex1_y}" '
          'stroke="#047857" stroke-width="1.8" marker-end="url(#a)"/>')
label((MAIN_X + (BOX_W + 30) / 2 + SIDE_X - 140) / 2, ex1_y - 10, "是", "#047857")
rect_box(SIDE_X, ex1_y, ["return CUBE_ONLY", "(能力硬钉,提前退出)"], w=280,
          fill=EXIT_BG, stroke=EXIT_STROKE, bold=True)
vline(MAIN_X, y2 + 38, y2 + 82)
label(MAIN_X + 14, y2 + 60, "否", "#334155", anchor="start")

# decision 2: memPolicy == WRITE?
y3 = y2 + 120
diamond(MAIN_X, y3, ["memPolicy ==", "WRITE ?", "(DAG.cpp:L357)"])
ex2_y = y3
L.append(f'<line x1="{MAIN_X+ (BOX_W+30)/2}" y1="{y3}" x2="{SIDE_X-140}" y2="{ex2_y}" '
          'stroke="#047857" stroke-width="1.8" marker-end="url(#a)"/>')
label((MAIN_X + (BOX_W + 30) / 2 + SIDE_X - 140) / 2, ex2_y - 10, "是", "#047857")
rect_box(SIDE_X, ex2_y, ["getWriteDataSource()"], w=280, h=40)
vline(MAIN_X, y3 + 38, y3 + 82)
label(MAIN_X + 14, y3 + 60, "否", "#334155", anchor="start")

# side chain: exactlyOneType?
y3b = ex2_y + 96
diamond(SIDE_X, y3b, ["exactlyOneType", "(数据源核)?"], w=280)
L.append(f'<line x1="{SIDE_X}" y1="{ex2_y+20}" x2="{SIDE_X}" y2="{y3b-38}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
ex2b_yes_y = y3b + 96
vline(SIDE_X, y3b + 38, ex2b_yes_y - 22)
rect_box(SIDE_X, ex2b_yes_y, ["return 数据源的核", "(store 跟被存数据走)"], w=280,
          fill=EXIT_BG, stroke=EXIT_STROKE, bold=True)
label(SIDE_X + 18, y3b + 62, "是", "#047857", anchor="start")
ex2c_no_x = SIDE_X + 430
L.append(f'<line x1="{SIDE_X+140}" y1="{y3b}" x2="{ex2c_no_x-115}" y2="{y3b}" '
          'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
label((SIDE_X + 140 + ex2c_no_x - 115) / 2, y3b - 12, "否", "#334155")
rect_box(ex2c_no_x, y3b, ["return VECTOR_ONLY", "(数据源非单核)"], w=230,
          fill=EXIT_BG, stroke=EXIT_STROKE, bold=True)

# decision 3(main line continues): for outputs switch
y4 = y3 + 240
rect_box(MAIN_X, y4, ["for output in outputs:", "switch(output.isOn())",
                      "3 case 按位或:", "CUBE_AND_VECTOR / CUBE_ONLY / VECTOR_ONLY"],
         h=104)
vline(MAIN_X, y3 + 38, y4 - 52)
label(MAIN_X + 14, (y3 + 38 + y4 - 52) / 2, "否(遍历下游诉求)", "#334155", anchor="start")

y5 = y4 + 100
rect_box(MAIN_X, y5, ["return newCoreType(按位或的并)"], w=340, h=40)
vline(MAIN_X, y4 + 52, y5 - 20)

foot_y = H - 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("注:switch 的 CUBE_AND_VECTOR case 无 break,故意 fall-through 到 CUBE_ONLY(DAG.cpp:L374-376)")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("三条出口互斥:能力硬钉 CUBE_ONLY / WRITE 跟数据源(单核则跟、否则 VECTOR_ONLY)/ 否则遍历 outputs 求并")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-absorb-branches.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
