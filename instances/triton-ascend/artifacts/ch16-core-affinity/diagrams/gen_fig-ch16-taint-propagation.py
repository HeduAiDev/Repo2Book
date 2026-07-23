#!/usr/bin/env python3
"""before-after 模板:isUpstreamOfCubeMem 沿『喂 cube 的读链』反向染色。
左态=能力(canRunOn 静态判定),右态=传染后放置(isOn 最终结果)。
两排对照:①矩阵乘操作数链(读喂 Cube,被染色) vs ②epilogue 链(bias/addf/store,留 Vector,不传染)。
修订(盲审 FAIL 后补):spec.claim 是完整 before/after 对比论点,原图只画了半边(matmul 链),
本版补上 epilogue 对照排,让两边核态的反差在同一张图上可验证。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
UND_BG = "#f1f5f9"
UND = "#64748b"
TAINT = "#ea580c"
NOTAINT = "#15803d"

BOX_W, BOX_H = 100, 52
PANEL_W = 4 * BOX_W  # 两排都是 4 节点,面板等宽对齐
GAP_MID = 190
PAD = 40
LX = PAD
RX = LX + PANEL_W + GAP_MID
W = RX + PANEL_W + PAD

TITLE_Y = 34
SUBTITLE1_Y = 58
SUBTITLE2_Y = 78
PANEL_TITLE_Y = 106
ROW1_BANNER_Y = 126
ROW1_TOP = 158
ROW1_BOT = ROW1_TOP + BOX_H            # 210
ROW2_BANNER_Y = ROW1_BOT + 32           # 242
ROW2_TOP = ROW2_BANNER_Y + 30           # 272
ROW2_BOT = ROW2_TOP + BOX_H             # 324
LEGEND_Y = ROW2_BOT + 55                # 379
FOOT1_Y = LEGEND_Y + 45                 # 424
FOOT2_Y = FOOT1_Y + 22                  # 446
H = FOOT2_Y + 40                        # 486

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
     '<marker id="at" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{TAINT}"/></marker>'
     '<marker id="an" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     f'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{NOTAINT}"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="18" font-weight="bold" '
     f'fill="{INK}">{esc("isUpstreamOfCubeMem:沿读链反向染色(两条链对照)")}</text>',
     f'<text x="{PAD}" y="{SUBTITLE1_Y}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">'
     f'{esc("① matmul 操作数链 %pa→load_a→%a→dot 被染成 Cube(对称链 %pb→load_b→%b→dot 未画)")}</text>',
     f'<text x="{PAD}" y="{SUBTITLE2_Y}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">'
     f'{esc("② epilogue 链 %bias→addf→%d→store 全程留在 Vector,不受传染(输出指针 %po 同落 Vector,见图下文字)")}</text>']


def draw_row(px, top, chain, states, taints):
    n = len(chain)
    xs_ = [px + i * BOX_W for i in range(n)]
    cy = top + BOX_H / 2
    for i, name in enumerate(chain):
        x = xs_[i]
        label, color, bg = states[i]
        L.append(f'<rect x="{x}" y="{top}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{bg}" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{x+BOX_W/2}" y="{top+21}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{INK}">{esc(name)}</text>')
        L.append(f'<text x="{x+BOX_W/2}" y="{top+40}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="{color}">{esc(label)}</text>')
        if taints[i]:
            bx, by = x + BOX_W - 9, top - 8
            L.append(f'<circle cx="{bx}" cy="{by}" r="10" fill="{TAINT}"/>')
            L.append(f'<text x="{bx}" y="{by+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" font-weight="bold" '
                      f'fill="white">T</text>')
        if i < n - 1:
            L.append(f'<line x1="{x+BOX_W}" y1="{cy}" x2="{xs_[i+1]}" y2="{cy}" '
                      f'stroke="#94a3b8" stroke-width="1.8" marker-end="url(#a)"/>')
    return xs_


# ---- 面板标题 ----
L.append(f'<text x="{LX+PANEL_W/2}" y="{PANEL_TITLE_Y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{INK}">'
          f'{esc("左:能力(canRunOn 静态判定)")}</text>')
L.append(f'<text x="{RX+PANEL_W/2}" y="{PANEL_TITLE_Y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{INK}">'
          f'{esc("右:传染后放置(isOn 最终结果)")}</text>')

# ---- 行组横幅(跨两栏) ----
L.append(f'<text x="{W/2}" y="{ROW1_BANNER_Y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="{TAINT}">'
          f'{esc("① 矩阵乘操作数链 —— 会被传染")}</text>')
L.append(f'<text x="{W/2}" y="{ROW2_BANNER_Y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="{NOTAINT}">'
          f'{esc("② epilogue 链 —— 全程 Vector,不传染")}</text>')

CUBE_CHAIN = ["%pa", "load_a", "%a", "dot"]
VEC_CHAIN = ["%bias", "addf", "%d", "store"]

LEFT_ROW1 = [
    ("(值,无 ability)", UND, UND_BG),
    ("PREFER_VECTOR", VEC, VEC_BG),
    ("(值,无 ability)", UND, UND_BG),
    ("CUBE_ONLY", CUBE, CUBE_BG),
]
LEFT_ROW1_T = [False, False, False, False]
RIGHT_ROW1 = [
    ("CUBE_ONLY(传染)", CUBE, CUBE_BG),
    ("CUBE_ONLY(传染)", CUBE, CUBE_BG),
    ("CUBE_ONLY(传染)", CUBE, CUBE_BG),
    ("CUBE_ONLY(硬钉)", CUBE, CUBE_BG),
]
RIGHT_ROW1_T = [True, True, True, False]

LEFT_ROW2 = [
    ("(值,无 ability)", UND, UND_BG),
    ("PREFER_VECTOR", VEC, VEC_BG),
    ("(值,无 ability)", UND, UND_BG),
    ("PREFER_VECTOR", VEC, VEC_BG),
]
LEFT_ROW2_T = [False, False, False, False]
RIGHT_ROW2 = [
    ("VECTOR_ONLY", VEC, VEC_BG),
    ("VECTOR_ONLY", VEC, VEC_BG),
    ("VECTOR_ONLY", VEC, VEC_BG),
    ("VECTOR_ONLY", VEC, VEC_BG),
]
RIGHT_ROW2_T = [False, False, False, False]

draw_row(LX, ROW1_TOP, CUBE_CHAIN, LEFT_ROW1, LEFT_ROW1_T)
draw_row(RX, ROW1_TOP, CUBE_CHAIN, RIGHT_ROW1, RIGHT_ROW1_T)
draw_row(LX, ROW2_TOP, VEC_CHAIN, LEFT_ROW2, LEFT_ROW2_T)
draw_row(RX, ROW2_TOP, VEC_CHAIN, RIGHT_ROW2, RIGHT_ROW2_T)

# ---- 中间连接箭头 ----
mid_cx = (LX + PANEL_W + RX) / 2

row1_cy = ROW1_TOP + BOX_H / 2
L.append(f'<line x1="{LX+PANEL_W+12}" y1="{row1_cy}" x2="{RX-12}" y2="{row1_cy}" '
          f'stroke="{TAINT}" stroke-width="2.5" marker-end="url(#at)"/>')
L.append(f'<text x="{mid_cx}" y="{row1_cy-30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{TAINT}">'
          f'{esc("READ 触发")}</text>')
L.append(f'<text x="{mid_cx}" y="{row1_cy-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{TAINT}">'
          f'{esc("taint 反向传播")}</text>')

row2_cy = ROW2_TOP + BOX_H / 2
L.append(f'<line x1="{LX+PANEL_W+12}" y1="{row2_cy}" x2="{RX-12}" y2="{row2_cy}" '
          f'stroke="{NOTAINT}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#an)"/>')
L.append(f'<text x="{mid_cx}" y="{row2_cy-30}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{NOTAINT}">'
          f'{esc("无 cube 数据源")}</text>')
L.append(f'<text x="{mid_cx}" y="{row2_cy-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="{NOTAINT}">'
          f'{esc("能力=放置,不传染")}</text>')

# ---- 图例 ----
legend = [("CUBE_ONLY", CUBE, CUBE_BG), ("VECTOR_ONLY/PREFER_VECTOR", VEC, VEC_BG),
          ("值,无 ability", UND, UND_BG), ("T=taint(isUpstreamOfCubeMem)", TAINT, None)]
lx = PAD
for name, color, bg in legend:
    if bg:
        L.append(f'<rect x="{lx}" y="{LEGEND_Y}" width="18" height="18" rx="3" fill="{bg}" '
                  f'stroke="{color}"/>')
    else:
        L.append(f'<circle cx="{lx+9}" cy="{LEGEND_Y+9}" r="9" fill="{color}"/>')
    L.append(f'<text x="{lx+26}" y="{LEGEND_Y+14}" font-family="sans-serif" font-size="12" '
              f'fill="{INK}">{esc(name)}</text>')
    lx += 26 + 9.5 * len(name) + 24

# ---- 脚注 ----
L.append(f'<text x="{PAD}" y="{FOOT1_Y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("触发首跳的 memPolicy 条件:READ(DAG.cpp:L379-387);load_a 的原始能力是 PREFER_VECTOR(Default 臂,DAG.cpp:L176-177)")}</text>')
L.append(f'<text x="{PAD}" y="{FOOT2_Y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("本例共 6 个节点被拉向 Cube(%pa %pb load_a load_b %a %b,仅画一路对称链);epilogue 链(%bias addf %d store,输出指针 %po 同落 Vector)全程 VECTOR_ONLY,不被 taint。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-taint-propagation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
