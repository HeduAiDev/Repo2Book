#!/usr/bin/env python3
"""swimlane 模板:aic region 与 aiv region 靠 static_flag_id 跨 region 配对的 buffer 就绪握手。
每个事件画成贴在生命线上的卡片(避免侧向文字溢出画布)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"

TITLE = "processFixpipeOpsInAIC:aic/aiv 两 region 靠 static_flag_id 配对的 buffer 就绪握手"
SUB = "每个 fixpipe 触发 3 个正程同步 op(1 set+2 wait) + 1 个回程 set,配对键=static_flag_id(DAGScope.cpp:L756-817,L885-939)"

CARD_W, CARD_H = 300, 52
LANE_GAP, TOP, STEP, PAD = 620, 150, 130, 40
W = PAD * 2 + LANE_GAP + CARD_W
H = TOP + STEP * 3 + CARD_H + 190

x1 = PAD + CARD_W / 2               # aiv lifeline x
x2 = x1 + LANE_GAP                  # aic lifeline x

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB)}</text>']

lane_colors = {"aiv": (VEC_BG, VEC), "aic": (CUBE_BG, CUBE)}
X = {"aiv": x1, "aic": x2}
for name, x in X.items():
    bg, fg = lane_colors[name]
    label = "aiv region" if name == "aiv" else "aic region"
    L.append(f'<rect x="{x-100}" y="{TOP-50}" width="200" height="32" rx="6" fill="{bg}" stroke="{fg}"/>')
    L.append(f'<text x="{x}" y="{TOP-28}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{fg}">{esc(label)}</text>')
    bottom_y = TOP + STEP * 3 + CARD_H + 30
    L.append(f'<line x1="{x}" y1="{TOP-14}" x2="{x}" y2="{bottom_y}" stroke="#cbd5e1" '
              f'stroke-dasharray="4,4"/>')

EVENTS = [
    (1, "aiv", "① aiv 头:set(VECTOR)", "flag = newflag"),
    (2, "aic", "② fixpipe 前:wait(CUBE)", "flag = newflag"),
    (3, "aic", "③ aic 尾 return 前:wait(CUBE)", "flag = newflag(同①)"),
]
centers = {}
# explicit y per step to keep vertical order 1 -> 2 -> 3, step4 below
Y = {1: TOP, 2: TOP + STEP, 3: TOP + STEP * 2}
for step_no, lane, text, flag in EVENTS:
    x = X[lane]
    y = Y[step_no]
    bg, fg = lane_colors[lane]
    centers[step_no] = (x, y + CARD_H / 2)
    L.append(f'<rect x="{x-CARD_W/2}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="8" '
              f'fill="{bg}" stroke="{fg}" stroke-width="1.6"/>')
    L.append(f'<text x="{x}" y="{y+21}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="{fg}">{esc(text)}</text>')
    L.append(f'<text x="{x}" y="{y+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="{fg}">{esc(flag)}</text>')

# step 4: aiv, placed below step3 row
y4 = TOP + STEP * 3
centers[4] = (x1, y4 + CARD_H / 2)
bg4, fg4 = lane_colors["aiv"]
L.append(f'<rect x="{x1-CARD_W/2}" y="{y4}" width="{CARD_W}" height="{CARD_H}" rx="8" '
          f'fill="{bg4}" stroke="{fg4}" stroke-width="1.6" stroke-dasharray="3,3"/>')
L.append(f'<text x="{x1}" y="{y4+21}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="{fg4}">④ 回程:wait(flag2)后插 set(VECTOR)</text>')
L.append(f'<text x="{x1}" y="{y4+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="{fg4}">flag = newflag(供下一轮复用)</text>')

# pairing arrows: card1(right edge) -> card2/card3(left edge), same flag
cx1, cy1 = centers[1]
cx2, cy2 = centers[2]
cx3, cy3 = centers[3]
L.append(f'<line x1="{cx1+CARD_W/2}" y1="{cy1}" x2="{cx2-CARD_W/2-4}" y2="{cy2}" stroke="{VEC}" '
          f'stroke-width="1.6" stroke-dasharray="6,4" marker-end="url(#a)"/>')
L.append(f'<line x1="{cx1+CARD_W/2}" y1="{cy1+6}" x2="{cx3-CARD_W/2-4}" y2="{cy3}" stroke="{VEC}" '
          f'stroke-width="1.3" stroke-dasharray="6,4" marker-end="url(#a)"/>')
L.append(f'<text x="{(cx1+cx2)/2}" y="{(cy1+cy2)/2-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="{VEC}">同 flag 配对</text>')

# reading order numbers
for step_no in [1, 2, 3, 4]:
    cx, cy = centers[step_no]
    bx = cx - CARD_W / 2 - 16
    L.append(f'<circle cx="{bx}" cy="{cy}" r="13" fill="#3b82f6"/>')
    L.append(f'<text x="{bx}" y="{cy+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="white">{step_no}</text>')

note_y = y4 + CARD_H + 26
L.append(f'<text x="{x1-CARD_W/2}" y="{note_y}" font-family="sans-serif" font-size="10.5" '
          f'font-style="italic" fill="{GRAY}">(为下一次同 flag 的 wait 预先释放,本图不展示其配对对象)</text>')

CAP1 = "两条泳道 = 两颗核的 region。箭头是同 flag 的 set→wait 配对：aiv 头 set 告诉 aic「buffer 腾好了」，"
CAP2 = "aic fixpipe 前 wait 等到才搬。每个 wait 都有源，才不会有人干等——这就是不死锁的保证。"
cap_y = note_y + 34
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP1)}</text>')
L.append(f'<text x="{PAD}" y="{cap_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP2)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m14-buffer-wait.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
