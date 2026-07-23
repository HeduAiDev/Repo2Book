#!/usr/bin/env python3
"""before-after 模板:一个 scf.for -> aiv 副本(裁去 cube 迭代参数/结果) + aic 副本(裁去 vector 的)。
1 对 2 扇出:左侧原骨架,右侧上下两份裁剪副本,两条箭头分别从左侧引出。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
NEU = "#475569"
NEU_BG = "#f1f5f9"
BAD = "#b91c1c"

TITLE = "SplitScope:一个骨架 → 两份按核裁剪的副本(clone+裁剪,非搬移)"
SUB = "aiv 遍丢 CUBE_ONLY 迭代参数/结果、aic 遍丢 VECTOR_ONLY 的;原 op 逆序 erase,先删用后删定义(DAGScope.cpp:L650-670)"

LEFT_W, RIGHT_W, PAD, TOP = 340, 360, 40, 150
GAP = 110
BOX_H = 210
VGAP = 40
W = PAD * 2 + LEFT_W + GAP + RIGHT_W
H = TOP + BOX_H * 2 + VGAP + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB)}</text>']

# ---- LEFT: original scf.for, vertically centered against the two right panels ----
left_h = BOX_H * 2 + VGAP
px = PAD
cx = px + LEFT_W / 2
L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="{NEU}">原 scf.for(切分前)</text>')
L.append(f'<rect x="{px}" y="{TOP}" width="{LEFT_W}" height="{left_h}" rx="10" '
          f'fill="{NEU_BG}" stroke="{NEU}" stroke-width="1.5"/>')
ly = TOP + 30
L.append(f'<text x="{px+20}" y="{ly}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{INK}">scf.for iter_args(</text>')
L.append(f'<text x="{px+20}" y="{ly+26}" font-family="sans-serif" font-size="12" fill="{VEC}">'
          f'  %acc: VECTOR 迭代参数</text>')
L.append(f'<text x="{px+20}" y="{ly+50}" font-family="sans-serif" font-size="12" fill="{CUBE}">'
          f'  %m: CUBE 迭代参数</text>')
L.append(f'<text x="{px+20}" y="{ly+74}" font-family="sans-serif" font-size="12.5" fill="{INK}">)</text>')
L.append(f'<rect x="{px+16}" y="{ly+92}" width="{LEFT_W-32}" height="36" rx="6" '
          f'fill="{CUBE_BG}" stroke="{CUBE}"/>')
L.append(f'<text x="{cx}" y="{ly+115}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="{CUBE}">%c = dot(...) (cube)</text>')
L.append(f'<rect x="{px+16}" y="{ly+136}" width="{LEFT_W-32}" height="36" rx="6" '
          f'fill="{VEC_BG}" stroke="{VEC}"/>')
L.append(f'<text x="{cx}" y="{ly+159}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="{VEC}">%v = addf(%acc,...) (vector)</text>')
L.append(f'<text x="{cx}" y="{ly+196}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="{INK}">yield %v, %c</text>')

# ---- RIGHT top: aiv copy ----
px2 = px + LEFT_W + GAP
cx2 = px2 + RIGHT_W / 2
y_aiv = TOP
L.append(f'<text x="{cx2}" y="{y_aiv-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="{VEC}">aiv 副本(丢 CUBE_ONLY)</text>')
L.append(f'<rect x="{px2}" y="{y_aiv}" width="{RIGHT_W}" height="{BOX_H}" rx="10" '
          f'fill="{VEC_BG}" stroke="{VEC}" stroke-width="1.8"/>')
L.append(f'<text x="{px2+20}" y="{y_aiv+28}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="{INK}">scf.for iter_args(</text>')
L.append(f'<text x="{px2+20}" y="{y_aiv+50}" font-family="sans-serif" font-size="11.5" fill="{VEC}">'
          f'  %acc: VECTOR 迭代参数</text>')
L.append(f'<text x="{px2+20}" y="{y_aiv+72}" font-family="sans-serif" font-size="11.5" fill="{BAD}" '
          f'text-decoration="line-through">  %m: CUBE 迭代参数(裁掉)</text>')
L.append(f'<text x="{px2+20}" y="{y_aiv+94}" font-family="sans-serif" font-size="12" fill="{INK}">)</text>')
L.append(f'<rect x="{px2+16}" y="{y_aiv+112}" width="{RIGHT_W-32}" height="34" rx="6" '
          f'fill="white" stroke="{VEC}"/>')
L.append(f'<text x="{cx2}" y="{y_aiv+134}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{VEC}">%v = addf(%acc,...) (vector)</text>')
L.append(f'<text x="{cx2}" y="{y_aiv+172}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{INK}">yield(裁去 %c 对应结果)</text>')

# ---- RIGHT bottom: aic copy ----
y_aic = TOP + BOX_H + VGAP
L.append(f'<text x="{cx2}" y="{y_aic-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="{CUBE}">aic 副本(丢 VECTOR_ONLY)</text>')
L.append(f'<rect x="{px2}" y="{y_aic}" width="{RIGHT_W}" height="{BOX_H}" rx="10" '
          f'fill="{CUBE_BG}" stroke="{CUBE}" stroke-width="1.8"/>')
L.append(f'<text x="{px2+20}" y="{y_aic+28}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="{INK}">scf.for iter_args(</text>')
L.append(f'<text x="{px2+20}" y="{y_aic+50}" font-family="sans-serif" font-size="11.5" fill="{BAD}" '
          f'text-decoration="line-through">  %acc: VECTOR 迭代参数(裁掉)</text>')
L.append(f'<text x="{px2+20}" y="{y_aic+72}" font-family="sans-serif" font-size="11.5" fill="{CUBE}">'
          f'  %m: CUBE 迭代参数</text>')
L.append(f'<text x="{px2+20}" y="{y_aic+94}" font-family="sans-serif" font-size="12" fill="{INK}">)</text>')
L.append(f'<rect x="{px2+16}" y="{y_aic+112}" width="{RIGHT_W-32}" height="34" rx="6" '
          f'fill="white" stroke="{CUBE}"/>')
L.append(f'<text x="{cx2}" y="{y_aic+134}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{CUBE}">%c = dot(...) (cube)</text>')
L.append(f'<text x="{cx2}" y="{y_aic+172}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{INK}">yield %c</text>')

# ---- fan-out arrows: left box right-edge -> each right box left-edge ----
lx = px + LEFT_W
def arrow(x1, y1, x2, y2, label, color="#64748b"):
    midx = (x1 + x2) / 2
    L.append(f'<path d="M {x1} {y1} L {midx} {y1} L {midx} {y2} L {x2-4} {y2}" fill="none" '
              f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<text x="{midx}" y="{min(y1,y2)-8 if y1==y2 else (y1+y2)/2-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="{color}">{esc(label)}</text>')

arrow(lx, TOP + left_h/2 - 40, px2, y_aiv + BOX_H/2, "aiv 遍", VEC)
arrow(lx, TOP + left_h/2 + 40, px2, y_aic + BOX_H/2, "aic 遍", CUBE)

# reading order badges
L.append(f'<circle cx="{px+16}" cy="{TOP-16}" r="12" fill="#3b82f6"/>')
L.append(f'<text x="{px+16}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="white">1</text>')
L.append(f'<circle cx="{px2+16}" cy="{y_aiv-16}" r="12" fill="#3b82f6"/>')
L.append(f'<text x="{px2+16}" y="{y_aiv-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="white">2</text>')
L.append(f'<circle cx="{px2+16}" cy="{y_aic-16}" r="12" fill="#3b82f6"/>')
L.append(f'<text x="{px2+16}" y="{y_aic-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="white">3</text>')

CAP1 = "不是简单 move 而是 clone+裁剪：循环骨架在两个 scope 各留一份，各自只跑本核那半迭代。"
CAP2 = "裁掉不属本核的 arg/result 是关键——否则副本会引用另一颗核的值；原 op 之后逆序 erase(先删用、后删定义)。"
cap_y = y_aic + BOX_H + 46
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP1)}</text>')
L.append(f'<text x="{PAD}" y="{cap_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP2)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m13-split-rebuild.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
