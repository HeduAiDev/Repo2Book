#!/usr/bin/env python3
"""before-after 模板:无同步的数据竞争 vs set/wait 握手。左右两个时间轴面板对比。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
BAD = "#b91c1c"
BAD_BG = "#fee2e2"
GOOD = "#15803d"

TITLE = "cube 写 buffer、vector 读同一 buffer:为什么必须同步"
SUBTITLE = "触发同步的核对组合只有 1 种(CUBE_ONLY↔VECTOR_ONLY);一次握手 = 1 set + 1 wait 共 2 个 op(DAGSync.cpp:L646-671)"

PANEL_W, PAD, TOP = 400, 40, 130
GAP = 90
W = PAD * 2 + PANEL_W * 2 + GAP
H = TOP + 340

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>']
# subtitle wraps to two lines
SUB1 = "触发同步的核对组合只有 1 种(CUBE_ONLY↔VECTOR_ONLY);一次握手 = 1 set + 1 wait 共 2 个 op"
SUB2 = "(needVectorCubeSync / insertSyncAndMovement, DAGSync.cpp:L243-247,L646-671)"
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">{esc(SUB1)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+40}" font-family="sans-serif" font-size="12.5" fill="{GRAY}">{esc(SUB2)}</text>')

BOX_W, BOX_H = 150, 46

def panel(px, title, title_color):
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{title_color}">{esc(title)}</text>')
    cube_x = px + 20
    vec_x = px + PANEL_W - 20 - BOX_W
    lane_y = TOP + 20
    L.append(f'<rect x="{cube_x}" y="{lane_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{CUBE_BG}" stroke="{CUBE}" stroke-width="1.5"/>')
    L.append(f'<text x="{cube_x+BOX_W/2}" y="{lane_y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" fill="{CUBE}">CUBE</text>')
    L.append(f'<rect x="{vec_x}" y="{lane_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{VEC_BG}" stroke="{VEC}" stroke-width="1.5"/>')
    L.append(f'<text x="{vec_x+BOX_W/2}" y="{lane_y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" fill="{VEC}">VECTOR</text>')
    # buffer box below, centered
    buf_y = lane_y + BOX_H + 110
    buf_w = 150
    buf_x = cx - buf_w / 2
    L.append(f'<rect x="{buf_x}" y="{buf_y}" width="{buf_w}" height="40" rx="6" '
              f'fill="#f1f5f9" stroke="{GRAY}" stroke-width="1.3"/>')
    L.append(f'<text x="{cx}" y="{buf_y+25}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" fill="{INK}">共享 buffer</text>')
    return cube_x, vec_x, lane_y, buf_y, buf_w

# ---- LEFT: no sync (race) ----
px = PAD
cube_x, vec_x, lane_y, buf_y, buf_w = panel(px, "无同步", BAD)
cx_c = cube_x + BOX_W / 2
cx_v = vec_x + BOX_W / 2
cx_buf = px + PANEL_W / 2
y0 = lane_y + BOX_H
y1 = buf_y
# both arrows arrive at same y level -> race
L.append(f'<line x1="{cx_c}" y1="{y0}" x2="{cx_buf-10}" y2="{y1}" stroke="{BAD}" '
          f'stroke-width="1.8" marker-end="url(#b)"/>')
L.append(f'<text x="{(cx_c+cx_buf)/2-8}" y="{(y0+y1)/2-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="{BAD}">写(t=?)</text>')
L.append(f'<line x1="{cx_v}" y1="{y0}" x2="{cx_buf+10}" y2="{y1}" stroke="{BAD}" '
          f'stroke-width="1.8" marker-end="url(#b)"/>')
L.append(f'<text x="{(cx_v+cx_buf)/2+10}" y="{(y0+y1)/2-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" fill="{BAD}">读(t=?)</text>')
L.append(f'<text x="{cx_buf}" y="{(y0+y1)/2+16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-style="italic" fill="{BAD}">读写时序未定</text>')
warn_y = buf_y + 70
L.append(f'<rect x="{px}" y="{warn_y}" width="{PANEL_W}" height="52" rx="8" fill="{BAD_BG}" '
          f'stroke="{BAD}" stroke-width="1.2"/>')
L.append(f'<text x="{px+PANEL_W/2}" y="{warn_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="{BAD}">编译能过,数值随机错</text>')
L.append(f'<text x="{px+PANEL_W/2}" y="{warn_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{BAD}">vector 可能读到写一半的半成品</text>')

# ---- RIGHT: with set/wait ----
px2 = PAD + PANEL_W + GAP
cube_x2, vec_x2, lane_y2, buf_y2, buf_w2 = panel(px2, "set/wait 握手", GOOD)
cx_c2 = cube_x2 + BOX_W / 2
cx_v2 = vec_x2 + BOX_W / 2
cx_buf2 = px2 + PANEL_W / 2
y0b = lane_y2 + BOX_H
badge_y = y0b + 34
badge_h = 26
# CUBE -> set badge (write happens, then set)
L.append(f'<line x1="{cx_c2}" y1="{y0b}" x2="{cx_c2}" y2="{badge_y-4}" stroke="{CUBE}" '
          f'stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{cx_c2+56}" y="{(y0b+badge_y)/2+4}" text-anchor="start" '
          f'font-family="sans-serif" font-size="11" fill="{CUBE}">写 buffer</text>')
# VECTOR -> wait badge
L.append(f'<line x1="{cx_v2}" y1="{y0b}" x2="{cx_v2}" y2="{badge_y-4}" stroke="{VEC}" '
          f'stroke-width="1.8" stroke-dasharray="3,3" marker-end="url(#a)"/>')
# set/wait badges
L.append(f'<rect x="{cx_c2-46}" y="{badge_y}" width="92" height="{badge_h}" rx="13" fill="{CUBE}"/>')
L.append(f'<text x="{cx_c2}" y="{badge_y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="white">set(CUBE)</text>')
L.append(f'<rect x="{cx_v2-50}" y="{badge_y}" width="100" height="{badge_h}" rx="13" fill="{VEC}"/>')
L.append(f'<text x="{cx_v2}" y="{badge_y+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="white">wait(VECTOR)</text>')
# arrow set -> wait (same flag) horizontally between badges
hs_y = badge_y + badge_h / 2
L.append(f'<line x1="{cx_c2+50}" y1="{hs_y}" x2="{cx_v2-54}" y2="{hs_y}" stroke="{GOOD}" '
          f'stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#g)"/>')
L.append(f'<text x="{cx_buf2}" y="{hs_y-8}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="{GOOD}">同 flag 配对</text>')
# set badge -> buffer (write completes)
seg_y = badge_y + badge_h
L.append(f'<line x1="{cx_c2}" y1="{seg_y}" x2="{cx_buf2-10}" y2="{buf_y2}" stroke="{CUBE}" '
          f'stroke-width="1.8" marker-end="url(#a)"/>')
# wait badge -> buffer (read only after wait passes)
L.append(f'<line x1="{cx_v2}" y1="{seg_y}" x2="{cx_buf2+10}" y2="{buf_y2}" stroke="{VEC}" '
          f'stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(cx_v2+cx_buf2)/2+20}" y="{(seg_y+buf_y2)/2-4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="{VEC}">读(wait 通过后)</text>')
ok_y = buf_y2 + 70
L.append(f'<rect x="{px2}" y="{ok_y}" width="{PANEL_W}" height="52" rx="8" fill="#dcfce7" '
          f'stroke="{GOOD}" stroke-width="1.2"/>')
L.append(f'<text x="{px2+PANEL_W/2}" y="{ok_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="{GOOD}">读到的一定是写完的数据</text>')
L.append(f'<text x="{px2+PANEL_W/2}" y="{ok_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="{GOOD}">set 在写后、wait 在读前,握手保序</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m2-race-vs-sync.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
