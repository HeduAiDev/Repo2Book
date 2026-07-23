#!/usr/bin/env python3
"""tensor-flow 模板:CUBE→VECTOR 同步+搬运链。srcOp(cube)->set->fixpipe(NZ2ND)->UB->wait->dstOp(vector)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
SYNC = "#b45309"
SYNC_BG = "#fef3c7"
MOVE = "#7c3aed"
MOVE_BG = "#ede9fe"

TITLE = "CUBE → VECTOR:同步管时序,搬运管位置+格式"
SUB1 = "srcOp(cube) 后插 set(CUBE,PIPE_FIX/PIPE_V) → fixpipe(NZ2ND) 把 L0C/nz 结果搬进 UB → dstOp(vector) 前插 wait(VECTOR) 才读"
SUB2 = "(insertCubeToVectorDataMovement / SyncBlockSetOp / SyncBlockWaitOp, DAGSync.cpp:L324-325,L649-652,L668-671)"

NODES = [
    ("srcOp", "cube 算子", CUBE_BG, CUBE, "core"),
    ("set(CUBE)", "PIPE_FIX/PIPE_V", SYNC_BG, SYNC, "sync"),
    ("fixpipe", "NZ2ND", MOVE_BG, MOVE, "move"),
    ("UB", "vector 可读地址空间", VEC_BG, VEC, "mem"),
    ("wait(VECTOR)", "PIPE_FIX/PIPE_V", SYNC_BG, SYNC, "sync"),
    ("dstOp", "vector 算子", VEC_BG, VEC, "core"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 200, 74, 46, 40, 150
W = PAD * 2 + BOX_W * len(NODES) + GAP * (len(NODES) - 1)
H = TOP + BOX_H + 200

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12" fill="{GRAY}">{esc(SUB1)}</text>',
     f'<text x="{PAD}" y="{PAD+42}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB2)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(len(NODES))]
for i, (name, sub, bg, fg, kind) in enumerate(NODES):
    x = xs_[i]
    y = TOP
    rx = 12 if kind in ("core", "mem") else 24
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="{rx}" '
              f'fill="{bg}" stroke="{fg}" stroke-width="1.6"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{fg}">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+52}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="{fg}">{esc(sub)}</text>')
    if i < len(NODES) - 1:
        ax1 = x + BOX_W
        ax2 = xs_[i + 1]
        ay = y + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-6}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# reading-order numbers
for i, x in enumerate(xs_):
    cx, cy = x + 18, TOP - 16
    L.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#3b82f6"/>')
    L.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="white">{i+1}</text>')

# legend
leg_y = TOP + BOX_H + 46
LEGEND = [(CUBE_BG, CUBE, "cube 侧核/地址空间"), (SYNC_BG, SYNC, "同步 op(set/wait)"),
          (MOVE_BG, MOVE, "搬运 op(fixpipe)"), (VEC_BG, VEC, "vector 侧核/地址空间")]
lx = PAD
for bg, fg, label in LEGEND:
    L.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="16" rx="3" fill="{bg}" stroke="{fg}"/>')
    L.append(f'<text x="{lx+22}" y="{leg_y+13}" font-family="sans-serif" font-size="11.5" '
              f'fill="{INK}">{esc(label)}</text>')
    lx += 22 + 13 * len(label) + 34

CAP = "同步管时序、搬运管位置+格式：set 在搬运源之后、wait 在搬运目的之前，fixpipe 负责把 nz 布局的 cube 结果 NZ2ND 落进 vector 能读的 UB。少了 fixpipe，vector 连地址空间都够不着。"
cap_y = leg_y + 46
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m5-cube-to-vector.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
