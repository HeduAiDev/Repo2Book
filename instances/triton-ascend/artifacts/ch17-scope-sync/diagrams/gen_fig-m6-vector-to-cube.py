#!/usr/bin/env python3
"""tensor-flow 模板:VECTOR→CUBE 同步+搬运链,比正方向多一层(要进 L1/CBUF 且按 32B 对齐重排nz)。"""
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

TITLE = "VECTOR → CUBE:反方向比正方向多一层——要进 L1(CBUF) 且按 32B 对齐重排 nz"
SUB1 = "srcOp(vector) 后插 set(VECTOR,PIPE_MTE3/PIPE_MTE1) → to_memref(UB) → CBUF(L1) nz alloc → copy → convert_layout(标 ND，物理仍 nz) → dstOp(cube) 前插 wait(CUBE)"
SUB2 = "(insertVectorToCubeDataMovement, DAGSync.cpp:L423-545,L681-683,L702-703)"

NODES = [
    ("srcOp", "vector 算子", VEC_BG, VEC),
    ("set(VECTOR)", "PIPE_MTE3/PIPE_MTE1", SYNC_BG, SYNC),
    ("to_memref", "出 UB", MOVE_BG, MOVE),
    ("CBUF(L1) nz alloc", "32B 对齐分形重排(见下图)", MOVE_BG, MOVE),
    ("copy", "进 CBUF", MOVE_BG, MOVE),
    ("convert_layout", "标 ND，物理仍 nz", MOVE_BG, MOVE),
    ("wait(CUBE)", "PIPE_MTE3/PIPE_MTE1", SYNC_BG, SYNC),
    ("dstOp", "cube 算子", CUBE_BG, CUBE),
]

BOX_W, BOX_H, GAP, PAD, TOP = 168, 78, 30, 40, 150
W = PAD * 2 + BOX_W * len(NODES) + GAP * (len(NODES) - 1)
H = TOP + BOX_H + 210

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" fill="{GRAY}">{esc(SUB1)}</text>',
     f'<text x="{PAD}" y="{PAD+42}" font-family="sans-serif" font-size="11" fill="{GRAY}">{esc(SUB2)}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(len(NODES))]
for i, (name, sub, bg, fg) in enumerate(NODES):
    x = xs_[i]
    y = TOP
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{bg}" stroke="{fg}" stroke-width="1.5"/>')
    # name may need two lines
    words = name.split(" ")
    if len(name) > 12 and len(words) > 1:
        mid = len(words) // 2 + (len(words) % 2)
        line1, line2 = " ".join(words[:mid]), " ".join(words[mid:])
        L.append(f'<text x="{x+BOX_W/2}" y="{y+26}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="bold" fill="{fg}">{esc(line1)}</text>')
        L.append(f'<text x="{x+BOX_W/2}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="bold" fill="{fg}">{esc(line2)}</text>')
        sub_y = y + 60
    else:
        L.append(f'<text x="{x+BOX_W/2}" y="{y+32}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="{fg}">{esc(name)}</text>')
        sub_y = y + 56
    L.append(f'<text x="{x+BOX_W/2}" y="{sub_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.3" fill="{fg}">{esc(sub)}</text>')
    if i < len(NODES) - 1:
        ax1 = x + BOX_W
        ax2 = xs_[i + 1]
        ay = y + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2-5}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

for i, x in enumerate(xs_):
    cx, cy = x + 16, TOP - 16
    L.append(f'<circle cx="{cx}" cy="{cy}" r="12" fill="#3b82f6"/>')
    L.append(f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="white">{i+1}</text>')

leg_y = TOP + BOX_H + 46
LEGEND = [(VEC_BG, VEC, "vector 侧"), (SYNC_BG, SYNC, "同步 op(set/wait)"),
          (MOVE_BG, MOVE, "搬运/重排/转布局 op"), (CUBE_BG, CUBE, "cube 侧")]
lx = PAD
for bg, fg, label in LEGEND:
    L.append(f'<rect x="{lx}" y="{leg_y}" width="16" height="16" rx="3" fill="{bg}" stroke="{fg}"/>')
    L.append(f'<text x="{lx+22}" y="{leg_y+13}" font-family="sans-serif" font-size="11.5" '
              f'fill="{INK}">{esc(label)}</text>')
    lx += 22 + 12 * len(label) + 30

CAP = "反方向比正方向多一层：cube 读数据要进 L1(CBUF) 且按 32B 对齐重排成 nz(见下图)，所以搬运链更长——to_memref 出 UB、copy 进 CBUF、convert_layout 只逻辑标 ND(物理仍是 nz，不拷贝不改字节)，全部搬完 set，cube 侧 wait 到才开算。"
cap_y = leg_y + 46
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m6-vector-to-cube.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
