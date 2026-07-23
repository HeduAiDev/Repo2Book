#!/usr/bin/env python3
"""fig-m8-cross-core-handshake: 跨核握手——Cube 把结果经 FIX 落 gm 后 sync_block_set
置位,Vector sync_block_wait 等到再从 gm 读。三泳道 swimlane:Cube / gm(全局内存)/
Vector。取自 inject-block-sync.mlir @test_block_sync_normal。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "跨核握手:Cube 经 FIX 落 gm 后置位,Vector 等到再读——两颗物理核经 FFTS+gm 交接"
SUBTITLE = "inject-block-sync.mlir @test_block_sync_normal:两侧 tpipe 都是 PIPE_FIX(同一 fixpipe 写 gm 事件配对),第三格都是 PIPE_S"

LANES = ["Cube(AIC)", "global memory (gm)", "Vector(AIV)"]
EVENTS = [
    ("Cube(AIC)", "Cube(AIC)", "matmul → fixpipe(FIX)写结果"),
    ("Cube(AIC)", "global memory (gm)", "结果写入 gm 缓冲(arg2)"),
    ("Cube(AIC)", "Cube(AIC)", "sync_block_set[<CUBE>,<PIPE_FIX>,<PIPE_S>] flag=0"),
    ("Vector(AIV)", "Vector(AIV)", "sync_block_wait[<VECTOR>,<PIPE_FIX>,<PIPE_S>] flag=0"),
    ("global memory (gm)", "Vector(AIV)", "load 读 gm 缓冲(arg2)"),
]
LANE_W, TOP, STEP_H, PAD = 260, 190, 62, 50
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 260
h = TOP + STEP_H * (len(EVENTS) + 1) + 110
X = {name: PAD + 90 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15" '
     f'fill="#1e40af" font-weight="bold">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

lane_colors = {"Cube(AIC)": "#dbeafe", "global memory (gm)": "#fef3c7", "Vector(AIV)": "#dcfce7"}
lane_strokes = {"Cube(AIC)": "#3b82f6", "global memory (gm)": "#d97706", "Vector(AIV)": "#16a34a"}
for name, x in X.items():
    L.append(f'<rect x="{x-95}" y="{TOP-42}" width="190" height="30" rx="6" '
              f'fill="{lane_colors[name]}" stroke="{lane_strokes[name]}" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{TOP-22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-12}" x2="{x}" y2="{TOP+STEP_H*len(EVENTS)+10}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP_H * i + 30
    x1, x2 = X[src], X[dst]
    is_flag = "sync_block" in label
    color = "#b91c1c" if is_flag else "#334155"
    if src == dst:
        L.append(f'<circle cx="{x1}" cy="{y}" r="4.5" fill="{color}"/>')
        anchor = "start" if x1 < w / 2 else "end"
        tx = x1 + 14 if anchor == "start" else x1 - 14
        L.append(f'<text x="{tx}" y="{y+4}" text-anchor="{anchor}" font-family="sans-serif" '
                  f'font-size="11" fill="{color}">{esc(label)}</text>')
    else:
        mk = "url(#r)" if is_flag else "url(#a)"
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
                  f'stroke-width="2" marker-end="{mk}"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{color}">{esc(label)}</text>')

hb_y = TOP + STEP_H * len(EVENTS) + 34
L.append(f'<text x="{(X[LANES[0]]+X[LANES[2]])/2}" y="{hb_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" font-style="italic" '
          f'fill="#0f172a">Cube 写 gm ≺ Vector 读 gm(happens-before,跨核经 FFTS 硬件传递)</text>')

foot_y = h - 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">跨核握手 flag 编号 = 0(inject-block-sync.mlir:L58-L59);'
          f'跨核 flag 全局上限 = 16(kBlockSyncSetWaitEventIdNum,SyncEventIdAllocation.h:L35)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">跨核 block sync 保留 event id = 2'
          f'(reservedBlockSyncEventIdNum,SyncEventIdAllocation.cpp:L170)</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m8-cross-core-handshake.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
