#!/usr/bin/env python3
"""fig-m2-handshake: set_flag/wait_flag 跨引擎握手 vs pipe_barrier 同引擎屏障。
左面板 swimlane(MTE2, V 两条泳道)+ 右面板单泳道 pipe_barrier 对照。
取自 inject-sync.mlir @test_mem_injcet_sync_basic 的真实指令与参数。
全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "跨引擎握手 set_flag/wait_flag vs 同引擎屏障 pipe_barrier"
SUBTITLE = "取自 inject-sync.mlir @test_mem_injcet_sync_basic:两条泳道共享 EVENT_ID0 建立 write ≺ read"

# 左面板:跨引擎握手(泳道)
LANES = ["MTE2(生产者)", "V(消费者)"]
STEPS = [
    ("MTE2(生产者)", "MTE2(生产者)", "load 写 %0"),
    ("MTE2(生产者)", "V(消费者)", "set_flag[<PIPE_MTE2>,<PIPE_V>,<EVENT_ID0>]"),
    ("V(消费者)", "V(消费者)", "wait_flag[<PIPE_MTE2>,<PIPE_V>,<EVENT_ID0>]"),
    ("V(消费者)", "V(消费者)", "vadd 读 %0"),
]

PAD, TOP = 50, 190
LANE_BOX_W, LANE_GAP_X = 190, 260
STEP_H = 54
X = {name: PAD + LANE_BOX_W // 2 + i * LANE_GAP_X for i, name in enumerate(LANES)}
left_panel_right = X[LANES[-1]] + LANE_BOX_W // 2

right_x0 = left_panel_right + 110
right_w = 260

h = TOP + STEP_H * (len(STEPS) + 1) + 130
w = right_x0 + right_w + PAD + 90  # 额外留白:右泳道 pipe_barrier[<PIPE_MTE2>] 标签较长,靠左标注会伸出泳道线

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 左面板标题
L.append(f'<text x="{PAD}" y="{TOP-46}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">① 异引擎:一对 set_flag / wait_flag(共享 EVENT_ID0)</text>')

for name, x in X.items():
    L.append(f'<rect x="{x-LANE_BOX_W/2}" y="{TOP-34}" width="{LANE_BOX_W}" height="26" rx="6" '
              'fill="#e2e8f0" stroke="#334155"/>')
    L.append(f'<text x="{x}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-6}" x2="{x}" y2="{TOP+STEP_H*len(STEPS)+8}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(STEPS):
    y = TOP + STEP_H * i + 24
    x1, x2 = X[src], X[dst]
    is_flag = "set_flag" in label or "wait_flag" in label
    color = "#dc2626" if is_flag else "#334155"
    if src == dst:
        L.append(f'<circle cx="{x1}" cy="{y}" r="4" fill="{color}"/>')
        L.append(f'<text x="{x1+14}" y="{y+4}" font-family="sans-serif" font-size="11.5" '
                  f'fill="{color}">{esc(label)}</text>')
    else:
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
                  f'stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{color}">{esc(label)}</text>')

hb_top = TOP + STEP_H * len(STEPS) + 30
L.append(f'<line x1="{X[LANES[0]]}" y1="{hb_top}" x2="{X[LANES[0]]}" y2="{hb_top+18}" '
          'stroke="#334155" stroke-width="1" stroke-dasharray="2,2"/>')
L.append(f'<text x="{(X[LANES[0]]+X[LANES[1]])/2}" y="{hb_top+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#0f172a" font-style="italic">'
          f'写%0 ≺ 读%0(happens-before)</text>')

# 右面板:pipe_barrier(单泳道)
rx = right_x0 + right_w / 2
L.append(f'<text x="{right_x0}" y="{TOP-46}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">② 同引擎:1 个 pipe_barrier</text>')
L.append(f'<rect x="{rx-LANE_BOX_W/2}" y="{TOP-34}" width="{LANE_BOX_W}" height="26" rx="6" '
          'fill="#e2e8f0" stroke="#334155"/>')
L.append(f'<text x="{rx}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#0f172a">MTE2(同一引擎)</text>')
r_bottom = TOP + STEP_H * 3
L.append(f'<line x1="{rx}" y1="{TOP-6}" x2="{rx}" y2="{r_bottom}" '
          'stroke="#94a3b8" stroke-dasharray="4,4"/>')
r_steps = [("load", False), ("pipe_barrier[<PIPE_MTE2>]", True), ("load(下一批)", False)]
for i, (label, is_barrier) in enumerate(r_steps):
    y = TOP + STEP_H * i + 24
    color = "#d97706" if is_barrier else "#334155"
    L.append(f'<circle cx="{rx}" cy="{y}" r="4" fill="{color}"/>')
    L.append(f'<text x="{rx+14}" y="{y+4}" font-family="sans-serif" font-size="11.5" '
              f'fill="{color}">{esc(label)}</text>')

foot_y = h - 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">set_flag/wait_flag 各 3 个参数(set_pipe, wait_pipe, event_id);'
          f'pipe_barrier 只 1 个参数(pipe)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">基本例握手用 EVENT_ID0(编号 0)——HIVMSynchronizationOps.td:L45-L48/L74</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m2-handshake.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
