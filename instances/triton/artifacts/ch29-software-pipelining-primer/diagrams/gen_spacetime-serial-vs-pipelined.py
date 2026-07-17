#!/usr/bin/env python3
"""figure_id: spacetime-serial-vs-pipelined
claim: 软件流水线让不同迭代的不同 stage 在同一时间片并行——num_stages=3 稳态里
t2 时刻迭代0在 dot、迭代1在 wait、迭代2在 load 三色并行,长 load 延迟被别的
迭代的 dot 盖住。
上面板:朴素串行(教学延迟单位 t_load=4,t_dot=1;每迭代 5 单位,严格不重叠)。
下面板:num_stages=3 的时空排期表(derive_schedule.out.json spacetime_ns3_it5,
时间片是逻辑排期格,stage0=load/stage1=wait/stage2=dot;稳态=t2,t3,t4,
并发=3)。数字全部取自 explainer/traces/derive_schedule.out.json。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

# ---------- 数据(全部来自 explainer/traces/derive_schedule.out.json) ----------
T_LOAD, T_DOT = 4, 1          # latency_scan[num_stages=1]: t_load=4, t_dot=1
SERIAL_PER_ITER = T_LOAD + T_DOT   # = 5   (latency_scan[num_stages=1].steady_per_iter)
PIPE_PER_ITER = 3                   # latency_scan[num_stages=3].steady_per_iter

NUM_STAGES = 3
NUM_ITERS = 5
# spacetime_ns3_it5.slots: time_slice -> [(iter, stage, op), ...]
SLOTS = {
    0: [(0, 0, "load")],
    1: [(0, 1, "wait"), (1, 0, "load")],
    2: [(0, 2, "dot"), (1, 1, "wait"), (2, 0, "load")],
    3: [(1, 2, "dot"), (2, 1, "wait"), (3, 0, "load")],
    4: [(2, 2, "dot"), (3, 1, "wait"), (4, 0, "load")],
    5: [(3, 2, "dot"), (4, 1, "wait")],
    6: [(4, 2, "dot")],
}
NUM_CONCURRENT = {0: 1, 1: 2, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1}
STEADY_SLICES = {2, 3, 4}          # spacetime_ns3_it5.steady_slices

STAGE_COLOR = {0: "#3b82f6", 1: "#f59e0b", 2: "#22c55e"}   # load / wait / dot
STAGE_LABEL = {0: "load", 1: "wait", 2: "dot"}

# ---------- 版式常量 ----------
PAD = 40
W = 1180

# --- 上面板:朴素串行 ---
TOP_TITLE_Y = PAD
TOP_Y = TOP_TITLE_Y + 34
UNIT = 34                       # 1 逻辑单位的像素宽
BAR_H = 46
SERIAL_ITERS = 2                 # 只画 2 个迭代示意即可说明"严格不重叠"
serial_x0 = PAD + 10

# --- 下面板:流水线时空表 ---
GAP_BETWEEN_PANELS = 78
LANE_H = 40
LANE_GAP = 6
CELL_W = 118
GRID_X0 = PAD + 130              # 留出左侧"迭代 k"标签列
TIME_AXIS_H = 26

bottom_title_y = TOP_Y + BAR_H + 46 + GAP_BETWEEN_PANELS
grid_top = bottom_title_y + 46
n_time = len(SLOTS)               # t0..t6 -> 7
grid_w = CELL_W * n_time
grid_h = LANE_H * NUM_ITERS + LANE_GAP * (NUM_ITERS - 1)

W = max(W, GRID_X0 + grid_w + PAD + 40)
H = grid_top + grid_h + TIME_AXIS_H + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# ============ 上面板:朴素串行 ============
L.append(f'<text x="{PAD}" y="{TOP_TITLE_Y}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("① 朴素串行:每迭代 load → wait → dot 顺序做完(5 单位/迭代,严格不重叠)")}</text>')

x = serial_x0
lane_y = TOP_Y
L.append(f'<text x="{PAD}" y="{lane_y + BAR_H/2 + 5}" text-anchor="start" '
          f'font-family="sans-serif" font-size="12" fill="#64748b">{esc("Tensor Core")}</text>')
label_col_w = 0
grid_start_x = PAD + 96
x = grid_start_x
for it in range(SERIAL_ITERS):
    load_w = T_LOAD * UNIT
    dot_w = T_DOT * UNIT
    # load 段(空转等待)
    L.append(f'<rect x="{x}" y="{lane_y}" width="{load_w}" height="{BAR_H}" rx="6" '
              f'fill="{STAGE_COLOR[0]}" fill-opacity="0.35" stroke="{STAGE_COLOR[0]}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + load_w/2}" y="{lane_y + BAR_H/2 + 4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#1e3a8a">{esc(f"迭代{it} load(4)")}</text>')
    x += load_w
    # dot 段
    L.append(f'<rect x="{x}" y="{lane_y}" width="{dot_w}" height="{BAR_H}" rx="6" '
              f'fill="{STAGE_COLOR[2]}" stroke="#15803d" stroke-width="1.5"/>')
    L.append(f'<text x="{x + dot_w/2}" y="{lane_y + BAR_H/2 + 4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="white" font-weight="bold">{esc("dot")}</text>')
    x += dot_w
    # 迭代总耗时标注 + 分隔虚线
    seg_x0 = grid_start_x + it * (load_w + dot_w)
    seg_x1 = x
    L.append(f'<line x1="{seg_x1}" y1="{lane_y-6}" x2="{seg_x1}" y2="{lane_y+BAR_H+6}" '
              'stroke="#94a3b8" stroke-dasharray="3,3"/>')
    L.append(f'<text x="{(seg_x0+seg_x1)/2}" y="{lane_y+BAR_H+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#475569">{esc("5 单位")}</text>')

L.append(f'<text x="{x + 30}" y="{lane_y + BAR_H/2 + 4}" font-family="sans-serif" '
          f'font-size="12" fill="#94a3b8">{esc("…")}</text>')
L.append(f'<text x="{grid_start_x}" y="{lane_y + BAR_H + 44}" font-family="sans-serif" '
          f'font-size="12" fill="#b91c1c">{esc("Tensor Core 每迭代空转 4/5 时间(等 load) → 每迭代耗时 5 单位")}</text>')

# ============ 下面板:流水线时空表 ============
L.append(f'<text x="{PAD}" y="{bottom_title_y}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc("② num_stages=3 稳态:不同迭代的不同 stage 在同一时间片并行(每迭代 3 单位)")}</text>')

# 稳态背景高亮带(t2,t3,t4)
for t in STEADY_SLICES:
    cx = GRID_X0 + t * CELL_W
    L.append(f'<rect x="{cx}" y="{grid_top-8}" width="{CELL_W}" height="{grid_h+16}" '
              'fill="#ecfdf5" stroke="none"/>')

# 泳道标签(迭代 0..4) + 网格线
for k in range(NUM_ITERS):
    ly = grid_top + k * (LANE_H + LANE_GAP)
    L.append(f'<text x="{GRID_X0 - 14}" y="{ly + LANE_H/2 + 4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(f"迭代 {k}")}</text>')

# 每个时间片列 -> 每个活动 (iter, stage, op) 画一个色块
for t, active in SLOTS.items():
    cx = GRID_X0 + t * CELL_W
    for (it, stage, op) in active:
        ly = grid_top + it * (LANE_H + LANE_GAP)
        L.append(f'<rect x="{cx+4}" y="{ly+3}" width="{CELL_W-8}" height="{LANE_H-6}" rx="6" '
                  f'fill="{STAGE_COLOR[stage]}" fill-opacity="0.85" stroke="#1e293b" stroke-width="1"/>')
        L.append(f'<text x="{cx+CELL_W/2}" y="{ly+LANE_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="white" '
                  f'font-weight="bold">{esc(STAGE_LABEL[stage])}</text>')

# 时间轴 + 并发数 + 相位
axis_y = grid_top + grid_h + 6
L.append(f'<line x1="{GRID_X0}" y1="{axis_y}" x2="{GRID_X0+grid_w}" y2="{axis_y}" stroke="#334155"/>')
for t in range(n_time):
    cx = GRID_X0 + t * CELL_W
    L.append(f'<text x="{cx+CELL_W/2}" y="{axis_y+16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#334155">{esc(f"t{t}")}</text>')
    conc = NUM_CONCURRENT[t]
    phase = "稳态(满并发)" if t in STEADY_SLICES else ("prologue 填流水" if t < min(STEADY_SLICES) else "epilogue 排空")
    L.append(f'<text x="{cx+CELL_W/2}" y="{axis_y+34}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#475569">{esc(f"并发={conc}")}</text>')
    color = "#15803d" if t in STEADY_SLICES else "#92400e"
    L.append(f'<text x="{cx+CELL_W/2}" y="{axis_y+50}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{color}">{esc(phase)}</text>')

# 图例
legend_y = axis_y + 70
legend_items = [(0, "stage0 load(异步引擎)"), (1, "stage1 wait+取数"), (2, "stage2 dot(Tensor Core)")]
lx = GRID_X0
for stage, label in legend_items:
    L.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" rx="3" '
              f'fill="{STAGE_COLOR[stage]}"/>')
    L.append(f'<text x="{lx+22}" y="{legend_y+13}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 20 + 11 * len(label) + 20

L.append(f'<text x="{PAD}" y="{legend_y+34}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("绿色高亮列 = 稳态时间片(t2,t3,t4);此时三种硬件资源(异步引擎/共享内存/Tensor Core)同时被 3 个不同迭代占满")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("spacetime-serial-vs-pipelined.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
