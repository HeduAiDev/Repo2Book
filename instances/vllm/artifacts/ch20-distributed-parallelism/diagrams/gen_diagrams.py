#!/usr/bin/env python3
"""Generate ch20 distributed-parallelism diagrams as SVG."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

BLUE = "#2563eb"      # CPU / gloo path
BLUE_BG = "#dbeafe"
ORANGE = "#ea580c"    # device / NCCL path
ORANGE_BG = "#ffedd5"
GREY = "#64748b"
DARK = "#1e293b"
GREEN = "#16a34a"
GREEN_BG = "#dcfce7"
PURPLE = "#7c3aed"
PURPLE_BG = "#ede9fe"


def box(x, y, w, h, fill, stroke, rx=6, sw=2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, size=14, fill=DARK, anchor="middle", weight="normal", family="sans-serif", style=""):
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{st}>{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke=GREY, sw=2, marker=True, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>'


def defs(color=GREY):
    return ('<defs>'
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
            f'<marker id="arrowo" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{ORANGE}"/></marker>'
            f'<marker id="arrowb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{BLUE}"/></marker>'
            '</defs>')


def svg_wrap(w, h, body):
    return ('\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
        defs(),
        f'<rect width="{w}" height="{h}" fill="white"/>',
        body,
        '</svg>']))


# ============================================================
# Diagram 1: GroupCoordinator anatomy
# ============================================================
def diagram1():
    w, h = 880, 560
    P = []
    P.append(txt(w/2, 30, "一个并行维度 = 一对进程组 + 一个 device communicator", 18, DARK, weight="bold"))

    # Right: 4 ghost instances (TP/PP/DP/EP)
    ghost_x = 700
    P.append(txt(ghost_x + 80, 70, "每维度一个同构实例", 12, GREY))
    labels = ["TP", "PP", "DP", "EP"]
    for i, lb in enumerate(labels):
        gx = ghost_x + i * 6
        gy = 88 + i * 6
        P.append(box(gx, gy, 150, 70, "white", "#cbd5e1", rx=6, sw=1.5))
    P.append(txt(ghost_x + 75 + 18, 130, "TP / PP / DP / EP", 13, GREY, weight="bold"))

    # Center: GroupCoordinator box
    gc_x, gc_y, gc_w, gc_h = 60, 90, 540, 150
    P.append(box(gc_x, gc_y, gc_w, gc_h, "#f8fafc", DARK, rx=10, sw=2.5))
    P.append(txt(gc_x + 16, gc_y + 28, "GroupCoordinator  (此处：TP)", 16, DARK, anchor="start", weight="bold"))
    fields = [
        "ranks = [0, 1]      world_size = 2",
        "rank_in_group = 本进程在组内的序号",
        "use_custom_op_call = 是否走 torch.ops.vllm.*",
    ]
    for i, f in enumerate(fields):
        P.append(txt(gc_x + 22, gc_y + 58 + i * 28, f, 13, "#334155", anchor="start", family="monospace"))

    # Two process groups below
    cpu_x, cpu_y, pw, ph = 80, 320, 240, 110
    dev_x = 520
    # cpu_group (blue)
    P.append(line(gc_x + 140, gc_y + gc_h, cpu_x + pw/2, cpu_y, BLUE, 2.5))
    P.append(box(cpu_x, cpu_y, pw, ph, BLUE_BG, BLUE, rx=8))
    P.append(txt(cpu_x + pw/2, cpu_y + 26, "cpu_group  (gloo)", 15, BLUE, weight="bold"))
    P.append(txt(cpu_x + pw/2, cpu_y + 52, "metadata / 对象广播", 12, "#1e40af"))
    P.append(txt(cpu_x + pw/2, cpu_y + 72, "broadcast_object", 12, "#1e40af", family="monospace"))
    P.append(txt(cpu_x + pw/2, cpu_y + 92, "barrier", 12, "#1e40af", family="monospace"))

    # device_group (orange)
    P.append(line(gc_x + 400, gc_y + gc_h, dev_x + pw/2, cpu_y, ORANGE, 2.5))
    P.append(box(dev_x, cpu_y, pw, ph, ORANGE_BG, ORANGE, rx=8))
    P.append(txt(dev_x + pw/2, cpu_y + 26, "device_group  (NCCL)", 15, ORANGE, weight="bold"))
    P.append(txt(dev_x + pw/2, cpu_y + 52, "张量集合通信", 12, "#9a3412"))
    P.append(txt(dev_x + pw/2, cpu_y + 72, "all_reduce / all_gather", 12, "#9a3412", family="monospace"))
    P.append(txt(dev_x + pw/2, cpu_y + 92, "reduce_scatter / send / recv", 11, "#9a3412", family="monospace"))

    # device_communicator hanging off device_group
    dc_x, dc_y, dcw, dch = 470, 470, 360, 70
    P.append(line(dev_x + pw/2, cpu_y + ph, dc_x + dcw/2, dc_y, ORANGE, 2.5))
    P.append(box(dc_x, dc_y, dcw, dch, "white", ORANGE, rx=8, sw=2))
    P.append(txt(dc_x + dcw/2, dc_y + 24, "device_communicator (CudaCommunicator)", 13, "#9a3412", weight="bold"))
    P.append(txt(dc_x + dcw/2, dc_y + 48, "CustomAllreduce → pynccl → torch.distributed", 12, "#9a3412", family="monospace"))

    # legend
    P.append(box(70, 470, 120, 24, BLUE_BG, BLUE, rx=4, sw=1.5))
    P.append(txt(130, 487, "CPU 路径", 12, BLUE))
    P.append(box(70, 504, 120, 24, ORANGE_BG, ORANGE, rx=4, sw=1.5))
    P.append(txt(130, 521, "device 路径", 12, ORANGE))

    return svg_wrap(w, h, '\n'.join(P))


# ============================================================
# Diagram 2: all_reduce dispatch paths
# ============================================================
def diagram2():
    w, h = 900, 720
    P = []
    P.append(txt(w/2, 30, "all_reduce 的两条派发路径", 18, DARK, weight="bold"))

    cx = w/2
    # entry
    def node(y, label, sub="", fill="#f1f5f9", stroke=DARK, ww=320, fs=14):
        x = cx - ww/2
        hh = 56 if sub else 42
        P.append(box(x, y, ww, hh, fill, stroke, rx=8))
        if sub:
            P.append(txt(cx, y + 24, label, fs, DARK, weight="bold"))
            P.append(txt(cx, y + 44, sub, 12, "#475569", family="monospace"))
        else:
            P.append(txt(cx, y + 26, label, fs, DARK, weight="bold"))
        return y + hh

    y = 56
    y2 = node(y, "tensor_model_parallel_all_reduce(x)", "communication_op.py", fill=GREEN_BG, stroke=GREEN)
    P.append(line(cx, y2, cx, y2 + 24))
    y = y2 + 24
    y2 = node(y, "get_tp_group().all_reduce(x)", "parallel_state.py", fill="#f1f5f9")
    P.append(line(cx, y2, cx, y2 + 24))
    y = y2 + 24

    # world_size == 1 short circuit
    dec_w = 260
    P.append(box(cx - dec_w/2, y, dec_w, 42, "#fef9c3", "#ca8a04", rx=8))
    P.append(txt(cx, y + 26, "world_size == 1 ?", 14, "#854d0e", weight="bold"))
    # short circuit to right
    P.append(line(cx + dec_w/2, y + 21, cx + dec_w/2 + 90, y + 21, GREY, 2))
    P.append(box(cx + dec_w/2 + 90, y, 150, 42, "white", GREY, rx=8, sw=1.5))
    P.append(txt(cx + dec_w/2 + 90 + 75, y + 20, "是 → 原样返回", 12, GREY))
    P.append(txt(cx + dec_w/2 + 90 + 75, y + 35, "（单卡短路）", 11, GREY))
    P.append(line(cx, y + 42, cx, y + 42 + 22))
    P.append(txt(cx + 14, y + 42 + 16, "否", 12, "#854d0e", anchor="start"))
    y = y + 42 + 22

    # use_custom_op_call branch
    P.append(box(cx - dec_w/2, y, dec_w, 42, "#fef9c3", "#ca8a04", rx=8))
    P.append(txt(cx, y + 26, "use_custom_op_call ?", 14, "#854d0e", weight="bold"))
    branch_y = y + 42
    y = branch_y + 36

    left_x = cx - 230
    right_x = cx + 230
    bw = 300
    # left = custom-op path (true)
    P.append(line(cx - dec_w/2, branch_y - 21, left_x, y, BLUE, 2.5))
    P.append(txt(cx - dec_w/2 - 80, branch_y + 4, "是（编译友好）", 12, BLUE, anchor="middle"))
    P.append(box(left_x - bw/2, y, bw, 80, BLUE_BG, BLUE, rx=8))
    P.append(txt(left_x, y + 22, "torch.ops.vllm.all_reduce", 13, BLUE, weight="bold", family="monospace"))
    P.append(txt(left_x, y + 42, "(x, group_name=unique_name)", 11, "#1e40af", family="monospace"))
    P.append(txt(left_x, y + 64, "只传字符串 → 运行期查回组", 11, "#1e40af"))

    # under custom-op: module-level lookup
    yy = y + 80 + 26
    P.append(line(left_x, y + 80, left_x, yy, BLUE, 2.5))
    P.append(box(left_x - bw/2, yy, bw, 60, "white", BLUE, rx=8, sw=2))
    P.append(txt(left_x, yy + 22, "模块级 all_reduce(x, group_name)", 12, BLUE, weight="bold", family="monospace"))
    P.append(txt(left_x, yy + 42, "_groups[group_name]() → group", 11, "#1e40af", family="monospace"))

    # right = direct path (false)
    P.append(line(cx + dec_w/2, branch_y - 21, right_x, y, ORANGE, 2.5))
    P.append(txt(cx + dec_w/2 + 80, branch_y + 4, "否（直接）", 12, ORANGE, anchor="middle"))
    P.append(box(right_x - bw/2, y, bw, 80, ORANGE_BG, ORANGE, rx=8))
    P.append(txt(right_x, y + 30, "直接调", 13, ORANGE, weight="bold"))
    P.append(txt(right_x, y + 54, "_all_reduce_out_place(x)", 13, ORANGE, weight="bold", family="monospace"))

    # converge to _all_reduce_out_place
    conv_y = yy + 60 + 40
    P.append(box(cx - 180, conv_y, 360, 50, "#fff7ed", ORANGE, rx=8, sw=2.5))
    P.append(txt(cx, conv_y + 30, "group._all_reduce_out_place(x)", 14, ORANGE, weight="bold", family="monospace"))
    P.append(line(left_x, yy + 60, cx - 110, conv_y, BLUE, 2.5))
    P.append(line(right_x, y + 80, cx + 110, conv_y, ORANGE, 2.5))

    # device communicator
    y = conv_y + 50 + 24
    P.append(line(cx, conv_y + 50, cx, y, ORANGE, 2.5))
    P.append(box(cx - 230, y, 460, 56, ORANGE_BG, ORANGE, rx=8))
    P.append(txt(cx, y + 22, "device_communicator.all_reduce(x)", 13, "#9a3412", weight="bold", family="monospace"))
    P.append(txt(cx, y + 42, "CustomAllreduce → pynccl → torch.distributed（device_group 上）", 11, "#9a3412"))

    return svg_wrap(w, h, '\n'.join(P))


# ============================================================
# Diagram 3: 5D rank tensor → groups
# ============================================================
def diagram3():
    w, h = 900, 600
    P = []
    P.append(txt(w/2, 28, "把 8 个 rank 切成各并行群组（TP=2, PP=2, DP=2）", 18, DARK, weight="bold"))
    P.append(txt(w/2, 50, "all_ranks = arange(8).reshape(DP, PP, TP)", 12, GREY, family="monospace"))

    # grid of 8 ranks: layout DP(rows of 4) x PP x TP. We'll draw as 2 DP blocks.
    # rank index = dp*4 + pp*2 + tp
    cell = 56
    gap = 10
    top = 80
    left = 60
    # draw 8 cells in a 2x4 arrangement grouped: [DP0: pp0(tp0,tp1) pp1(tp0,tp1)] [DP1: ...]
    coords = {}
    for dp in range(2):
        for pp in range(2):
            for tp in range(2):
                r = dp * 4 + pp * 2 + tp
                col = pp * 2 + tp
                x = left + col * (cell + gap) + dp * 30
                y = top + dp * (cell + gap)
                coords[r] = (x, y)
                P.append(box(x, y, cell, cell, "#f1f5f9", DARK, rx=6))
                P.append(txt(x + cell/2, y + cell/2 + 5, f"g{r}", 16, DARK, weight="bold"))
    P.append(txt(left + 2*(cell+gap), top - 10, "← 同一行 = 同一 DP 副本，TP 相邻 →", 11, GREY, anchor="middle"))

    # Tables of group_ranks
    tbl_y = top + 2 * (cell + gap) + 40
    def table(x, y, title, op, groups, color, bg):
        tw = 240
        P.append(box(x, y, tw, 150, bg, color, rx=8))
        P.append(txt(x + tw/2, y + 24, title, 14, color, weight="bold"))
        P.append(txt(x + tw/2, y + 44, op, 11, "#475569", family="monospace"))
        for i, g in enumerate(groups):
            P.append(txt(x + tw/2, y + 70 + i * 20, g, 13, DARK, family="monospace"))

    table(40, tbl_y, "TP 组", "view(-1, 2)",
          ["[g0, g1]   [g2, g3]", "[g4, g5]   [g6, g7]"], GREEN, GREEN_BG)
    table(330, tbl_y, "PP 组", "transpose(1,2).reshape(-1,2)",
          ["[g0, g2]   [g1, g3]", "[g4, g6]   [g5, g7]"], ORANGE, ORANGE_BG)
    table(620, tbl_y, "DP 组", "transpose(0,2).reshape(-1,2)",
          ["[g0, g4]   [g2, g6]", "[g1, g5]   [g3, g7]"], BLUE, BLUE_BG)

    P.append(txt(w/2, tbl_y + 180, "套路统一：把目标维 transpose 到末尾 → reshape 成 2D → unbind 出每个组的 rank 列表", 12, DARK))
    P.append(txt(w/2, tbl_y + 202, "TP 放最内层，同组 rank 相邻 → all_reduce 走机内 NVLink 带宽最高", 12, GREEN, weight="bold"))

    return svg_wrap(w, h, '\n'.join(P))


# ============================================================
# Diagram 4: MultiprocExecutor RPC
# ============================================================
def diagram4():
    w, h = 940, 580
    P = []
    P.append(txt(w/2, 28, "一次 collective_rpc：控制面广播 + 数据面 NCCL + 单点回收", 18, DARK, weight="bold"))

    # Executor box on left
    ex_x, ex_y, ex_w, ex_h = 40, 90, 220, 200
    P.append(box(ex_x, ex_y, ex_w, ex_h, "#f8fafc", DARK, rx=10, sw=2.5))
    P.append(txt(ex_x + ex_w/2, ex_y + 28, "MultiprocExecutor", 15, DARK, weight="bold"))
    P.append(box(ex_x + 16, ex_y + 46, ex_w - 32, 40, GREEN_BG, GREEN, rx=6))
    P.append(txt(ex_x + ex_w/2, ex_y + 64, "rpc_broadcast_mq", 12, GREEN, weight="bold", family="monospace"))
    P.append(txt(ex_x + ex_w/2, ex_y + 80, "(共享内存，输入端)", 11, "#15803d"))
    P.append(box(ex_x + 16, ex_y + 100, ex_w - 32, 40, PURPLE_BG, PURPLE, rx=6))
    P.append(txt(ex_x + ex_w/2, ex_y + 118, "response_mqs[]", 12, PURPLE, weight="bold", family="monospace"))
    P.append(txt(ex_x + ex_w/2, ex_y + 134, "(每 worker 一个，输出端)", 11, "#6d28d9"))
    P.append(txt(ex_x + ex_w/2, ex_y + 168, "FutureWrapper.result()", 11, GREY, family="monospace"))
    P.append(txt(ex_x + ex_w/2, ex_y + 186, "← 取回最终输出", 11, GREY))

    # 4 workers
    wk_x0 = 420
    wk_w, wk_h = 160, 90
    wk_gap = 22
    wk_ys = [70 + i * (wk_h + wk_gap) for i in range(4)]
    for i, wy in enumerate(wk_ys):
        is_out = (i == 3)
        P.append(box(wk_x0, wy, wk_w, wk_h, "#fff7ed" if is_out else "white", ORANGE if is_out else GREY, rx=8, sw=2.5 if is_out else 2))
        P.append(txt(wk_x0 + wk_w/2, wy + 22, f"WorkerProc r{i}", 13, DARK, weight="bold"))
        P.append(txt(wk_x0 + wk_w/2, wy + 42, "worker_busy_loop", 11, "#475569", family="monospace"))
        P.append(txt(wk_x0 + wk_w/2, wy + 60, "→ worker.execute_model", 10, "#475569", family="monospace"))
        if is_out:
            P.append(txt(wk_x0 + wk_w/2, wy + 80, "output_rank（PP末段 TP0）", 10, ORANGE, weight="bold"))

    # broadcast lines (control plane, solid green) from mq to each worker
    mq_right = ex_x + ex_w
    fan_x = 360
    P.append(line(mq_right, ex_y + 66, fan_x, ex_y + 66, GREEN, 2.5, marker=False))
    for wy in wk_ys:
        P.append(f'<line x1="{fan_x}" y1="{ex_y+66}" x2="{wk_x0}" y2="{wy+wk_h/2}" stroke="{GREEN}" stroke-width="2" marker-end="url(#arrow)"/>')
    P.append(txt(fan_x + 30, 60, "enqueue 一次 → 广播", 11, GREEN, anchor="start", weight="bold"))

    # data plane: NCCL between workers (dashed orange) — TP all_reduce + PP send/recv
    # vertical dashed links between adjacent workers on the right
    dx = wk_x0 + wk_w + 30
    for i in range(3):
        y1 = wk_ys[i] + wk_h/2
        y2 = wk_ys[i + 1] + wk_h/2
        P.append(f'<path d="M{wk_x0+wk_w},{y1} C{dx+20},{y1} {dx+20},{y2} {wk_x0+wk_w},{y2}" fill="none" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>')
    P.append(txt(dx + 40, wk_ys[0] + wk_h, "数据面 NCCL", 12, ORANGE, anchor="start", weight="bold"))
    P.append(txt(dx + 40, wk_ys[0] + wk_h + 18, "TP all_reduce", 11, "#9a3412", anchor="start", family="monospace"))
    P.append(txt(dx + 40, wk_ys[1] + wk_h, "PP send → recv", 11, "#9a3412", anchor="start", family="monospace"))
    P.append(txt(dx + 40, wk_ys[1] + wk_h + 18, "（device_group）", 10, "#9a3412", anchor="start"))

    # response from output_rank worker back to response_mqs (purple)
    oy = wk_ys[3] + wk_h/2
    P.append(f'<path d="M{wk_x0},{oy} C{340},{oy} {340},{ex_y+120} {mq_right},{ex_y+120}" fill="none" stroke="{PURPLE}" stroke-width="2.5" marker-end="url(#arrow)"/>')
    P.append(txt(310, oy + 18, "仅 output_rank 回写", 11, PURPLE, anchor="middle", weight="bold"))

    # legend
    ly = 520
    P.append(line(60, ly, 100, ly, GREEN, 2.5, marker=False))
    P.append(txt(110, ly + 4, "控制面（经 MQ 广播 RPC）", 11, DARK, anchor="start"))
    P.append(f'<line x1="360" y1="{ly}" x2="400" y2="{ly}" stroke="{ORANGE}" stroke-width="2" stroke-dasharray="6 4"/>')
    P.append(txt(410, ly + 4, "数据面（经 NCCL / device_group）", 11, DARK, anchor="start"))
    P.append(line(700, ly, 740, ly, PURPLE, 2.5, marker=False))
    P.append(txt(750, ly + 4, "输出回收", 11, DARK, anchor="start"))

    return svg_wrap(w, h, '\n'.join(P))


import os
HERE = os.path.dirname(os.path.abspath(__file__))
for name, fn in [("01-group-coordinator-anatomy", diagram1),
                 ("02-allreduce-dispatch-path", diagram2),
                 ("03-rank-layout-and-groups", diagram3),
                 ("04-mp-executor-rpc", diagram4)]:
    p = os.path.join(HERE, name + ".svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(fn())
    print("wrote", p)
