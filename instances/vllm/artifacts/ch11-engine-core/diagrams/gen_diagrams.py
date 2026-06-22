#!/usr/bin/env python3
"""ch11 diagrams: step orchestration / busy loop / batch queue / lifecycle states."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
        '<defs>'
        '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
        '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
        'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
        '</defs>')


def box(x, y, w, h, fill, stroke, rx=8, sw=2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, s, size=13, anchor="middle", fill="#0f172a", weight="normal"):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-family="sans-serif" font-weight="{weight}" fill="{fill}">{esc(s)}</text>')


def line(x1, y1, x2, y2, stroke="#64748b", sw=2, marker="a", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def title(w, t, sub=None):
    out = [txt(w // 2, 30, t, 19, weight="bold")]
    if sub:
        out.append(txt(w // 2, 52, sub, 12, fill="#64748b"))
    return out


def write(name, w, h, body):
    svg = HEAD.format(w=w, h=h) + f'<rect width="{w}" height="{h}" fill="white"/>' + ''.join(body) + '</svg>'
    with open(name + ".svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", name + ".svg")


# ---------------------------------------------------------------------------
# 1. step orchestration: CPU bitmask || GPU forward overlap
# ---------------------------------------------------------------------------
def step_orchestration():
    w, h = 880, 540
    B = []
    B += title(w, "EngineCore.step() 一次迭代", "CPU 算掩码 与 GPU 跑前向 在同一段时间重叠")
    # swimlane headers
    cpu_x, gpu_x = 230, 600
    B.append(txt(cpu_x, 90, "主线程 (CPU)", 14, weight="bold", fill="#1d4ed8"))
    B.append(txt(gpu_x, 90, "Worker / GPU", 14, weight="bold", fill="#b45309"))
    # vertical lane lines
    B.append(line(cpu_x, 100, cpu_x, 500, stroke="#cbd5e1", sw=1, marker="a"))
    # nodes on cpu lane
    bw, bh = 250, 46
    def cnode(y, s, sub=""):
        b = [box(cpu_x - bw // 2, y, bw, bh, "#eff6ff", "#1d4ed8")]
        b.append(txt(cpu_x, y + (24 if sub else 28), s, 12.5, weight="bold", fill="#1e3a8a"))
        if sub:
            b.append(txt(cpu_x, y + 40, sub, 10.5, fill="#3b82f6"))
        return b
    def gnode(y, s, sub=""):
        b = [box(gpu_x - bw // 2, y, bw, bh, "#fffbeb", "#b45309")]
        b.append(txt(gpu_x, y + (24 if sub else 28), s, 12.5, weight="bold", fill="#92400e"))
        if sub:
            b.append(txt(gpu_x, y + 40, sub, 10.5, fill="#d97706"))
        return b

    y0 = 110
    B += cnode(y0, "scheduler.schedule()", "排出本拍的批")
    B += cnode(y0 + 75, "execute_model(non_block=True)", "发起前向, 不等它, 拿 future")
    # arrow from execute_model to gpu lane (dispatch)
    B.append(line(cpu_x + bw // 2, y0 + 75 + bh // 2, gpu_x - bw // 2 - 2, y0 + 160 + bh // 2,
                  stroke="#b45309", sw=2, marker="a"))
    B += gnode(y0 + 160, "GPU 前向 (异步进行中)", "model forward")
    # overlap region: cpu computes bitmask in parallel
    B += cnode(y0 + 160, "get_grammar_bitmask()", "趁前向在跑, 算结构化掩码")
    # overlap bracket
    B.append(f'<rect x="40" y="{y0+155}" width="800" height="{bh+10}" rx="10" '
             f'fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="6 4"/>')
    B.append(txt(440, y0 + 160 + bh + 30, "↑ 这两段并行 —— 掩码计算藏进前向的影子里", 12,
                 fill="#16a34a", weight="bold"))
    # join point
    B += cnode(y0 + 255, "future.result()", "到这里才等前向; 返回 None = 采样归我")
    # arrow gpu->join
    B.append(line(gpu_x - bw // 2 - 2, y0 + 160 + bh // 2, cpu_x + bw // 2, y0 + 255 + 10,
                  stroke="#b45309", sw=2, marker="a"))
    B += cnode(y0 + 320, "sample_tokens(grammar_output)", "用刚算好的掩码采样")
    B += cnode(y0 + 360, "", "")
    return w, h, B[:-3]  # drop empty cnode


def step_orchestration2():
    # cleaner re-layout
    w, h = 900, 640
    B = []
    B += title(w, "EngineCore.step() 一次迭代", "CPU 算掩码 与 GPU 跑前向 在同一段时间重叠")
    cpu_x, gpu_x = 250, 660
    B.append(txt(cpu_x, 88, "主线程 (CPU)", 14, weight="bold", fill="#1d4ed8"))
    B.append(txt(gpu_x, 88, "Worker / GPU", 14, weight="bold", fill="#b45309"))
    bw, bh, gap = 270, 44, 36

    def cnode(i, s, sub=""):
        y = 105 + i * (bh + gap)
        b = [box(cpu_x - bw // 2, y, bw, bh, "#eff6ff", "#1d4ed8")]
        b.append(txt(cpu_x, y + (22 if sub else 27), s, 12, weight="bold", fill="#1e3a8a"))
        if sub:
            b.append(txt(cpu_x, y + 37, sub, 10, fill="#3b82f6"))
        return b, y

    rows = [
        ("scheduler.schedule()", "排出本拍要算的批"),
        ("execute_model(non_block=True)", "发起前向, 立刻返回 future"),
        ("get_grammar_bitmask()", "趁前向在跑, 算结构化掩码"),
        ("future.result()", "到此才等前向; None = 采样归主进程"),
        ("sample_tokens(grammar)", "用掩码采样"),
        ("_process_aborts_queue()", "执行期 abort 批量落地"),
        ("update_from_output()", "转 EngineCoreOutputs 收口"),
    ]
    ys = []
    for i, (s, sub) in enumerate(rows):
        nb, y = cnode(i, s, sub)
        B += nb
        ys.append(y)
    # cpu vertical arrows between consecutive nodes
    for i in range(len(rows) - 1):
        B.append(line(cpu_x, ys[i] + bh, cpu_x, ys[i + 1] - 2, stroke="#94a3b8", sw=1.5, marker="a"))
    # GPU forward box aligned with execute->result span
    gy = ys[1] + bh // 2
    gh = ys[3] + bh // 2 - gy
    B.append(box(gpu_x - bw // 2, gy, bw, gh, "#fffbeb", "#b45309"))
    B.append(txt(gpu_x, gy + gh // 2 - 6, "GPU 前向", 13, weight="bold", fill="#92400e"))
    B.append(txt(gpu_x, gy + gh // 2 + 12, "model forward (异步)", 10.5, fill="#d97706"))
    # dispatch arrow (label placed high, near execute_model row)
    B.append(line(cpu_x + bw // 2, ys[1] + bh // 2, gpu_x - bw // 2 - 2, gy + 14,
                  stroke="#b45309", sw=2, marker="a"))
    B.append(txt((cpu_x + gpu_x) // 2, ys[1] + bh // 2 - 12, "发起前向", 11, fill="#b45309"))
    # join arrow back (label placed low, near future.result row)
    B.append(line(gpu_x - bw // 2 - 2, gy + gh - 14, cpu_x + bw // 2, ys[3] + bh // 2,
                  stroke="#b45309", sw=2, marker="a"))
    B.append(txt((cpu_x + gpu_x) // 2, ys[3] + bh // 2 + 16, "等到结果", 11, fill="#b45309"))
    # overlap bracket spanning bitmask node and gpu box
    oy = ys[2] - 8
    B.append(f'<rect x="{cpu_x - bw//2 - 12}" y="{oy}" width="{gpu_x + bw//2 + 12 - (cpu_x - bw//2 - 12)}" '
             f'height="{bh + 16}" rx="10" fill="none" stroke="#16a34a" stroke-width="2" '
             f'stroke-dasharray="6 4"/>')
    # caption to the right of the bracket midpoint, clear of the cpu-lane vertical arrow
    mid_x = (cpu_x + bw // 2 + gpu_x - bw // 2) // 2
    B.append(txt(mid_x, ys[2] + bh // 2 - 8, "重叠区", 12, anchor="middle",
                 fill="#16a34a", weight="bold"))
    B.append(txt(mid_x, ys[2] + bh // 2 + 10, "掩码(CPU) 与 前向(GPU) 并行", 10.5,
                 anchor="middle", fill="#16a34a"))
    return w, h, B


# ---------------------------------------------------------------------------
# 2. busy loop + IO threads + shutdown states
# ---------------------------------------------------------------------------
def busy_loop():
    w, h = 940, 560
    B = []
    B += title(w, "run_busy_loop 与两条 IO 线程", "忙循环靠一对内存队列与 ZMQ IO 解耦")
    # shutdown state machine (top)
    sm_y = 78
    states = [("RUNNING", "#dcfce7", "#16a34a"), ("REQUESTED", "#fef9c3", "#ca8a04"),
              ("SHUTTING_DOWN", "#fee2e2", "#dc2626")]
    sx0, sbw, sgap = 270, 130, 60
    sxs = []
    for i, (s, fill, stroke) in enumerate(states):
        x = sx0 + i * (sbw + sgap)
        B.append(box(x, sm_y, sbw, 36, fill, stroke))
        B.append(txt(x + sbw // 2, sm_y + 23, s, 12, weight="bold", fill=stroke))
        sxs.append(x)
    for i in range(2):
        B.append(line(sxs[i] + sbw, sm_y + 18, sxs[i + 1] - 2, sm_y + 18))
    B.append(line(sxs[2] + sbw, sm_y + 18, sxs[2] + sbw + 50, sm_y + 18, stroke="#dc2626", marker="ar"))
    B.append(txt(sxs[2] + sbw + 95, sm_y + 22, "exit", 12, fill="#dc2626", weight="bold"))
    B.append(txt(sx0 - 60, sm_y + 22, "关停三态", 12, fill="#475569", weight="bold"))

    # input_queue (left), busy loop (center), output_queue (right)
    iq_x, ce_x, oq_x = 130, 470, 810
    qy, qh = 200, 230
    # input queue
    B.append(box(iq_x - 60, qy, 120, qh, "#f1f5f9", "#94a3b8"))
    B.append(txt(iq_x, qy - 10, "input_queue", 12, weight="bold", fill="#475569"))
    for i in range(4):
        B.append(box(iq_x - 45, qy + 20 + i * 40, 90, 28, "#e0e7ff", "#6366f1"))
        B.append(txt(iq_x, qy + 39 + i * 40, ["ADD", "ABORT", "UTILITY", "WAKEUP"][i], 10.5, fill="#3730a3"))
    # output queue
    B.append(box(oq_x - 60, qy, 120, qh, "#f1f5f9", "#94a3b8"))
    B.append(txt(oq_x, qy - 10, "output_queue", 12, weight="bold", fill="#475569"))
    for i in range(3):
        B.append(box(oq_x - 45, qy + 30 + i * 45, 90, 30, "#dcfce7", "#16a34a"))
        B.append(txt(oq_x, qy + 50 + i * 45, "Outputs", 10.5, fill="#166534"))

    # busy loop center: two steps in a loop
    B.append(box(ce_x - 150, qy - 10, 300, qh + 30, "white", "#1d4ed8", rx=14, sw=2.5))
    B.append(txt(ce_x, qy + 8, "run_busy_loop", 14, weight="bold", fill="#1d4ed8"))
    B.append(box(ce_x - 130, qy + 25, 260, 50, "#eff6ff", "#1d4ed8"))
    B.append(txt(ce_x, qy + 46, "1) _process_input_queue", 12, weight="bold", fill="#1e3a8a"))
    B.append(txt(ce_x, qy + 63, "取请求 / dispatch / 无活则阻塞睡", 10, fill="#3b82f6"))
    B.append(box(ce_x - 130, qy + 95, 260, 50, "#eff6ff", "#1d4ed8"))
    B.append(txt(ce_x, qy + 116, "2) _process_engine_step", 12, weight="bold", fill="#1e3a8a"))
    B.append(txt(ce_x, qy + 133, "step_fn() → 塞 output / post_step", 10, fill="#3b82f6"))
    # loop arrow
    B.append(line(ce_x - 130, qy + 120, ce_x - 145, qy + 120, stroke="#1d4ed8", sw=2, marker="a"))
    B.append(f'<path d="M{ce_x-145},{qy+120} C{ce_x-185},{qy+120} {ce_x-185},{qy+50} {ce_x-130},{qy+50}" '
             f'fill="none" stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')
    B.append(txt(ce_x - 185, qy + 88, "每圈", 10, fill="#1d4ed8"))

    # IO threads
    B.append(box(iq_x - 70, 470, 140, 50, "#fef3c7", "#d97706"))
    B.append(txt(iq_x, 491, "process_input", 11, weight="bold", fill="#92400e"))
    B.append(txt(iq_x, 507, "_sockets (IO线程)", 10, fill="#b45309"))
    B.append(box(oq_x - 70, 470, 140, 50, "#fef3c7", "#d97706"))
    B.append(txt(oq_x, 491, "process_output", 11, weight="bold", fill="#92400e"))
    B.append(txt(oq_x, 507, "_sockets (IO线程)", 10, fill="#b45309"))
    # arrows: io->input_queue, input_queue->loop, loop->output_queue, output_queue->io
    B.append(line(iq_x, 470, iq_x, qy + qh + 2, stroke="#d97706", sw=2, marker="a"))
    B.append(line(iq_x + 60, qy + qh // 2, ce_x - 152, qy + 50, stroke="#64748b", sw=2, marker="a"))
    B.append(line(ce_x + 152, qy + 120, oq_x - 60, qy + qh // 2, stroke="#64748b", sw=2, marker="a"))
    B.append(line(oq_x, qy + qh + 2, oq_x, 470, stroke="#d97706", sw=2, marker="a"))
    B.append(txt(iq_x, qy + qh + 38, "ZMQ recv → 入队", 10, fill="#b45309"))
    B.append(txt(oq_x, qy + qh + 38, "出队 → ZMQ PUSH", 10, fill="#b45309"))
    return w, h, B


# ---------------------------------------------------------------------------
# 3. batch queue pipeline vs plain step
# ---------------------------------------------------------------------------
def batch_queue():
    w, h = 920, 480
    B = []
    B += title(w, "step  vs  step_with_batch_queue", "填满流水线 优先于 取结果, 消除 PP 气泡")
    # top track: plain step (serial)
    ty = 110
    B.append(txt(70, ty - 8, "step (普通)", 13, anchor="start", weight="bold", fill="#475569"))
    stages = ["schedule", "exec", "sample", "update"]
    sx0, sbw, sgap = 90, 150, 20
    for i, s in enumerate(stages):
        x = sx0 + i * (sbw + sgap)
        B.append(box(x, ty, sbw, 40, "#f1f5f9", "#94a3b8"))
        B.append(txt(x + sbw // 2, ty + 25, s, 12, fill="#334155"))
        if i < len(stages) - 1:
            B.append(line(x + sbw, ty + 20, x + sbw + sgap - 2, ty + 20))
    B.append(txt(w // 2, ty + 62, "一拍一批, 各 stage 串行 —— 单批走完才下一批", 11, fill="#64748b"))

    # bottom track: batch queue deque
    by = 240
    B.append(txt(70, by - 8, "step_with_batch_queue", 13, anchor="start", weight="bold", fill="#1d4ed8"))
    B.append(box(80, by, 760, 130, "#eff6ff", "#1d4ed8", rx=12))
    B.append(txt(180, by + 22, "batch_queue (deque, 深度 N)", 12, weight="bold", fill="#1e3a8a"))
    # batches in flight at different stages
    bx0, bbw, bgap = 110, 165, 25
    labels = [("批 #3", "appendleft\n刚调度", "#dbeafe", "#2563eb"),
              ("批 #2", "exec 中", "#fde68a", "#d97706"),
              ("批 #1", "pop\n取结果", "#bbf7d0", "#16a34a")]
    for i, (b, st, fill, stroke) in enumerate(labels):
        x = bx0 + i * (bbw + bgap)
        B.append(box(x, by + 40, bbw, 70, fill, stroke))
        B.append(txt(x + bbw // 2, by + 67, b, 13, weight="bold", fill=stroke))
        for j, lineel in enumerate(st.split("\n")):
            B.append(txt(x + bbw // 2, by + 87 + j * 14, lineel, 10, fill=stroke))
        if i < len(labels) - 1:
            B.append(line(x + bbw, by + 75, x + bbw + bgap - 2, by + 75))
    # deferred branch note
    B.append(box(bx0 + 3 * (bbw + bgap) - 5, by + 40, 150, 70, "#fee2e2", "#dc2626", rx=8))
    B.append(txt(bx0 + 3 * (bbw + bgap) + 70, by + 65, "deferred", 11, weight="bold", fill="#b91c1c"))
    B.append(txt(bx0 + 3 * (bbw + bgap) + 70, by + 83, "投机解码+", 9.5, fill="#dc2626"))
    B.append(txt(bx0 + 3 * (bbw + bgap) + 70, by + 96, "结构化输出 (第12章)", 9, fill="#dc2626"))
    B.append(txt(w // 2, by + 152, "队未满且队尾未完成 → return (None, True) 立刻去调度下一批", 11.5,
                 fill="#1d4ed8", weight="bold"))
    return w, h, B


# ---------------------------------------------------------------------------
# 4. lifecycle states: PauseState + sleep levels
# ---------------------------------------------------------------------------
def lifecycle():
    w, h = 900, 520
    B = []
    B += title(w, "生命周期: pause 三模式 + sleep 三级", "停调度 与 让出显存 是两组正交旋钮")
    # PauseState row
    py = 100
    B.append(txt(70, py - 6, "PauseState (停调度)", 12, anchor="start", weight="bold", fill="#475569"))
    pstates = [("UNPAUSED", "正常", "#dcfce7", "#16a34a"),
               ("PAUSED_NEW", "不收新, 跑完在途", "#fef9c3", "#ca8a04"),
               ("PAUSED_ALL", "全停, 保留状态", "#fee2e2", "#dc2626")]
    px0, pbw, pgap = 110, 200, 60
    pxs = []
    for i, (s, sub, fill, stroke) in enumerate(pstates):
        x = px0 + i * (pbw + pgap)
        B.append(box(x, py, pbw, 56, fill, stroke))
        B.append(txt(x + pbw // 2, py + 24, s, 13, weight="bold", fill=stroke))
        B.append(txt(x + pbw // 2, py + 44, sub, 10.5, fill=stroke))
        pxs.append(x)
    # transitions
    B.append(line(pxs[0] + pbw, py + 20, pxs[1] - 2, py + 20))
    B.append(txt((pxs[0] + pbw + pxs[1]) // 2, py + 14, "abort", 10, fill="#475569"))
    B.append(line(pxs[1] + pbw, py + 20, pxs[2] - 2, py + 20))
    B.append(txt((pxs[1] + pbw + pxs[2]) // 2, py + 14, "keep", 10, fill="#475569"))
    # resume arrows back
    B.append(f'<path d="M{pxs[1]},{py+50} C{pxs[1]-30},{py+90} {pxs[0]+pbw+20},{py+90} {pxs[0]+pbw},{py+50}" '
             f'fill="none" stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>')
    B.append(txt((pxs[0] + pbw + pxs[1]) // 2, py + 100, "resume_scheduler → UNPAUSED", 10, fill="#16a34a"))

    # sleep levels stacked below
    sy = 250
    B.append(txt(70, sy - 6, "sleep level (让出显存, 叠加在 pause 之上)", 12, anchor="start",
                 weight="bold", fill="#475569"))
    levels = [("level 0", "只停调度", "无显存变化, 醒得最快", "#e0f2fe", "#0284c7"),
              ("level 1", "卸权重→CPU, 弃 KV", "委托 executor.sleep(1)", "#fef3c7", "#d97706"),
              ("level 2", "弃全部 GPU 显存", "委托 executor.sleep(2), 醒最贵", "#fee2e2", "#dc2626")]
    lx0, lbw, lgap = 110, 220, 30
    for i, (lv, a, b, fill, stroke) in enumerate(levels):
        x = lx0 + i * (lbw + lgap)
        B.append(box(x, sy, lbw, 80, fill, stroke))
        B.append(txt(x + lbw // 2, sy + 24, lv, 14, weight="bold", fill=stroke))
        B.append(txt(x + lbw // 2, sy + 46, a, 11, fill=stroke))
        B.append(txt(x + lbw // 2, sy + 64, b, 9.5, fill=stroke))
    B.append(txt(w // 2, sy + 105, "sleep 不论哪级都先 pause_scheduler; level≥1 才委托 executor 动显存", 11.5,
                 fill="#475569"))

    # async future side note
    ny = sy + 130
    B.append(box(180, ny, 540, 56, "#f8fafc", "#94a3b8", rx=10))
    B.append(txt(w // 2, ny + 23, "多进程版: 在途未排空 → 挂 idle 回调, 返回未完成 Future", 12,
                 weight="bold", fill="#334155"))
    B.append(txt(w // 2, ny + 43, "引擎转空闲时 _notify_idle_state_callbacks 触发 → 清缓存 → set_result", 10.5,
                 fill="#64748b"))
    return w, h, B


if __name__ == "__main__":
    write("ch11-step-orchestration", *step_orchestration2())
    write("ch11-busy-loop", *busy_loop())
    write("ch11-batch-queue-pipeline", *batch_queue())
    write("ch11-lifecycle-states", *lifecycle())
