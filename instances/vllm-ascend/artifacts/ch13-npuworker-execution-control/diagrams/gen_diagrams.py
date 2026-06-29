"""生成 ch13 三张配图：重写 vs 继承 / 四步生命周期 / KV 显存预算。
全部坐标由 Python 计算，无手填魔数。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

FONT = 'font-family="sans-serif"'


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, size=15, anchor="middle", fill="#1e293b", weight="normal", mono=False):
    fam = 'font-family="monospace"' if mono else FONT
    return (f'<text x="{x}" y="{y}" {fam} font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


# ----------------------------------------------------------------------------
# 图 1：重写 vs 继承
# ----------------------------------------------------------------------------
def diagram_rewrite_vs_inherit(path):
    W, H = 940, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>'
             '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
             '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
             '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 36, "WorkerBase 一个抽象，两个平级实现", 22, weight="bold"))

    # 顶层抽象盒
    bw, bh = 520, 130
    bx = (W - bw) / 2
    by = 60
    L.append(box(bx, by, bw, bh, "#eef2ff", "#6366f1", sw=2.5))
    L.append(text(bx + bw / 2, by + 28, "WorkerBase（硬件无关抽象基类）", 17, weight="bold", fill="#4338ca"))
    methods = ["init_device", "determine_available_memory",
               "compile_or_warm_up_model", "execute_model"]
    mx0 = bx + 30
    for i, m in enumerate(methods):
        col = i % 2
        row = i // 2
        mx = mx0 + col * 250
        my = by + 58 + row * 32
        L.append(text(mx, my, m + "()", 13, anchor="start", fill="#3730a3", mono=True))
        L.append(text(mx, my + 16, "raise NotImplementedError", 11.5, anchor="start", fill="#b91c1c", mono=True))

    # 中层两个兄弟盒
    cy = 300
    cw, ch = 360, 200
    gx = 30
    npux = W - gx - cw
    # GPU Worker
    L.append(box(gx, cy, cw, ch, "#f8fafc", "#94a3b8", sw=2))
    L.append(text(gx + cw / 2, cy + 26, "Worker（GPU 实现，基座）", 16, weight="bold", fill="#475569"))
    L.append(text(gx + 20, cy + 56, "init_device():", 12.5, anchor="start", fill="#334155", mono=True))
    L.append(text(gx + 28, cy + 80, 'if device_type == "cuda":', 12, anchor="start", fill="#334155", mono=True))
    L.append(text(gx + 44, cy + 100, "cuda:N · NCCL · CUDA inductor …", 11.5, anchor="start", fill="#334155", mono=True))
    L.append(text(gx + 28, cy + 124, "else:", 12, anchor="start", fill="#334155", mono=True))
    L.append(text(gx + 44, cy + 144, "raise RuntimeError(Not support …)", 12, anchor="start", fill="#b91c1c", weight="bold", mono=True))
    L.append(text(gx + cw / 2, cy + 178, "设备层钉死 cuda → 继承只能走到 raise", 12.5, fill="#b91c1c"))

    # NPUWorker
    L.append(box(npux, cy, cw, ch, "#ecfdf5", "#10b981", sw=2.5))
    L.append(text(npux + cw / 2, cy + 26, "NPUWorker（昇腾实现）", 16, weight="bold", fill="#047857"))
    swaps = [
        "torch.npu.set_device(npu:N)",
        "torch_npu._inductor（triton）",
        'init_distributed(…, "hccl")',
        "ACLGraph / profile_cudagraph_memory",
        "_warm_up_atb（ATB 预热）",
    ]
    for i, s in enumerate(swaps):
        L.append(text(npux + 20, cy + 56 + i * 26, "• " + s, 12, anchor="start", fill="#065f46", mono=True))
    L.append(text(npux + cw / 2, cy + 192, "每处设备调用换成 torch_npu / ATB / ACLGraph", 12, fill="#047857"))

    # is-a 虚线箭头（两子 → 抽象底边）
    L.append(f'<line x1="{gx + cw / 2}" y1="{cy}" x2="{bx + bw * 0.32}" y2="{by + bh}" '
             f'stroke="#64748b" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#ar)"/>')
    L.append(f'<line x1="{npux + cw / 2}" y1="{cy}" x2="{bx + bw * 0.68}" y2="{by + bh}" '
             f'stroke="#64748b" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#ar)"/>')
    L.append(text(bx + bw * 0.30 - 26, by + bh + 48, "is-a", 12, fill="#64748b"))
    L.append(text(bx + bw * 0.70 + 26, by + bh + 48, "is-a", 12, fill="#64748b"))

    # Adapted from 注释线（NPUWorker → GPU Worker，灰色）
    midy = cy + ch + 34
    L.append(f'<line x1="{npux}" y1="{midy}" x2="{gx + cw}" y2="{midy}" '
             f'stroke="#94a3b8" stroke-width="2" stroke-dasharray="3,3" marker-end="url(#arg)"/>')
    L.append(text(W / 2, midy - 8, "Adapted from gpu_worker.py（搬结构、换设备层）", 13, fill="#64748b", weight="bold"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 2：四步生命周期管线
# ----------------------------------------------------------------------------
def diagram_lifecycle(path):
    W, H = 1180, 420
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>'
             '<marker id="fa" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
             '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 36, "一个 Worker 的四步生命周期", 22, weight="bold"))

    steps = [
        ("① init_device", "#dbeafe", "#3b82f6",
         ["torch.npu.set_device", "torch_npu._inductor", "MemorySnapshot / hccl"],
         "产物：self.device"),
        ("② determine_available_memory", "#fef3c7", "#f59e0b",
         ["memory_profiling + profile_run", "profile_cudagraph_memory", "算 KV 预算 + 回退建议"],
         "产物：available_kv_cache_memory_bytes"),
        ("③ compile_or_warm_up_model", "#ede9fe", "#8b5cf6",
         ["_dummy_run(warmup_sizes)", "capture_model（ACLGraph）", "_warm_up_atb（ATB 预热）"],
         "产物：CompilationTimes"),
        ("④ execute_model", "#dcfce7", "#22c55e",
         ["profile_memory / profiler.step", "→ model_runner.execute_model", "（派发给 NPUModelRunner）"],
         "产物：ModelRunnerOutput"),
    ]
    n = len(steps)
    bw, bh = 250, 170
    gap = (W - 40 - n * bw) / (n - 1)
    y = 90
    centers = []
    for i, (title, fill, stroke, calls, out) in enumerate(steps):
        x = 20 + i * (bw + gap)
        centers.append((x, x + bw))
        L.append(box(x, y, bw, bh, fill, stroke, sw=2.5))
        L.append(text(x + bw / 2, y + 30, title, 13, weight="bold", fill="#0f172a"))
        L.append(f'<line x1="{x + 16}" y1="{y + 42}" x2="{x + bw - 16}" y2="{y + 42}" stroke="{stroke}" stroke-width="1.2"/>')
        for j, c in enumerate(calls):
            L.append(text(x + 16, y + 66 + j * 24, c, 11.5, anchor="start", fill="#334155", mono=True))
        L.append(text(x + bw / 2, y + bh - 14, out, 11, fill=stroke, weight="bold"))
        # 阶段编号下标：执行次数
        freq = "每个 step 一次（热路径）" if i == n - 1 else "启动期一次"
        L.append(text(x + bw / 2, y + bh + 26, freq, 11.5, fill="#64748b"))

    # 串联箭头
    for i in range(n - 1):
        x1 = centers[i][1]
        x2 = centers[i + 1][0]
        L.append(f'<line x1="{x1}" y1="{y + bh / 2}" x2="{x2 - 2}" y2="{y + bh / 2}" '
                 f'stroke="#475569" stroke-width="2.5" marker-end="url(#fa)"/>')

    # execute_model 自循环
    lx0, lx1 = centers[-1]
    cx = (lx0 + lx1) / 2
    loop_y = y + bh + 50
    L.append(f'<path d="M {lx1 - 30} {y + bh} '
             f'C {lx1 + 40} {y + bh}, {lx1 + 40} {loop_y + 30}, {cx} {loop_y + 30} '
             f'C {lx0 - 40} {loop_y + 30}, {lx0 - 40} {y + bh}, {lx0 + 30} {y + bh}" '
             f'fill="none" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="5,4" marker-end="url(#fa)"/>')
    L.append(text(cx, loop_y + 50, "稳态：调度器每拍回到这里", 12.5, fill="#16a34a", weight="bold"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 3：KV 显存预算分解
# ----------------------------------------------------------------------------
def diagram_memory_budget(path):
    W, H = 980, 470
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 36, "KV cache 显存预算分解（一台 100 GiB NPU 的算例）", 21, weight="bold"))

    bx, by = 60, 80
    bw, bh = 860, 64
    # 总显存条
    L.append(box(bx, by, bw, bh, "#f1f5f9", "#cbd5e1"))
    util_w = bw * 0.9
    # requested 区（util=0.9）
    L.append(f'<line x1="{bx + util_w}" y1="{by - 8}" x2="{bx + util_w}" y2="{by + bh + 8}" '
             f'stroke="#ef4444" stroke-width="2" stroke-dasharray="4,3"/>')
    L.append(text(bx + bw, by - 12, "total_memory = 100 GiB", 13, anchor="end", fill="#64748b"))
    L.append(text(bx + util_w, by + bh + 24, "× gpu_memory_utilization 0.9  →  requested = 90 GiB", 13, anchor="end", fill="#b91c1c", weight="bold"))
    # 利用率之外的灰区标注
    L.append(text(bx + util_w + (bw - util_w) / 2, by + bh / 2 + 5, "预留 10", 12, fill="#94a3b8"))

    # 分解条：requested = weights + peak_activation + non_torch + ACLGraph + KV
    segs = [
        ("weights", 10, "#fca5a5", "#991b1b"),
        ("peak_activation", 7, "#fdba74", "#9a3412"),
        ("non_torch", 5, "#fde047", "#854d0e"),
        ("ACLGraph 估算", 3, "#c4b5fd", "#5b21b6"),
        ("KV cache", 65, "#86efac", "#166534"),
    ]
    by2 = by + 138
    L.append(text(bx, by2 - 16, "requested 90 GiB 的去向：", 14, anchor="start", weight="bold", fill="#334155"))
    xacc = bx
    scale = util_w / 90.0
    for name, val, fill, txt in segs:
        sw_ = val * scale
        L.append(box(xacc, by2, sw_, bh, fill, txt, rx=4, sw=1.5))
        if sw_ > 120:  # 仅 KV 段宽得下内嵌文字
            L.append(text(xacc + sw_ / 2, by2 + bh / 2 - 2, f"{name} = {val}", 14, fill=txt, weight="bold"))
            L.append(text(xacc + sw_ / 2, by2 + bh / 2 + 18, f"{val} GiB", 12.5, fill=txt))
        xacc += sw_

    # 图例（色块 + 名称 + 值），均匀分槽，避免窄段文字相撞
    ly = by2 + bh + 30
    slot = bw / len(segs)
    for i, (name, val, fill, txt) in enumerate(segs):
        sx = bx + i * slot
        L.append(box(sx, ly, 16, 16, fill, txt, rx=3, sw=1.2))
        L.append(text(sx + 24, ly + 13, f"{name} {val}", 12.5, anchor="start", fill=txt, weight="bold"))

    # 大括号式公式说明
    fy = ly + 64
    L.append(text(W / 2, fy, "available_kv = requested(90) − [weights(10)+peak_act(7)+non_torch(5)] − ACLGraph(3) = 65 GiB",
                  14, fill="#166534", weight="bold", mono=False))
    L.append(text(W / 2, fy + 30, "关 ACLGraph 估算开关：不扣那 3 GiB → KV = 68 GiB（但运行期 graph pool 仍要吃，易 OOM）",
                  12.5, fill="#64748b"))
    L.append(text(W / 2, fy + 56, "回退建议：suggested_util = min(0.9 + 3/100, 1.0) = 0.93  （把被 ACLGraph 占走的 KV 补回来）",
                  12.5, fill="#7c3aed", weight="bold"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    import os
    d = os.path.dirname(os.path.abspath(__file__))
    diagram_rewrite_vs_inherit(os.path.join(d, "rewrite-vs-inherit.svg"))
    diagram_lifecycle(os.path.join(d, "four-step-lifecycle.svg"))
    diagram_memory_budget(os.path.join(d, "kv-memory-budget.svg"))
    print("done")
