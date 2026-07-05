#!/usr/bin/env python3
"""Generate ch12 diagrams: three-things boundary, standard-path objects,
cadence swimlane, block-granularity, simple-path pathway."""
import os, subprocess
import xml.sax.saxutils as xs

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return xs.escape(s)


def box(x, y, w, h, fill, stroke, rx=6, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, fill="#1e293b", anchor="middle", weight="normal", mono=False, italic=False):
    fam = "monospace" if mono else "sans-serif"
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}"{st}>{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color="#64748b", sw=2, dash=None, marker="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def line(x1, y1, x2, y2, color="#64748b", sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'


def defs():
    out = ['<defs>']
    for mid, col in [("a", "#64748b"), ("b", "#7c3aed"), ("g", "#0d9488"),
                     ("r", "#dc2626"), ("o", "#ea580c"), ("bl", "#2563eb")]:
        out.append(
            f'<marker id="{mid}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L10,3 L0,6 Z" fill="{col}"/></marker>')
    out.append('</defs>')
    return "".join(out)


def save(name, w, h, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
           + defs() + f'<rect width="{w}" height="{h}" fill="white"/>'
           + "".join(body) + '</svg>')
    sp = os.path.join(HERE, name + ".svg")
    pp = os.path.join(HERE, name + ".png")
    with open(sp, "w") as f:
        f.write(svg)
    assert subprocess.run(["xmllint", "--noout", sp]).returncode == 0, name
    subprocess.run(["rsvg-convert", "-z", "2", sp, "-o", pp], check=True)
    print("wrote", pp)


# ── Diagram 1: three-things boundary ─────────────────────────────────
def three_things():
    w, h = 1000, 560
    L = [text(w / 2, 38, "把 KV 从 HBM 挪走的三件事：搬去哪、省什么、跨不跨节点", 19, "#0f172a", weight="bold")]

    cols = [
        ("ch10 · PD 分离", "#fef2f2", "#fca5a5", "#b91c1c",
         ["跨节点 P2P 直传", "prefill→decode", "一对实例点对点"],
         "省 prefill 算力", "跨节点"),
        ("ch11 · KV 池化", "#f5f3ff", "#ddd6fe", "#6d28d9",
         ["写进外部共享 store", "按内容寻址", "跨请求 / 跨实例复用"],
         "省重算", "跨节点"),
        ("ch12 · KV 卸载", "#ecfdf5", "#a7f3d0", "#047857",
         ["device↔host 分层搬运", "NPU HBM ↔ CPU DRAM", "同机一上一下"],
         "省显存（HBM）", "同机"),
    ]
    cw, gap = 290, 35
    x0 = (w - (cw * 3 + gap * 2)) / 2
    top = 70
    bh = 210
    for i, (title, fill, stroke, tcol, bullets, save_what, where) in enumerate(cols):
        x = x0 + i * (cw + gap)
        L.append(box(x, top, cw, bh, fill, stroke, rx=12, sw=2.5))
        L.append(text(x + cw / 2, top + 34, title, 17, tcol, weight="bold"))
        L.append(line(x + 24, top + 50, x + cw - 24, top + 50, stroke, 1.5))
        for j, b in enumerate(bullets):
            L.append(text(x + cw / 2, top + 86 + j * 34, b, 13.5, "#334155"))
    # comparison strip
    sy = top + bh + 30
    rows = [("省什么", [c[5] for c in cols]), ("范围", [c[6] for c in cols])]
    rh = 46
    for ri, (label, vals) in enumerate(rows):
        ry = sy + ri * rh
        for i, v in enumerate(vals):
            x = x0 + i * (cw + gap)
            fill = ["#fef2f2", "#f5f3ff", "#ecfdf5"][i]
            stroke = ["#fca5a5", "#ddd6fe", "#a7f3d0"][i]
            L.append(box(x, ry, cw, rh - 6, fill, stroke, rx=6, sw=1.5))
            tcol = ["#b91c1c", "#6d28d9", "#047857"][i]
            L.append(text(x + cw / 2, ry + rh / 2, v, 14, tcol, weight="bold"))
    # cross-cut footer
    fy = sy + rh * 2 + 26
    L.append(box(x0, fy, cw * 3 + gap * 2, 44, "#fffbeb", "#fcd34d", rx=10, sw=2))
    L.append(text(w / 2, fy + 27,
                  "横切同源：权重预取 NPUPrefetchOffloader 也是 device↔host 搬运（本章点名不展开）",
                  13.5, "#92400e", weight="bold"))
    save("three-things", w, h, L)


# ── Diagram 2: standard-path object diagram ──────────────────────────
def standard_objects():
    w, h = 1000, 600
    L = [text(w / 2, 36, "标准路径：只新写 Handler，Manager 直接复用基座", 19, "#0f172a", weight="bold")]

    # Spec box
    sx, sw_, sy_ = w / 2 - 170, 340, 64
    spec_y = 64
    L.append(box(sx, spec_y, sw_, sy_, "#e0f2fe", "#38bdf8", rx=10, sw=2.5))
    L.append(text(w / 2, spec_y + 26, "NPUOffloadingSpec", 16, "#0369a1", weight="bold", mono=True))
    L.append(text(w / 2, spec_y + 47, "继承基座 OffloadingSpec · 卸载接入点", 12.5, "#0c4a6e"))

    # two branches
    sch_x = 200
    wrk_x = 800
    br_y = spec_y + sy_ + 50
    L.append(arrow(w / 2 - 70, spec_y + sy_, sch_x, br_y - 8, "#2563eb", 2.2, marker="bl"))
    L.append(arrow(w / 2 + 70, spec_y + sy_, wrk_x, br_y - 8, "#7c3aed", 2.2, marker="b"))
    L.append(text(310, br_y - 40, "scheduler 侧", 12.5, "#2563eb", weight="bold"))
    L.append(text(690, br_y - 40, "worker 侧", 12.5, "#7c3aed", weight="bold"))

    # scheduler: get_manager -> CPUOffloadingManager (reuse base)
    L.append(box(sch_x - 130, br_y, 260, 44, "#eff6ff", "#bfdbfe", rx=8))
    L.append(text(sch_x, br_y + 27, "get_manager()", 14, "#1e40af", weight="bold", mono=True))
    mgr_y = br_y + 80
    L.append(arrow(sch_x, br_y + 44, sch_x, mgr_y - 8, "#2563eb", 2))
    L.append(box(sch_x - 140, mgr_y, 280, 60, "#dbeafe", "#3b82f6", rx=8, sw=2.5))
    L.append(text(sch_x, mgr_y + 25, "CPUOffloadingManager", 14.5, "#1e3a8a", weight="bold", mono=True))
    L.append(text(sch_x, mgr_y + 45, "vLLM 基座 · 直接复用", 12.5, "#1d4ed8"))

    # worker: get_handlers -> CpuNpuOffloadingHandler (ascend)
    L.append(box(wrk_x - 130, br_y, 260, 44, "#f5f3ff", "#ddd6fe", rx=8))
    L.append(text(wrk_x, br_y + 27, "get_handlers()", 14, "#6d28d9", weight="bold", mono=True))
    hdl_y = br_y + 80
    L.append(arrow(wrk_x, br_y + 44, wrk_x, hdl_y - 8, "#7c3aed", 2))
    L.append(box(wrk_x - 150, hdl_y, 300, 60, "#ede9fe", "#8b5cf6", rx=8, sw=2.5))
    L.append(text(wrk_x, hdl_y + 25, "CpuNpuOffloadingHandler", 14, "#5b21b6", weight="bold", mono=True))
    L.append(text(wrk_x, hdl_y + 45, "昇腾自带 · 搬运执行体", 12.5, "#6d28d9"))

    # implements OffloadingHandler abstract; dashed to base contrast
    abs_y = hdl_y + 110
    L.append(box(wrk_x - 150, abs_y, 300, 40, "#faf5ff", "#c4b5fd", rx=8, dash="5,4"))
    L.append(text(wrk_x, abs_y + 25, "implements  OffloadingHandler（基座抽象）", 12.5, "#7c3aed", mono=True))
    L.append(arrow(wrk_x, hdl_y + 60, wrk_x, abs_y - 6, "#7c3aed", 1.8, dash="4,3"))

    # base CUDA reference (contrast)
    ref_y = abs_y - 6
    L.append(box(wrk_x - 470, abs_y - 20, 250, 78, "#f8fafc", "#94a3b8", rx=8, dash="6,4"))
    L.append(text(wrk_x - 345, abs_y + 4, "SingleDirection-", 12.5, "#475569", weight="bold", mono=True))
    L.append(text(wrk_x - 345, abs_y + 20, "OffloadingHandler", 12.5, "#475569", weight="bold", mono=True))
    L.append(text(wrk_x - 345, abs_y + 40, "CUDA 参考：每方向一 handler", 11.5, "#64748b"))
    L.append(text(wrk_x - 345, abs_y + 55, "每 transfer 一条新流", 11.5, "#64748b"))
    L.append(line(wrk_x - 220, abs_y + 19, wrk_x - 150, abs_y + 19, "#94a3b8", 1.6, dash="5,4"))
    L.append(text(wrk_x - 185, abs_y + 12, "对位", 11, "#64748b", italic=True))

    # bidirectional registration + converge to swap_blocks_batch
    swp_y = abs_y + 95
    L.append(text(w / 2, swp_y - 12, "Handler 双向注册：(GPU→CPU) 卸载  ·  (CPU→GPU) 加载", 13, "#334155"))
    swp_box_y = swp_y + 6
    L.append(arrow(wrk_x, abs_y + 40, w / 2 + 140, swp_box_y - 6, "#7c3aed", 2))
    L.append(box(w / 2 - 200, swp_box_y, 400, 50, "#fef9c3", "#eab308", rx=10, sw=2.5))
    L.append(text(w / 2, swp_box_y + 22, "torch.ops._C_ascend.swap_blocks_batch", 14, "#854d0e", weight="bold", mono=True))
    L.append(text(w / 2, swp_box_y + 40, "两条路径最终收口的昇腾批量搬运算子", 11.5, "#a16207"))
    save("standard-objects", w, h, L)


# ── Diagram 3: cadence swimlane (store / d2h) ────────────────────────
def cadence():
    w, h = 1040, 520
    L = [text(w / 2, 34, "卸载（store / d2h）节拍：搬运与下一步 forward 重叠", 19, "#0f172a", weight="bold")]
    lanes = [
        ("compute 默认流", "#fff7ed", "#fdba74", "#c2410c"),
        ("d2h 搬运流", "#ecfeff", "#67e8f9", "#0e7490"),
        ("主线程（轮询）", "#f0fdf4", "#86efac", "#15803d"),
    ]
    lx, lw = 180, 800
    ly0, lh = 70, 100
    for i, (name, fill, stroke, tcol) in enumerate(lanes):
        y = ly0 + i * (lh + 14)
        L.append(box(lx, y, lw, lh, fill, stroke, rx=10, sw=1.8))
        L.append(box(20, y, lx - 30, lh, "#f8fafc", stroke, rx=8, sw=1.8))
        L.append(text(20 + (lx - 30) / 2, y + lh / 2 + 5, name, 13.5, tcol, weight="bold"))
    # time axis arrow
    ty = ly0 + 3 * (lh + 14) + 6
    L.append(arrow(lx, ty, lx + lw, ty, "#94a3b8", 2))
    L.append(text(lx + lw, ty + 20, "时间 →", 12.5, "#64748b", anchor="end"))

    yc = [ly0 + lh / 2, ly0 + (lh + 14) + lh / 2, ly0 + 2 * (lh + 14) + lh / 2]

    # compute: forward写KV
    L.append(box(lx + 30, yc[0] - 22, 200, 44, "#ffedd5", "#fb923c", rx=8))
    L.append(text(lx + 130, yc[0] + 5, "forward 写 KV", 13, "#9a3412", weight="bold"))
    # wait_stream arrow compute -> d2h
    L.append(arrow(lx + 230, yc[0] + 16, lx + 290, yc[1] - 20, "#c2410c", 2))
    L.append(text(lx + 300, yc[0] + 40, "wait_stream(compute)：等算完再读", 12, "#c2410c", weight="bold", anchor="start"))
    # ★ 下一步 forward 与 d2h 搬运在时间轴上重叠 —— 异步省墙钟的视觉证据
    fx = lx + 455
    L.append(box(fx, yc[0] - 22, 210, 44, "#fff7ed", "#fb923c", rx=8, dash="6,4"))
    L.append(text(fx + 105, yc[0] + 5, "下一步 forward", 13, "#9a3412", weight="bold"))
    L.append(text(fx + 105, yc[0] - 30, "与搬运并行 → 摊薄墙钟", 11, "#15803d", weight="bold"))
    # 竖向虚线把「下一步 forward」与 swap_blocks_batch 钉在同一时间列（lx+545 = swap 框心）
    L.append(line(fx + 105, yc[0] + 22, lx + 545, yc[1] - 26, "#16a34a", 1.6, dash="4,3"))

    # d2h lane events
    ev = [
        (lx + 290, "record\nstart_event", "#cffafe"),
        (lx + 470, "swap_blocks_\nbatch(dir=1)", "#a5f3fc"),
        (lx + 670, "record\nend_event", "#cffafe"),
    ]
    bw = 150
    for ex, label, fill in ev:
        L.append(box(ex, yc[1] - 26, bw, 52, fill, "#0891b2", rx=8))
        parts = label.split("\n")
        for k, p in enumerate(parts):
            L.append(text(ex + bw / 2, yc[1] - 4 + k * 16, p, 12.5, "#0e7490", weight="bold", mono=True))
    # arrows between d2h events
    L.append(arrow(lx + 290 + bw, yc[1], lx + 470, yc[1], "#0891b2", 1.8))
    L.append(arrow(lx + 470 + bw, yc[1], lx + 670, yc[1], "#0891b2", 1.8))

    # main thread poll
    L.append(box(lx + 600, yc[2] - 22, 230, 44, "#dcfce7", "#4ade80", rx=8))
    L.append(text(lx + 715, yc[2] + 5, "get_finished：end_event.query()", 11.5, "#166534", weight="bold", mono=True))
    # end_event -> poll (dashed up)
    L.append(arrow(lx + 745, yc[1] + 26, lx + 715, yc[2] - 22, "#15803d", 1.8, dash="4,3"))
    L.append(text(lx + lw - 12, yc[1] + 52, "完成即回收 Event", 11.5, "#15803d", anchor="end"))

    # deque serial note
    L.append(box(lx + 30, ty + 36, lw - 60, 40, "#f8fafc", "#cbd5e1", rx=8))
    L.append(text(lx + lw / 2, ty + 61,
                  "同方向多搬运经 deque 串行：transfer[i].wait_event( transfer[i-1].end_event )  →  保证提交序执行",
                  12.5, "#475569", mono=False))
    save("cadence", w, h, L)


# ── Diagram 4: block granularity (block_size_factor=4) ───────────────
def block_granularity():
    w, h = 1000, 480
    L = [text(w / 2, 36, "标准路径的粒度换算：一个 CPU block = factor 个 GPU sub-block", 18, "#0f172a", weight="bold")]
    L.append(text(w / 2, 60, "expand_block_ids（block_size_factor = 4）", 13.5, "#475569", mono=True))

    # CPU blocks row —— 非连续示例 [0,3]，与正文 docstring 逐字对齐（block 2 跳过）
    cpu_y = 100
    cpu_ids = [0, 3]
    cw_ = 360
    cx0 = 60
    L.append(text(cx0 - 8, cpu_y + 30, "CPU", 13, "#6d28d9", weight="bold", anchor="end"))
    for i, bid in enumerate(cpu_ids):
        x = cx0 + i * (cw_ + 20)
        L.append(box(x, cpu_y, cw_, 50, "#ede9fe", "#8b5cf6", rx=8, sw=2.5))
        L.append(text(x + cw_ / 2, cpu_y + 31, f"CPU block {bid}（粗粒度）", 14, "#5b21b6", weight="bold"))

    # GPU sub-blocks row —— 0→[0,1,2,3]，3→[12,13,14,15]，中间 block 2 缺位（无 8–11）
    gpu_y = 210
    gpu_w = 80
    gx0 = 60
    L.append(text(gx0 - 8, gpu_y + 28, "GPU", 13, "#0e7490", weight="bold", anchor="end"))
    # 槽位：4 实 + 1 缺位 + 4 实
    slots = [("0", 0), ("1", 0), ("2", 0), ("3", 0), ("gap", None),
             ("12", 1), ("13", 1), ("14", 1), ("15", 1)]
    pos_x = {}
    for idx, (label, grp) in enumerate(slots):
        x = gx0 + idx * (gpu_w + 9)
        if grp is None:
            L.append(box(x, gpu_y, gpu_w, 48, "#fef2f2", "#fca5a5", rx=6, sw=2, dash="4,3"))
            L.append(text(x + gpu_w / 2, gpu_y + 21, "8–11", 12, "#b91c1c", weight="bold", mono=True))
            L.append(text(x + gpu_w / 2, gpu_y + 38, "缺位", 11, "#b91c1c"))
        else:
            L.append(box(x, gpu_y, gpu_w, 48, "#cffafe", "#0891b2", rx=6, sw=2))
            L.append(text(x + gpu_w / 2, gpu_y + 30, label, 15, "#0e7490", weight="bold", mono=True))
            pos_x[label] = x + gpu_w / 2
    # mapping lines：CPU block 0(粗) → [0,1,2,3]；CPU block 3(粗) → [12,13,14,15]
    for cpu_i, labels in [(0, ["0", "1", "2", "3"]), (1, ["12", "13", "14", "15"])]:
        cx = cx0 + cpu_i * (cw_ + 20) + cw_ / 2
        for lb in labels:
            gx = pos_x[lb]
            col = "#8b5cf6" if cpu_i == 0 else "#0891b2"
            L.append(line(cx, cpu_y + 50, gx, gpu_y, col, 1.4, dash="3,3"))
    L.append(text(w / 2, gpu_y + 80,
                  "block_id · factor + [0 .. factor)  →  0→[0,1,2,3]、3→[12,13,14,15]（block 2 缺位则无 8–11）",
                  13, "#475569", mono=True))

    # skip_count callout
    sk_y = 330
    L.append(box(60, sk_y, w - 120, 110, "#fffbeb", "#fcd34d", rx=10, sw=2))
    L.append(text(w / 2, sk_y + 26, "首块不整除：skip_count 对齐落位", 14, "#92400e", weight="bold"))
    L.append(text(w / 2, sk_y + 50,
                  "搬 5 个 GPU sub-block、factor=4  →  dst_sub_blocks_to_skip = (-5) % 4 = 3", 13, "#a16207", mono=True))
    L.append(text(w / 2, sk_y + 72,
                  "目标 CPU 首块前 3 个 slot 已被占（部分块），跳过它们让细粒度精确对齐", 12.5, "#a16207"))
    L.append(text(w / 2, sk_y + 94,
                  "极简路径无此换算：CPU / NPU block 1:1 同粒度，copy_blocks 直接映射", 12.5, "#047857", weight="bold"))
    save("block-granularity", w, h, L)


# ── Diagram 5: simple-path pathway ───────────────────────────────────
def simple_pathway():
    w, h = 1040, 600
    L = [text(w / 2, 34, "极简路径数据通路：注册期重建视图，运行期发了不等、轮询回收", 18, "#0f172a", weight="bold")]

    # register-time chain (top)
    L.append(text(70, 70, "注册期  register_kv_caches", 14, "#6d28d9", weight="bold", anchor="start"))
    reg = [
        ("_flatten_kv_value", "拍平 (K, V) 各自张量", "#ede9fe", "#8b5cf6", "#5b21b6"),
        ("按 storage.data_ptr 去重", "K/V 分开分配，不漏 V", "#ede9fe", "#8b5cf6", "#5b21b6"),
        ("_build_block_views", "[num_blocks, block_bytes] int8 视图", "#ede9fe", "#8b5cf6", "#5b21b6"),
        ("分配 pinned CPU 镜像", "+ 起 load/store npu stream", "#ede9fe", "#8b5cf6", "#5b21b6"),
        ("_backend.init", "预建 BatchMemcpyParams", "#ede9fe", "#8b5cf6", "#5b21b6"),
    ]
    bw, bh, gap = 178, 60, 16
    x0 = 40
    ry = 86
    for i, (t, sub, fill, stroke, tcol) in enumerate(reg):
        x = x0 + i * (bw + gap)
        L.append(box(x, ry, bw, bh, fill, stroke, rx=8, sw=2))
        L.append(text(x + bw / 2, ry + 24, t, 12.5, tcol, weight="bold", mono=True))
        L.append(text(x + bw / 2, ry + 44, sub, 10.5, "#6d28d9"))
        if i < len(reg) - 1:
            L.append(arrow(x + bw, ry + bh / 2, x + bw + gap, ry + bh / 2, "#7c3aed", 1.8, marker="b"))

    # view-rebuild callout
    cy = ry + bh + 24
    L.append(box(x0, cy, bw * 5 + gap * 4, 48, "#fffbeb", "#fcd34d", rx=10, sw=2))
    L.append(text(x0 + (bw * 5 + gap * 4) / 2, cy + 20,
                  "为何重建视图：runner 给每个 KV 张量超额分配 +2 MiB 对齐再切回", 13, "#92400e", weight="bold"))
    L.append(text(x0 + (bw * 5 + gap * 4) / 2, cy + 39,
                  "storage.nbytes() 含前导偏移 + 尾部 padding → 改用 shape/stride 精确裁出数据区", 12, "#a16207"))

    # runtime chain (bottom)
    rt_y = cy + 92
    L.append(text(70, rt_y - 6, "运行期  异步搬运", 14, "#0e7490", weight="bold", anchor="start"))
    rt_box_y = rt_y + 12
    nodes = [
        (70, "launch_copy", "主线程提交 job", "#cffafe", "#0891b2", "#0e7490"),
        (300, "SimpleQueue", "FIFO 队列", "#e0f2fe", "#38bdf8", "#0369a1"),
        (520, "_copy_loop", "后台守护线程", "#cffafe", "#0891b2", "#0e7490"),
        (745, "copy_blocks", "拼指针 → swap_blocks_batch", "#fef9c3", "#eab308", "#854d0e"),
    ]
    nbw = 200
    for x, t, sub, fill, stroke, tcol in nodes:
        L.append(box(x, rt_box_y, nbw, 56, fill, stroke, rx=8, sw=2))
        L.append(text(x + nbw / 2, rt_box_y + 23, t, 13, tcol, weight="bold", mono=True))
        L.append(text(x + nbw / 2, rt_box_y + 42, sub, 11, tcol))
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + nbw
        x2 = nodes[i + 1][0]
        L.append(arrow(x1, rt_box_y + 28, x2, rt_box_y + 28, "#0891b2", 1.8))

    # record event -> events_list -> poll
    ev_y = rt_box_y + 92
    L.append(box(520, ev_y, 200, 46, "#f0fdf4", "#4ade80", rx=8, sw=2))
    L.append(text(620, ev_y + 20, "record Event", 12.5, "#166534", weight="bold", mono=True))
    L.append(text(620, ev_y + 38, "→ events_list", 11.5, "#15803d", mono=True))
    L.append(arrow(845, rt_box_y + 56, 700, ev_y, "#15803d", 1.8, dash="4,3"))
    L.append(box(80, ev_y, 360, 46, "#f0fdf4", "#4ade80", rx=8, sw=2))
    L.append(text(260, ev_y + 20, "_poll_stream_events（继承）", 12.5, "#166534", weight="bold", mono=True))
    L.append(text(260, ev_y + 38, "主线程 Event.query() 无阻塞轮询完成", 11.5, "#15803d"))
    L.append(arrow(520, ev_y + 23, 440, ev_y + 23, "#15803d", 1.8, dash="4,3"))
    save("simple-pathway", w, h, L)


if __name__ == "__main__":
    three_things()
    standard_objects()
    cadence()
    block_granularity()
    simple_pathway()
