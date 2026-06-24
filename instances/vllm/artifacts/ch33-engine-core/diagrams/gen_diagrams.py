#!/usr/bin/env python3
"""ch33 章节配图：弹性 EP 扩缩状态机（scale_up 双泳道 / scale_down 双链）、
KV-init 前扩时序、Responses 多轮共享 list 时序。全部坐标由 Python 计算。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


def svg_header(w, h, arrow_color="#64748b"):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
    L.append(
        '<defs>'
        f'<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{arrow_color}"/></marker>'
        f'<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
        '</defs>'
    )
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    return L


def box(L, x, y, w, h, label, fill="#eef2ff", stroke="#6366f1", fs=13, sub=None, tw="normal", rx=6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    cy = y + h / 2 + fs / 3 if sub is None else y + h / 2 - 2
    L.append(f'<text x="{x + w/2}" y="{cy}" text-anchor="middle" font-size="{fs}" font-weight="{tw}" fill="#1e293b">{esc(label)}</text>')
    if sub:
        L.append(f'<text x="{x + w/2}" y="{cy + fs + 2}" text-anchor="middle" font-size="{fs-2}" fill="#475569">{esc(sub)}</text>')


def text(L, x, y, s, fs=12, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(s)}</text>')


def vline(L, x, y1, y2, color="#94a3b8", dash=None, arrow=False, w=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = ' marker-end="url(#a)"' if arrow else ''
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="{w}"{d}{m}/>')


def hline(L, x1, x2, y, color="#94a3b8", dash=None, arrow=False, marker="a", w=1.5):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    m = f' marker-end="url(#{marker})"' if arrow else ''
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="{w}"{d}{m}/>')


# ===========================================================================
# 图1: scale_up 双泳道状态机（existing 9 态 / new 4 态 + 跨进程通知）
# ===========================================================================
def diagram_scaleup():
    w, h = 940, 660
    L = svg_header(w, h)
    text(L, w / 2, 30, "scale_up：增引擎时两类引擎的状态推进与跨进程握手", 17, "middle", "#0f172a", "bold")

    lane_w = 360
    ex_x = 40
    nw_x = w - lane_w - 40
    top = 60
    # 泳道背景
    L.append(f'<rect x="{ex_x-12}" y="{top}" width="{lane_w+24}" height="{h-top-20}" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
    L.append(f'<rect x="{nw_x-12}" y="{top}" width="{lane_w+24}" height="{h-top-20}" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
    text(L, ex_x + lane_w / 2, top + 22, "存在引擎  worker_type=\"existing\"", 14, "middle", "#1e293b", "bold")
    text(L, nw_x + lane_w / 2, top + 22, "新引擎  worker_type=\"new\"", 14, "middle", "#1e293b", "bold")

    ex_states = [
        ("WAIT_NEW_CORE_ENGINES_INIT", "空转：progress() 返回 False"),
        ("CREATE_STANDBY_GROUPS", "建 standby DP 组"),
        ("TRANSFER_EXPERT_MAPPING", "广播专家映射"),
        ("WAIT_..._WEIGHTS_INIT", "空转：等新引擎权重就绪"),
        ("TRANSFER_WEIGHTS", "向新 worker 传权重"),
        ("SYNC_KV_CACHE_MEMORY_SIZE", "同步 KV 显存额度"),
        ("SWITCH_AND_PREPARE", "销毁旧组→切新组→对齐 wave"),
        ("EPLB_RESHUFFLE", "专家负载再均衡"),
        ("COMPLETE", "_update_parallel_config"),
    ]
    nw_states = [
        ("PRE_KV_INIT", "收权重 + 同步 KV 显存额度"),
        ("PREPARE", "all_reduce(MAX) 拿 wave 状态"),
        ("EPLB_RESHUFFLE", "专家负载再均衡"),
        ("COMPLETE", ""),
    ]

    bh, gap = 46, 12
    bx_w = lane_w - 20
    ex_y0 = top + 36
    nw_y0 = top + 36
    ex_ys, nw_ys = [], []
    for i, (s, sub) in enumerate(ex_states):
        y = ex_y0 + i * (bh + gap)
        ex_ys.append(y)
        passive = "空转" in sub
        box(L, ex_x, y, bx_w, bh, s, "#fef9c3" if passive else "#e0e7ff",
            "#ca8a04" if passive else "#6366f1", 12, sub, "bold")
        if i < len(ex_states) - 1:
            vline(L, ex_x + bx_w / 2, y + bh, ex_ys[i] + bh + gap, "#6366f1", arrow=True)

    # new 引擎纵向居中铺开
    nw_gap = (h - top - 60 - 4 * bh) / 3
    for i, (s, sub) in enumerate(nw_states):
        y = nw_y0 + i * (bh + nw_gap)
        nw_ys.append(y)
        box(L, nw_x, y, bx_w, bh, s, "#dcfce7", "#16a34a", 12, sub if sub else None, "bold")
        if i < len(nw_states) - 1:
            vline(L, nw_x + bx_w / 2, y + bh, nw_ys[i] + bh + nw_gap, "#16a34a", arrow=True)

    # 跨进程通知（横向箭头）—— 从 new 指向 existing 的两个 WAIT 态
    def notify(from_y, to_y, txt):
        x1 = nw_x - 12
        x2 = ex_x + bx_w + 12
        ymid = (from_y + to_y) / 2
        L.append(f'<path d="M{x1},{from_y+bh/2} C{(x1+x2)/2},{from_y+bh/2} {(x1+x2)/2},{to_y+bh/2} {x2+4},{to_y+bh/2}" '
                 f'fill="none" stroke="#b45309" stroke-width="1.6" stroke-dasharray="5,3" marker-end="url(#ah)"/>')
        text(L, (x1 + x2) / 2, ymid - 4, txt, 11, "middle", "#b45309", "bold")

    # NEW_CORE_ENGINES_INIT_READY: new 启动后通知 → existing WAIT_NEW_CORE_ENGINES_INIT 推进
    notify(nw_ys[0] - 8, ex_ys[0], "①NEW_CORE_ENGINES_INIT_READY")
    # WEIGHTS_INIT_READY: new 进 PRE_KV_INIT 时发 → existing WAIT_..._WEIGHTS_INIT 推进
    notify(nw_ys[0] + 8, ex_ys[3], "②WEIGHTS_INIT_READY")

    text(L, w / 2, h - 6,
         "黄=被动空转(靠 ①② 通知经 handle_notification 推进)　绿/紫=主动推进　每框=busy loop 一轮 progress() 一步",
         11, "middle", "#64748b")
    return w, h, L


# ===========================================================================
# 图2: scale_down 双链（余留 4 态 / 移除 3 态）
# ===========================================================================
def diagram_scaledown():
    w, h = 860, 540
    L = svg_header(w, h)
    text(L, w / 2, 30, "scale_down：减引擎时余留引擎与被裁引擎的两条短链", 17, "middle", "#0f172a", "bold")

    bh, bw, gap = 64, 240, 30
    col1_x = 70
    col2_x = w - bw - 70
    y0 = 80

    remain = [
        ("PREPARE", "登记 eep_barrier_engine_count"),
        ("EPLB_RESHUFFLE", "reshuffle→建组→switch→update_config"),
        ("SWITCH_AND_PREPARE", "(并入上一态内顺序完成)"),
        ("COMPLETE", ""),
    ]
    remove = [
        ("PREPARE", "登记 eep_barrier_engine_count"),
        ("EPLB_RESHUFFLE", "reshuffle→switch_and_remove"),
        ("COMPLETE", "发 SHUTDOWN_COMPLETE"),
    ]
    text(L, col1_x + bw / 2, y0 - 14, "余留引擎  ScaleDownRemainingEngineState", 13, "middle", "#1e293b", "bold")
    text(L, col2_x + bw / 2, y0 - 14, "被裁引擎  ScaleDownRemovingEngineState", 13, "middle", "#1e293b", "bold")

    ys1 = []
    for i, (s, sub) in enumerate(remain):
        y = y0 + i * (bh + gap)
        ys1.append(y)
        faded = "并入上一态" in sub
        box(L, col1_x, y, bw, bh, s, "#f1f5f9" if faded else "#e0e7ff",
            "#94a3b8" if faded else "#6366f1", 13, sub if sub else None, "bold")
        if i:
            vline(L, col1_x + bw / 2, ys1[i - 1] + bh, y, "#6366f1", arrow=True)

    ys2 = []
    for i, (s, sub) in enumerate(remove):
        y = y0 + i * (bh + gap)
        ys2.append(y)
        box(L, col2_x, y, bw, bh, s, "#fee2e2", "#dc2626", 13, sub if sub else None, "bold")
        if i:
            vline(L, col2_x + bw / 2, ys2[i - 1] + bh, y, "#dc2626", arrow=True)

    # 移除引擎结局：SystemExit
    ey = ys2[-1] + bh + gap
    box(L, col2_x, ey, bw, 40, "run_busy_loop → raise SystemExit", "#fecaca", "#b91c1c", 12, None, "bold")
    vline(L, col2_x + bw / 2, ys2[-1] + bh, ey, "#dc2626", arrow=True)

    text(L, w / 2, h - 14,
         "余留引擎在一个 EPLB_RESHUFFLE 态内连做 reshuffle→建组→切换→更新 config，比 scale_up 紧凑",
         12, "middle", "#64748b")
    return w, h, L


# ===========================================================================
# 图3: KV-init 前扩时序
# ===========================================================================
def diagram_kvinit():
    w, h = 820, 540
    L = svg_header(w, h)
    text(L, w / 2, 30, "新引擎：扩入发生在 KV cache 初始化之前", 17, "middle", "#0f172a", "bold")

    steps = [
        ("EngineCore.__init__", "见 VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=1", "#eef2ff", "#6366f1"),
        ("_eep_scale_up_before_kv_init()", "建 worker_type=\"new\" 状态机", "#dcfce7", "#16a34a"),
        ("run_pre_kv_init_states()  [PRE_KV_INIT]", "发 WEIGHTS_INIT_READY · receive_weights", "#dcfce7", "#16a34a"),
        ("sync_kv_cache_memory_size(new_dp_group)", "→ available_gpu_memory_for_kv_cache（全组统一额度）", "#fef3c7", "#d97706"),
        ("_initialize_kv_caches()", "用同步额度分配，而非各自 determine_available_memory()", "#eef2ff", "#6366f1"),
    ]
    bw, bh, gap = 560, 56, 28
    x = (w - bw) / 2
    y0 = 70
    ys = []
    for i, (s, sub, fill, stroke) in enumerate(steps):
        y = y0 + i * (bh + gap)
        ys.append(y)
        box(L, x, y, bw, bh, s, fill, stroke, 13, sub, "bold")
        if i:
            vline(L, x + bw / 2, ys[i - 1] + bh, y, "#475569", arrow=True)

    # 高亮“额度从 step4 流到 step5”
    y4 = ys[3] + bh / 2
    y5 = ys[4] + bh / 2
    L.append(f'<path d="M{x+bw+10},{y4} C{x+bw+70},{y4} {x+bw+70},{y5} {x+bw+10},{y5}" '
             f'fill="none" stroke="#b45309" stroke-width="2" marker-end="url(#ah)"/>')
    text(L, x + bw + 76, (y4 + y5) / 2, "复用", 12, "start", "#b45309", "bold")
    text(L, x + bw + 76, (y4 + y5) / 2 + 16, "额度", 12, "start", "#b45309", "bold")

    text(L, w / 2, h - 16,
         "若新引擎自己 profiling，DP 组内各 rank 显存/块数会不一致；先同步、再分配，从源头消除分歧",
         12, "middle", "#64748b")
    return w, h, L


# ===========================================================================
# 图4: Responses 多轮共享 list 时序（两轮）
# ===========================================================================
def diagram_multiturn():
    w, h = 900, 540
    L = svg_header(w, h)
    text(L, w / 2, 28, "Responses 多轮：msg_store 的 list 与 context._messages 是同一对象", 16, "middle", "#0f172a", "bold")

    # 三条泳道：客户端 / serving / 共享存储
    lanes = [("客户端", 70), ("OpenAIServingResponses", 360), ("response_store / msg_store", 720)]
    for name, x in lanes:
        L.append(f'<line x1="{x}" y1="60" x2="{x}" y2="{h-50}" stroke="#cbd5e1" stroke-width="1.5"/>')
        text(L, x, 52, name, 13, "middle", "#1e293b", "bold")

    cx, sx, stx = 70, 360, 720

    def msg(y, x1, x2, label, color="#475569", dash=None):
        hline(L, x1, x2, y, color, dash=dash, arrow=True)
        mx = (x1 + x2) / 2
        text(L, mx, y - 6, label, 11, "middle", color, "bold")

    # 轮1
    text(L, 70, 80, "── 轮 1 ──", 13, "start", "#0369a1", "bold")
    msg(100, cx, sx, "create_responses(input=\"my name is Bob\")")
    box(L, sx + 10, 116, 300, 34, "construct_input_messages → messages", "#eef2ff", "#6366f1", 11, None)
    msg(176, sx + 160, stx, "store: msg_store[id1] = messages", "#16a34a")
    box(L, sx - 150, 196, 300, 38, "生成 → context.append_output", "#dcfce7", "#16a34a", 11, "self._messages.extend(output)")
    # 共享对象指示
    L.append(f'<path d="M{sx+150},215 C{(sx+stx)/2},215 {(sx+stx)/2},176 {stx-6},176" fill="none" stroke="#b45309" stroke-width="1.6" stroke-dasharray="5,3" marker-end="url(#ah)"/>')
    text(L, (sx + stx) / 2 + 20, 250, "同一 list 对象 → output 自动留存", 11, "middle", "#b45309", "bold")
    msg(290, sx + 160, stx, "response_store[id1] = response", "#16a34a")

    # 分隔
    L.append(f'<line x1="40" y1="316" x2="{w-40}" y2="316" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')

    # 轮2
    text(L, 70, 340, "── 轮 2 ──", 13, "start", "#0369a1", "bold")
    msg(360, cx, sx, "create_responses(prev_id=id1, input=\"what is my name?\")")
    msg(404, sx + 160, stx, "prev = response_store.get(id1)  ·  hist = msg_store[id1]", "#0369a1")
    hline(L, stx, sx + 160, 404, "#0369a1", arrow=True)
    box(L, sx - 150, 424, 300, 42, "拼历史 + 本轮 input", "#eef2ff", "#6366f1", 12,
        "construct_input_messages(_with_harmony)")
    msg(496, sx, cx, "resp2（已知 \"Bob\"）", "#475569")

    text(L, w / 2, h - 18,
         "上轮 output 无需显式写回——因 msg_store[id] 与 context._messages 共享同一 list，续轮取回即得完整历史",
         12, "middle", "#64748b")
    return w, h, L


def write(name, tup):
    w, h, L = tup
    L.append('</svg>')
    path = f"/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch33-engine-core/diagrams/{name}.svg"
    with open(path, "w", encoding="utf-8") as f:
        f.write('\n'.join(L))
    print("wrote", path)


write("01-eep-scaleup-state-machine", diagram_scaleup())
write("02-eep-scaledown-states", diagram_scaledown())
write("03-kv-init-before-scale", diagram_kvinit())
write("04-responses-multiturn-stores", diagram_multiturn())
