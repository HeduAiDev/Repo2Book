#!/usr/bin/env python3
"""Generate ch19 diagrams: two-phase timeline, f13 writeback, cudagraph dispatch."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))


def header(w, h, arrow_fill="#64748b"):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append(
        '<defs>'
        f'<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{arrow_fill}"/></marker>'
        '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
        '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
        '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#059669"/></marker>'
        '</defs>'
    )
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    return L


def box(L, x, y, w, h, fill, stroke, lines, fs=14, tcol="#1e293b", rx=8, weight=None):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 4) / 2 + fs / 2 - 2
    for i, (txt, *rest) in enumerate(lines):
        lf = rest[0] if rest else fs
        lc = rest[1] if len(rest) > 1 else tcol
        lw = rest[2] if len(rest) > 2 else (weight or "normal")
        L.append(
            f'<text x="{x + w/2}" y="{cy + i*(fs+4)}" font-family="sans-serif" '
            f'font-size="{lf}" fill="{lc}" font-weight="{lw}" text-anchor="middle">{esc(txt)}</text>'
        )


def text(L, x, y, s, fs=13, col="#1e293b", anchor="start", weight="normal", style=""):
    L.append(
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" fill="{col}" '
        f'font-weight="{weight}" text-anchor="{anchor}" {style}>{esc(s)}</text>'
    )


# ============================================================ #
# Diagram 1: two-phase timeline (execute_model / sample_tokens overlap)
# ============================================================ #
def two_phase_timeline():
    w, h = 1180, 560
    L = header(w, h)
    text(L, w/2, 34, "execute_model() 发起前向 / sample_tokens() 采样 —— 两阶段在时间上交错", 18, "#0f172a", "middle", "bold")

    lane_x = 250
    lane_w = w - lane_x - 30
    lanes = [
        ("worker / EngineCore", "#475569"),
        ("execute_model 前向", "#2563eb"),
        ("sample_tokens 采样+簿记", "#059669"),
    ]
    lane_y0 = 80
    lane_h = 130
    gap = 8
    for i, (name, col) in enumerate(lanes):
        y = lane_y0 + i * (lane_h + gap)
        L.append(f'<rect x="{lane_x}" y="{y}" width="{lane_w}" height="{lane_h}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
        text(L, lane_x - 12, y + lane_h/2 - 8, name, 13, col, "end", "bold")
        text(L, lane_x - 12, y + lane_h/2 + 10, "" , 11, "#94a3b8", "end")

    # time grid: ticks for 拍 n-1, n, n+1
    t0 = lane_x + 20
    span = lane_w - 40
    unit = span / 6.0
    # blocks: (lane_idx, t_start_unit, t_len_unit, fill, stroke, label_lines)
    def blk(lane, ts, tl, fill, stroke, label, sub=""):
        x = t0 + ts * unit
        bw = tl * unit
        y = lane_y0 + lane * (lane_h + gap) + 22
        bh = lane_h - 44
        L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        text(L, x + bw/2, y + bh/2 - 4, label, 13, "#0f172a", "middle", "bold")
        if sub:
            text(L, x + bw/2, y + bh/2 + 14, sub, 11, "#475569", "middle")
        return x, bw, y, bh

    # 拍 n
    blk(1, 0.0, 1.8, "#dbeafe", "#2563eb", "EXEC 第 n 拍", "前向 issue→GPU")
    blk(2, 1.8, 1.6, "#d1fae5", "#059669", "SAMPLE 第 n 拍", "解包 state→采样→写回")
    # 拍 n+1 — overlaps with sample n
    x1, bw1, y1, bh1 = blk(1, 2.2, 1.8, "#dbeafe", "#2563eb", "EXEC 第 n+1 拍", "前向 issue→GPU")
    blk(2, 4.0, 1.6, "#d1fae5", "#059669", "SAMPLE 第 n+1 拍", "解包 state→采样→写回")
    # worker lane: scheduling work bridging
    blk(0, 0.2, 1.6, "#f1f5f9", "#94a3b8", "调度第 n+1 拍", "schedule")
    blk(0, 2.4, 1.6, "#f1f5f9", "#94a3b8", "调度第 n+2 拍", "schedule")

    # overlap highlight band between sample-n and exec-(n+1)
    ox = t0 + 2.2 * unit
    ow = (3.4 - 2.2) * unit
    oy = lane_y0 + 1 * (lane_h + gap)
    oh = 2 * lane_h + gap
    L.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" fill="#fde68a" fill-opacity="0.28" stroke="#f59e0b" stroke-dasharray="5 4" stroke-width="1.5"/>')
    text(L, ox + ow/2, oy - 6, "重叠区", 12, "#b45309", "middle", "bold")

    # bridge slot annotation
    by = lane_y0 + 3 * (lane_h + gap) + 18
    L.append(f'<rect x="{lane_x}" y="{by}" width="{lane_w}" height="46" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>')
    text(L, lane_x + 16, by + 28, "self.execute_model_state（单槽桥）：EXEC 填入 ExecuteModelState → return None；SAMPLE 解包后置 None。入口断言强制两阶段严格配对。", 13, "#4338ca", "start")

    text(L, lane_x + 20, lane_y0 + 3*(lane_h+gap) + 86, "时间 →", 12, "#94a3b8", "start", "italic")
    return w, h, '\n'.join(L) + '\n</svg>'


# ============================================================ #
# Diagram 2: f13 persistent-batch writeback / readback closure
# ============================================================ #
def f13_writeback():
    w, h = 1180, 700
    L = header(w, h)
    text(L, w/2, 34, "f13 闭环：新 token 写回持久批次 token_ids_cpu → 活到下一拍 → 读回作输入", 18, "#0f172a", "middle", "bold")

    # grid: rows = slots, cols = token positions
    cols = 8
    rows = 3
    cell = 58
    rgap = 18
    gx = 210
    gy = 120
    # column header
    for c in range(cols):
        text(L, gx + c*cell + cell/2, gy - 12, f"pos {c}", 11, "#94a3b8", "middle")
    # row labels
    slot_labels = ["slot 0 (req A)", "slot 1 (req B)", "slot 2 (req C)"]

    # row r: filled positions [0..start) existing, [start:end) newly written
    starts = [4, 3, 5]   # num_tokens_no_spec before step n  (= 写入起点)
    vals = [
        ["7","3","9","1"],      # existing tokens A
        ["2","8","5"],          # existing tokens B
        ["4","6","0","2","9"],  # existing tokens C
    ]
    new_tok = ["Z", "Y", None]

    def row_y(r):
        return gy + r*(cell+rgap)

    for r in range(rows):
        y = row_y(r)
        text(L, gx - 16, y + cell/2 + 4, slot_labels[r], 12, "#334155", "end", "bold")
        st = starts[r]
        for c in range(cols):
            x = gx + c*cell
            if c < st:
                fill, stroke, txt, tc = "#f1f5f9", "#cbd5e1", (vals[r][c] if c < len(vals[r]) else ""), "#64748b"
            elif c == st and new_tok[r] is not None:
                fill, stroke, txt, tc = "#fecaca", "#dc2626", new_tok[r], "#7f1d1d"
            else:
                fill, stroke, txt, tc = "#ffffff", "#e2e8f0", "", "#94a3b8"
            L.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
            if txt:
                text(L, x + cell/2, y + cell/2 + 5, txt, 15, tc, "middle", "bold")

    # num_tokens_no_spec pointer label in the row gap, under the written cell
    for r in range(rows):
        if new_tok[r] is None:
            continue
        st = starts[r]
        x = gx + st*cell + cell/2
        y = row_y(r)
        text(L, x, y + cell + 13, "↑ start=num_tokens_no_spec", 10, "#dc2626", "middle")

    # req C note
    yc = row_y(2)
    text(L, gx + cols*cell + 14, yc + cell/2 + 4, "(本拍无采样)", 11, "#94a3b8", "start", "italic")

    grid_bottom = row_y(rows-1) + cell

    # write-back equation block
    eqy = grid_bottom + 40
    L.append(f'<rect x="{gx-14}" y="{eqy}" width="{640}" height="120" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.2"/>')
    text(L, gx, eqy + 26, "_bookkeeping_sync 第 n 拍写回（红格）：", 14, "#b91c1c", "start", "bold")
    text(L, gx, eqy + 48, "start = num_tokens_no_spec[r];  end = start + 1", 13, "#7f1d1d")
    text(L, gx, eqy + 68, "token_ids_cpu[r, start:end] = sampled_ids", 13, "#7f1d1d")
    text(L, gx, eqy + 88, "num_tokens_no_spec[r] = end          # 计数右移一格", 13, "#7f1d1d")
    text(L, gx, eqy + 108, "req_state.output_token_ids.extend(sampled_ids)  # 同一 list 别名同步增长", 13, "#7f1d1d")

    # aliasing arrow: req_output_token_ids[slot] is output_token_ids
    alx = gx + cols*cell + 130
    aly = gy
    L.append(f'<rect x="{alx}" y="{aly}" width="{300}" height="160" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>')
    text(L, alx + 150, aly + 24, "同一 list 对象（别名）", 13, "#4338ca", "middle", "bold")
    box(L, alx+20, aly+44, 110, 46, "#e0e7ff", "#6366f1", [("InputBatch", 11, "#3730a3"), (".req_output_token_ids[r]", 9, "#4338ca")])
    box(L, alx+170, aly+44, 110, 46, "#e0e7ff", "#6366f1", [("CachedRequestState", 10, "#3730a3"), (".output_token_ids", 10, "#4338ca")])
    L.append(f'<line x1="{alx+130}" y1="{aly+62}" x2="{alx+170}" y2="{aly+62}" stroke="#6366f1" stroke-width="1.5" marker-end="url(#ab)"/>')
    L.append(f'<line x1="{alx+170}" y1="{aly+78}" x2="{alx+130}" y2="{aly+78}" stroke="#6366f1" stroke-width="1.5" marker-end="url(#ab)"/>')
    text(L, alx + 150, aly + 120, "extend 一处 = 两个视图", 11, "#4338ca", "middle")
    text(L, alx + 150, aly + 138, "同步长出新 token", 11, "#4338ca", "middle")

    # write-back equation block (left, under grid)
    # read-back arrow block (right, under grid)
    rby = eqy
    L.append(f'<rect x="{alx-30}" y="{rby}" width="{360}" height="130" rx="8" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.2"/>')
    text(L, alx-14, rby + 26, "下一拍 _prepare_inputs 读回（绿）：", 14, "#047857", "start", "bold")
    text(L, alx-14, rby + 50, "positions = num_computed_tokens_cpu + query_pos", 12, "#065f46")
    text(L, alx-14, rby + 72, "token_indices = positions + r * max_model_len", 12, "#065f46")
    text(L, alx-14, rby + 94, "index_select 取到上拍写回的同一格", 12, "#065f46")
    text(L, alx-14, rby + 114, "        → 本拍 input_ids", 12, "#065f46")

    # curved closure arrow: from written cell (row 0) over the top to readback box
    wx = gx + starts[0]*cell + cell/2
    wy = row_y(0)
    rbx = alx + 150
    L.append(f'<path d="M {wx} {wy} C {wx} {wy-56}, {rbx} {rby-70}, {rbx} {rby}" fill="none" stroke="#059669" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ag)"/>')
    text(L, (wx+rbx)/2, wy-50, "写回的 token 在下一拍被读回", 12, "#059669", "middle", "bold")

    return w, h, '\n'.join(L) + '\n</svg>'


# ============================================================ #
# Diagram 3: cudagraph dispatch decision tree
# ============================================================ #
def cudagraph_dispatch():
    w, h = 1160, 620
    L = header(w, h)
    text(L, w/2, 34, "CudagraphDispatcher.dispatch：FULL → PIECEWISE → NONE 分级选择", 18, "#0f172a", "middle", "bold")

    cx = w/2
    # input box
    box(L, cx-180, 60, 360, 50, "#f1f5f9", "#94a3b8",
        [("输入：num_tokens / uniform_decode / has_lora", 13, "#334155")])

    # guard diamond-ish (use rounded box)
    gy = 140
    box(L, cx-210, gy, 420, 56, "#fff7ed", "#fb923c",
        [("未初始化 / mode==NONE / num_tokens > max_size ?", 13, "#9a3412", "bold")])
    L.append(f'<line x1="{cx}" y1="110" x2="{cx}" y2="{gy}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

    # NONE early (right)
    none1_x = cx + 330
    none_y = gy + 4
    box(L, none1_x-90, none_y, 180, 48, "#fee2e2", "#ef4444",
        [("NONE (eager)", 14, "#991b1b", "bold"), ("launch O(算子数)", 10, "#b91c1c")])
    L.append(f'<line x1="{cx+210}" y1="{gy+28}" x2="{none1_x-90}" y2="{none_y+24}" stroke="#dc2626" stroke-width="1.5" marker-end="url(#ar)"/>')
    text(L, (cx+210+none1_x-90)/2, gy+18, "是 → 回退", 11, "#dc2626", "middle", "bold")

    # build descriptor
    by = gy + 100
    box(L, cx-200, by, 400, 50, "#eef2ff", "#6366f1",
        [("构造 BatchDescriptor(num_tokens 填充后, num_reqs, uniform)", 12, "#3730a3")])
    L.append(f'<line x1="{cx}" y1="{gy+56}" x2="{cx}" y2="{by}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    text(L, cx+12, gy+82, "否", 11, "#475569", "start")

    # check FULL
    fy = by + 90
    box(L, cx-200, fy, 400, 54, "#faf5ff", "#a855f7",
        [("查 FULL keys（要求精确 num_reqs 命中）?", 13, "#6b21a8", "bold")])
    L.append(f'<line x1="{cx}" y1="{by+50}" x2="{cx}" y2="{fy}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

    # FULL result (left)
    full_x = cx - 300
    box(L, full_x-80, fy+2, 170, 48, "#dcfce7", "#22c55e",
        [("FULL", 14, "#166534", "bold"), ("整图 replay · launch ~O(1)", 10, "#15803d")])
    L.append(f'<line x1="{cx-200}" y1="{fy+27}" x2="{full_x+90}" y2="{fy+26}" stroke="#059669" stroke-width="1.5" marker-end="url(#ag)"/>')
    text(L, (cx-200+full_x+90)/2, fy+18, "命中", 11, "#059669", "middle", "bold")

    # check PIECEWISE
    py = fy + 100
    box(L, cx-200, py, 400, 54, "#faf5ff", "#a855f7",
        [("查 PIECEWISE relaxed keys（num_reqs=None）?", 13, "#6b21a8", "bold")])
    L.append(f'<line x1="{cx}" y1="{fy+54}" x2="{cx}" y2="{py}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    text(L, cx+12, fy+82, "未命中", 11, "#475569", "start")

    # PIECEWISE result (left)
    box(L, full_x-80, py+2, 170, 50, "#dbeafe", "#3b82f6",
        [("PIECEWISE", 14, "#1e40af", "bold"), ("算子段 replay·attn 动态", 9, "#1d4ed8"), ("launch O(段数)", 9, "#1d4ed8")])
    L.append(f'<line x1="{cx-200}" y1="{py+27}" x2="{full_x+90}" y2="{py+27}" stroke="#2563eb" stroke-width="1.5" marker-end="url(#ab)"/>')
    text(L, (cx-200+full_x+90)/2, py+18, "命中", 11, "#2563eb", "middle", "bold")

    # NONE fallback (bottom)
    ny = py + 100
    box(L, cx-90, ny, 180, 48, "#fee2e2", "#ef4444",
        [("NONE (eager)", 14, "#991b1b", "bold"), ("都未命中 → 回退", 10, "#b91c1c")])
    L.append(f'<line x1="{cx}" y1="{py+54}" x2="{cx}" y2="{ny}" stroke="#dc2626" stroke-width="1.5" marker-end="url(#ar)"/>')
    text(L, cx+12, py+82, "未命中", 11, "#475569", "start")

    # note
    text(L, cx, ny+78, "顺序即 launch 开销升序：先试最省的 FULL，退而求其次 PIECEWISE，再不行 eager。", 12, "#475569", "middle", "italic")
    return w, h, '\n'.join(L) + '\n</svg>'


for name, fn in [
    ("two-phase-timeline", two_phase_timeline),
    ("f13-writeback", f13_writeback),
    ("cudagraph-dispatch", cudagraph_dispatch),
]:
    w, h, svg = fn()
    with open(f"{name}.svg", "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {name}.svg ({w}x{h})")
