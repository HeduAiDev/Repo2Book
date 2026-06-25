#!/usr/bin/env python3
"""Generate ch14 scheduler diagrams (preemption loop / dual-queue / lifecycle)."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

DEFS = (
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ared" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '<marker id="agrn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '</defs>'
)

def box(x, y, w, h, fill, stroke, rx=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

def txt(x, y, s, size=13, anchor="middle", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

def line(x1, y1, x2, y2, color="#475569", marker="a", dash=None, width=1.6):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{d}{m}/>')

def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
            + DEFS + f'<rect width="{w}" height="{h}" fill="white"/>' + body + '</svg>')


# ============ Diagram 1: preemption-loop-flow ============
def diagram1():
    w, h = 760, 620
    b = []
    cx = 300          # center column x for boxes
    bw = 280          # box width
    diamond_w = 210
    b.append(txt(w/2, 32, "RUNNING 阶段：allocate_slots 失败时的抢占循环", 16, weight="bold"))

    # node y positions
    ys = {}
    # entry
    y = 60
    b.append(box(cx-bw/2, y, bw, 44, "#e0f2fe", "#0369a1"))
    b.append(txt(cx, y+20, "遍历 self.running 取一个 request", 12.5))
    b.append(txt(cx, y+36, "num_new_tokens = 追赶量", 11.5, mono=True, fill="#475569"))
    ys['entry'] = (y, 44)

    # allocate_slots diamond-ish
    y = 138
    b.append(box(cx-diamond_w/2, y, diamond_w, 50, "#fef9c3", "#a16207"))
    b.append(txt(cx, y+22, "allocate_slots(request)", 12.5, mono=True))
    b.append(txt(cx, y+40, "能分到 KV 块吗？", 12.5))
    ys['alloc'] = (y, 50)
    b.append(line(cx, ys['entry'][0]+ys['entry'][1], cx, y))

    # success branch (right)
    sx = 620
    b.append(box(sx-90, y, 180, 50, "#dcfce7", "#15803d"))
    b.append(txt(sx, y+22, "能 → break", 12.5, weight="bold", fill="#15803d"))
    b.append(txt(sx, y+40, "调度该 request", 11.5))
    b.append(line(cx+diamond_w/2, y+25, sx-90, y+25, color="#15803d", marker="agrn"))
    b.append(txt((cx+diamond_w/2+sx-90)/2, y+16, "new_blocks≠None", 11, fill="#15803d"))

    # fail -> preempt pick (down)
    y2 = 250
    b.append(box(cx-bw/2, y2, bw, 50, "#fee2e2", "#b91c1c"))
    b.append(txt(cx, y2+21, "FCFS: preempted_req = running.pop()", 12, mono=True, fill="#b91c1c"))
    b.append(txt(cx, y2+39, "抢 RUNNING 末尾（LIFO，最晚加入者）", 11.5))
    b.append(line(cx, ys['alloc'][0]+ys['alloc'][1], cx, y2, color="#b91c1c", marker="ared"))
    b.append(txt(cx+88, (ys['alloc'][0]+ys['alloc'][1]+y2)/2+2, "不能", 11, fill="#b91c1c", anchor="start"))

    # _preempt_request
    y3 = 332
    b.append(box(cx-bw/2-30, y3, bw+60, 86, "#fff7ed", "#c2410c"))
    b.append(txt(cx, y3+20, "_preempt_request(preempted_req)", 12.5, mono=True, weight="bold"))
    b.append(txt(cx, y3+38, "free KV 块（丢弃，不换出到 CPU）", 11.5, anchor="middle"))
    b.append(txt(cx, y3+54, "status → PREEMPTED ；num_computed_tokens = 0", 11, mono=True))
    b.append(txt(cx, y3+70, "num_preemptions++ ；waiting.prepend_request 回队头", 11, mono=True))
    b.append(line(cx, y2+50, cx, y3, color="#b91c1c", marker="ared"))

    # decision: preempted_req == request?
    y4 = 452
    b.append(box(cx-diamond_w/2, y4, diamond_w, 50, "#fef9c3", "#a16207"))
    b.append(txt(cx, y4+22, "preempted_req == request ?", 11.5, mono=True))
    b.append(txt(cx, y4+40, "（连自己都抢了？）", 11.5))
    b.append(line(cx, y3+86, cx, y4))

    # no -> back to allocate (left loop arrow)
    lx = 70
    b.append(line(cx-diamond_w/2, y4+25, lx, y4+25, color="#475569", marker=None))
    b.append(line(lx, y4+25, lx, ys['alloc'][0]+25, color="#475569", marker=None))
    b.append(line(lx, ys['alloc'][0]+25, cx-diamond_w/2, ys['alloc'][0]+25, color="#475569", marker="a"))
    b.append(txt(lx+8, (y4+25+ys['alloc'][0]+25)/2, "否：重试分配", 11, anchor="start", fill="#475569"))
    b.append(txt(lx+30, y4+18, "否", 11, fill="#475569"))

    # yes -> give up
    y5 = 556
    b.append(box(cx-bw/2, y5, bw, 44, "#f1f5f9", "#475569"))
    b.append(txt(cx, y5+19, "break：本请求这一拍排不进", 12, weight="bold"))
    b.append(txt(cx, y5+35, "外层 new_blocks is None → 退出 RUNNING 循环", 10.5, fill="#475569"))
    b.append(line(cx, y4+50, cx, y5, marker="a"))
    b.append(txt(cx+18, (y4+50+y5)/2, "是", 11, anchor="start", fill="#475569"))

    return svg(w, h, "".join(b))


# ============ Diagram 2: dual-queue-anti-hol ============
def diagram2():
    w, h = 980, 470
    b = []
    b.append(txt(w/2, 30, "队头阻塞 vs waiting / skipped_waiting 双队列", 16, weight="bold"))

    # ---- LEFT: single queue, HOL blocking ----
    lx = 40
    b.append(txt(lx+160, 64, "单队列：阻塞态卡住队头", 13.5, weight="bold", fill="#b91c1c"))
    # queue slots horizontal
    slot_w, slot_h, gap = 64, 40, 8
    qy = 90
    labels = [("R0", "#fee2e2", "#b91c1c", "等远程KV\n(阻塞)"),
              ("R1", "#f1f5f9", "#94a3b8", "可调度"),
              ("R2", "#f1f5f9", "#94a3b8", "可调度"),
              ("R3", "#f1f5f9", "#94a3b8", "可调度")]
    for i, (name, fill, st, sub) in enumerate(labels):
        x = lx + i*(slot_w+gap)
        b.append(box(x, qy, slot_w, slot_h, fill, st))
        b.append(txt(x+slot_w/2, qy+18, name, 13, weight="bold", mono=True))
        b.append(txt(x+slot_w/2, qy+33, sub.split("\n")[0], 9, fill="#64748b"))
    b.append(txt(lx, qy+slot_h+18, "↑ 队头", 11, anchor="start", fill="#b91c1c"))
    # blocked marker
    bx = lx + slot_w/2
    b.append(txt(lx+150, qy+slot_h+18, "R0 提升失败 → 整队卡死，R1..R3 全部饿死", 11, anchor="start", fill="#b91c1c"))
    # consequence box
    cy = qy+slot_h+40
    b.append(box(lx, cy, 360, 46, "#fef2f2", "#b91c1c"))
    b.append(txt(lx+180, cy+19, "吞吐塌缩：本可调度的请求被一个", 11.5, fill="#b91c1c"))
    b.append(txt(lx+180, cy+35, "阻塞请求堵在身后", 11.5, fill="#b91c1c"))

    # divider
    b.append(line(440, 60, 440, 420, color="#cbd5e1", marker=None, dash="4,4"))

    # ---- RIGHT: dual queue ----
    rx = 470
    b.append(txt(rx+190, 64, "双队列：阻塞态隔离到 skipped_waiting", 13.5, weight="bold", fill="#15803d"))

    # waiting queue
    wy = 90
    b.append(txt(rx, wy-6, "waiting（可调度）", 11.5, anchor="start", fill="#0369a1"))
    wlabels = [("R1", "#dbeafe", "#0369a1"), ("R2", "#dbeafe", "#0369a1"), ("R3", "#dbeafe", "#0369a1")]
    for i, (name, fill, st) in enumerate(wlabels):
        x = rx + i*(slot_w+gap)
        b.append(box(x, wy, slot_w, slot_h, fill, st))
        b.append(txt(x+slot_w/2, wy+24, name, 13, weight="bold", mono=True))

    # skipped queue
    sy = wy + 78
    b.append(txt(rx, sy-6, "skipped_waiting（阻塞态隔离区）", 11.5, anchor="start", fill="#b45309"))
    b.append(box(rx, sy, slot_w, slot_h, "#fef3c7", "#b45309"))
    b.append(txt(rx+slot_w/2, sy+18, "R0", 13, weight="bold", mono=True))
    b.append(txt(rx+slot_w/2, sy+33, "等远程KV", 9, fill="#b45309"))

    # _select picks waiting -> schedule R1..R3
    schy = sy + 84
    b.append(box(rx, schy, 230, 44, "#dcfce7", "#15803d"))
    b.append(txt(rx+115, schy+19, "_select → 调度 R1, R2, R3", 12, fill="#15803d"))
    b.append(txt(rx+115, schy+35, "可调度请求照常前进", 10.5, fill="#15803d"))
    # green waiting→_select arrow routed down the LEFT of R0 so it never crosses
    # the skipped_waiting box (R0 spans rx..rx+slot_w).
    gx = rx - 14
    b.append(line(rx+slot_w/2, wy+slot_h, gx, wy+slot_h+10, color="#15803d", marker=None))
    b.append(line(gx, wy+slot_h+10, gx, schy+22, color="#15803d", marker=None))
    b.append(line(gx, schy+22, rx, schy+22, color="#15803d", marker="agrn"))

    # R0 try_promote fail -> step_skipped_waiting -> prepend back
    b.append(box(rx+270, schy, 220, 44, "#fff7ed", "#b45309"))
    b.append(txt(rx+380, schy+18, "R0 提升失败 → pop 出来", 10.5, fill="#b45309"))
    b.append(txt(rx+380, schy+34, "step 末 prepend 回 skipped", 10.5, fill="#b45309"))
    b.append(line(rx+slot_w, sy+slot_h/2, rx+270, schy+22, color="#b45309", marker="ared"))

    return svg(w, h, "".join(b))


# ============ Diagram 3: request-lifecycle-statemachine ============
def diagram3():
    w, h = 820, 540
    b = []
    b.append(txt(w/2, 30, "请求状态机：PREEMPTED 回流环 + is_finished 分界线", 16, weight="bold"))

    bw, bh = 150, 46
    # WAITING
    wx, wy = 70, 130
    b.append(box(wx, wy, bw, bh, "#dbeafe", "#0369a1"))
    b.append(txt(wx+bw/2, wy+22, "WAITING", 14, weight="bold", mono=True, fill="#0369a1"))
    b.append(txt(wx+bw/2, wy+38, "在 waiting 队列等内存", 10, fill="#475569"))

    # RUNNING
    rx, ry = 330, 130
    b.append(box(rx, ry, bw, bh, "#dcfce7", "#15803d"))
    b.append(txt(rx+bw/2, ry+22, "RUNNING", 14, weight="bold", mono=True, fill="#15803d"))
    b.append(txt(rx+bw/2, ry+38, "在 running 队列推进", 10, fill="#475569"))

    # PREEMPTED
    px, py = 330, 280
    b.append(box(px, py, bw, bh, "#fee2e2", "#b91c1c"))
    b.append(txt(px+bw/2, py+22, "PREEMPTED", 14, weight="bold", mono=True, fill="#b91c1c"))
    b.append(txt(px+bw/2, py+38, "被抢占，回 waiting 队头", 10, fill="#475569"))

    # WAITING -> RUNNING (schedule)
    b.append(line(wx+bw, wy+bh/2, rx, ry+bh/2, color="#15803d", marker="agrn"))
    b.append(txt((wx+bw+rx)/2, wy+bh/2-8, "schedule 拉起", 11, fill="#15803d"))
    b.append(txt((wx+bw+rx)/2, wy+bh/2+16, "(scheduled_new)", 9.5, fill="#64748b", mono=True))

    # RUNNING -> PREEMPTED (preempt)
    b.append(line(rx+bw/2-20, ry+bh, px+bw/2-20, py, color="#b91c1c", marker="ared"))
    b.append(txt(rx+bw+78, (ry+bh+py)/2, "_preempt_request", 10.5, fill="#b91c1c", mono=True, anchor="middle"))
    b.append(txt(rx+bw+78, (ry+bh+py)/2+16, "free KV / computed=0", 9.5, fill="#64748b", anchor="middle"))

    # PREEMPTED -> WAITING (reflux): curved via left
    b.append(line(px, py+bh/2, wx+bw/2, py+bh/2, color="#b91c1c", marker=None))
    b.append(line(wx+bw/2, py+bh/2, wx+bw/2, wy+bh, color="#b91c1c", marker="ared"))
    b.append(txt(px-86, py+bh/2-8, "prepend 回队头", 11, fill="#b91c1c", anchor="middle"))

    # PREEMPTED -> RUNNING (resumed)
    b.append(line(px+bw/2+20, py, rx+bw/2+20, ry+bh, color="#0369a1", marker="a"))
    b.append(txt(rx+bw+58, (ry+bh+py)/2+34, "scheduled_resumed", 9.5, fill="#0369a1", mono=True, anchor="middle"))

    # dividing line is_finished = status > PREEMPTED
    divy = 380
    b.append(line(40, divy, w-40, divy, color="#7c3aed", marker=None, dash="6,4", width=2))
    b.append(txt(w-44, divy-8, "is_finished = status > PREEMPTED", 12, anchor="end", fill="#7c3aed", weight="bold", mono=True))
    b.append(txt(44, divy-8, "↑ 活跃态", 11, anchor="start", fill="#7c3aed"))
    b.append(txt(44, divy+18, "↓ 终止态（IntEnum 顺序排在 PREEMPTED 之后）", 11, anchor="start", fill="#7c3aed"))

    # FINISHED states row
    fy = 430
    fins = [("FINISHED_STOPPED", "EOS / stop_token_id"),
            ("FINISHED_LENGTH_CAPPED", "max_tokens / max_len"),
            ("FINISHED_REPETITION", "重复检测"),
            ("FINISHED_ABORTED", "外部 abort")]
    fw = 178
    fgap = 12
    total = len(fins)*fw + (len(fins)-1)*fgap
    startx = (w-total)/2
    for i, (name, sub) in enumerate(fins):
        x = startx + i*(fw+fgap)
        b.append(box(x, fy, fw, 50, "#ede9fe", "#7c3aed"))
        b.append(txt(x+fw/2, fy+21, name, 11, weight="bold", mono=True, fill="#6d28d9"))
        b.append(txt(x+fw/2, fy+38, sub, 9.5, fill="#64748b"))

    # arrows RUNNING/PREEMPTED -> finished band (check_stop), routed straight down
    # from RUNNING: down its right side to the band
    rdx = rx + bw + 30
    b.append(line(rx+bw, ry+bh/2, rdx, ry+bh/2, color="#6d28d9", marker=None, dash="4,3"))
    b.append(line(rdx, ry+bh/2, rdx, fy-4, color="#6d28d9", marker="a", dash="4,3"))
    b.append(txt(rdx+8, ry+bh/2-8, "check_stop 命中 → 终态", 11, fill="#6d28d9", anchor="start"))
    # from PREEMPTED: stopped while preempted (rare) -> FINISHED_LENGTH_CAPPED top-center
    # route: down to dividing line level, then left to box center, then down to box top
    flc_cx = startx + fw + fgap + fw/2   # center x of FINISHED_LENGTH_CAPPED = 315
    b.append(line(px+bw/2, py+bh, px+bw/2, divy+6, color="#6d28d9", marker=None, dash="4,3"))
    b.append(line(px+bw/2, divy+6, flc_cx, divy+6, color="#6d28d9", marker=None, dash="4,3"))
    b.append(line(flc_cx, divy+6, flc_cx, fy-4, color="#6d28d9", marker="a", dash="4,3"))

    return svg(w, h, "".join(b))


import os
outdir = os.path.dirname(os.path.abspath(__file__))
for name, fn in [("preemption-loop-flow", diagram1),
                 ("dual-queue-anti-hol", diagram2),
                 ("request-lifecycle-statemachine", diagram3)]:
    path = os.path.join(outdir, name + ".svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write(fn())
    print("wrote", path)
