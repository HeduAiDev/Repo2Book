#!/usr/bin/env python3
"""schedule() 两阶段流程：先 RUNNING（含抢占回环），再 WAITING（守卫），最后组装 SchedulerOutput。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 860, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w//2}" y="32" text-anchor="middle" font-size="21" '
         'font-weight="bold" fill="#0f172a">schedule()：先 RUNNING 后 WAITING 的两阶段</text>')


def box(x, y, bw, bh, lines, fill, stroke, fs=13.5, tc="#0f172a"):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n = len(lines)
    for i, t in enumerate(lines):
        ty = y + bh / 2 - (n - 1) * (fs + 3) / 2 + i * (fs + 3) + fs / 2
        fw = "bold" if i == 0 and n > 1 else "normal"
        L.append(f'<text x="{x+bw/2}" y="{ty:.1f}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{fw}" fill="{tc}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, color="#475569", mid=None, dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ''
    mk = "ar" if color == "#dc2626" else "a"
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
             f'stroke-width="1.8"{d} marker-end="url(#{mk})"/>')
    if mid:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        L.append(f'<text x="{mx+6}" y="{my-3}" font-size="11.5" fill="{color}">{esc(mid)}</text>')


cx = 280  # main column center
bw = 300
# Phase 1 header band
L.append(f'<rect x="40" y="58" width="380" height="360" rx="9" fill="#f0fdf4" '
         'stroke="#86efac" stroke-width="1.5"/>')
L.append('<text x="60" y="80" font-size="14.5" font-weight="bold" fill="#166534">'
         '阶段 1 · 遍历 self.running（RUNNING 优先）</text>')

box(130, 92, bw, 46, ["token_budget = max_num_scheduled_tokens"], "#dcfce7", "#22c55e", 12.5)
box(130, 156, bw, 56,
    ["num_new_tokens =", "num_tokens_with_spec − num_computed_tokens"],
    "#dcfce7", "#22c55e", 12.5)
box(130, 230, bw, 44, ["allocate_slots(request, num_new_tokens)"], "#dcfce7", "#22c55e", 12.5)
# decision diamond (allocate None?)
dy = 300
L.append(f'<polygon points="280,{dy} 360,{dy+34} 280,{dy+68} 200,{dy+34}" '
         'fill="#fef9c3" stroke="#eab308" stroke-width="1.8"/>')
L.append(f'<text x="280" y="{dy+30}" text-anchor="middle" font-size="12" fill="#854d0e">'
         'new_blocks</text>')
L.append(f'<text x="280" y="{dy+46}" text-anchor="middle" font-size="12" fill="#854d0e">'
         '是 None?</text>')

arrow(280, 138, 280, 156)
arrow(280, 212, 280, 230)
arrow(280, 274, 280, dy)

# preempt box (right)
box(470, dy + 6, 320, 56,
    ["否则抢占：self.running.pop() → _preempt_request",
     "被抢者回 waiting、num_computed_tokens = 0"],
    "#fee2e2", "#ef4444", 11.5)
arrow(360, dy + 34, 470, dy + 34, "#dc2626", "是")
# loop back from preempt up-and-over to allocate box (right edge -> top of allocate)
L.append(f'<path d="M790,{dy+6} L820,{dy+6} L820,252 L430,252" fill="none" '
         'stroke="#dc2626" stroke-width="1.8" stroke-dasharray="5,4" marker-end="url(#ar)"/>')
L.append(f'<text x="700" y="247" font-size="11" fill="#dc2626">重试分配</text>')

# success out
arrow(200, dy + 34, 120, dy + 34, "#16a34a", "否")
box(40, dy + 50, 150, 56,
    ["记账：", "num_scheduled_tokens[id]", "budget −= n"],
    "#dcfce7", "#22c55e", 10.5)
L.append(f'<line x1="115" y1="{dy+34}" x2="115" y2="{dy+50}" stroke="#16a34a" '
         'stroke-width="1.8" marker-end="url(#a)"/>')

# Guard diamond between phases
gy = 440
L.append(f'<polygon points="230,{gy} 360,{gy+30} 230,{gy+60} 100,{gy+30}" '
         'fill="#fef9c3" stroke="#eab308" stroke-width="1.8"/>')
L.append(f'<text x="230" y="{gy+26}" text-anchor="middle" font-size="12" fill="#854d0e">'
         'if not preempted_reqs</text>')
L.append(f'<text x="230" y="{gy+42}" text-anchor="middle" font-size="12" fill="#854d0e">'
         '本拍发生过抢占?</text>')
arrow(280, 418, 230, gy)

# Phase 2 band
L.append(f'<rect x="40" y="{gy+78}" width="500" height="170" rx="9" fill="#eff6ff" '
         'stroke="#93c5fd" stroke-width="1.5"/>')
L.append(f'<text x="60" y="{gy+100}" font-size="14.5" font-weight="bold" fill="#1e40af">'
         '阶段 2 · 遍历 self.waiting（仅当未抢占且 UNPAUSED）</text>')
box(130, gy + 112, 320, 50,
    ["前缀缓存命中 → num_computed_tokens",
     "num_new_tokens = num_tokens − num_computed_tokens"],
    "#dbeafe", "#3b82f6", 11)
box(130, gy + 174, 320, 50,
    ["allocate_slots 成功 → self.running.append",
     "WAITING→new_reqs / PREEMPTED→resumed_reqs"],
    "#dbeafe", "#3b82f6", 11)
arrow(290, gy + 162, 290, gy + 174)
arrow(230, gy + 60, 290, gy + 112, "#2563eb", "否")

# skip-to-output when preempted
box(560, gy + 4, 270, 52,
    ["是：跳过 WAITING 阶段", "（刚释放的内存不让新请求占走）"],
    "#fee2e2", "#ef4444", 11.5)
arrow(360, gy + 30, 560, gy + 30, "#dc2626", "是")

# Output box
oy = h - 56
box(560, oy - 70, 270, 64,
    ["组装 SchedulerOutput", "new_reqs(全量) + cached_reqs(增量)",
     "+ _update_after_schedule 乐观推进"],
    "#f3e8ff", "#a855f7", 11)
arrow(450, gy + 199, 560, oy - 40, "#7c3aed")
arrow(695, gy + 56, 695, oy - 70, "#dc2626")

L.append('</svg>')
open("13-schedule-two-phase.svg", "w").write('\n'.join(L))
print("wrote 13-schedule-two-phase.svg")
