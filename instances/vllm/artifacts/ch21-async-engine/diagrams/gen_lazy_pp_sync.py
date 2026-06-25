"""Lazy PP sync timeline: three swimlanes (stage k-1 / NCCL / stage k)."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


w, h = 1080, 470
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append('<text x="20" y="32" font-size="20" font-weight="bold" fill="#0f172a" '
         'font-family="sans-serif">惰性 PP 同步：一格 execute_model 的时序</text>')

lanes = [
    ("stage k-1", "#1d4ed8"),
    ("NCCL 传输", "#7c3aed"),
    ("stage k (本 rank)", "#047857"),
]
lane_x = 175
lane_w = w - lane_x - 20
lane_h = 80
lane_y0 = 70
for i, (name, color) in enumerate(lanes):
    y = lane_y0 + i * (lane_h + 24)
    L.append(f'<rect x="{lane_x}" y="{y}" width="{lane_w}" height="{lane_h}" '
             f'fill="#f8fafc" stroke="#e2e8f0"/>')
    L.append(f'<text x="20" y="{y + lane_h // 2 + 5}" font-size="15" '
             f'font-weight="bold" fill="{color}" font-family="sans-serif">{esc(name)}</text>')

# time axis ticks (relative)
t0 = lane_x + 10
tw = lane_w - 20


def box(lane_idx, x, bw, label, color, sub=None):
    y = lane_y0 + lane_idx * (lane_h + 24) + 18
    bh = 44
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" '
             f'fill="white" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{x + bw // 2}" y="{y + (26 if sub else 28)}" font-size="13" '
             f'text-anchor="middle" fill="{color}" font-family="sans-serif" '
             f'font-weight="bold">{esc(label)}</text>')
    if sub:
        L.append(f'<text x="{x + bw // 2}" y="{y + 40}" font-size="11" '
                 f'text-anchor="middle" fill="#64748b" '
                 f'font-family="sans-serif">{esc(sub)}</text>')
    return x, y, bw, bh


# stage k lane events
x = t0
b_wait = box(2, x, 120, "wait 上轮 isend", "#047857", "_pp_send_work")
x += 130
b_irecv = box(2, x, 110, "irecv 发出", "#047857", "不阻塞")
x += 120
b_prep = box(2, x, 230, "input / KV / attn metadata 准备", "#047857", "与接收重叠")
x += 240
b_read = box(2, x, 130, "首次读 .tensors", "#dc2626", "触发 wait_for_comm")
x += 140
b_fwd = box(2, x, 110, "model forward", "#047857")
x += 120
b_send = box(2, x, 90, "isend 发出", "#047857", "句柄留到下轮")

# NCCL lane: the receive in flight spanning irecv→read
ncl_x = b_irecv[0] + b_irecv[2] // 2
ncl_end = b_read[0] + 10
y_ncl = lane_y0 + 1 * (lane_h + 24) + 28
L.append(f'<rect x="{ncl_x}" y="{y_ncl}" width="{ncl_end - ncl_x}" height="30" rx="5" '
         f'fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>')
L.append(f'<text x="{(ncl_x + ncl_end) // 2}" y="{y_ncl + 20}" font-size="12" '
         f'text-anchor="middle" fill="#7c3aed" font-family="sans-serif">'
         'NCCL 接收在途（与准备并行）</text>')

# stage k-1 lane: its isend that feeds our irecv
y_up = lane_y0 + 0 * (lane_h + 24) + 28
L.append(f'<rect x="{t0}" y="{y_up}" width="120" height="30" rx="5" '
         f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{t0 + 60}" y="{y_up + 20}" font-size="12" text-anchor="middle" '
         f'fill="#1d4ed8" font-family="sans-serif">isend 隐藏状态</text>')

# arrow stage k-1 isend → NCCL
L.append(f'<line x1="{t0 + 60}" y1="{y_up + 30}" x2="{ncl_x}" y2="{y_ncl}" '
         'stroke="#7c3aed" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#a)"/>')

# the deferred-wait dashed line: from irecv to first read
ir_cx = b_irecv[0] + b_irecv[2] // 2
rd_cx = b_read[0] + b_read[2] // 2
y_band = lane_y0 + 2 * (lane_h + 24) + 18 + 44 + 18
L.append(f'<line x1="{ir_cx}" y1="{y_band}" x2="{rd_cx}" y2="{y_band}" '
         'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ar)"/>')
L.append(f'<text x="{(ir_cx + rd_cx) // 2}" y="{y_band + 18}" font-size="12.5" '
         f'text-anchor="middle" fill="#dc2626" font-family="sans-serif" '
         'font-weight="bold">推迟的 wait：irecv 发出 → 直到真正用到才阻塞</text>')

# NCCL receive completion arrow up into read box
L.append(f'<line x1="{ncl_end}" y1="{y_ncl + 15}" x2="{b_read[0] + 20}" '
         f'y2="{b_read[1]}" stroke="#7c3aed" stroke-width="1.5" '
         'stroke-dasharray="4 3" marker-end="url(#a)"/>')

# next-round wait note under isend — placed below swimlane + red arrow label
L.append(f'<text x="{b_send[0] + b_send[2] // 2}" y="{b_send[1] + b_send[3] + 45}" '
         'font-size="11" text-anchor="middle" fill="#64748b" '
         'font-family="sans-serif">下一轮开头才 wait</text>')

L.append('</svg>')
open("instances/vllm/artifacts/ch21-async-engine/diagrams/lazy-pp-sync-timeline.svg",
     "w", encoding="utf-8").write('\n'.join(L))
print("wrote lazy-pp-sync-timeline.svg")
