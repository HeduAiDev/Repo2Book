#!/usr/bin/env python3
"""figure_id: shared-mem-ring-buffer
claim: num_stages 深度直接决定共享内存里几份 tile buffer 环形轮转:非 MMAv3
直取时 numBuffers=num_stages-1,第 i 次迭代 AsyncCopy 写 insertIdx=i%numBuffers
格、dot 从 extractIdx 格读,首尾轮转;每加一个 stage 多一份 tile 共享内存。
数字来自 explainer/traces/derive_schedule.out.json
schedule_by_num_stages[num_stages=4](num_buffers_ampere=3, num_buffers_mmav3=4,
smem_KB_ampere=48, per_buffer_KB=16)。
"""
import math
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

# ---------- 数据 ----------
NUM_STAGES = 4                 # 本例选定,使环有 3 格便于画环
NUM_BUFFERS = 3                 # num_buffers_ampere(非 MMAv3,直取)
DIST_TO_USE = NUM_BUFFERS        # 本例 numBuffers=max(distToUse)=distToUse=3
NUM_BUFFERS_MMAV3 = 4           # num_buffers_mmav3 = +1
PER_BUFFER_KB = 16               # A 8KB + B 8KB
TOTAL_KB = NUM_BUFFERS * PER_BUFFER_KB   # 48
NUM_ITERS_SHOWN = 6              # 迭代 0..5,足够看出 i%3 的轮转

BUF_COLOR = ["#3b82f6", "#22c55e", "#f59e0b"]   # buffer0/1/2
WRITE_COLOR = "#1d4ed8"          # insertIdx(写,load)
READ_COLOR = "#b45309"           # extractIdx(读,dot)

W, H = 1300, 720
PAD = 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
          '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
          '<marker id="arrRing" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
          'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
          '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc(f"num_stages={NUM_STAGES}(非 MMAv3、直取 load):共享内存里 {NUM_BUFFERS} 份 tile buffer 环形轮转")}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">'
          f'{esc("第 i 次迭代的 load 写 insertIdx=i%3 格(为自己将来用);distToUse=3 拍后,同一迭代 i 自己的 dot 从 extractIdx=i%3 格读出——同一个格,写和读相隔 3 拍")}</text>')

# ============ 上半:迭代时间线 -> 每格同时标 insertIdx(写,现在)与 extractIdx(读,+3拍后) ============
top_y = PAD + 70
box_w, box_h, gap = 150, 82, 16
start_x = PAD + 10
L.append(f'<text x="{PAD}" y="{top_y - 14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#334155">'
          f'{esc("迭代时间线——每格是同一个物理 buffer 格的写(insertIdx)与读(extractIdx)两次使用")}</text>')
iter_centers = []
for i in range(NUM_ITERS_SHOWN):
    x = start_x + i * (box_w + gap)
    buf = i % NUM_BUFFERS
    color = BUF_COLOR[buf]
    cx = x + box_w / 2
    iter_centers.append((cx, top_y + box_h))
    L.append(f'<rect x="{x}" y="{top_y}" width="{box_w}" height="{box_h}" rx="8" '
              f'fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{top_y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(f"迭代 {i}")}</text>')
    L.append(f'<text x="{cx}" y="{top_y+39}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{WRITE_COLOR}">'
              f'{esc(f"写 insertIdx={buf}")}</text>')
    L.append(f'<text x="{cx}" y="{top_y+56}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{READ_COLOR}">'
              f'{esc(f"读 extractIdx={buf}")}</text>')
    L.append(f'<text x="{cx}" y="{top_y+72}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9" fill="#64748b">{esc("同一格,隔 3 拍两次用")}</text>')

end_x = start_x + NUM_ITERS_SHOWN * (box_w + gap) - gap
L.append(f'<text x="{end_x + 14}" y="{top_y + box_h/2 + 4}" font-family="sans-serif" '
          f'font-size="14" fill="#94a3b8">{esc("…")}</text>')

# ============ 下半:环形 buffer(3 格) ============
ring_cy = top_y + box_h + 300
ring_cx = PAD + 230
ring_r = 150
node_r = 74
angles = [-90, 30, 150]   # 3 个 buffer 均匀分布(度)

node_centers = []
for idx, ang in enumerate(angles):
    rad = math.radians(ang)
    nx = ring_cx + ring_r * math.cos(rad)
    ny = ring_cy + ring_r * math.sin(rad)
    node_centers.append((nx, ny))

# 环形箭头(buf0 -> buf1 -> buf2 -> buf0),沿圆弧
for idx in range(3):
    a0 = angles[idx]
    a1 = angles[(idx + 1) % 3]
    if idx == 2:
        a1 += 360
    # 圆弧起止角:从 node 边缘偏移,避免箭头压在节点上
    start_ang = a0 + 26
    end_ang = a1 - 26
    steps = 24
    pts = []
    for s in range(steps + 1):
        a = math.radians(start_ang + (end_ang - start_ang) * s / steps)
        px = ring_cx + ring_r * math.cos(a)
        py = ring_cy + ring_r * math.sin(a)
        pts.append(f"{px:.1f},{py:.1f}")
    path_d = "M " + " L ".join(pts)
    L.append(f'<path d="{path_d}" fill="none" stroke="#94a3b8" stroke-width="2" '
              f'marker-end="url(#arrRing)"/>')

for idx, (nx, ny) in enumerate(node_centers):
    color = BUF_COLOR[idx]
    L.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{node_r}" fill="{color}" '
              f'fill-opacity="0.16" stroke="{color}" stroke-width="2.5"/>')
    L.append(f'<text x="{nx:.1f}" y="{ny-32:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(f"buffer {idx}")}</text>')
    L.append(f'<text x="{nx:.1f}" y="{ny-14:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#334155">{esc("A 8KB + B 8KB")}</text>')
    L.append(f'<text x="{nx:.1f}" y="{ny+2:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="{color}">{esc("= 16KB")}</text>')
    writers = [i for i in range(NUM_ITERS_SHOWN) if i % NUM_BUFFERS == idx]
    writers_s = ",".join(str(w) for w in writers)
    L.append(f'<text x="{nx:.1f}" y="{ny+21:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" font-weight="bold" fill="{WRITE_COLOR}">'
              f'{esc(f"写 insertIdx:迭代 {writers_s} 的 load")}</text>')
    L.append(f'<text x="{nx:.1f}" y="{ny+37:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" font-weight="bold" fill="{READ_COLOR}">'
              f'{esc(f"读 extractIdx:同一迭代自己的 dot")}</text>')
    L.append(f'<text x="{nx:.1f}" y="{ny+53:.1f}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9" fill="{READ_COLOR}">'
              f'{esc(f"(distToUse=3 拍后读出)")}</text>')

# 迭代时间线与环的映射改用同色标注(而非连线跨图穿过其它节点造成压字)——
# 每个迭代方块的边框色已与其 insertIdx 对应的 buffer 节点同色,足以传达映射关系。
legend_y = top_y + box_h + 30
L.append(f'<text x="{start_x}" y="{legend_y}" font-family="sans-serif" font-size="11" '
          f'fill="#94a3b8">'
          f'{esc("(同色 = 同一 buffer;迭代方块边框色对应下方环上同色节点)")}</text>')
L.append(f'<rect x="{start_x + 620}" y="{legend_y-13}" width="14" height="14" rx="3" '
          f'fill="{WRITE_COLOR}" fill-opacity="0.75"/>')
L.append(f'<text x="{start_x + 640}" y="{legend_y}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="{WRITE_COLOR}">{esc("写(insertIdx,load,现在)")}</text>')
L.append(f'<rect x="{start_x + 840}" y="{legend_y-13}" width="14" height="14" rx="3" '
          f'fill="{READ_COLOR}" fill-opacity="0.75"/>')
L.append(f'<text x="{start_x + 860}" y="{legend_y}" font-family="sans-serif" font-size="11" '
          f'font-weight="bold" fill="{READ_COLOR}">'
          f'{esc("读(extractIdx,dot,distToUse=3 拍后)")}</text>')

# ============ 右侧:共享内存总量 + MMAv3 对比 ============
box_x = ring_cx + ring_r + node_r + 130
box_y = ring_cy - 150
box_w2 = 330
L.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w2}" height="150" rx="10" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{box_x+18}" y="{box_y+28}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc("共享内存占用(非 MMAv3)")}</text>')
L.append(f'<text x="{box_x+18}" y="{box_y+52}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(f"numBuffers = {NUM_BUFFERS} × 16KB")}</text>')
L.append(f'<text x="{box_x+18}" y="{box_y+72}" font-family="sans-serif" font-size="16" '
          f'font-weight="bold" fill="#1d4ed8">{esc(f"= {TOTAL_KB}KB")}</text>')
L.append(f'<line x1="{box_x+18}" y1="{box_y+86}" x2="{box_x+box_w2-18}" y2="{box_y+86}" '
          'stroke="#e2e8f0"/>')
L.append(f'<text x="{box_x+18}" y="{box_y+106}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("Hopper MMAv3:numBuffers 再 +1")}</text>')
L.append(f'<text x="{box_x+18}" y="{box_y+126}" font-family="sans-serif" font-size="14" '
          f'font-weight="bold" fill="#b45309">'
          f'{esc(f"→ numBuffers={NUM_BUFFERS_MMAV3}")}</text>')

foot_y = H - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("每加深一个 stage(num_stages+1)恰好多一份 16KB 的 buffer 轮转格——这是 num_stages 调多爆共享内存的物理落点")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("shared-mem-ring-buffer.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
