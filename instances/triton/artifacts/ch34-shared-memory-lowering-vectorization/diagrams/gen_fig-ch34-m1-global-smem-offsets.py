#!/usr/bin/env python3
"""layout 模板:AllocateSharedMemory 把每个 op 的 allocation.offset 钉成 i32 属性,
降级期所有分配都 gep 进同一个 global_smem extern 数组;总量钉进 triton_gpu.shared。
buffer 个数/offset 数值为教学摆放(provenance_note),字节区间语义忠实源码。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 教学摆放的 3 个 buffer(op, offset, size) —— 区间语义忠实源码,数值示意
BUFS = [
    ("op %0 (local_alloc)", 0, 64),
    ("op %1 (local_alloc)", 64, 32),
    ("op %2 (convert_layout tmp)", 96, 48),
]
TOTAL = 144  # triton_gpu.shared
BUF_COLORS = ["#93c5fd", "#86efac", "#fcd34d"]

PAD, TOP = 44, 150
BYTE_PX = 4.2  # 每字节像素宽
BAR_H = 56
OP_BOX_W, OP_BOX_H = 190, 54
OP_GAP_Y = 96

bar_w = TOTAL * BYTE_PX
w = PAD * 2 + max(bar_w, 640)
bar_x0 = PAD + (w - PAD * 2 - bar_w) / 2

op_top = TOP
bar_top = op_top + OP_BOX_H + OP_GAP_Y
h = bar_top + BAR_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("分配(钉 offset 属性)与物化(gep 进同一数组)解耦")}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("每个 op 的 allocation.offset 是钉在它自己身上的一个 i32 属性;降级期所有分配 gep 进唯一的 global_smem 大数组")}</text>',
     f'<text x="{PAD}" y="{TOP-24}" font-family="monospace" font-size="12" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("@global_smem : i8[0] extern, linkage=External, align 16 字节")}</text>']

# op 属性框(顶部)——均匀铺开避免相撞,gep 箭头斜连到各自的 bar 段中点
n = len(BUFS)
op_slot_w = (w - 2 * PAD) / n
for i, (name, off, size) in enumerate(BUFS):
    seg_x0 = bar_x0 + off * BYTE_PX
    seg_w = size * BYTE_PX
    ox = PAD + i * op_slot_w + (op_slot_w - OP_BOX_W) / 2
    L.append(f'<rect x="{ox}" y="{op_top}" width="{OP_BOX_W}" height="{OP_BOX_H}" rx="8" '
              f'fill="{BUF_COLORS[i]}" stroke="#334155" stroke-width="1.5"/>')
    L.append(f'<text x="{ox+OP_BOX_W/2}" y="{op_top+20}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" font-weight="bold" fill="#0f172a">'
              f'{esc(name)}</text>')
    L.append(f'<text x="{ox+OP_BOX_W/2}" y="{op_top+40}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" fill="#334155">'
              f'{esc(f"allocation.offset = {off} (i32)")}</text>')
    # gep 箭头:op 底部 -> bar 段中点
    ax = ox + OP_BOX_W / 2
    ay0 = op_top + OP_BOX_H
    ay1 = bar_top
    L.append(f'<line x1="{ax}" y1="{ay0}" x2="{seg_x0+seg_w/2}" y2="{ay1}" '
              'stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')
    mid_y = (ay0 + ay1) / 2
    L.append(f'<text x="{ax+6}" y="{mid_y}" font-family="sans-serif" font-size="10" '
              f'fill="#475569">{esc("gep")}</text>')

# global_smem 大数组条(底部),按 buffer 切段着色
for i, (name, off, size) in enumerate(BUFS):
    seg_x = bar_x0 + off * BYTE_PX
    seg_w = size * BYTE_PX
    L.append(f'<rect x="{seg_x}" y="{bar_top}" width="{seg_w}" height="{BAR_H}" '
              f'fill="{BUF_COLORS[i]}" stroke="#334155" stroke-width="1"/>')
    L.append(f'<text x="{seg_x+seg_w/2}" y="{bar_top+BAR_H/2+4}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" fill="#0f172a">'
              f'{esc(f"[{off},{off+size})")}</text>')
# 未分配尾段(到 TOTAL)
tail_off = BUFS[-1][1] + BUFS[-1][2]
if tail_off < TOTAL:
    seg_x = bar_x0 + tail_off * BYTE_PX
    seg_w = (TOTAL - tail_off) * BYTE_PX
    L.append(f'<rect x="{seg_x}" y="{bar_top}" width="{seg_w}" height="{BAR_H}" '
              f'fill="#f1f5f9" stroke="#cbd5e1" stroke-dasharray="3,3"/>')

L.append(f'<rect x="{bar_x0}" y="{bar_top}" width="{bar_w}" height="{BAR_H}" '
          'fill="none" stroke="#0f172a" stroke-width="2"/>')

# 总量徽标
badge_y = bar_top + BAR_H + 30
badge_w = 340
badge_x = bar_x0 + bar_w / 2 - badge_w / 2
L.append(f'<rect x="{badge_x}" y="{badge_y}" width="{badge_w}" height="40" rx="8" '
          'fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>')
L.append(f'<text x="{badge_x+badge_w/2}" y="{badge_y+25}" text-anchor="middle" '
          f'font-family="monospace" font-size="13" font-weight="bold" fill="#3730a3">'
          f'{esc(f"mod.triton_gpu.shared = {TOTAL} (总量属性)")}</text>')

# 会合点注记
foot_y = badge_y + 40 + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="monospace" font-size="12" '
          f'fill="#334155">'
          f'{esc("会合点: gep(i8, AddressOf(@global_smem), offset) —— 任一 op 读自己的 offset 就地从大数组里取基址")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m1-global-smem-offsets.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
