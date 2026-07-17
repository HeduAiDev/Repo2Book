#!/usr/bin/env python3
"""layout 模板:redundantDataMask 用 icmp_slt(threadDim*sizePerThread, shape) 只放行
warp0/warp1 写全局内存,warp2/warp3 谓词为假被屏蔽——广播布局(shape 64 < CTA tile 128)
下数据被复制 2x,store 流量减半。数据取自 explainer/traces/redundant_mask.out。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

WARPS = [
    ("warp 0", "threadDim 0..31", "WRITE"),
    ("warp 1", "threadDim 32..63", "WRITE"),
    ("warp 2", "threadDim 64..95", "masked"),
    ("warp 3", "threadDim 96..127", "masked"),
]
COLORS = {"WRITE": ("#86efac", "#166534"), "masked": ("#e2e8f0", "#64748b")}

CELL_W, CELL_H, GAP, PAD, TOP = 220, 90, 14, 40, 150
n = len(WARPS)
w = PAD * 2 + n * CELL_W + (n - 1) * GAP
h = TOP + CELL_H + 220

TITLE = "redundantDataMask:shape(64) < CTA tile(128) → 只放行 warp0/1 写全局内存"
SUBTITLE = "谓词 icmp_slt(threadDim·sizePerThread, shape);threadDim = warpId·32+laneId,阈值在 threadDim=64 处翻转"
min_w_text = PAD * 2 + max(cjk_text_width(TITLE, 16) / 16 * 16, cjk_text_width(SUBTITLE, 12))
w = max(w, min_w_text)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

# CTA tile 外框(128 宽的逻辑刻度,顶部标注)
tile_x0 = PAD
tile_w = n * CELL_W + (n - 1) * GAP
L.append(f'<text x="{tile_x0}" y="{TOP-46}" font-family="monospace" font-size="12" '
          f'fill="#334155">{esc("shapePerCTATile = 128(4 warp x 32 lane),shape = 64(实际逻辑元素数)")}</text>')

for i, (name, rng, action) in enumerate(WARPS):
    x = PAD + i * (CELL_W + GAP)
    fill, tcolor = COLORS[action]
    L.append(f'<rect x="{x}" y="{TOP}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              f'fill="{fill}" stroke="{tcolor}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{TOP+30}" text-anchor="middle" '
              f'font-family="monospace" font-size="14" font-weight="bold" '
              f'fill="{tcolor}">{esc(name)}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{TOP+52}" text-anchor="middle" '
              f'font-family="monospace" font-size="11" fill="{tcolor}">{esc(rng)}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{TOP+74}" text-anchor="middle" '
              f'font-family="monospace" font-size="13" font-weight="bold" '
              f'fill="{tcolor}">{esc(action)}</text>')

# shape=64 阈值线(在 warp1/warp2 之间)
boundary_x = PAD + 2 * CELL_W + GAP
L.append(f'<line x1="{boundary_x}" y1="{TOP-20}" x2="{boundary_x}" y2="{TOP+CELL_H+14}" '
          'stroke="#dc2626" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{boundary_x+8}" y="{TOP-24}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#dc2626">{esc("threadDim = 64 = shape 边界")}</text>')

# 图例
legend_y = TOP + CELL_H + 40
lx = PAD
for key, label in [("WRITE", "谓词真:唯一写者"), ("masked", "谓词假:复制副本,屏蔽")]:
    fill, tcolor = COLORS[key]
    L.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{tcolor}" stroke-width="1.5"/>')
    L.append(f'<text x="{lx+22}" y="{legend_y+13}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + cjk_text_width(label, 11) + 40

# 底部结论条
foot_y = legend_y + 32
foot_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{foot_w}" height="66" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y+26}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("写者 64/128 线程(warp0/1),屏蔽 64/128(warp2/3) —— store 流量 x0.50")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+46}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("覆盖检查:64/64 个逻辑索引各写一次(True)——写者恰好覆盖 [0,shape),复制副本不重复写")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m6-writer-mask.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
