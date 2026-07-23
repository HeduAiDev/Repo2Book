#!/usr/bin/env python3
"""fig-ch23-dma-pipes — flow 模板。
DMA 算子族每个成员固定一条搬运方向 + 一条硬件流水引擎：
load GM→UB(MTE2)、store UB→GM(MTE3)、fixpipe L0C→GM/L1/UB(FIX)。
上半：Vector 核完整数据流骨架（GM→UB→(向量算子)→UB→GM）。
下半：fixpipe 的三路扇出（L0C→GM/L1/UB）。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)

def fit(s, maxw, base, floor=9.0):
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size

TITLE = "DMA 算子族：方向与硬件流水引擎一一对应"
SUBTITLE = "load(GM→UB,MTE2) / store(UB→GM,MTE3) / fixpipe(L0C→GM|L1|UB,FIX) —— HIVMDMAOps.td"

PAD = 50
BOX_W, BOX_H = 190, 76

# ── 上半：Vector 数据流骨架 GM -> UB -> (向量算子,自环) -> GM ──────────
SEC1_LABEL_Y = 100
row1_y = 260
gm1_x = PAD
ub1_x = gm1_x + BOX_W + 220
row1_w = ub1_x + BOX_W - PAD

# ── 下半：fixpipe 扇出 L0C -> {GM, L1, UB} ─────────────────────────
row2_y = row1_y + BOX_H + 250
l0c_x = PAD
targets_x0 = l0c_x + BOX_W + 220
TARGETS = ["GM(OUT)", "L1", "UB"]
target_gap = 40
targets_w = len(TARGETS) * BOX_W + (len(TARGETS)-1) * target_gap

w = max(row1_w, targets_x0 + targets_w - PAD) + PAD
h = row2_y + BOX_H + 170

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="mte2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '<marker id="mte3" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#c2410c"/></marker>'
     '<marker id="fix" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7e22ce"/></marker>'
     '<marker id="pv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# ── 上半标题（放在 arc 区域之上，避免与回弧/箭头重叠）──
L.append(f'<text x="{PAD}" y="{SEC1_LABEL_Y}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#475569">{esc("① Vector 核数据流骨架：GM → (load) UB → 向量算子 → (store) GM")}</text>')

def box(x, y, name, sub, fill, stroke, tf):
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="17" font-weight="bold" fill="{tf}">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+52}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fit(sub, BOX_W-16, 11)}" fill="{tf}">{esc(sub)}</text>')

box(gm1_x, row1_y, "GM", "全局内存", "#e0e7ff", "#4338ca", "#3730a3")
box(ub1_x, row1_y, "UB", "Vector 工作缓冲", "#bbf7d0", "#15803d", "#14532d")

cy1 = row1_y + BOX_H/2
# load: GM -> UB (MTE2)
L.append(f'<line x1="{gm1_x+BOX_W}" y1="{cy1}" x2="{ub1_x}" y2="{cy1}" '
          f'stroke="#1d4ed8" stroke-width="2.6" marker-end="url(#mte2)"/>')
L.append(f'<text x="{(gm1_x+BOX_W+ub1_x)/2}" y="{cy1-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1d4ed8">{esc("load：PIPE_MTE2")}</text>')
L.append(f'<text x="{(gm1_x+BOX_W+ub1_x)/2}" y="{cy1+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1d4ed8">{esc("ins(gm) outs(ub)")}</text>')

# store: UB -> GM (MTE3, 走上方回弧；弧顶留在 SEC1_LABEL_Y 下方，不与该标题重叠)
arc_top = SEC1_LABEL_Y + 50
L.append(f'<path d="M {ub1_x+BOX_W*0.25} {row1_y} C {ub1_x+BOX_W*0.25} {arc_top} '
          f'{gm1_x+BOX_W*0.75} {arc_top} {gm1_x+BOX_W*0.75} {row1_y}" fill="none" '
          f'stroke="#c2410c" stroke-width="2.4" marker-end="url(#mte3)"/>')
L.append(f'<text x="{(gm1_x+ub1_x+BOX_W)/2}" y="{arc_top-8}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#c2410c">{esc("store：PIPE_MTE3")}</text>')

# 向量算子自环（UB 内部，多个算子）
loop_y0 = row1_y + BOX_H + 14
loop_y1 = row1_y + BOX_H + 60
loop_cx = ub1_x + BOX_W/2
L.append(f'<path d="M {loop_cx-40} {loop_y0} C {loop_cx-70} {loop_y1} {loop_cx+70} {loop_y1} '
          f'{loop_cx+40} {loop_y0}" fill="none" stroke="#15803d" stroke-width="2.2" '
          f'marker-end="url(#pv)"/>')
L.append(f'<text x="{loop_cx}" y="{loop_y1+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#15803d">'
          f'{esc("PIPE_V 向量算子（多个，在 UB 上算）")}</text>')

# provenance 行
prov1_y = row1_y + BOX_H + 118
PROV1 = [
    (gm1_x+BOX_W/2, "全局内存"),
]
L.append(f'<text x="{gm1_x+BOX_W/2}" y="{prov1_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#94a3b8">{esc("HIVMDMAOps.td:L64/L81-82（load）")}</text>')
L.append(f'<text x="{ub1_x+BOX_W/2}" y="{prov1_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="#94a3b8">{esc("HIVMDMAOps.td:L146/L160（store）")}</text>')

# ── 下半：fixpipe 扇出 ────────────────────────────────────────────
L.append(f'<text x="{PAD}" y="{row2_y-24}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#475569">{esc("② fixpipe：Cube 收尾，L0C 三路扇出（PIPE_FIX）")}</text>')

box(l0c_x, row2_y, "L0C", "Cube 累加器", "#fed7aa", "#c2410c", "#7c2d12")

cy2 = row2_y + BOX_H/2
target_fills = [("#e0e7ff", "#4338ca", "#3730a3"), ("#fef9c3", "#a16207", "#78350f"),
                 ("#bbf7d0", "#15803d", "#14532d")]
tys = []
n_t = len(TARGETS)
total_th = n_t * BOX_H + (n_t-1) * 24
ty0 = cy2 - total_th/2
for i, name in enumerate(TARGETS):
    ty = ty0 + i * (BOX_H + 24)
    tys.append(ty)
    fill, stroke, tf = target_fills[i]
    box(targets_x0, ty, name, "", fill, stroke, tf)
    tcy = ty + BOX_H/2
    L.append(f'<path d="M {l0c_x+BOX_W} {cy2} L {targets_x0} {tcy}" fill="none" '
              f'stroke="#7e22ce" stroke-width="2.2" marker-end="url(#fix)"/>')

fix_label_x = l0c_x + BOX_W + (targets_x0 - l0c_x - BOX_W) / 2
L.append(f'<text x="{fix_label_x}" y="{cy2-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#7e22ce">{esc("fixpipe：PIPE_FIX")}</text>')

prov2_y = row2_y + BOX_H + 30
L.append(f'<text x="{l0c_x}" y="{prov2_y}" font-family="sans-serif" font-size="10" '
          f'fill="#94a3b8">{esc("HIVMDMAOps.td:L272/L280-283 L0C to OUT/L1/UB")}</text>')

foot_y0 = h - 60
FOOT = [
    "内存墙的「搬运」在 IR 上就是 DMA 算子：每个算子锁定一个源→目标方向和一条专用硬件传输引擎（MTE2/MTE3/FIX）。",
    "Vector 核数据流 = load(MTE2) 进 UB → 一串向量算子在 UB 上算 → store(MTE3) 回 GM；Cube 核收尾由 fixpipe(FIX) 把 L0C 累加结果搬出。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*22}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch23-dma-pipes.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
