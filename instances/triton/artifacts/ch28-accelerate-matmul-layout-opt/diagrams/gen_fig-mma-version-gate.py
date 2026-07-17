#!/usr/bin/env python3
"""fig-mma-version-gate (flow 模板)
MMA 版本判定链:compute capability 分档给候选表,{3,2} 分支逐门验票——
过不了 v3 的 K/shape/warps/dtype 门就退 v2,v2 的 dtype 门也过不了就退回 chosen=0(FMA)。
全部坐标由常量/循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD = 50
TOP = 128
COL_W = 270
COL_GAP = 34
BOX_H = 54
ROW_GAP = 28

SIDE_W = 170  # 右侧 MMAv3/MMAv2 终局框宽度
SIDE_GAP = 30
D_COL_W = 210  # cap>=100 独立列(置于最右,不与 side box 同列)
w = PAD * 2 + COL_W * 3 + COL_GAP * 2 + SIDE_GAP + SIDE_W + COL_GAP + D_COL_W
# C 列往下还要走 4 级(候选→v3门→v2门→终局),比 A/B 列(候选→终局)多 2 级
LEVELS_AB = 2
LEVELS_C = 4
h = TOP + BOX_H * LEVELS_C + ROW_GAP * (LEVELS_C - 1) + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
     '<marker id="n" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="46" font-family="sans-serif" font-size="19" '
          f'font-weight="bold" fill="#0f172a">{esc("getMMAVersionSafe:按站台等级给候选表,逐档验票,验不过就退档")}</text>')
L.append(f'<text x="{PAD}" y="70" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("cap 分档阈值 75 / 90 / 100(AccelerateMatmul.cpp:L30-L38);sm90 的 {3,2} 候选需逐个过 K/shape/warps/dtype 门")}</text>')

cols_x = [PAD + i * (COL_W + COL_GAP) for i in range(3)]
d_col_x = cols_x[2] + COL_W + SIDE_GAP + SIDE_W + COL_GAP  # cap>=100 独立列,置于 side box 右侧

def box(x, y, wdt, hgt, fill, stroke, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{wdt}" height="{hgt}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def multi_text(cx, cy, lines, size, color, bold=False):
    fw = 'font-weight="bold" ' if bold else ''
    y0 = cy - (len(lines) - 1) * (size * 0.6)
    for i, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{y0+i*size*1.25+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{size}" {fw}fill="{color}">{esc(ln)}</text>')

def arrow(x1, y1, x2, y2, marker="n", stroke="#64748b", sw=1.8, dash=False):
    d = ' stroke-dasharray="4,3"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
              f'stroke="{stroke}" stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')

# --- 第 0 级:三档候选表 ---
row0_y = TOP
HEADERS = [
    ("sm70,cap<75", "候选 = {1}"),
    ("sm8x,75<=cap<90", "候选 = {2}"),
    ("sm90,90<=cap<100", "候选 = {3,2}"),
]
for i, (cond, cand) in enumerate(HEADERS):
    cx = cols_x[i] + COL_W / 2
    box(cols_x[i], row0_y, COL_W, BOX_H, "#eef2ff", "#6366f1")
    multi_text(cx, row0_y + BOX_H / 2, [cond, cand], 13, "#312e81", bold=True)

# --- A/B 列:直接终局(v1/v2 恒过) ---
final_y_ab = row0_y + BOX_H + ROW_GAP
for i, label in [(0, "选中版本 = v1"), (1, "选中版本 = v2")]:
    cx = cols_x[i] + COL_W / 2
    box(cols_x[i], final_y_ab, COL_W, BOX_H, "#dcfce7", "#16a34a")
    multi_text(cx, final_y_ab + BOX_H / 2, [label, "f16 恒支持,无附加门"], 12.5, "#14532d")
    arrow(cx, row0_y + BOX_H, cx, final_y_ab, marker="g", stroke="#16a34a")

# --- D 列(cap>=100):尚不支持,独立列置于最右,不与 side box 同列 ---
cx_d = d_col_x + D_COL_W / 2
box(d_col_x, row0_y, D_COL_W, BOX_H, "#f1f5f9", "#94a3b8")
multi_text(cx_d, row0_y + BOX_H / 2, ["cap>=100(未来架构)", "本 pass 未覆盖"], 12, "#475569")
box(d_col_x, final_y_ab, D_COL_W, BOX_H, "#f1f5f9", "#94a3b8")
multi_text(cx_d, final_y_ab + BOX_H / 2, ["assert(false)", "编译期断言失败,非退档"], 11.5, "#475569")
arrow(cx_d, row0_y + BOX_H, cx_d, final_y_ab, marker="n", stroke="#94a3b8")

# --- C 列:v3 门 ---
cx_c = cols_x[2] + COL_W / 2
gate3_y = row0_y + BOX_H + ROW_GAP
box(cols_x[2], gate3_y, COL_W, BOX_H + 24, "#fff7ed", "#ea580c")
multi_text(cx_c, gate3_y + (BOX_H + 24) / 2 - 6,
           ["先试 v3 —— K>=256/位宽(f16 即 16)", "且 M%64=0, N%8=0, numWarps%4=0",
            "(f32/f32 还需 inputPrecision==TF32)"],
           11.3, "#7c2d12")
arrow(cx_c, row0_y + BOX_H, cx_c, gate3_y, marker="n", stroke="#64748b")

# v3 通过 → MMAv3(向右分支到独立终局框,避免与 v2 退档竖线重叠)
v3_ok_x = cols_x[2] + COL_W + SIDE_GAP
v3_ok_y = gate3_y
box(v3_ok_x, v3_ok_y, SIDE_W, BOX_H, "#dcfce7", "#16a34a")
multi_text(v3_ok_x + 85, v3_ok_y + BOX_H / 2, ["全过 -> MMAv3"], 13, "#14532d", bold=True)
L.append(f'<path d="M {cols_x[2]+COL_W} {gate3_y + (BOX_H+24)/2 - 6} '
          f'L {v3_ok_x} {v3_ok_y + BOX_H/2}" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#g)"/>')
L.append(f'<text x="{(cols_x[2]+COL_W+v3_ok_x)/2}" y="{gate3_y-6}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#16a34a">{esc("四门全过")}</text>')

# v3 不过 → 退到 v2 门
gate2_y = gate3_y + (BOX_H + 24) + ROW_GAP
box(cols_x[2], gate2_y, COL_W, BOX_H, "#fff7ed", "#ea580c")
multi_text(cx_c, gate2_y + BOX_H / 2, ["退到 v2 —— f32/f32 需 TF32 且 version>=2", "(f16 恒过)"], 11.5, "#7c2d12")
arrow(cx_c, gate3_y + BOX_H + 24, cx_c, gate2_y, marker="r", stroke="#dc2626")
L.append(f'<text x="{cx_c+16}" y="{gate3_y + BOX_H + 24 + ROW_GAP/2 + 4}" '
          f'font-family="sans-serif" font-size="11" fill="#dc2626">{esc("任一门不过")}</text>')

# v2 通过 → MMAv2(右侧终局)
v2_ok_x = v3_ok_x
v2_ok_y = gate2_y
box(v2_ok_x, v2_ok_y, SIDE_W, BOX_H, "#dcfce7", "#16a34a")
multi_text(v2_ok_x + 85, v2_ok_y + BOX_H / 2, ["过 -> MMAv2"], 13, "#14532d", bold=True)
L.append(f'<path d="M {cols_x[2]+COL_W} {gate2_y + BOX_H/2} '
          f'L {v2_ok_x} {v2_ok_y + BOX_H/2}" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#g)"/>')

# v2 不过 → chosen=0
final0_y = gate2_y + BOX_H + ROW_GAP
box(cols_x[2], final0_y, COL_W, BOX_H, "#fee2e2", "#dc2626")
multi_text(cx_c, final0_y + BOX_H / 2, ["chosen = 0 -> 退回 blocked/FMA", "(不进 Tensor Core)"], 12, "#7f1d1d", bold=True)
arrow(cx_c, gate2_y + BOX_H, cx_c, final0_y, marker="r", stroke="#dc2626")

L.append(f'<text x="{PAD}" y="{h-16}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("这条判定链是『matmul 没提速』的第一诊断点:K 太小、shape/warps 不整除、f32 没开 TF32,都会在这里悄悄退档。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-mma-version-gate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
