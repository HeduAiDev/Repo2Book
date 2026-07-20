#!/usr/bin/env python3
"""fig-ch32-mapping-expand: 行映射把「调度序的紧凑掩码行」按 cu_num_logits 区间展开成
「batch 序的 logits 行下标」,一个投机请求一次展开 1+k 个下标。
template: tensor-flow"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1300, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("行映射把紧凑掩码行按 cu_num_logits 展开成 batch 序的 logits 行下标")}</text>')
L.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("调度序 grammar_req_ids=[rA, rC];rA 带 2 个草稿占 3 行 logits")}</text>')

PAD = 60
COL_L_X = PAD
COL_M_X = W / 2 - 60
COL_R_X = W - PAD - 220
CELL_W = 220
CELL_H = 50
TOP = 110
GAP = 8

LOGITS_TOP_PRE = TOP - (CELL_H + GAP) / 2  # 与下方 LOGITS_TOP 计算一致,供表头定位
L.append(f'<text x="{COL_L_X+CELL_W/2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#334155">{esc("紧凑掩码行(调度序)")}</text>')
L.append(f'<text x="{COL_M_X+30}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#334155">{esc("mapping[k]")}</text>')
L.append(f'<text x="{COL_R_X+CELL_W/2}" y="{LOGITS_TOP_PRE-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#334155">{esc("logits 行(batch 序)")}</text>')

MASK_ROWS = [
    ("掩码行 0", "rA 位置 0", "#2563eb"),
    ("掩码行 1", "rA 位置 1", "#2563eb"),
    ("掩码行 2", "rA 位置 2", "#2563eb"),
    ("掩码行 3", "rC 位置 0", "#7c3aed"),
]
MAPPING = [1, 2, 3, 4]
LOGITS_ROWS = [
    ("logits 行 0", "rB(非结构化)", "#94a3b8", False),
    ("logits 行 1", "rA 位置 0", "#2563eb", True),
    ("logits 行 2", "rA 位置 1", "#2563eb", True),
    ("logits 行 3", "rA 位置 2", "#2563eb", True),
    ("logits 行 4", "rC 位置 0", "#7c3aed", True),
]

mask_y = {}
for i, (rlabel, sub, color) in enumerate(MASK_ROWS):
    y = TOP + i * (CELL_H + GAP)
    mask_y[i] = y + CELL_H / 2
    L.append(f'<rect x="{COL_L_X}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
              f'fill="white" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{COL_L_X+CELL_W/2}" y="{y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" font-weight="bold" fill="{color}">{esc(rlabel)}</text>')
    L.append(f'<text x="{COL_L_X+CELL_W/2}" y="{y+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(sub)}</text>')

logits_y = {}
LOGITS_TOP = TOP - (CELL_H + GAP) / 2
for j, (rlabel, sub, color, active) in enumerate(LOGITS_ROWS):
    y = LOGITS_TOP + j * (CELL_H + GAP)
    logits_y[j] = y + CELL_H / 2
    fill = "white" if active else "#f8fafc"
    L.append(f'<rect x="{COL_R_X}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="6" '
              f'fill="{fill}" stroke="{color}" stroke-width="{2 if active else 1.5}"/>')
    L.append(f'<text x="{COL_R_X+CELL_W/2}" y="{y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" font-weight="bold" fill="{color}">{esc(rlabel)}</text>')
    L.append(f'<text x="{COL_R_X+CELL_W/2}" y="{y+38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(sub)}</text>')

# mapping 中间小方块 + 箭头:掩码行 i -> mapping[i] -> logits 行 mapping[i]
for i, m in enumerate(MAPPING):
    my = mask_y[i]
    color = MASK_ROWS[i][2]
    mx = COL_M_X
    L.append(f'<line x1="{COL_L_X+CELL_W}" y1="{my}" x2="{mx}" y2="{my}" '
              f'stroke="{color}" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<rect x="{mx+6}" y="{my-14}" width="48" height="28" rx="5" '
              f'fill="#eef2ff" stroke="#6366f1"/>')
    L.append(f'<text x="{mx+30}" y="{my+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#3730a3">{esc(str(m))}</text>')
    ly = logits_y[m]
    L.append(f'<path d="M {mx+54} {my} C {mx+120} {my}, {COL_R_X-70} {ly}, {COL_R_X} {ly}" '
              f'fill="none" stroke="{color}" stroke-width="1.8" marker-end="url(#a)"/>')

FOOT_Y = LOGITS_TOP + len(LOGITS_ROWS) * (CELL_H + GAP) + 20
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="90" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+24}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("cu_num_logits = [0, 1, 4, 5];batch 顺序 = [rB, rA, rC](rA 占区间 [1,4),rC 占 [4,5))")}</text>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+46}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("对账:num_masks=4 == len(mapping)=4(assert 钉死),不等就在 kernel 启动前直接崩掉")}</text>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+68}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("logits 行 0(rB)未被任何掩码行映射到——96 个 token 全部有限,未被误伤")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-mapping-expand.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
