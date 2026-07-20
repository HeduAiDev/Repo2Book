#!/usr/bin/env python3
"""fig-ch32-row-order-and-budget: 掩码是一张「调度顺序的行 x ceil(|V|/32) 列」的紧凑表,
某一行归谁由随行同传的 req_id 列表决定,不是由行号与 batch 顺序的巧合决定。
template: layout"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1200, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("掩码行归属由 req_id 决定,不由行号决定")}</text>')

PAD = 50
TOP = 70

# --- 预分配缓冲:上界 4 行,本步裁到 2 行 ---
L.append(f'<text x="{PAD}" y="{TOP}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#1e293b">{esc("① 预分配缓冲:max_num_seqs x (1 + num_spec) = 4 x 1 = 4 行上界,本步裁到 2 行")}</text>')
BUF_Y = TOP + 26
CELL_W, CELL_H, GAP = 110, 56, 10
BUF_LABELS = ["rA", "rD", "(未用)", "(未用)"]
BUF_USED = [True, True, False, False]
for i, (lab, used) in enumerate(zip(BUF_LABELS, BUF_USED)):
    x = PAD + i * (CELL_W + GAP)
    fill = "#dbeafe" if used else "#f1f5f9"
    stroke = "#2563eb" if used else "#94a3b8"
    L.append(f'<rect x="{x}" y="{BUF_Y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if used else 1}"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{BUF_Y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(f"缓冲行 {i}")}</text>')
    L.append(f'<text x="{x+CELL_W/2}" y="{BUF_Y+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{"#1d4ed8" if used else "#94a3b8"}">{esc(lab)}</text>')
L.append(f'<text x="{PAD + 2*(CELL_W+GAP)}" y="{BUF_Y+CELL_H+22}" font-family="sans-serif" '
          f'font-size="11.5" fill="#64748b">{esc("裁到 cumulative_index=2 后返回,后 2 行不进 GrammarOutput")}</text>')

# --- 调度序装配(左)与 worker 端 batch 顺序(右),按 id 交叉映射 ---
MID_Y = BUF_Y + CELL_H + 70
L.append(f'<text x="{PAD}" y="{MID_Y}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
          f'fill="#1e293b">{esc("② 两侧顺序刻意不同,靠随行同传的 req_id 列表(而非行号)对号入座")}</text>')

ROW_Y0 = MID_Y + 30
ROW_H = 58
LEFT_X = PAD
RIGHT_X = W - PAD - CELL_W

SCHED = ["rA", "rD"]          # 调度序 = 掩码行序
BATCH = ["rD", "rB", "rA", "rC"]  # worker batch 顺序

L.append(f'<text x="{LEFT_X+CELL_W/2}" y="{ROW_Y0-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#334155">{esc("掩码行(调度序)")}</text>')
L.append(f'<text x="{RIGHT_X+CELL_W/2}" y="{ROW_Y0-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#334155">{esc("logits 行(batch 顺序)")}</text>')

ID_COLOR = {"rA": "#2563eb", "rD": "#059669", "rB": "#94a3b8", "rC": "#94a3b8"}
sched_y = {}
for i, rid in enumerate(SCHED):
    y = ROW_Y0 + i * ROW_H
    sched_y[rid] = y + CELL_H / 2
    L.append(f'<rect x="{LEFT_X}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              f'fill="white" stroke="{ID_COLOR[rid]}" stroke-width="2.5"/>')
    L.append(f'<text x="{LEFT_X+CELL_W/2}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(f"掩码行 {i}")}</text>')
    L.append(f'<text x="{LEFT_X+CELL_W/2}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="{ID_COLOR[rid]}">{esc(rid)}</text>')

batch_y = {}
for j, rid in enumerate(BATCH):
    y = ROW_Y0 + j * ROW_H
    batch_y[rid] = y + CELL_H / 2
    is_target = rid in ("rA", "rD")
    fill = "white" if is_target else "#f8fafc"
    stroke = ID_COLOR[rid] if is_target else "#cbd5e1"
    L.append(f'<rect x="{RIGHT_X}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.5 if is_target else 1.5}"/>')
    L.append(f'<text x="{RIGHT_X+CELL_W/2}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(f"logits 行 {j}")}</text>')
    L.append(f'<text x="{RIGHT_X+CELL_W/2}" y="{y+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" '
              f'fill="{ID_COLOR[rid] if is_target else "#94a3b8"}">{esc(rid)}</text>')

# 交叉箭头:掩码行(rA=0, rD=1) -> 对应 batch 位置(rA 在 batch idx2, rD 在 batch idx0)
for rid in SCHED:
    x1 = LEFT_X + CELL_W
    y1 = sched_y[rid]
    x2 = RIGHT_X
    y2 = batch_y[rid]
    L.append(f'<path d="M {x1} {y1} C {x1+90} {y1}, {x2-90} {y2}, {x2} {y2}" '
              f'fill="none" stroke="{ID_COLOR[rid]}" stroke-width="2.2" marker-end="url(#a)"/>')
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    L.append(f'<rect x="{mx-46}" y="{my-11}" width="92" height="22" rx="5" '
              f'fill="white" stroke="{ID_COLOR[rid]}" stroke-width="1"/>')
    L.append(f'<text x="{mx}" y="{my+4}" text-anchor="middle" font-family="sans-serif" font-size="10.5" '
              f'fill="{ID_COLOR[rid]}">{esc(f"req_id={rid}")}</text>')

note_y = ROW_Y0 + len(BATCH) * ROW_H + 6
L.append(f'<rect x="{PAD}" y="{note_y}" width="{W-2*PAD}" height="46" rx="8" '
          f'fill="#fef2f2" stroke="#dc2626" stroke-dasharray="5 3"/>')
L.append(f'<text x="{W/2}" y="{note_y+20}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#7f1d1d">{esc("若按行号硬对齐(掩码行 0 -> logits 行 0),rA 的许可会错发给 rD——不报错,静默生成错的 JSON")}</text>')
L.append(f'<text x="{W/2}" y="{note_y+38}" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#7f1d1d">{esc("真实映射:掩码行 0(rA) -> logits 行 2;掩码行 1(rD) -> logits 行 0")}</text>')

# --- 列数说明 ---
COL_Y = note_y + 70
L.append(f'<rect x="{PAD}" y="{COL_Y}" width="{W-2*PAD}" height="40" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{COL_Y+25}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("列数 = ceil(|V|/32):本例 |V|=96 -> 3 个 int32;真实 |V|=152064 时为 4752 列 = 18.5625 KiB/行")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-row-order-and-budget.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
