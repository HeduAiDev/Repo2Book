#!/usr/bin/env python3
"""swimlane 模板(单泳道时序变体):同一 buffer b 的 op1..op5 读写序列,
在 RAW/WAR 相邻边界处插 gpu.barrier,RAR 处不插。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Membar 插入序列 — buffer b 分配区间 [0, 512)"
SUBTITLE = "同一地址区间上的读写流:每跨越一次写<->读边界(RAW/WAR)落一道 gpu.barrier;连续读(RAR)不插"

OPS = [
    ("op1", "local_store", "W", "账本空,不插", None),
    ("op2", "local_load", "R", "RAW 命中", "barrier #1"),
    ("op3", "local_store", "W", "WAR 命中", "barrier #2"),
    ("op4", "local_load", "R", "RAW 命中", "barrier #3"),
    ("op5", "local_load", "R", "RAR,不插", None),
]
TOTAL_BARRIERS = 3

BOX_W, BOX_H = 128, 64
GAP = 96
PAD, TOP = 46, 118
n = len(OPS)
w = PAD * 2 + BOX_W * n + GAP * (n - 1)
h = TOP + BOX_H + 190

op_x = [PAD + i * (BOX_W + GAP) for i in range(n)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-16}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+6}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# op boxes
for i, (name, kind, rw, note, barrier_before_next) in enumerate(OPS):
    x = op_x[i]
    fill, stroke = ("#dbeafe", "#1e40af") if rw == "R" else ("#fee2e2", "#b91c1c")
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+42}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(kind + " (" + rw + ")")}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+58}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#64748b">{esc(note)}</text>')

# connectors between ops, with barrier gates on the ones that fire
for i in range(n - 1):
    x1 = op_x[i] + BOX_W
    x2 = op_x[i + 1]
    y = TOP + BOX_H / 2
    _, _, _, _, barrier_label = OPS[i]
    if barrier_label:
        # draw a red gate icon at midpoint
        mx = (x1 + x2) / 2
        L.append(f'<line x1="{x1}" y1="{y}" x2="{mx-14}" y2="{y}" stroke="#334155" stroke-width="1.5"/>')
        L.append(f'<line x1="{mx+14}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" stroke-width="1.5" '
                  'marker-end="url(#a)"/>')
        L.append(f'<rect x="{mx-13}" y="{y-22}" width="26" height="44" rx="4" '
                  'fill="#fecaca" stroke="#b91c1c" stroke-width="2"/>')
        L.append(f'<text x="{mx}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="16" font-weight="bold" fill="#7f1d1d">B</text>')
        L.append(f'<text x="{mx}" y="{y-30}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" font-weight="bold" fill="#b91c1c">{esc(barrier_label)}</text>')
    else:
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#94a3b8" '
                  'stroke-width="1.5" stroke-dasharray="4,4" marker-end="url(#a)"/>')
        mx = (x1 + x2) / 2
        L.append(f'<text x="{mx}" y="{y-14}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="#64748b">{esc("RAR: 不插")}</text>')

# buffer interval bar at bottom
bar_y = TOP + BOX_H + 66
bar_x1, bar_x2 = op_x[0], op_x[-1] + BOX_W
L.append(f'<rect x="{bar_x1}" y="{bar_y}" width="{bar_x2-bar_x1}" height="24" rx="4" '
          'fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>')
L.append(f'<text x="{(bar_x1+bar_x2)/2}" y="{bar_y+16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#4c1d95">{esc("buffer b 分配区间 [0, 512) — 全序列共享同一地址")}</text>')

# legend
leg_y = bar_y + 56
L.append(f'<rect x="{PAD}" y="{leg_y}" width="16" height="16" rx="3" fill="#dbeafe" stroke="#1e40af"/>')
L.append(f'<text x="{PAD+24}" y="{leg_y+13}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("读(R)")}</text>')
L.append(f'<rect x="{PAD+90}" y="{leg_y}" width="16" height="16" rx="3" fill="#fee2e2" stroke="#b91c1c"/>')
L.append(f'<text x="{PAD+114}" y="{leg_y+13}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("写(W)")}</text>')
L.append(f'<rect x="{PAD+180}" y="{leg_y}" width="16" height="16" rx="3" fill="#fecaca" stroke="#b91c1c" stroke-width="2"/>')
L.append(f'<text x="{PAD+204}" y="{leg_y+13}" font-family="sans-serif" font-size="11.5" fill="#334155">{esc("gpu.barrier")}</text>')

foot_y = leg_y + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">{esc(f"barrier 总数 = {TOTAL_BARRIERS}(op2 前 RAW、op3 前 WAR、op4 前 RAW);op1 无前序、op5 为 RAR 均不插")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-membar-sequence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
