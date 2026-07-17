#!/usr/bin/env python3
"""fig-m8-prologue-staircase: prologue 是 2 级台阶——段 0 发 iter0 的取数、段 1
发 iter1 的取数,各由 trip>0、trip>1 谓词兜住,把环形缓冲预填到 2 槽。
数字来自 explainer.json m8.figure_specs[0].numbers:
  2 = prologue 段数 = maxStage
  0 = 段0 写槽 subview %42[0]
  1 = 段1 写槽 subview %42[1]
  6 = sm90_ns3.ttgir_async_copy = 6(2 段x2 + 稳态2)
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

STEPS = [
    ("段 0 — 守卫 trip > 0", [
        "守卫谓词: trip > 0 (%44 = cmpi sgt %39, 0)",
        "写入槽: 槽 0 (subview %42[0])",
        "发射: async_copy A、B -> 槽0 + commit",
    ]),
    ("段 1 — 守卫 trip > 1", [
        "守卫谓词: trip > 1 (%53 = cmpi sgt %39, 1)",
        "写入槽: 槽 1 (subview %42[1])",
        "发射: 先 addptr 推进指针,再 async_copy A、B -> 槽1 + commit",
    ]),
    ("稳态入口 iter_args 初值", [
        "insertIdx = 1(写游标从槽1起)",
        "extractIdx = -1(读游标预留2拍)",
        "prologue 已填槽 0、1,槽 2 留给稳态首拍写",
    ]),
]

BOX_W, BOX_H, VGAP = 560, 110, 44
PAD, TOP = 44, 108
BUF_W, BUF_GAP = 150, 60

body_h = len(STEPS) * (BOX_H + VGAP) - VGAP
w = PAD * 2 + BOX_W + BUF_GAP + BUF_W
h = TOP + body_h + 96

cx = PAD + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">Prologue 两级台阶:把环形缓冲预填到 2 槽</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">maxStage=2 => prologue 段数=2;num_stages=3,sm90(matmul_sm90_ns3.ttgir.mlir)</text>']

boxy = []
for i, (title, lines) in enumerate(STEPS):
    y = TOP + i * (BOX_H + VGAP)
    boxy.append(y)
    is_last = (i == len(STEPS) - 1)
    fill = "#eef2ff" if is_last else "#e0f2fe"
    stroke = "#6366f1" if is_last else "#0284c7"
    L.append(f'<rect x="{PAD}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{PAD+18}" y="{y+26}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for k, line in enumerate(lines):
        L.append(f'<text x="{PAD+18}" y="{y+50+k*20}" font-family="sans-serif" '
                  f'font-size="12" fill="#1e293b">{esc(line)}</text>')
    if i < len(STEPS) - 1:
        y2 = y + BOX_H
        L.append(f'<line x1="{cx}" y1="{y2}" x2="{cx}" y2="{y2+VGAP-6}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 右侧环形缓冲状态可视化:prologue 结束后槽0、槽1 已填,槽2 空
buf_x = PAD + BOX_W + BUF_GAP
buf_title_y = TOP - 40
L.append(f'<text x="{buf_x+BUF_W/2}" y="{buf_title_y}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#0f172a">Prologue 结束后</text>')
slot_h = 70
SLOTS = [("槽 0", "iter0", True), ("槽 1", "iter1", True), ("槽 2", "空(留给稳态)", False)]
for i, (name, tag, filled) in enumerate(SLOTS):
    y = TOP + i * (slot_h + 14)
    fill = "#dcfce7" if filled else "#f1f5f9"
    stroke = "#16a34a" if filled else "#94a3b8"
    L.append(f'<rect x="{buf_x}" y="{y}" width="{BUF_W}" height="{slot_h}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{buf_x+BUF_W/2}" y="{y+28}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    tcol = "#15803d" if filled else "#64748b"
    L.append(f'<text x="{buf_x+BUF_W/2}" y="{y+50}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="{tcol}">{esc(tag)}</text>')
    # 与对应 prologue 段用虚线连一下(仅前两槽)
    if i < 2:
        by = boxy[i] + BOX_H / 2
        L.append(f'<line x1="{PAD+BOX_W}" y1="{by}" x2="{buf_x}" y2="{y+slot_h/2}" '
                  'stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4,3"/>')

foot_y = TOP + body_h + 44
L.append(f'<line x1="{PAD}" y1="{foot_y-22}" x2="{w-PAD}" y2="{foot_y-22}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#374151">async_copy 总数 6 = prologue 2 段 x 2 个 load + 稳态 1 拍 x 2 个 load'
          f'(pipeline_dump_summary.json)。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m8-prologue-staircase.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
