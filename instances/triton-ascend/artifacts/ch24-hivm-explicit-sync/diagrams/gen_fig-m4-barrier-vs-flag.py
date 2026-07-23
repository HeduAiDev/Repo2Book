#!/usr/bin/env python3
"""fig-m4-barrier-vs-flag: 同引擎依赖插 1 个 pipe_barrier;异引擎依赖插 1 对
set_flag/wait_flag。before-after 模板:左=裸 IR(3 条计算 op),右=注入同步后
(4 个 flag + 1 个收尾 barrier);右下角小面板对照 if_else 例的同引擎 pipe_barrier。
取自 inject-sync.mlir @test_mem_injcet_sync_basic。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "决策二分:同引擎插 pipe_barrier,异引擎插 set_flag/wait_flag 一对"
SUBTITLE = "inject-sync.mlir @test_mem_injcet_sync_basic:3 条裸 op → 注入 4 个 flag + 1 个收尾 barrier"

BEFORE = ["load(MTE2) 写 %0", "vadd(V) 读/写 %0", "store(MTE3) 读 %0"]
AFTER = [
    ("load(MTE2) 写 %0", None),
    ("set_flag[<PIPE_MTE2>,<PIPE_V>,<EVENT_ID0>]", "flag"),
    ("wait_flag[<PIPE_MTE2>,<PIPE_V>,<EVENT_ID0>]", "flag"),
    ("vadd(V) 读/写 %0", None),
    ("set_flag[<PIPE_V>,<PIPE_MTE3>,<EVENT_ID0>]", "flag"),
    ("wait_flag[<PIPE_V>,<PIPE_MTE3>,<EVENT_ID0>]", "flag"),
    ("store(MTE3) 读 %0", None),
    ("pipe_barrier[<PIPE_ALL>]", "barrier"),
]

BOX_W, BOX_H, VGAP, PAD, TOP = 260, 40, 18, 44, 130
AFTER_BOX_W = 350
PANEL_GAP = 90
left_x = PAD
left_cx = left_x + BOX_W / 2
right_x = left_x + BOX_W + PANEL_GAP
right_cx = right_x + AFTER_BOX_W / 2

h_before = TOP + len(BEFORE) * (BOX_H + VGAP)
h_after = TOP + len(AFTER) * (BOX_H + VGAP)
h_main = max(h_before, h_after) + 10

# 右下小面板:if_else 同引擎对照
SIDE_TOP = h_main + 50
SIDE_STEPS = ["load(MTE2)", "load(MTE2,相邻)"]
side_w = 300
side_x = right_x + AFTER_BOX_W - side_w
side_cx = side_x + side_w / 2

h = SIDE_TOP + 40 + 2 * (BOX_H + VGAP) + PAD + 50
w = right_x + AFTER_BOX_W + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

L.append(f'<text x="{left_cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">裸 IR(无同步)</text>')
for i, label in enumerate(BEFORE):
    y = TOP + i * (BOX_H + VGAP)
    L.append(f'<rect x="{left_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
    L.append(f'<text x="{left_cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#0f172a">{esc(label)}</text>')
    if i < len(BEFORE) - 1:
        L.append(f'<line x1="{left_cx}" y1="{y+BOX_H}" x2="{left_cx}" y2="{y+BOX_H+VGAP-4}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

L.append(f'<text x="{right_cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">注入同步后</text>')
for i, (label, kind) in enumerate(AFTER):
    y = TOP + i * (BOX_H + VGAP)
    if kind == "flag":
        fill, stroke, tf = "#fee2e2", "#b91c1c", "#7f1d1d"
    elif kind == "barrier":
        fill, stroke, tf = "#fef3c7", "#d97706", "#78350f"
    else:
        fill, stroke, tf = "#e2e8f0", "#64748b", "#0f172a"
    L.append(f'<rect x="{right_x}" y="{y}" width="{AFTER_BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if kind else 1.5}"/>')
    L.append(f'<text x="{right_cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{tf}">{esc(label)}</text>')
    if i < len(AFTER) - 1:
        mk = "url(#b)" if kind == "barrier" else "url(#a)"
        col = "#d97706" if kind == "barrier" else "#64748b"
        L.append(f'<line x1="{right_cx}" y1="{y+BOX_H}" x2="{right_cx}" y2="{y+BOX_H+VGAP-4}" '
                  f'stroke="{col}" stroke-width="1.5" marker-end="{mk}"/>')

# 中间对照箭头
mid_y = (TOP + h_before) / 2 - BOX_H
L.append(f'<line x1="{left_x+BOX_W+8}" y1="{mid_y}" x2="{right_x-8}" y2="{mid_y}" '
          'stroke="#1e40af" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(left_x+BOX_W+right_x)/2}" y="{mid_y-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#1e40af">插入同步</text>')

# 右下小面板:if_else 同引擎对照
L.append(f'<text x="{side_cx}" y="{SIDE_TOP}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">对照(if_else 例):同引擎相邻 load</text>')
side_top2 = SIDE_TOP + 20
side_steps_full = [("load(MTE2)", None), ("pipe_barrier[<PIPE_MTE2>]", "barrier"), ("load(MTE2,相邻)", None)]
for i, (label, kind) in enumerate(side_steps_full):
    y = side_top2 + i * (BOX_H + VGAP)
    fill, stroke, tf = ("#fef3c7", "#d97706", "#78350f") if kind == "barrier" else ("#e2e8f0", "#64748b", "#0f172a")
    L.append(f'<rect x="{side_x}" y="{y}" width="{side_w}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if kind else 1.5}"/>')
    L.append(f'<text x="{side_cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="{tf}">{esc(label)}</text>')
    if i < len(side_steps_full) - 1:
        mk = "url(#b)" if kind == "barrier" else "url(#a)"
        col = "#d97706" if kind == "barrier" else "#64748b"
        L.append(f'<line x1="{side_cx}" y1="{y+BOX_H}" x2="{side_cx}" y2="{y+BOX_H+VGAP-4}" '
                  f'stroke="{col}" stroke-width="1.5" marker-end="{mk}"/>')

foot_y = h - 36
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">异 pipe 边 2 对(MTE2→V、V→MTE3)= 4 个 flag op;'
          f'收尾 pipe_barrier[&lt;PIPE_ALL&gt;] 1 个</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">对照:if_else 例同引擎相邻 load 只插 1 个 pipe_barrier(无 flag)</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m4-barrier-vs-flag.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
