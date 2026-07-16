#!/usr/bin/env python3
"""loop-carried-scope-diff: 上半两趟流程条(swimlane 风)+ 下半韦恩图。
Triton 用 dry-run 收集 local_defs、再取 local_defs∩liveins 认 loop-carried——
scope 差集，不是 Cytron 的 φ 放置算法。全部坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path
import math

def esc(s):
    return xs.escape(s)

W = 1220
PAD = 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 1"></svg>']  # 占位，最后重算 h

# ---------- 标题 ----------
title = "loop-carried = local_defs ∩ liveins：dry-run + scope 差集，不是 Cytron φ 放置算法"
subtitle = "|liveins|=5（acc,m,N,k,…）、|local_defs|=3（acc,m,tmp）、|交集|=2 → init_args 长度=2"

# ---------- 第一部分：两趟流程条 ----------
PASS_BOX_W, PASS_BOX_H = 210, 64
PASS_GAP = 46
PASS_TOP1 = 100
PASS_ROW_GAP = 100
LANE_LABEL_W = 90

pass1 = [
    ("dry-run 循环体", "跑一遍循环体"),
    ("填 local_defs", "set_value 填充"),
    ("block.erase 擦除", "L971：丢弃试跑结果"),
]
pass2 = [
    ("取交集", "local_defs ∩ liveins"),
    ("create_for_op", "init_args，L990"),
    ("绑块参数 arg(i+1)", "L1002"),
]

pass_start_x = PAD + LANE_LABEL_W
n_boxes = len(pass1)
row_w = n_boxes * PASS_BOX_W + (n_boxes - 1) * PASS_GAP

def flow_row(y, lane_label, boxes, fill, stroke, note):
    out = []
    out.append(f'<text x="{PAD}" y="{y+PASS_BOX_H/2+5}" font-family="sans-serif" '
                f'font-size="13" font-weight="bold" fill="#0f172a">{esc(lane_label)}</text>')
    for i, (title_t, sub_t) in enumerate(boxes):
        x = pass_start_x + i * (PASS_BOX_W + PASS_GAP)
        out.append(f'<rect x="{x}" y="{y}" width="{PASS_BOX_W}" height="{PASS_BOX_H}" rx="9" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
        out.append(f'<text x="{x+PASS_BOX_W/2}" y="{y+27}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                    f'fill="#0f172a">{esc(title_t)}</text>')
        out.append(f'<text x="{x+PASS_BOX_W/2}" y="{y+46}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="#475569">{esc(sub_t)}</text>')
        if i < len(boxes) - 1:
            x2 = x + PASS_BOX_W
            out.append(f'<line x1="{x2}" y1="{y+PASS_BOX_H/2}" x2="{x2+PASS_GAP-4}" '
                        f'y2="{y+PASS_BOX_H/2}" stroke="#334155" stroke-width="1.6" '
                        'marker-end="url(#a)"/>')
    out.append(f'<text x="{pass_start_x+row_w+16}" y="{y+PASS_BOX_H/2+5}" '
                f'font-family="sans-serif" font-size="11" fill="#64748b">{esc(note)}</text>')
    return '\n'.join(out)

row1_y = PASS_TOP1
row2_y = PASS_TOP1 + PASS_BOX_H + PASS_ROW_GAP

part1 = []
part1.append(flow_row(row1_y, "第一趟", pass1, '#dbeafe', '#2563eb',
                       "唯一目的：探循环体定义了哪些名字"))
part1.append(flow_row(row2_y, "第二趟", pass2, '#fef3c7', '#d97706', "认出 loop-carried"))
# 竖向衔接箭头：第一趟末框 -> 第二趟首框
last1_x = pass_start_x + (len(pass1) - 1) * (PASS_BOX_W + PASS_GAP) + PASS_BOX_W / 2
first2_x = pass_start_x + PASS_BOX_W / 2
part1.append(f'<path d="M {last1_x},{row1_y+PASS_BOX_H} '
              f'C {last1_x},{row1_y+PASS_BOX_H+40} {first2_x},{row2_y-40} {first2_x},{row2_y}" '
              'fill="none" stroke="#7c3aed" stroke-width="1.6" stroke-dasharray="5,3" '
              'marker-end="url(#a)"/>')

# ---------- 第二部分：韦恩图 ----------
VENN_TOP = row2_y + PASS_BOX_H + 90
R = 130
CX_L = W / 2 - 90
CX_R = W / 2 + 90
CY = VENN_TOP + R

venn = []
venn.append(f'<text x="{W/2}" y="{VENN_TOP-52}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">'
             f'{esc("两张点名册对照：只出现在两册中的名字才 loop-carried")}</text>')
venn.append(f'<text x="{CX_L-R+10}" y="{VENN_TOP-16}" text-anchor="start" '
             f'font-family="sans-serif" font-size="12" font-weight="bold" '
             f'fill="#2563eb">{esc("liveins（循环前已在场）")}</text>')
venn.append(f'<text x="{CX_R+R-10}" y="{VENN_TOP-16}" text-anchor="end" '
             f'font-family="sans-serif" font-size="12" font-weight="bold" '
             f'fill="#16a34a">{esc("local_defs（循环体内点过名）")}</text>')

venn.append(f'<circle cx="{CX_L}" cy="{CY}" r="{R}" fill="#dbeafe" fill-opacity="0.55" '
             'stroke="#2563eb" stroke-width="1.8"/>')
venn.append(f'<circle cx="{CX_R}" cy="{CY}" r="{R}" fill="#bbf7d0" fill-opacity="0.55" '
             'stroke="#16a34a" stroke-width="1.8"/>')

# 左新月标签 {N, k}
venn.append(f'<text x="{CX_L-R*0.55}" y="{CY-10}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#1d4ed8">{esc("N, k")}</text>')
venn.append(f'<text x="{CX_L-R*0.55}" y="{CY+12}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#1e3a8a">{esc("只读/归纳")}</text>')
venn.append(f'<text x="{CX_L-R*0.55}" y="{CY+26}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#1e3a8a">{esc("不 carry")}</text>')

# 重叠区 {acc, m}
venn.append(f'<text x="{(CX_L+CX_R)/2}" y="{CY-10}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#7c2d12">{esc("acc, m")}</text>')
venn.append(f'<text x="{(CX_L+CX_R)/2}" y="{CY+12}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" font-weight="bold" fill="#9a3412">{esc("loop-carried")}</text>')
venn.append(f'<text x="{(CX_L+CX_R)/2}" y="{CY+26}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" font-weight="bold" fill="#9a3412">{esc("→ iter_arg")}</text>')

# 右新月标签 {tmp}
venn.append(f'<text x="{CX_R+R*0.55}" y="{CY-10}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#15803d">{esc("tmp")}</text>')
venn.append(f'<text x="{CX_R+R*0.55}" y="{CY+12}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#14532d">{esc("纯临时")}</text>')
venn.append(f'<text x="{CX_R+R*0.55}" y="{CY+26}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#14532d">{esc("不 carry")}</text>')

# 外侧省略号，表示 liveins 还有未列出的名字（|liveins|≈5，本图只标出 4 个 + …）
venn.append(f'<text x="{CX_L-R*1.05}" y="{CY+R*0.55}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#64748b">{esc("…")}</text>')

VENN_H = 2 * R + 70

# ---------- 底部数字小结 + 红线note ----------
sum_y = VENN_TOP + VENN_H
summary = "|交集|=2 loop-carried（acc,m）→ create_for_op(init_args) 长度=2 → 块参数 arg(1),arg(2)"

note_lines = [
    "红线：",
    "全过程 = 一次 dry-run（跑一遍循环体）",
    "+ 一次集合交，零支配边界计算——",
    "因为 Python for/if 边界天然已知，",
    "Triton 不需要 Cytron 算法在一般 CFG 上",
    "求最小 φ 插入点。",
]
NOTE_H = 26 + len(note_lines) * 20 + 14
note_y = sum_y + 30

H = note_y + NOTE_H + PAD

# ---------- 组装 ----------
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15" '
          f'font-weight="bold" fill="#0f172a">{esc(title)}</text>')
L.append(f'<text x="{PAD}" y="46" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc(subtitle)}</text>')
L.extend(part1)
L.extend(venn)
L.append(f'<text x="{W/2}" y="{sum_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" fill="#0f172a">{esc(summary)}</text>')
L.append(f'<rect x="{PAD}" y="{note_y}" width="{W-2*PAD}" height="{NOTE_H}" rx="10" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5,3"/>')
for i, t in enumerate(note_lines):
    fw = 'bold' if i == 0 else 'normal'
    L.append(f'<text x="{PAD+18}" y="{note_y+26+i*20}" font-family="sans-serif" '
              f'font-size="12" font-weight="{fw}" fill="#7f1d1d">{esc(t)}</text>')
L.append('</svg>')

out = Path(__file__).with_name('loop-carried-scope-diff.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
