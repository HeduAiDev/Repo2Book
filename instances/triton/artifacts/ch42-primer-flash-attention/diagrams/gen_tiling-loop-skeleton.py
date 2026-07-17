#!/usr/bin/env python3
"""figure: tiling-loop-skeleton (layout 模板改)
claim: 外层锁一块 Q(BLOCK_M 行)常驻 SRAM,内层 for start_n in range(lo,hi,BLOCK_N)
遍历 K/V 块逐块流过增量更新 acc;running 状态 m_i[BLOCK_M]/l_i[BLOCK_M]/
acc[BLOCK_M,HEAD_DIM] 全 O(block)、与 N 无关、全程无 N x N 落地。
数据来源: explainer/explainer.json mechanism m05-tiling-loop-skeleton
(python/tutorials/06-fused-attention.py)。全坐标计算,零手写魔数。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "分块遍历骨架:外层锁 Q、内层流过 K/V 块"
SUBTITLE = "任一时刻片上只有一块 BLOCK_M x BLOCK_N 的打分,running 状态与序列长 N 无关"

PAD, TOP = 40, 108
SRAM_W, SRAM_H = 260, 300
KV_W, KV_H = 190, 70
KV_GAP = 26
N_KV = 4

w = PAD * 2 + SRAM_W + 130 + KV_W + 60
kv_col_h = N_KV * (KV_H + KV_GAP) - KV_GAP
content_h = TOP + 26 + max(SRAM_H, kv_col_h)  # kv_top = TOP+26
h = content_h + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>',
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>',
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
     'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1e40af"/></marker>',
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# SRAM box: 外层锁定的 Q + running 状态,常驻不动
sram_x, sram_y = PAD, TOP
L.append(f'<rect x="{sram_x}" y="{sram_y}" width="{SRAM_W}" height="{SRAM_H}" rx="10" '
          'fill="#eff6ff" stroke="#1e40af" stroke-width="2.5" stroke-dasharray="0"/>')
L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+26}" text-anchor="middle" '
          'font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#1e3a5f">{esc("SRAM(片上,一个 program)")}</text>')

# Q block inside SRAM
q_x, q_y, q_w, q_h = sram_x + 20, sram_y + 44, SRAM_W - 40, 56
L.append(f'<rect x="{q_x}" y="{q_y}" width="{q_w}" height="{q_h}" rx="6" '
          'fill="#93c5fd" stroke="#1e40af" stroke-width="2"/>')
L.append(f'<text x="{q_x+q_w/2}" y="{q_y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#1e3a5f">{esc("q = tl.load(Q_block_ptr)")}</text>')
L.append(f'<text x="{q_x+q_w/2}" y="{q_y+40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#1e3a5f">{esc("[BLOCK_M 行] 常驻,内层反复读")}</text>')

# running state boxes
state_labels = [
    ("m_i", "[BLOCK_M]"),
    ("l_i", "[BLOCK_M]"),
    ("acc", "[BLOCK_M, HEAD_DIM]"),
]
st_top = q_y + q_h + 22
st_h = 40
for i, (name, shape) in enumerate(state_labels):
    sy = st_top + i * (st_h + 10)
    L.append(f'<rect x="{q_x}" y="{sy}" width="{q_w}" height="{st_h}" rx="6" '
              'fill="#dcfce7" stroke="#047857" stroke-width="1.5"/>')
    L.append(f'<text x="{q_x+14}" y="{sy+st_h/2+5}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#047857">{esc(name)}</text>')
    L.append(f'<text x="{q_x+q_w-14}" y="{sy+st_h/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" fill="#047857">{esc(shape)}</text>')

L.append(f'<text x="{sram_x+SRAM_W/2}" y="{sram_y+SRAM_H-14}" text-anchor="middle" '
         'font-family="sans-serif" font-size="11" '
         f'fill="#1e40af">{esc("running 状态全 O(block),与 N 无关")}</text>')

# KV conveyor: 4 blocks streaming through, only current one "on the table"
kv_x = sram_x + SRAM_W + 130
kv_top = TOP + 26
labels = [f"K{j}V{j}" for j in range(N_KV)]
for j, name in enumerate(labels):
    ky = kv_top + j * (KV_H + KV_GAP)
    is_current = (j == 1)  # 高亮"当前台面上"这一块
    fill = "#fde68a" if is_current else "#e2e8f0"
    stroke = "#d97706" if is_current else "#94a3b8"
    sw = 2.5 if is_current else 1.2
    dash = "" if is_current else 'stroke-dasharray="4,3"'
    L.append(f'<rect x="{kv_x}" y="{ky}" width="{KV_W}" height="{KV_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {dash}/>')
    L.append(f'<text x="{kv_x+KV_W/2}" y="{ky+KV_H/2-3}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{"#92400e" if is_current else "#475569"}">{esc(name)}</text>')
    label2 = "当前台面:BLOCK_M x BLOCK_N" if is_current else "等待/已流过"
    L.append(f'<text x="{kv_x+KV_W/2}" y="{ky+KV_H/2+15}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" '
              f'fill="{"#92400e" if is_current else "#94a3b8"}">{esc(label2)}</text>')
    # arrow: SRAM -> current KV block (qk = tl.dot(q,k)), else faint down-arrow for conveyor
    if is_current:
        L.append(f'<line x1="{sram_x+SRAM_W}" y1="{sram_y+SRAM_H/2}" x2="{kv_x}" y2="{ky+KV_H/2}" '
                  'stroke="#1e40af" stroke-width="2.5" marker-end="url(#b)"/>')
        L.append(f'<text x="{(sram_x+SRAM_W+kv_x)/2}" y="{(sram_y+SRAM_H/2+ky+KV_H/2)/2-10}" '
                  'text-anchor="middle" font-family="sans-serif" font-size="11" '
                  f'font-weight="bold" fill="#1e40af">{esc("qk=tl.dot(q,k)")}</text>')
    if j < len(labels) - 1:
        y1 = ky + KV_H
        y2 = kv_top + (j + 1) * (KV_H + KV_GAP)
        L.append(f'<line x1="{kv_x+KV_W/2}" y1="{y1}" x2="{kv_x+KV_W/2}" y2="{y2}" '
                  'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#a)"/>')

L.append(f'<text x="{kv_x+KV_W/2}" y="{kv_top-8}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#334155">{esc("K/V 沿序列方向逐块流过")}</text>')
loop_label_y = kv_top + kv_col_h + 30
L.append(f'<text x="{kv_x+KV_W/2}" y="{loop_label_y}" text-anchor="middle" '
         'font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc("for start_n in range(lo, hi, BLOCK_N)")}</text>')

foot_y = content_h + 44
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("黄底=当前台面上的打分块(用完即扔);绿底=running 状态,全程 O(block) 大小,不随 N 增长。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc("锚点 python/tutorials/06-fused-attention.py:L46(内层循环)、L164-165(q 常驻 SRAM)、L158-160(三件套 O(block))")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("tiling-loop-skeleton.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
