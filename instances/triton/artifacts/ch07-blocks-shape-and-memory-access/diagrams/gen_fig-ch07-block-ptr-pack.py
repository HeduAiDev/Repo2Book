#!/usr/bin/env python3
"""layout 模板(两阶段版):make_block_ptr 只打包 shape/strides/offsets/
block_shape/order 五组元信息,落进一个 tt.make_tensor_ptr 节点;boundary_check/
padding 是随后 tl.load(ptr, boundary_check=…, padding=…) 单独传入的参数,落进
另一个独立的 tt.load 节点——两次调用、两个 IR 节点,不是一次打包。
左侧仍画父张量(20x20)+ 本 block(16x16,offset=16,16)在边界外溢出的部分,
溢出区用 padding 语义色,标注挂在阶段②(load 才决定边界怎么处理)附近。
数字全部来自 dossier m7-block-pointer(M=N=20, BM=BN=16, pid_m=pid_n=1)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "make_block_ptr 只打包 5 组元信息；边界留给随后的 tl.load"
SUBTITLE = "M=N=20, BLOCK=16×16, offsets=(16,16)（pid_m=pid_n=1）→ 尾块界内 4×4=16 / 共 256"

PAD, TOP = 40, 118
UNIT = 9  # px per tensor unit
M = N = 20
BM = BN = 16
OFF = 16

parent_w = parent_h = M * UNIT
px0, py0 = PAD + 30, TOP

block_w = block_h = BM * UNIT
bx0 = px0 + OFF * UNIT
by0 = py0 + OFF * UNIT

in_bound_w = (M - OFF) * UNIT  # 4*UNIT
in_bound_h = (N - OFF) * UNIT

# 左侧绘图区实际最底边 = max(父张量底边, block 溢出底边)
diagram_bottom = max(py0 + parent_h, by0 + block_h)

CARD_X = px0 + parent_w + 150
CARD_W = 480

w = CARD_X + CARD_W + PAD

# ---- 先算右侧两阶段卡片的几何,再定总高 ----
FIELDS1 = [
    ("shape", "(M, N) = (20, 20)"),
    ("strides", "(N, 1) = (20, 1)"),
    ("offsets", "(pid_m·BM, pid_n·BN) = (16, 16)"),
    ("block_shape", "(BM, BN) = (16, 16)"),
    ("order", "(1, 0)  行主序"),
]
FIELDS2 = [
    ("boundary_check", "(0, 1)  两维都检查"),
    ("padding", "'zero' → IR padding=1"),
]

card1_y = TOP - 10
card1_h = 34 + len(FIELDS1) * 34 + 54
CONNECT_GAP = 74  # 两阶段之间留给弱连接箭头+说明文字
card2_y = card1_y + card1_h + CONNECT_GAP
card2_h = 34 + len(FIELDS2) * 34 + 54

legend_y = diagram_bottom + 46
FOOT = [
    "结论:①一个 tt.make_tensor_ptr 只携带 shape/strides/offsets/block_shape/order 五组元信息,",
    "advance 只挪 offsets、其余原样不变;②边界由随后的 tl.load(boundary_check, padding) 单独",
    "传入,落进另一个 tt.load 节点——两次调用、两个 IR 节点,不是一次打包。",
]

diagram_side_bottom = legend_y + 22 + 14 + 20  # 图例最后一行底边 + 余量
card_side_bottom = card2_y + card2_h
content_bottom = max(diagram_side_bottom, card_side_bottom)
foot_y0 = content_bottom + 40
h = foot_y0 + len(FOOT) * 19 + 14

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     '<defs><marker id="arrow" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
     'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
     '<marker id="arrow-weak" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
     'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12.5" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# ---- 左侧:父张量 + 越界 block ----
L.append(f'<text x="{px0}" y="{py0-10}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">{esc("父张")}'
          f'<tspan font-weight="normal">{esc("量")}</tspan>'
          f'{esc(" shape=(20,20)")}</text>')
L.append(f'<rect x="{px0}" y="{py0}" width="{parent_w}" height="{parent_h}" '
          'fill="#f8fafc" stroke="#0f172a" stroke-width="2"/>')
for k in range(1, 5):
    gx = px0 + k * 4 * UNIT
    gy = py0 + k * 4 * UNIT
    L.append(f'<line x1="{gx}" y1="{py0}" x2="{gx}" y2="{py0+parent_h}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<line x1="{px0}" y1="{gy}" x2="{px0+parent_w}" y2="{gy}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

hatch_id = "hatch"
L.append(f'<defs><pattern id="{hatch_id}" width="8" height="8" patternTransform="rotate(45)" '
          'patternUnits="userSpaceOnUse"><line x1="0" y1="0" x2="0" y2="8" '
          'stroke="#fbbf24" stroke-width="4"/></pattern></defs>')
L.append(f'<rect x="{bx0}" y="{by0}" width="{block_w}" height="{block_h}" '
          f'fill="url(#{hatch_id})" fill-opacity="0.55" stroke="#b45309" '
          'stroke-width="2" stroke-dasharray="5,4"/>')

L.append(f'<rect x="{bx0}" y="{by0}" width="{in_bound_w}" height="{in_bound_h}" '
          'fill="#93c5fd" fill-opacity="0.85" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{bx0+in_bound_w/2}" y="{by0+in_bound_h/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#1e3a8a">{esc("4×4 界内")}</text>')
L.append(f'<text x="{bx0+block_w-4}" y="{by0-8}" text-anchor="end" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#92400e">{esc("越界区→②load 时 padding=0")}</text>')

for val in [0, OFF, M]:
    tx = px0 + val * UNIT
    L.append(f'<line x1="{tx}" y1="{py0+parent_h}" x2="{tx}" y2="{py0+parent_h+6}" '
              'stroke="#64748b" stroke-width="1"/>')
    L.append(f'<text x="{tx}" y="{py0+parent_h+20}" text-anchor="middle" '
              f'font-family="monospace" font-size="10.5" fill="#475569">{val}</text>')
for val in [0, OFF, N]:
    ty = py0 + val * UNIT
    L.append(f'<text x="{px0-8}" y="{ty+4}" text-anchor="end" '
              f'font-family="monospace" font-size="10.5" fill="#475569">{val}</text>')

L.append(f'<rect x="{px0}" y="{legend_y}" width="14" height="14" rx="3" '
          'fill="#93c5fd" stroke="#1d4ed8"/>')
L.append(f'<text x="{px0+20}" y="{legend_y+11}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("界内(真实读到的 16 个元素)")}</text>')
L.append(f'<rect x="{px0}" y="{legend_y+22}" width="14" height="14" rx="3" '
          f'fill="url(#{hatch_id})" fill-opacity="0.7" stroke="#b45309"/>')
L.append(f'<text x="{px0+20}" y="{legend_y+33}" font-family="sans-serif" font-size="11" '
          f'fill="#334155">{esc("越界(240 个,②load 时按 padding 补 0)")}</text>')

# ---- 右侧阶段① make_block_ptr:五组元信息 → tt.make_tensor_ptr ----
def draw_stage_card(y, h_card, badge, header, fields, ir_lines, header_color, badge_color):
    L.append(f'<rect x="{CARD_X}" y="{y}" width="{CARD_W}" height="{h_card}" '
              f'rx="10" fill="#eef2ff" stroke="{header_color}" stroke-width="1.6"/>')
    L.append(f'<circle cx="{CARD_X+16}" cy="{y+16}" r="11" fill="{badge_color}"/>')
    L.append(f'<text x="{CARD_X+16}" y="{y+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="white">{esc(badge)}</text>')
    L.append(f'<text x="{CARD_X+34}" y="{y+20}" font-family="monospace" font-size="13" '
              f'font-weight="bold" fill="{header_color}">{esc(header)}</text>')
    for i, (k, v) in enumerate(fields):
        fy = y + 44 + i * 34
        L.append(f'<text x="{CARD_X+16}" y="{fy}" font-family="monospace" font-size="12.5" '
                  f'font-weight="bold" fill="#4338ca">{esc(k)}</text>')
        L.append(f'<text x="{CARD_X+16}" y="{fy+16}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#312e81">{esc(v)}</text>')
    ir_y = y + 34 + len(fields) * 34 + 20
    for i, line in enumerate(ir_lines):
        L.append(f'<text x="{CARD_X+16}" y="{ir_y+i*16}" font-family="monospace" font-size="11" '
                  f'fill="#78350f">{esc(line)}</text>')

draw_stage_card(card1_y, card1_h, "①", "make_block_ptr(...)", FIELDS1,
                 ["→ tt.make_tensor_ptr ...", ": <tensor<16x16xf32>>"],
                 "#3730a3", "#4338ca")

# 弱连接:阶段① → 阶段②,说明"先造指针、随后 load 才指定边界"
conn_x = CARD_X + CARD_W / 2
conn_y1 = card1_y + card1_h
conn_y2 = card2_y
L.append(f'<line x1="{conn_x}" y1="{conn_y1}" x2="{conn_x}" y2="{conn_y2}" '
          'stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="6,4" '
          'marker-end="url(#arrow-weak)"/>')
mid_y = (conn_y1 + conn_y2) / 2
label_x = conn_x + 16
L.append(f'<text x="{label_x}" y="{mid_y-4}" text-anchor="start" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("先造指针,")}</text>')
L.append(f'<text x="{label_x}" y="{mid_y+11}" text-anchor="start" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("随后 load 时才指定边界(另一次调用)")}</text>')

draw_stage_card(card2_y, card2_h, "②", "tl.load(ptr, boundary_check=…, padding=…)", FIELDS2,
                 ["→ tt.load { boundaryCheck = array<i32: 0, 1>,", "    padding = 1 }"],
                 "#9a3412", "#c2410c")

for i, line in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*19}" font-family="sans-serif" font-size="12" '
              f'fill="#64748b">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch07-block-ptr-pack.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
