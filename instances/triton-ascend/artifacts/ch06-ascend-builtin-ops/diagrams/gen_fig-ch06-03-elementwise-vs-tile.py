#!/usr/bin/env python3
"""fig-ch06-03-elementwise-vs-tile — tiling 模板。
gather_out_to_ub 按 index 的每个元素取一个数;index_select_simd 按 index
的每个元素取一整条连续 tile;后者没有 index_boundary,越界就直接读到隔壁。
数据取自 traces/mem_semantics.json(dossier m5 worked_example)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)


W, H = 1360, 820
PAD = 50
BLUE, RED, GREEN, GRAY, ORANGE = "#1d4ed8", "#b91c1c", "#15803d", "#94a3b8", "#c2410c"

TITLE = "逐元素 vs 逐 tile:同一块结果,两种访存粒度"
SUB = "src 是 4x3 的 fp32 表(值 0..11),紧邻其后是隔壁数据(900..907)"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     f'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{RED}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUB)}</text>']

SRC_VALS = [[r * 3 + c for c in range(3)] for r in range(4)]  # 0..11
CELL = 46

def draw_src_table(x0, y0, extra_rows=0):
    for r in range(4):
        for c in range(3):
            x, y = x0 + c * CELL, y0 + r * CELL
            L.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" rx="4" '
                     f'fill="#f8fafc" stroke="{GRAY}" stroke-width="1.2"/>')
            L.append(f'<text x="{x+(CELL-3)/2}" y="{y+(CELL-3)/2+5}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="12" fill="#334155">'
                     f'{esc(str(SRC_VALS[r][c]))}</text>')
    for er in range(extra_rows):
        r = 4 + er
        extra_vals = [900 + er * 3 + c for c in range(3)]
        for c in range(3):
            x, y = x0 + c * CELL, y0 + r * CELL
            L.append(f'<rect x="{x}" y="{y}" width="{CELL-3}" height="{CELL-3}" rx="4" '
                     f'fill="#fff7ed" stroke="{ORANGE}" stroke-width="1.4" stroke-dasharray="4,3"/>')
            L.append(f'<text x="{x+(CELL-3)/2}" y="{y+(CELL-3)/2+5}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10.5" fill="{ORANGE}">'
                     f'{esc(str(extra_vals[c]))}</text>')

# ── 左面板:gather_out_to_ub(逐元素) ───────────────────────────────────
PANEL_Y = 90
LX0 = 90
L.append(f'<text x="{LX0}" y="{PANEL_Y-8}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1d4ed8">{esc("左:gather_out_to_ub —— 4 次逐元素地址计算")}</text>')
draw_src_table(LX0, PANEL_Y)
IDX_GATHER = [(0, 0), (3, 1), (5, 1), (1, 1)]  # 演示用:index tile 的四个来源(与 fig-02 同源)
targets = [(0, 0), (3, 1), None, (1, 1)]
IDX_X = LX0 + 3 * CELL + 60
IDX_Y0 = PANEL_Y + 10
for i, tgt in enumerate(targets):
    ty = IDX_Y0 + i * 44
    L.append(f'<rect x="{IDX_X}" y="{ty}" width="120" height="34" rx="6" fill="#eff6ff" '
             f'stroke="{BLUE}" stroke-width="1.6"/>')
    L.append(f'<text x="{IDX_X+60}" y="{ty+22}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" fill="{BLUE}">{esc(f"元素 {i}")}</text>')
    if tgt:
        r, c = tgt
        sx = LX0 + c * CELL + (CELL - 3) / 2
        sy = PANEL_Y + r * CELL + (CELL - 3) / 2
        L.append(f'<line x1="{IDX_X}" y1="{ty+17}" x2="{sx+10}" y2="{sy}" stroke="{BLUE}" '
                 f'stroke-width="1.3" marker-end="url(#a)" opacity="0.75"/>')
L.append(f'<text x="{LX0}" y="{PANEL_Y+4*CELL+30}" font-family="sans-serif" font-size="12" '
         f'fill="#475569">{esc("4 个 index → 4 次独立地址计算(1 个越界,换 other)")}</text>')

# ── 右面板:index_select_simd(逐 tile) ─────────────────────────────────
RX0 = 780
L.append(f'<text x="{RX0}" y="{PANEL_Y-8}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="{ORANGE}">'
         f'{esc("右:index_select_simd —— 2 次整条 tile 访存,无 index_boundary")}</text>')
draw_src_table(RX0, PANEL_Y, extra_rows=2)
IDX2_X = RX0 + 3 * CELL + 60
for i, (idxval, rowlabel, oob) in enumerate([(2, "src 第 2 行", False), (0, "src 第 0 行", False)]):
    ty = IDX_Y0 + i * 50
    fill, stroke = ("#eff6ff", BLUE)
    L.append(f'<rect x="{IDX2_X}" y="{ty}" width="150" height="40" rx="6" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{IDX2_X+75}" y="{ty+17}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" font-weight="bold" fill="{stroke}">{esc(f"index={idxval}")}</text>')
    L.append(f'<text x="{IDX2_X+75}" y="{ty+33}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10" fill="{stroke}">{esc(rowlabel + ",前2个")}</text>')
    sy = PANEL_Y + idxval * CELL + (CELL - 3) / 2
    L.append(f'<path d="M {IDX2_X} {ty+20} L {RX0+2*CELL} {sy}" fill="none" '
             f'stroke="{stroke}" stroke-width="2" marker-end="url(#a)"/>')

# 越界那一行(index=5,单独标红,箭头指到隔壁数据行)
oob_y = IDX_Y0 + 2 * 50 + 6
L.append(f'<rect x="{IDX2_X}" y="{oob_y}" width="150" height="40" rx="6" fill="#fee2e2" '
         f'stroke="{RED}" stroke-width="1.8"/>')
L.append(f'<text x="{IDX2_X+75}" y="{oob_y+17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="{RED}">{esc("index=5(越界)")}</text>')
L.append(f'<text x="{IDX2_X+75}" y="{oob_y+33}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10" fill="{RED}">{esc("没人拦,按公式照算")}</text>')
sy5 = PANEL_Y + 5 * CELL + (CELL - 3) / 2  # row5,col0 = 元素 15 = 值 903
L.append(f'<path d="M {IDX2_X} {oob_y+20} L {RX0+CELL*0.5} {sy5}" fill="none" '
         f'stroke="{RED}" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#r)"/>')

L.append(f'<text x="{RX0}" y="{PANEL_Y+6*CELL+30}" font-family="sans-serif" font-size="12" '
         f'fill="#475569">{esc("2 个 index → 2 次 tile 访存;index=5 读到隔壁数据 903/904")}</text>')

# ── 底部:结果对照 ─────────────────────────────────────────────────────
RES_Y = PANEL_Y + 6 * CELL + 70
RES = [
    ("合法 index=[2,0]", "[[6.0,7.0],[0.0,1.0]]", GREEN),
    ("越界 index=[2,5]", "[[6.0,7.0],[903.0,904.0]]  ← 读到 src 之外", RED),
]
for i, (lab, val, color) in enumerate(RES):
    ry = RES_Y + i * 50
    L.append(f'<rect x="{PAD}" y="{ry}" width="{W-2*PAD}" height="38" rx="8" '
             f'fill="{"#ecfdf5" if color==GREEN else "#fee2e2"}" stroke="{color}" stroke-width="1.6"/>')
    L.append(f'<text x="{PAD+16}" y="{ry+24}" font-family="sans-serif" font-size="12.5" '
             f'font-weight="bold" fill="{color}">{esc(lab)}</text>')
    L.append(f'<text x="{PAD+240}" y="{ry+24}" font-family="sans-serif" font-size="12.5" '
             f'fill="{color}">{esc(val)}</text>')

FOOT_Y = RES_Y + len(RES) * 50 + 30
FOOT = [
    "同一块 2x2 数据:gather_out_to_ub 要 4 次逐元素地址计算,index_select_simd 只发 2 次 tile 读——访存请求数从 numel(index) 降到 len(index)。",
    "代价:index_select_simd 没有 index_boundary 形参(mem_ops.py:L485-521),docstring 明写 does not check if index contains out-of-bounds values。",
    "占位协议:read_shape[dim] 必须是 -1,src_offset[dim] 被忽略;返回 shape = index 长度(dim 轴)拼 read_shape 其余轴,如 index 长 4 + read_shape(4,-1,128) → (4,4,128)。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD}" y="{FOOT_Y+i*20}" font-family="sans-serif" font-size="11" '
             f'fill="#64748b">{esc(ln)}</text>')

H_ACTUAL = FOOT_Y + len(FOOT) * 20 + 16
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-03-elementwise-vs-tile.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
