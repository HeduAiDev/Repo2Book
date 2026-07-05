#!/usr/bin/env python3
"""before-after 模板改造:CSA 块窗口相邻重叠(borrow) vs HCA 块源区间互不相交。
左panel:CSA 块1 借块0 的 token0-3;右panel:HCA 块0/块1 源区间 0-3 / 4-7 完全不相交。
数字来自 traces/csa_overlap.json 与 traces/hca_dense.json。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "CSA 块窗口重叠 vs HCA 块源区间不相交"
SUBTITLE = "n=8 个 token;CSA 相邻块借用边界 token,HCA(压缩率 128 远大于 CSA 的 4)源区间完全不重叠"

TOKENS = list(range(8))
PANEL_W, PAD, TOP, TOK_SIZE, TOK_GAP = 340, 40, 130, 34, 8
w = PAD * 2 + PANEL_W * 2 + 80
h = TOP + 300

def token_row(L, px, highlight_map):
    cx = px + PANEL_W / 2
    row_w = len(TOKENS) * (TOK_SIZE + TOK_GAP) - TOK_GAP
    start_x = cx - row_w / 2
    xs_ = []
    for i in TOKENS:
        x = start_x + i * (TOK_SIZE + TOK_GAP)
        xs_.append(x)
        fill, stroke, tcol = highlight_map[i]
        L.append(f'<rect x="{x}" y="{TOP}" width="{TOK_SIZE}" height="{TOK_SIZE}" rx="5" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+TOK_SIZE/2}" y="{TOP+TOK_SIZE/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{tcol}" '
                  f'font-weight="bold">{i}</text>')
    return xs_, start_x, row_w

def bracket(L, x0, x1, y, color, label, dashed=False):
    dash = ' stroke-dasharray="5,3"' if dashed else ''
    L.append(f'<path d="M {x0} {y} L {x0} {y+8} L {x1} {y+8} L {x1} {y}" '
              f'fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    L.append(f'<text x="{(x0+x1)/2}" y="{y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="{color}">{esc(label)}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+14}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# ---- 左:CSA(重叠) ----
px0 = PAD
L.append(f'<text x="{px0+PANEL_W/2}" y="{TOP-42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">左:CSA —— 块1 窗口借用块0 的 token</text>')
csa_map = {i: (("#3b82f6", "#1e3a5f", "white") if i < 4 else ("#059669", "#065f46", "white")) for i in TOKENS}
xs0, start_x0, row_w0 = token_row(L, px0, csa_map)
bracket(L, start_x0, start_x0 + 4*(TOK_SIZE+TOK_GAP)-TOK_GAP, TOP+TOK_SIZE+14, "#1e3a5f", "块0 窗口:token 0-3")
bracket(L, start_x0, start_x0 + row_w0, TOP+TOK_SIZE+56, "#d97706",
        "块1 窗口:借 token 0-3 + 本块 token 4-7(8 个位置,重叠)", dashed=True)
L.append(f'<rect x="{px0}" y="{TOP+TOK_SIZE+96}" width="{PANEL_W}" height="46" rx="6" '
          'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{px0+PANEL_W/2}" y="{TOP+TOK_SIZE+116}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#1e40af">相邻块索引交叠</text>')
L.append(f'<text x="{px0+PANEL_W/2}" y="{TOP+TOK_SIZE+134}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">但净压缩率仍 1/m(=1/4),而非 1/(2m)</text>')

# ---- 右:HCA(不重叠) ----
px1 = PAD + PANEL_W + 80
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP-42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">右:HCA —— 块间源区间互不相交</text>')
hca_map = {i: (("#3b82f6", "#1e3a5f", "white") if i < 4 else ("#7c3aed", "#4c1d95", "white")) for i in TOKENS}
xs1, start_x1, row_w1 = token_row(L, px1, hca_map)
bracket(L, start_x1, start_x1 + 4*(TOK_SIZE+TOK_GAP)-TOK_GAP, TOP+TOK_SIZE+14, "#1e3a5f", "块0 源区间:token 0-3")
bracket(L, start_x1 + 4*(TOK_SIZE+TOK_GAP), start_x1 + row_w1, TOP+TOK_SIZE+14, "#4c1d95", "块1 源区间:token 4-7")
L.append(f'<rect x="{px1}" y="{TOP+TOK_SIZE+96}" width="{PANEL_W}" height="46" rx="6" '
          'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP+TOK_SIZE+116}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#7c3aed">区间完全不相交(overlap=false)</text>')
L.append(f'<text x="{px1+PANEL_W/2}" y="{TOP+TOK_SIZE+134}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">真实 m\'=128(此例缩为 4);块少故省略重叠收益</text>')

# 中间箭头指示对比
midy = TOP + TOK_SIZE / 2
L.append(f'<line x1="{px0+PANEL_W+8}" y1="{midy}" x2="{px0+PANEL_W+68}" y2="{midy}" '
          'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{px0+PANEL_W+38}" y="{midy-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#94a3b8">对比</text>')

# 底部:稠密 attend callout
callout_y = TOP + TOK_SIZE + 156
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="66" rx="6" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#92400e">HCA 之后不做 top-k:稠密 MQA attend 全部 2 个压缩块(num_kv_entries_attended=2)</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">压缩率 128 大到块数极少时,选择的收益(省 top-k)小于代价(可能漏选),故 HCA 索性全 attend</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+60}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">CSA(压缩率 4)块数仍多,重叠窗口 + top-k 两件事都做;HCA(压缩率 128)块数已少,两件事都省</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig36-5-hca-vs-csa-overlap.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
