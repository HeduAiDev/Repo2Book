#!/usr/bin/env python3
"""state-table 模板:.td 五个手算例子逐格核对。
行=例 1-5,列=参数/phase(r)序列/复现表第 0、1 行/bit-exact。
行背景按"这一例相对例 1 新拧了哪个旋钮"分组染色(perPhase/maxPhase/vec),
呼应正文"逐步加料"的叙事;全部数字取自 verify_swizzle.py 对 .td 的逐位复现。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "TritonGPUAttrDefs.td 五个 swizzle 手算例子——逐格 bit-exact 复现"
SUBTITLE = "phase(r) = floor(r / perPhase) mod maxPhase ; out[r][c] = in[r][(floor(c/vec) xor phase(r))*vec + c%vec]"

# (例名, (vec,perPhase,maxPhase), phase(r)序列, row0, row1, 相对例1改了哪个旋钮)
ROWS = [
    ("例 1 基础",        "(1, 1, 4)", "[0,1,2,3]",
     "[0,1,2,3]", "[5,4,7,6]", "base"),
    ("例 2 perPhase=2",  "(1, 2, 4)", "[0,0,1,1]",
     "[0,1,2,3]", "[4,5,6,7]", "perPhase"),
    ("例 3 maxPhase=2",  "(1, 1, 2)", "[0,1,0,1,0,1,0,1]",
     "[0,1,2,3]", "[5,4,7,6]", "maxPhase"),
    ("例 4 per+max",     "(1, 2, 2)", "[0,0,1,1,0,0,1,1]",
     "[0,1,2,3]", "[4,5,6,7]", "both"),
    ("例 5 vec=2",       "(2, 1, 4)", "[0,1,2,3]",
     "[0,1,2,3,4,5,6,7]", "[10,11,8,9,14,15,12,13]", "vec"),
]
STATUS_COLOR = {
    "base":     ("#f8fafc", "#334155"),
    "perPhase": ("#dbeafe", "#1d4ed8"),
    "maxPhase": ("#dcfce7", "#166534"),
    "both":     ("#fef3c7", "#92400e"),
    "vec":      ("#ede9fe", "#5b21b6"),
}
LEGEND = [
    ("base", "例 1:基线"), ("perPhase", "拧 perPhase"),
    ("maxPhase", "拧 maxPhase"), ("both", "两者都拧"), ("vec", "拧 vec"),
]

COL_LABELS = ["(vec, perPhase, maxPhase)", "phase(r) 序列",
              "row 0(逻辑值)", "row 1(逻辑值,swizzle 后)", "bit-exact?"]
LABEL_W = 130
COL_W = [190, 190, 170, 230, 90]
HEADER_H, ROW_H = 40, 46
PAD, TOP = 36, 90

w = PAD * 2 + LABEL_W + sum(COL_W)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 40 + PAD
col_x = [PAD + LABEL_W]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD - 6}" font-family="sans-serif" font-size="15" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD + 14}" font-family="monospace" font-size="11" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, label in enumerate(COL_LABELS):  # 列头
    x, cw = col_x[j], COL_W[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw - 6}" height="{HEADER_H}" rx="4" '
              f'fill="#3b82f6" stroke="#1e3a5f"/>')
    L.append(f'<text x="{x + (cw-6)/2}" y="{TOP + HEADER_H/2 + 4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="white">{esc(label)}</text>')

for i, (name, params, phase, row0, row1, grp) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    fill, text_c = STATUS_COLOR[grp]
    L.append(f'<rect x="{PAD}" y="{ry}" width="{w - PAD*2}" height="{ROW_H - 4}" rx="4" '
              f'fill="{fill}"/>')
    L.append(f'<text x="{PAD + 12}" y="{ry + (ROW_H-4)/2 + 4}" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{text_c}">{esc(name)}</text>')
    cells = [params, phase, row0, row1]
    for j, val in enumerate(cells):
        x, cw = col_x[j], COL_W[j]
        L.append(f'<text x="{x + (cw-6)/2}" y="{ry + (ROW_H-4)/2 + 4}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11" '
                  f'fill="{text_c}">{esc(val)}</text>')
    # bit-exact 勾选列
    x, cw = col_x[4], COL_W[4]
    L.append(f'<text x="{x + (cw-6)/2}" y="{ry + (ROW_H-4)/2 + 6}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="16" font-weight="bold" '
              f'fill="#16a34a">{esc("是")}</text>')

# 图例
ly = TOP + HEADER_H + ROW_H * len(ROWS) + 22
lx = PAD
L.append(f'<text x="{lx}" y="{ly - 8}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("行色 = 相对例 1 新拧了哪个旋钮:")}</text>')
lx2 = PAD + 190
for grp, label in LEGEND:
    fill, text_c = STATUS_COLOR[grp]
    L.append(f'<rect x="{lx2}" y="{ly - 18}" width="14" height="14" rx="3" '
              f'fill="{fill}" stroke="{text_c}"/>')
    L.append(f'<text x="{lx2 + 20}" y="{ly - 6}" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(label)}</text>')
    lx2 += 20 + 8 + len(label) * 11 + 20

L.append('</svg>')
out = Path(__file__).with_name("fig-five-examples-phase.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
