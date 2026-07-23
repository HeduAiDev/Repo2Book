#!/usr/bin/env python3
"""layout 模板改造:两套 2-bit 枚举位对齐——OpAbility(能力) vs CoreType(放置),
toCoreType 位重解释。左表 3 行(无 00),右表 4 行(含 00);按位值对齐横向连线。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
VEC = "#15803d"
UND = "#64748b"

# (bit_value, name, color) — index 决定纵向对齐槽位
LEFT = {1: ("PREFER_VECTOR", VEC), 2: ("CUBE_ONLY", CUBE), 3: ("CUBE_AND_VECTOR", "#7c3aed")}
RIGHT = {0: ("UNDETERMINED", UND), 1: ("VECTOR_ONLY", VEC), 2: ("CUBE_ONLY", CUBE),
         3: ("CUBE_AND_VECTOR", "#7c3aed")}
BITS = {0: "00", 1: "01", 2: "10", 3: "11"}

W, H = 940, 480
PAD = 40
TOP = 120
ROW_H = 76
BOX_W = 300
LX = PAD
RX = W - PAD - BOX_W

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="18" font-weight="bold" '
     f'fill="{INK}">{esc("两套 2-bit 枚举:能力(OpAbility) 对齐 放置(CoreType)")}</text>',
     f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="13" fill="{GRAY}">'
     f'{esc("toCoreType 只做位重解释(原样搬运底层整数),不改变数值")}</text>']

L.append(f'<text x="{LX+BOX_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{INK}">{esc("OpAbility(能力,DAG.h:L27-32)")}</text>')
L.append(f'<text x="{RX+BOX_W/2}" y="{TOP-16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{INK}">{esc("CoreType(放置,DAG.h:L34-39)")}</text>')

for bit in (0, 1, 2, 3):
    y = TOP + bit * ROW_H
    # left
    if bit in LEFT:
        name, color = LEFT[bit]
        L.append(f'<rect x="{LX}" y="{y}" width="{BOX_W}" height="{ROW_H-14}" rx="8" '
                  f'fill="white" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{LX+18}" y="{y+28}" font-family="sans-serif" font-size="13" '
                  f'font-weight="bold" fill="{color}">{esc(name)}</text>')
        L.append(f'<text x="{LX+18}" y="{y+50}" font-family="sans-serif" font-size="12" '
                  f'fill="{GRAY}">{esc("bit = " + BITS[bit] + f" ({bit})")}</text>')
    # right
    name, color = RIGHT[bit]
    L.append(f'<rect x="{RX}" y="{y}" width="{BOX_W}" height="{ROW_H-14}" rx="8" '
              f'fill="white" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{RX+18}" y="{y+28}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<text x="{RX+18}" y="{y+50}" font-family="sans-serif" font-size="12" '
              f'fill="{GRAY}">{esc("bit = " + BITS[bit] + f" ({bit})")}</text>')
    # alignment arrow (only where both sides exist)
    if bit in LEFT:
        ay = y + (ROW_H - 14) / 2
        color = LEFT[bit][1]
        L.append(f'<line x1="{LX+BOX_W}" y1="{ay}" x2="{RX}" y2="{ay}" '
                  f'stroke="{color}" stroke-width="2" stroke-dasharray="6,4" '
                  'marker-end="url(#a)"/>')
        L.append(f'<rect x="{(LX+BOX_W+RX)/2-40}" y="{ay-30}" width="80" height="20" rx="4" '
                  f'fill="white" stroke="{color}"/>')
        L.append(f'<text x="{(LX+BOX_W+RX)/2}" y="{ay-16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="{color}">{esc("位=" + BITS[bit])}</text>')

# annotate the row without a left counterpart
y0 = TOP
L.append(f'<text x="{(LX+BOX_W+RX)/2}" y="{y0+(ROW_H-14)/2+4}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="{GRAY}">'
          f'{esc("OpAbility 无 00 态")}</text>')

# toCoreType label bracket at top
L.append(f'<text x="{(LX+BOX_W+RX)/2}" y="{TOP-40}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#334155">'
          f'{esc("toCoreType(位重解释)")}</text>')

foot_y = H - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("PREFER_VECTOR(01) 对齐 VECTOR_ONLY(01);CUBE_ONLY(10) 两侧同位;CUBE_AND_VECTOR(11)=VECTOR_ONLY|CUBE_ONLY")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-two-enums.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
