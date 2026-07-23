#!/usr/bin/env python3
"""state-machine 模板改造为 Hasse 菱形格:CoreType 2-bit 幂集格。
底 UNDETERMINED(0),中 VECTOR_ONLY(1)/CUBE_ONLY(2),顶 CUBE_AND_VECTOR(3);边=按位或。
右侧旁注 toHivm 映射。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
VEC = "#15803d"
TOP_C = "#7c3aed"
BOT_C = "#64748b"

W, H = 1200, 560
PAD = 40
CX = 460
NODE_W, NODE_H = 220, 66
BOT_Y = 440
MID_Y = 290
TOP_Y = 140
LX = CX - 260
RX = CX + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="18" font-weight="bold" '
     # 粗体避开 CJK 紧邻拉丁字符(rsvg 字形回退 bug),CoreType 单独一段常规字重
     f'fill="{INK}">{esc("四态幂集格")}<tspan font-weight="normal">{esc("(CoreType)")}</tspan></text>',
     f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="13" fill="{GRAY}">'
     f'{esc("operator| 是格的并(按位或);底=拿不准,顶=双核皆可")}</text>']


def node(cx, cy, name, sub, color):
    x = cx - NODE_W / 2
    y = cy - NODE_H / 2
    L.append(f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="white" stroke="{color}" stroke-width="2.5"/>')
    L.append(f'<text x="{cx}" y="{y+27}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<text x="{cx}" y="{y+48}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="{GRAY}">{esc(sub)}</text>')


node(CX, TOP_Y, "CUBE_AND_VECTOR", "= 3 (⊤,双核皆可)", TOP_C)
node(LX, MID_Y, "VECTOR_ONLY", "= 1", VEC)
node(RX, MID_Y, "CUBE_ONLY", "= 2", CUBE)
node(CX, BOT_Y, "UNDETERMINED", "= 0 (⊥,拿不准)", BOT_C)

edges = [
    (CX, BOT_Y - NODE_H / 2, LX, MID_Y + NODE_H / 2, VEC),
    (CX, BOT_Y - NODE_H / 2, RX, MID_Y + NODE_H / 2, CUBE),
    (LX, MID_Y - NODE_H / 2, CX, TOP_Y + NODE_H / 2, VEC),
    (RX, MID_Y - NODE_H / 2, CX, TOP_Y + NODE_H / 2, CUBE),
]
for x1, y1, x2, y2, color in edges:
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
              'stroke-width="2" marker-end="url(#a)"/>')
mid_lbl_1 = ((CX + LX) / 2 - 8, (BOT_Y - NODE_H / 2 + MID_Y + NODE_H / 2) / 2)
mid_lbl_2 = ((CX + RX) / 2 + 8, (BOT_Y - NODE_H / 2 + MID_Y + NODE_H / 2) / 2)
L.append(f'<text x="{mid_lbl_1[0]}" y="{mid_lbl_1[1]}" text-anchor="end" '
          f'font-family="sans-serif" font-size="11" fill="{VEC}">{esc("⊔ = |")}</text>')
L.append(f'<text x="{mid_lbl_2[0]}" y="{mid_lbl_2[1]}" text-anchor="start" '
          f'font-family="sans-serif" font-size="11" fill="{CUBE}">{esc("⊔ = |")}</text>')

# toHivm side annotations
hivm = [
    (BOT_Y, "toHivm(0) → CUBE_OR_VECTOR", BOT_C),
    (MID_Y, "toHivm(1) → VECTOR", VEC),
    (MID_Y - 26, "toHivm(2) → CUBE", CUBE),
    (TOP_Y, "toHivm(3) → CUBE_AND_VECTOR", TOP_C),
]
ann_x = RX + NODE_W / 2 + 40
L.append(f'<text x="{ann_x}" y="{TOP_Y-70}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{INK}">{esc("toHivm(DAG.h:L76-90)")}</text>')
seen_y = set()
order = [(TOP_Y, "toHivm(3) → CUBE_AND_VECTOR", TOP_C),
         (MID_Y - 30, "toHivm(2) → CUBE", CUBE),
         (MID_Y + 30, "toHivm(1) → VECTOR", VEC),
         (BOT_Y, "toHivm(0) → CUBE_OR_VECTOR", BOT_C)]
for y, text, color in order:
    L.append(f'<text x="{ann_x}" y="{y}" font-family="sans-serif" font-size="12" '
              f'fill="{color}">{esc(text)}</text>')

foot_y = H - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="{GRAY}">'
          f'{esc("operator!(取补,DAG.h:L62-74)只在 {CUBE_ONLY,VECTOR_ONLY} 内定义,不作用于顶/底两态")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch16-lattice.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
