#!/usr/bin/env python3
"""before-after 模板:扁平函数体 -> [VECTOR scope{全部op}] + [CUBE scope{空}]。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
CUBE = "#1e40af"
CUBE_BG = "#dbeafe"
VEC = "#15803d"
VEC_BG = "#dcfce7"
NEU = "#475569"
NEU_BG = "#f1f5f9"

TITLE = "encapsulateWithScope:先建两个 scope,全部塞进 VECTOR、CUBE 留空"
SUB = "第一个 scope 恒打 VECTOR、第二个恒打 CUBE(DAGScope.cpp:L69-151,L146-149);真正按核分发在 SplitScope 完成"

PANEL_W, PAD, TOP = 420, 40, 130
GAP = 100
W = PAD * 2 + PANEL_W * 2 + GAP
H = TOP + 440

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12" fill="{GRAY}">{esc(SUB)}</text>']

# LEFT panel: flat function body
px = PAD
cx = px + PANEL_W / 2
L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="{NEU}">扁平函数体(切分前)</text>')
L.append(f'<rect x="{px}" y="{TOP}" width="{PANEL_W}" height="300" rx="10" '
          f'fill="{NEU_BG}" stroke="{NEU}" stroke-width="1.5"/>')
OPS = ["dot", "addf", "store", "…"]
op_h = 56
op_gap = 12
oy = TOP + 24
for op in OPS:
    L.append(f'<rect x="{px+30}" y="{oy}" width="{PANEL_W-60}" height="{op_h}" rx="8" '
              f'fill="white" stroke="#94a3b8" stroke-width="1.2"/>')
    L.append(f'<text x="{cx}" y="{oy+op_h/2+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" fill="{INK}">{esc(op)}</text>')
    oy += op_h + op_gap

# arrow
ax1 = px + PANEL_W + 12
ax2 = px + PANEL_W + GAP - 12
ay = TOP + 150
L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="#64748b" stroke-width="2.2" '
          f'marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{ay-28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="{GRAY}">encapsulate</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{ay-16}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10" fill="{GRAY}">WithScope</text>')

# RIGHT panel: two scopes
px2 = PAD + PANEL_W + GAP
cx2 = px2 + PANEL_W / 2
L.append(f'<text x="{cx2}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="{NEU}">两个 scope.scope(切分后即刻状态)</text>')

# VECTOR scope (top, full)
vec_y = TOP
vec_h = 190
L.append(f'<rect x="{px2}" y="{vec_y}" width="{PANEL_W}" height="{vec_h}" rx="10" '
          f'fill="{VEC_BG}" stroke="{VEC}" stroke-width="1.8"/>')
L.append(f'<text x="{px2+16}" y="{vec_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{VEC}">scope.scope [tcore_type=VECTOR]</text>')
oy2 = vec_y + 42
for op in OPS:
    L.append(f'<rect x="{px2+24}" y="{oy2}" width="{PANEL_W-48}" height="30" rx="6" '
              f'fill="white" stroke="{VEC}" stroke-width="1"/>')
    L.append(f'<text x="{cx2}" y="{oy2+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="{INK}">{esc(op)}</text>')
    oy2 += 36

# CUBE scope (bottom, empty)
cube_y = vec_y + vec_h + 24
cube_h = 100
L.append(f'<rect x="{px2}" y="{cube_y}" width="{PANEL_W}" height="{cube_h}" rx="10" '
          f'fill="{CUBE_BG}" stroke="{CUBE}" stroke-width="1.8" stroke-dasharray="6,4"/>')
L.append(f'<text x="{px2+16}" y="{cube_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="{CUBE}">scope.scope [tcore_type=CUBE]</text>')
L.append(f'<text x="{cx2}" y="{cube_y+cube_h/2+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-style="italic" fill="{CUBE}">(空,待 SplitScope 填入)</text>')

# reading order
L.append(f'<circle cx="{px+16}" cy="{TOP-16}" r="12" fill="#3b82f6"/>')
L.append(f'<text x="{px+16}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="white">1</text>')
L.append(f'<circle cx="{px2+16}" cy="{TOP-16}" r="12" fill="#3b82f6"/>')
L.append(f'<text x="{px2+16}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="white">2</text>')

CAP1 = "scope.scope 是中性容器，靠挂 #hivm.tcore_type<CUBE|VECTOR> 属性标记归哪颗物理核。"
CAP2 = "先全塞进 VECTOR、CUBE 留空，是为让下一步 SplitScope 用「复制+裁剪」而非「搬移」来填 CUBE。"
cap_y = cube_y + cube_h + 46
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP1)}</text>')
L.append(f'<text x="{PAD}" y="{cap_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP2)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m11-encapsulate.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
