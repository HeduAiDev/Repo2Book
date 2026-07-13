#!/usr/bin/env python3
"""fig31-1-kv-bottleneck: 标准 MHA 每 token 每层缓存 2*n_h*d_h 个 K/V 元素,
随上下文线性累积,是长上下文/大 batch 的显存瓶颈。tensor-flow 骨架 + 增长条示意。
坐标全由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 940
PAD = 40

CALL_TOP_PLACEHOLDER = None  # computed below after strip geometry is known
H = 84 + 60 + 66 + 22*6 + 60 + 90 + PAD  # FLOW_TOP+BOX_H+gap+strip+gap+call_h+bottom pad
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ared" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("标准 MHA:每个 token 都要为每个头各留一份完整 K、V")}</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("每头各存一份完整 K、V——随头数与上下文长度双重线性")}</text>']

# --- Row 1: flow  h_t -> n_h heads -> k_i,v_i(全部写入缓存) ---
FLOW_TOP = 84
BOX_H = 60
b1 = (PAD, FLOW_TOP, 150, BOX_H)
b2 = (b1[0] + b1[2] + 70, FLOW_TOP, 230, BOX_H)
b3 = (b2[0] + b2[2] + 70, FLOW_TOP, 300, BOX_H)

def box(x, y, w, h, fill, stroke, sw=1.5, rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def ctext(cx, cy, s, size=13, fill="#0f172a", weight=""):
    wt = f'font-weight="{weight}" ' if weight else ''
    return f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="sans-serif" font-size="{size}" {wt}fill="{fill}">{esc(s)}</text>'

L.append(box(*b1, "#e2e8f0", "#64748b"))
L.append(ctext(b1[0]+b1[2]/2, b1[1]+b1[3]/2-6, "token 隐状态", 13, "#0f172a", "bold"))
L.append(ctext(b1[0]+b1[2]/2, b1[1]+b1[3]/2+14, "h_t", 13, "#334155"))

L.append(box(*b2, "#e2e8f0", "#64748b"))
L.append(ctext(b2[0]+b2[2]/2, b2[1]+b2[3]/2-6, "n_h=128 个头,每头 d_h=128", 13, "#0f172a", "bold"))
L.append(ctext(b2[0]+b2[2]/2, b2[1]+b2[3]/2+14, "各头独立投影出 k_i, v_i", 12, "#334155"))

L.append(box(*b3, "#fee2e2", "#b91c1c", 2))
L.append(ctext(b3[0]+b3[2]/2, b3[1]+b3[3]/2-6, "全部 k_i, v_i 都要写入缓存", 13, "#991b1b", "bold"))
L.append(ctext(b3[0]+b3[2]/2, b3[1]+b3[3]/2+14, "2·n_h·d_h = 2·128·128 = 32768", 13, "#991b1b", "bold"))

for src, dst in [(b1, b2), (b2, b3)]:
    y = src[1] + src[3] / 2
    x1 = src[0] + src[2]
    x2 = dst[0]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2-4}" y2="{y}" stroke="#64748b" '
             'stroke-width="1.8" marker-end="url(#a)"/>')

# --- Row 2: growth strip: linear accumulation across tokens (no invented numbers,
#     only the given 32768 per-token increment, shown as repeating unit) ---
STRIP_TOP = FLOW_TOP + BOX_H + 66
strip_label_y = STRIP_TOP - 14
L.append(f'<text x="{PAD}" y="{strip_label_y}" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#0f172a">{esc("上下文每增长 1 个 token,缓存就再摞上一层 32768 个元素——线性累积,无上限")}</text>')

N_BARS = 6
BAR_W = 60
BAR_GAP = 14
BASE_H = 22
strip_x0 = PAD
for i in range(N_BARS):
    x = strip_x0 + i * (BAR_W + BAR_GAP)
    bar_h = BASE_H * (i + 1)
    y = STRIP_TOP + (BASE_H * N_BARS - bar_h)
    shade = "#b91c1c" if i == N_BARS - 1 else "#fca5a5"
    L.append(box(x, y, BAR_W, bar_h, shade, "#991b1b", 1, 4))
    L.append(ctext(x + BAR_W/2, STRIP_TOP + BASE_H*N_BARS + 18, f"t{i}", 11, "#64748b"))
L.append(f'<text x="{strip_x0 + N_BARS*(BAR_W+BAR_GAP) + 10}" y="{STRIP_TOP + BASE_H*N_BARS/2}" '
         f'font-family="sans-serif" font-size="12" fill="#64748b">{esc("上下文长度 →")}</text>')

# --- Row 3: full-model callout ---
CALL_TOP = STRIP_TOP + BASE_H*N_BARS + 60
call_w, call_h = W - 2*PAD, 90
L.append(box(PAD, CALL_TOP, call_w, call_h, "#fef3c7", "#d97706", 2, 10))
L.append(ctext(PAD + call_w/2, CALL_TOP + 32, "全模型(60 层)每 token 累计缓存元素", 14, "#92400e", "bold"))
L.append(ctext(PAD + call_w/2, CALL_TOP + 60,
                "32768 × 60 层 = 1,966,080 个元素/token", 15, "#92400e", "bold"))

L.append('</svg>')
out = Path(__file__).with_name("fig31-1-kv-bottleneck.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
