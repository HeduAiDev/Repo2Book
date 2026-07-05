#!/usr/bin/env python3
"""fig31-2-joint-compression: h 经 W_DKV 压到 d_c 维潜向量 c_kv,K/V 都从这份共享
c_kv 上投影;推理期只有 c_kv 入 cache(实线=真实写盘,虚线=按需现场重算、从不落盘)。
tensor-flow 骨架。全坐标计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W = 940
PAD = 40

def box(x, y, w, h, fill, stroke, sw=1.5, rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def ctext(cx, cy, s, size=13, fill="#0f172a", weight=""):
    wt = f'font-weight="{weight}" ' if weight else ''
    return f'<text x="{cx}" y="{cy}" text-anchor="middle" font-family="sans-serif" font-size="{size}" {wt}fill="{fill}">{esc(s)}</text>'

TOP = 96
BOX_H = 64
h_box = (PAD, TOP, 150, BOX_H)
c_box = (h_box[0] + h_box[2] + 90, TOP, 230, BOX_H)
k_box = (c_box[0] + c_box[2] + 110, TOP - 60, 220, BOX_H)
v_box = (c_box[0] + c_box[2] + 110, TOP + 60, 220, BOX_H)

H = v_box[1] + BOX_H + 40 + 40 + 20 + 110 + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("联合压缩:K、V 共享同一段潜向量 c_kv,只有它写入缓存")}</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("不囤整箱零件,只囤一张能现场复原全部零件的图纸")}</text>']

L.append(box(*h_box, "#e2e8f0", "#64748b"))
L.append(ctext(h_box[0]+h_box[2]/2, h_box[1]+h_box[3]/2-6, "token h", 13, "#0f172a", "bold"))
L.append(ctext(h_box[0]+h_box[2]/2, h_box[1]+h_box[3]/2+14, "(隐状态)", 11, "#334155"))

L.append(box(*c_box, "#bfdbfe", "#1d4ed8", 2.5))
L.append(ctext(c_box[0]+c_box[2]/2, c_box[1]+c_box[3]/2-8, "c_kv (潜向量)", 14, "#1e3a8a", "bold"))
L.append(ctext(c_box[0]+c_box[2]/2, c_box[1]+c_box[3]/2+12, "d_c=4,唯一写入缓存", 12, "#1e3a8a", "bold"))

# h -> c_kv (solid, real compute + cache write)
y_mid = h_box[1] + h_box[3]/2
L.append(f'<line x1="{h_box[0]+h_box[2]}" y1="{y_mid}" x2="{c_box[0]-4}" y2="{y_mid}" '
         'stroke="#1d4ed8" stroke-width="2" marker-end="url(#a)"/>')
L.append(ctext((h_box[0]+h_box[2]+c_box[0])/2, y_mid-10, "W_DKV", 12, "#1d4ed8", "bold"))

for kv_box, label, wname in [(k_box, "k_c (物化 key)", "W_UK"), (v_box, "v_c (物化 value)", "W_UV")]:
    L.append(box(*kv_box, "#f1f5f9", "#94a3b8", 1.5))
    L.append(f'<text x="{kv_box[0]+8}" y="{kv_box[1]+kv_box[3]/2-6}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#475569">{esc(label)}</text>')
    L.append(f'<text x="{kv_box[0]+8}" y="{kv_box[1]+kv_box[3]/2+14}" font-family="sans-serif" '
             f'font-size="12" fill="#64748b">{esc("物化维度 n_h·d_h = 8(从不落盘)")}</text>')
    cy = kv_box[1] + kv_box[3]/2
    src_x = c_box[0] + c_box[2] + 10
    L.append(f'<line x1="{src_x}" y1="{y_mid}" x2="{kv_box[0]-4}" y2="{cy}" '
             'stroke="#94a3b8" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#ag)"/>')
    lx = (src_x+kv_box[0])/2
    ly = (y_mid+cy)/2 - 8
    L.append(ctext(lx, ly, wname, 12, "#475569", "bold"))

# legend (placed below the lowest box: v_box)
leg_y = v_box[1] + BOX_H + 40
L.append(f'<line x1="{PAD}" y1="{leg_y}" x2="{PAD+40}" y2="{leg_y}" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{PAD+50}" y="{leg_y+4}" font-family="sans-serif" font-size="12" fill="#334155">{esc("实线 = 真实写入 KV cache")}</text>')
L.append(f'<line x1="{PAD+280}" y1="{leg_y}" x2="{PAD+320}" y2="{leg_y}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,4"/>')
L.append(f'<text x="{PAD+330}" y="{leg_y+4}" font-family="sans-serif" font-size="12" fill="#334155">{esc("虚线 = 按需现场上投影,从不落盘")}</text>')

# comparison bars: MLA per-token(6) vs MHA baseline(16)
cmp_top = leg_y + 40
L.append(f'<text x="{PAD}" y="{cmp_top}" font-family="sans-serif" font-size="13" font-weight="bold" '
         f'fill="#0f172a">{esc("每 token 每层缓存元素数对比")}</text>')
bar_top = cmp_top + 20
bar_h_max = 90
bars = [("MLA (c_kv=4 + 解耦 k_pe=2)", 6, "#1d4ed8"), ("MHA 基线", 16, "#94a3b8")]
bx = PAD
for label, val, color in bars:
    bh = bar_h_max * val / 16
    by = bar_top + (bar_h_max - bh)
    L.append(box(bx, by, 140, bh, color, "#334155", 1, 4))
    L.append(ctext(bx+70, by-10, str(val), 14, "#0f172a", "bold"))
    L.append(ctext(bx+70, bar_top+bar_h_max+20, label, 12, "#334155"))
    bx += 220

L.append('</svg>')
out = Path(__file__).with_name("fig31-2-joint-compression.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
