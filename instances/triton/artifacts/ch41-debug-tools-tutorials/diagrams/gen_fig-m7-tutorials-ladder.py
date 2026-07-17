#!/usr/bin/env python3
"""fig-m7-tutorials-ladder: flow 模板(自定义阶梯)。python/tutorials 01->09
每级一个新概念, 三级(01/03/06)回指本书对应章节。数字来自
python/tutorials/*.py 文件与 dossier data_flow ③。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# (文件名, 新概念, 回指章节或 None)
STEPS = [
    ("01-vector-add", "programming model", "回指第 3 章"),
    ("02-fused-softmax", "fusion + reduction", None),
    ("03-matmul", "block matmul + swizzle", "回指第 27/28 章"),
    ("04-dropout", "并行 RNG", None),
    ("05-layer-norm", "反向 + 并行归约", None),
    ("06-fused-attention", "在线 softmax 融合", "预告第 42/43 章"),
    ("07-extern", "libdevice 扩展", None),
    ("08-grouped-gemm", "device 端静态调度", None),
    ("09-persistent", "persistent + TMA", None),
]

BOX_W, BOX_H, HGAP, PAD, TOP = 168, 58, 20, 40, 96
CALLOUT_H = 46
n = len(STEPS)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
h = TOP + BOX_H + CALLOUT_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#0f172a">'
         f'{esc("python/tutorials 01→09:一道认知阶梯,每级一个新概念")}</text>')

X = [PAD + i * (BOX_W + HGAP) for i in range(n)]

for i, (name, concept, back) in enumerate(STEPS):
    x = X[i]
    hl = back is not None
    fill = "#fef3c7" if hl else "#e2e8f0"
    stroke = "#d97706" if hl else "#64748b"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1}"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{"#78350f" if hl else "#0f172a"}">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+40}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" '
              f'fill="{"#92400e" if hl else "#475569"}">{esc(concept)}</text>')
    if i < n - 1:
        y1 = TOP + BOX_H / 2
        L.append(f'<line x1="{x+BOX_W}" y1="{y1}" x2="{X[i+1]}" y2="{y1}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    if back:
        cy = TOP + BOX_H + 18
        L.append(f'<line x1="{x+BOX_W/2}" y1="{TOP+BOX_H}" x2="{x+BOX_W/2}" y2="{cy}" '
                  'stroke="#d97706" stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<rect x="{x+BOX_W/2-70}" y="{cy}" width="140" height="{CALLOUT_H-8}" rx="6" '
                  f'fill="#fff7ed" stroke="#ea580c"/>')
        L.append(f'<text x="{x+BOX_W/2}" y="{cy+22}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#7c2d12">{esc(back)}</text>')

note_y = TOP + BOX_H + CALLOUT_H + 62
L.append(f'<text x="{PAD}" y="{note_y}" font-family="sans-serif" font-size="13" '
         f'fill="#0f172a">'
         f'{esc("06-fused-attention 是收束点:把前面 dot/reduce/block-ptr 全用上,是 FlashAttention v2 的在线 softmax 内层。")}</text>')
L.append(f'<text x="{PAD}" y="{note_y+24}" font-family="sans-serif" font-size="12" '
         f'fill="#475569">'
         f'{esc("别从 06 硬啃——先沿阶梯每级吃一个新概念,tutorials 的顺序就是本书章节的顺序。")}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m7-tutorials-ladder.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
