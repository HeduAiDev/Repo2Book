#!/usr/bin/env python3
"""fig36-3-head-layout: layout 模板。Autoregression Head 数据流水线(左→右),
雪花标记冻结模块(Embedding/LM Head 复用目标参数),FC+decoder 为可训练模块。
下方小表列不同目标模型规模对应的可训练参数量。
数字来自 explainer.json fig36-3 numbers(vllm/model_executor/models/llama_eagle.py:L87-L95;paper.md L67/L110)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STAGES = [
    ("token\n(bs,seq)", "#f1f5f9", "#64748b", False),
    ("Embedding\n共享目标参数", "#dbeafe", "#1e40af", True),
    ("拼接 (bs,seq,2h)\n[embed ⊕ feature]", "#e2e8f0", "#475569", False),
    ("FC (2h→h)", "#fed7aa", "#c2410c", False),
    ("decoder\n单层", "#fed7aa", "#c2410c", False),
    ("下一特征\n(bs,seq,h)", "#e2e8f0", "#475569", False),
    ("LM Head\n共享目标参数", "#dbeafe", "#1e40af", True),
    ("token", "#f1f5f9", "#64748b", False),
]
BOX_W, BOX_H, HGAP, PAD, TOP = 128, 74, 26, 40, 108
n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * HGAP
h = TOP + BOX_H + 230

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#0f172a">{esc("Autoregression Head：复用 Embedding/LM Head + 轻量 FC+decoder")}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("融合 [token_embed ⊕ feature] 的拼接维度为 2×hidden_dim")}</text>']

xs_pos = [PAD + i * (BOX_W + HGAP) for i in range(n)]
for i, (label, fill, stroke, frozen) in enumerate(STAGES):
    x = xs_pos[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines = label.split("\n")
    ny = TOP + BOX_H/2 - (len(lines)-1)*8
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{ny+k*16}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="{"bold" if k==0 else "normal"}" '
                  f'fill="{stroke}">{esc(line)}</text>')
    if frozen:
        L.append(f'<text x="{x+BOX_W-10}" y="{TOP+16}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="14" fill="#0369a1">❄</text>')
    if i < n - 1:
        y = TOP + BOX_H/2
        L.append(f'<line x1="{x+BOX_W}" y1="{y}" x2="{x+BOX_W+HGAP-4}" y2="{y}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    if i == 3 or i == 4:
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+BOX_H+20}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="#c2410c">{esc("需训练")}</text>')

L.append(f'<text x="{PAD}" y="{TOP+BOX_H+56}" font-family="sans-serif" font-size="12" '
          f'fill="#0369a1">{esc("❄ = 冻结（沿用目标模型参数，不参与训练）")}</text>')

# small table: target size -> trainable params
TABLE = [("7B", "0.24B"), ("13B", "0.37B"), ("33B", "0.56B"), ("70B", "0.99B")]
tab_top = TOP + BOX_H + 84
col_w = 150
tab_x0 = PAD
L.append(f'<text x="{PAD}" y="{tab_top-10}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#0f172a">{esc("草稿头可训练参数量（随目标模型规模）")}</text>')
for j, (tgt, params) in enumerate(TABLE):
    cx = tab_x0 + j * col_w
    L.append(f'<rect x="{cx}" y="{tab_top}" width="{col_w-12}" height="30" rx="4" '
              'fill="#fff7ed" stroke="#fdba74" stroke-width="1"/>')
    L.append(f'<text x="{cx+(col_w-12)/2}" y="{tab_top+13}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#7c2d12">{esc("目标 "+tgt)}</text>')
    L.append(f'<text x="{cx+(col_w-12)/2}" y="{tab_top+26}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#7c2d12">{esc(params)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-3-head-layout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
