#!/usr/bin/env python3
"""fig31-5-kv-cache-table: DeepSeek-V2 维度(n_h=128,d_h=128)下四种注意力机制每
token 每层 KV cache 元素数对比。state-table 骨架(单行数值 + 条形可视化)。
数字全部来自 traces/cache_compare.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "KV Cache 账单:四种注意力机制每 token 每层缓存多少元素(DeepSeek-V2 n_h=128, d_h=128)"
SUBTITLE = "MLA 一栏与 MQA 同量级,却因潜向量携带全部头信息而保住 MHA 级能力"
ROWS = [
    ("MHA", 32768, "#94a3b8", "每头各存一份完整 K/V(2·n_h·d_h)"),
    ("GQA-8", 2048, "#fbbf24", "8 组共享(2·(n_h/8)·d_h)"),
    ("MQA", 256, "#fca5a5", "全部头共享一份(2·d_h)"),
    ("MLA", 576, "#1d4ed8", "潜向量 d_c=512 + 解耦位置 d_h_r=64"),
]
PAD, TOP = 40, 96
LABEL_W = 110
ROW_H = 78
MAX_VAL = 32768
BAR_MAX_W = 640
w = PAD*2 + LABEL_W + BAR_MAX_W + 220
h = TOP + ROW_H*len(ROWS) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

import math
def barlen(v):
    # log-scale so MQA(256)/MLA(576) stay visible next to MHA(32768)
    return BAR_MAX_W * math.log10(v+1) / math.log10(MAX_VAL+1)

for i, (name, val, color, note) in enumerate(ROWS):
    y = TOP + i*ROW_H
    L.append(f'<text x="{PAD+LABEL_W-14}" y="{y+ROW_H/2-4}" text-anchor="end" '
             f'font-family="sans-serif" font-size="15" font-weight="bold" '
             f'fill="#0f172a">{esc(name)}</text>')
    bw = barlen(val)
    bx = PAD + LABEL_W
    by = y + ROW_H/2 - 20
    L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="34" rx="6" '
             f'fill="{color}" stroke="#334155" stroke-width="1"/>')
    L.append(f'<text x="{bx+bw+12}" y="{y+ROW_H/2+5}" font-family="sans-serif" '
             f'font-size="15" font-weight="bold" fill="#0f172a">{val}</text>')
    L.append(f'<text x="{PAD+LABEL_W}" y="{y+ROW_H/2+34}" font-family="sans-serif" '
             f'font-size="11.5" fill="#64748b">{esc(note)}</text>')

foot_top = TOP + ROW_H*len(ROWS) + 20
foot_w = w - 2*PAD
L.append(f'<rect x="{PAD}" y="{foot_top}" width="{foot_w}" height="96" rx="10" '
         'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{PAD+foot_w/2}" y="{foot_top+30}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#92400e">'
         f'{esc("MLA/MHA 压缩比 = 32768 ÷ 576 = 56.89×,等效 2.25 组 GQA")}</text>')
L.append(f'<text x="{PAD+foot_w/2}" y="{foot_top+62}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" fill="#92400e">'
         f'{esc("全模型(60 层)每 token 缓存:MHA 1,966,080 个元素 → MLA 34,560 个元素")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig31-5-kv-cache-table.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
