#!/usr/bin/env python3
"""flow 模板改造:NSA 三支路(压缩 cmp / 选择 slc / 滑窗 win)各产出紧凑 KV,
门控 g^c 加权求和为 o_t*。下方两行数字对比 t=64 vs t=1024 的稀疏比。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "NSA 三支路:压缩+选择+滑窗各出紧凑 KV,门控加权求和"
SUBTITLE = "query q_t 分别对三份紧凑 KV 做注意力;g^c 加权求和 = o_t*;门控和 = 1.0(0.2+0.5+0.3)"

BRANCHES = [
    ("压缩支路 cmp", "K̃^cmp, Ṽ^cmp", "g=0.2", "#3b82f6"),
    ("选择支路 slc", "K̃^slc, Ṽ^slc", "g=0.5", "#7c3aed"),
    ("滑窗支路 win", "K̃^win, Ṽ^win", "g=0.3", "#059669"),
]

BOX_W, BOX_H, COL_GAP, PAD, TOP = 190, 60, 40, 40, 110
n = len(BRANCHES)
w = PAD * 2 + BOX_W * n + COL_GAP * (n - 1)
GATE_Y = TOP + BOX_H + 70
SUM_Y = GATE_Y + 90
h = SUM_Y + 130

col_x = [PAD + i * (BOX_W + COL_GAP) for i in range(n)]
q_x = w / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# query 源框
qy = PAD + 26
L.append(f'<rect x="{q_x-70}" y="{qy}" width="140" height="40" rx="6" '
          'fill="#0f172a" stroke="#0f172a" stroke-width="2"/>')
L.append(f'<text x="{q_x}" y="{qy+25}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" fill="white" font-weight="bold">query q_t</text>')

box_top = qy + 40 + 34
for i, (name, kv, gate, color) in enumerate(BRANCHES):
    x = col_x[i]
    # 从 query 到每支路的箭头
    L.append(f'<line x1="{q_x}" y1="{qy+40}" x2="{x+BOX_W/2}" y2="{box_top}" '
              f'stroke="{color}" stroke-width="1.5" marker-end="url(#a)" opacity="0.7"/>')
    L.append(f'<rect x="{x}" y="{box_top}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{color}" stroke="#1e293b" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_top+24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{box_top+44}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="#e2e8f0">{esc(kv)}</text>')
    # 箭头到门控
    gy = box_top + BOX_H
    L.append(f'<line x1="{x+BOX_W/2}" y1="{gy}" x2="{x+BOX_W/2}" y2="{GATE_Y}" '
              f'stroke="{color}" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<rect x="{x+BOX_W/2-34}" y="{GATE_Y}" width="68" height="30" rx="5" '
              f'fill="#fef3c7" stroke="#d97706" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{GATE_Y+20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="#92400e" '
              f'font-weight="bold">{esc(gate)}</text>')
    # 箭头到求和框
    sy = GATE_Y + 30
    L.append(f'<line x1="{x+BOX_W/2}" y1="{sy}" x2="{w/2}" y2="{SUM_Y}" '
              f'stroke="{color}" stroke-width="1.5" marker-end="url(#a)" opacity="0.8"/>')

# 求和框
SUM_BOX_W = 400
L.append(f'<rect x="{w/2-SUM_BOX_W/2}" y="{SUM_Y}" width="{SUM_BOX_W}" height="46" rx="8" '
          'fill="#0f172a" stroke="#0f172a" stroke-width="2"/>')
L.append(f'<text x="{w/2}" y="{SUM_Y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" fill="white" font-weight="bold">o_t* = Σ_c g_t^c · Attn(q_t, K̃^c, Ṽ^c)</text>')
L.append(f'<text x="{w/2}" y="{SUM_Y+38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#93c5fd">门控和 = 1.0 (0.2+0.5+0.3)</text>')

# 数字对比 callout
callout_y = SUM_Y + 62
box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{box_w}" height="52" rx="6" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#047857">t=64 → N_t=16(4+8+4), 稀疏比 0.25 ｜ t=1024 → N_t=32(8+16+8), 稀疏比 0.0312</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+40}" font-family="sans-serif" font-size="11.5" '
          f'fill="#047857">t 涨 16 倍,N_t 只涨 2 倍——支路预算不随 t 线性扩张,稀疏比随 t 增大单调趋 0</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig32-nsa-three-branch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
