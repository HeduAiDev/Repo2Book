#!/usr/bin/env python3
"""fig34-5-orchestration — flow 模板：rebalance_experts 五步编排 + 0.95 变更闸门。
5 个步骤框横排，箭头端点由框边缘计算；末尾闸门框展示判决式与本例代入结果。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

STEPS = ["① 折叠热度\nfold_layer_load", "② 贪心复制\nreplicate_hot_experts",
         "③ LPT 装箱\npack_replicated_experts", "④ 就地映射\nlocal_exchange",
         "⑤ 变更闸门\n0.95 阈值"]

TOTAL_BEFORE = 135.0
TOTAL_AFTER = 87.5
GATE_RATIO = 0.95
THRESHOLD = 128.25
IMPROVEMENT_RATIO = 0.648
CHANGE = 1

BOX_W, BOX_H = 156, 70
GAP = 46
PAD = 34
TOP = 96

n = len(STEPS)
w = PAD * 2 + BOX_W * n + GAP * (n - 1)
gate_y = TOP + BOX_H + 60
gate_h = 108
h = gate_y + gate_h + PAD + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#1e40af">rebalance_experts 五步编排</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'折热度→贪心复制→LPT 装箱→就地映射→0.95 变更闸门，只有真正变好才触发迁移</text>']

box_x = []
for i, step in enumerate(STEPS):
    x = PAD + i * (BOX_W + GAP)
    box_x.append(x)
    is_gate = (i == n - 1)
    fill = "#fef3c7" if is_gate else "#dbeafe"
    stroke = "#d97706" if is_gate else "#1e40af"
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = step.split("\n")
    y0 = TOP + BOX_H / 2 - (len(lines) - 1) * 9 + 2
    for k, line in enumerate(lines):
        weight = 'font-weight="bold" ' if k == 0 else ''
        size = 13 if k == 0 else 11
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*18}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{size}" {weight}'
                  f'fill="#0f172a">{esc(line)}</text>')
    if i < n - 1:
        ax1 = x + BOX_W
        ax2 = box_x_next = PAD + (i + 1) * (BOX_W + GAP)
        y = TOP + BOX_H / 2
        L.append(f'<line x1="{ax1}" y1="{y}" x2="{ax2}" y2="{y}" '
                  'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

# gate decision box below step 5, connected by a vertical arrow
gate_x_center = box_x[-1] + BOX_W / 2
arrow_top = TOP + BOX_H
arrow_bot = gate_y
L.append(f'<line x1="{gate_x_center}" y1="{arrow_top}" x2="{gate_x_center}" y2="{arrow_bot}" '
          'stroke="#d97706" stroke-width="2" marker-end="url(#a)"/>')

gate_box_w = w - PAD * 2
gate_box_x = PAD
L.append(f'<rect x="{gate_box_x}" y="{gate_y}" width="{gate_box_w}" height="{gate_h}" rx="10" '
          'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
line1 = (f'total_after={TOTAL_AFTER:g}  <  {GATE_RATIO:g} × total_before '
         f'= {GATE_RATIO:g} × {TOTAL_BEFORE:g} = {THRESHOLD:g} ？　→ 是（improvement_ratio='
         f'{IMPROVEMENT_RATIO:g} < {GATE_RATIO:g}）')
line2 = f'change = {CHANGE:g}（触发本层迁移；per_layer_priority 按改善比 argsort 给出迁移次序）'
L.append(f'<text x="{gate_box_x+22}" y="{gate_y+34}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#047857">{esc(line1)}</text>')
L.append(f'<text x="{gate_box_x+22}" y="{gate_y+68}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#047857">{esc(line2)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-5-orchestration.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
