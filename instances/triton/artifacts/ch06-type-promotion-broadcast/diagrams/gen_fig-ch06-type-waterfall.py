#!/usr/bin/env python3
"""fig-ch06-type-waterfall: computation_type_impl 的前置标量判断 + 六档 if 瀑布。
自上而下先命中先赢；每档右侧挂真实取证的命中样例（provenance: traces/ch06_run.json）。
全坐标由循环/常量计算，文本全 esc()。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PRE = {
    "title": "前置判断（step 0）：标量 kind ≤ 张量 kind？",
    "hit": "命中→提前返回张量自身 dtype（跳过下面 6 档）",
}

# (档号, 条件文本, 命中结果文本, 命中样例 或 None)
TIERS = [
    (1, "有一侧是 fp64？", "→ fp64", None),
    (2, "有一侧是 fp32？", "→ fp32", "fp16 × fp32 → fp32"),
    (3, "有一侧是 fp16？（除/模例外）", "→ fp16；除/模 → fp32", "fp16 × fp16 → fp16　|　除/模 → fp32"),
    (4, "有一侧是 bf16？（双侧才保）", "双侧 bf16 → bf16，否则 → fp32", "bf16 × bf16 → bf16"),
    (5, "两侧都是 fp8？", "同变体保留，异变体 → fp16", "fp8e4nv × fp8e5 → fp16"),
    (6, "两侧都是整数", "→ integer_promote_impl", "int32 × int64 → int64"),
]

BOX_W, BOX_H = 400, 56
GAP_Y = 30
PAD = 40
TOP = 110
CHIP_GAP = 36
CHIP_W = 240
EX_GAP = 30
EX_W = 330

box_x = PAD
chip_x = box_x + BOX_W + CHIP_GAP
ex_x = chip_x + CHIP_W + EX_GAP

w = ex_x + EX_W + PAD
row_h = BOX_H + GAP_Y
h = TOP + BOX_H + GAP_Y + (BOX_H + GAP_Y) * len(TIERS) + PAD + 10

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="arrGreen" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# Title
L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="19" font-weight="bold" '
          f'fill="#0f172a">{esc("computation_type_impl：6 档 if 瀑布，自上而下先命中先赢")}</text>')
L.append(f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="13" '
          f'fill="#475569">{esc("规则档数 = 6（不含前置标量判断 step 0）—— semantic.py:L61-L108")}</text>')

# Precondition box (dashed amber)
py = TOP
L.append(f'<rect x="{box_x}" y="{py}" width="{BOX_W}" height="{BOX_H}" rx="10" '
          f'fill="#fef3c7" stroke="#d97706" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{box_x+16}" y="{py+23}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#78350f">{esc(PRE["title"])}</text>')
L.append(f'<text x="{box_x+16}" y="{py+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#92400e">{esc("标量档 ≤ 张量档 →不参与提升（除模+fp16/bf16 例外仍升 fp32）")}</text>')

# arrow from precondition to hit chip (amber)
chip_y = py
L.append(f'<line x1="{box_x+BOX_W}" y1="{py+BOX_H/2}" x2="{chip_x}" y2="{py+BOX_H/2}" '
          'stroke="#d97706" stroke-width="1.6" marker-end="url(#arr)"/>')
L.append(f'<text x="{(box_x+BOX_W+chip_x)/2}" y="{py+BOX_H/2-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#92400e">{esc("是")}</text>')
L.append(f'<rect x="{chip_x}" y="{chip_y}" width="{CHIP_W}" height="{BOX_H}" rx="10" '
          'fill="#fffbeb" stroke="#d97706" stroke-width="1.3"/>')
L.append(f'<text x="{chip_x+CHIP_W/2}" y="{chip_y+23}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#78350f">{esc("提前返回")}</text>')
L.append(f'<text x="{chip_x+CHIP_W/2}" y="{chip_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#92400e">{esc("跳过档 1-6")}</text>')

prev_bottom = py + BOX_H

for idx, (tier, cond, hit, example) in enumerate(TIERS):
    y = TOP + BOX_H + GAP_Y + idx * row_h
    # downward connector from previous box bottom to this box top, label "否"
    conn_x = box_x + BOX_W * 0.15
    L.append(f'<line x1="{conn_x}" y1="{prev_bottom}" x2="{conn_x}" y2="{y}" '
              'stroke="#64748b" stroke-width="1.5" stroke-dasharray="3,3" marker-end="url(#arr)"/>')
    L.append(f'<text x="{conn_x+8}" y="{(prev_bottom+y)/2+4}" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc("否，落下一档")}</text>')

    # condition box
    L.append(f'<rect x="{box_x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/>')
    L.append(f'<text x="{box_x+16}" y="{y+23}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#1e3a8a">{esc(f"档 {tier}：{cond}")}</text>')
    L.append(f'<text x="{box_x+16}" y="{y+42}" font-family="sans-serif" font-size="11.5" '
              f'fill="#1e40af">{esc(hit)}</text>')

    # arrow to hit chip
    ay = y + BOX_H / 2
    L.append(f'<line x1="{box_x+BOX_W}" y1="{ay}" x2="{chip_x}" y2="{ay}" '
              'stroke="#16a34a" stroke-width="1.6" marker-end="url(#arrGreen)"/>')
    L.append(f'<text x="{(box_x+BOX_W+chip_x)/2}" y="{ay-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#166534">{esc("是")}</text>')

    L.append(f'<rect x="{chip_x}" y="{y}" width="{CHIP_W}" height="{BOX_H}" rx="10" '
              'fill="#dcfce7" stroke="#16a34a" stroke-width="1.4"/>')
    L.append(f'<text x="{chip_x+CHIP_W/2}" y="{y+23}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#14532d">{esc(f"命中→{hit}")}</text>')

    if example:
        L.append(f'<line x1="{chip_x+CHIP_W}" y1="{ay}" x2="{ex_x}" y2="{ay}" '
                  'stroke="#94a3b8" stroke-width="1.2" marker-end="url(#arr)"/>')
        L.append(f'<rect x="{ex_x}" y="{y+6}" width="{EX_W}" height="{BOX_H-12}" rx="8" '
                  'fill="#f8fafc" stroke="#94a3b8" stroke-width="1"/>')
        L.append(f'<text x="{ex_x+14}" y="{ay-4}" font-family="sans-serif" font-size="11" '
                  f'fill="#334155">{esc("真实取证（Triton 3.2.0 headless）")}</text>')
        L.append(f'<text x="{ex_x+14}" y="{ay+14}" font-family="sans-serif" font-size="12.5" '
                  f'font-weight="bold" fill="#0f172a">{esc(example)}</text>')

    prev_bottom = y + BOX_H

L.append('</svg>')
out = Path(__file__).with_name('fig-ch06-type-waterfall.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}  size={w}x{h}')
