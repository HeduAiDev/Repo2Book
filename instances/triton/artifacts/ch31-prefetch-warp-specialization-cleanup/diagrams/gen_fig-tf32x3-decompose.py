#!/usr/bin/env python3
"""fig-tf32x3-decompose: flow 模板——一个 fp32 dot 拆成 3 个 tf32 dot 的累加链,
保留 aSmall*bBig、aBig*bSmall、aBig*bBig 三项、丢弃最小的 aSmall*bSmall。
数字来源见 explainer.json mechanism f32-dot-tc-tf32x3 / traces/tf32x3.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "F32DotTC 的 TF32x3 —— 为什么恰好三次"

# 四个交叉项(累加顺序:从小到大),第 4 项丢弃
TERMS = [
    ("aSmall·bBig", "0.002575", "补正项 · 累加链首", "#e0f2fe", "#0284c7", False),
    ("aBig·bSmall", "0.001967", "补正项 · 累加链中", "#e0f2fe", "#0284c7", False),
    ("aBig·bBig",   "10.298326", "主项 · 累加链末(=单次 tf32 结果)", "#fef3c7", "#d97706", False),
    ("aSmall·bSmall", "0.00000049", "占结果 0.000000047 · 快尺读不出", "#f1f5f9", "#94a3b8", True),
]

BOX_W, BOX_H = 250, 92
GAP = 30
PAD, TOP = 46, 150
w = PAD * 2 + BOX_W * 4 + GAP * 3
h = TOP + BOX_H + 300

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" '
     f'fill="#475569">ab = (aBig+aSmall)(bBig+bSmall) 精确展开四项，dot(a,b,inputPrecision=tf32x3) 只留三项</text>']

# 顶部:精确展开公式框
L.append(f'<rect x="{w/2-260}" y="76" width="520" height="34" rx="8" fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{w/2}" y="98" text-anchor="middle" font-family="sans-serif" font-size="13.5" '
          f'fill="#3730a3">ab = aSmall·bBig + aBig·bSmall + aBig·bBig + aSmall·bSmall</text>')

x_positions = [PAD + i * (BOX_W + GAP) for i in range(4)]
for i, (name, val, note, fill, stroke, dropped) in enumerate(TERMS):
    x = x_positions[i]
    y = TOP
    # 从公式框连线下来
    fx = w/2 - 260 + 520 * (i + 0.5) / 4
    L.append(f'<line x1="{fx}" y1="110" x2="{x+BOX_W/2}" y2="{y-4}" '
              f'stroke="#94a3b8" stroke-width="1.3" marker-end="url(#a)"/>')
    dash = 'stroke-dasharray="5,4"' if dropped else ''
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2" {dash}/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+28}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    val_fill = "#94a3b8" if dropped else "#0f172a"
    L.append(f'<text x="{x+BOX_W/2}" y="{y+52}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" fill="{val_fill}">{esc(val)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{y+75}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">{esc(note)}</text>')
    if dropped:
        # 打叉标记「丢弃」
        L.append(f'<text x="{x+BOX_W/2}" y="{y+BOX_H+20}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#dc2626">✕ 丢弃(第 4 次不做)</text>')

# 前三项 → 累加箭头 → 累加器
acc_y = TOP + BOX_H + 60
acc_x = w/2 - 100
L.append(f'<rect x="{acc_x}" y="{acc_y}" width="200" height="50" rx="10" '
          f'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{w/2}" y="{acc_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#14532d">Σ 三项 = tf32x3 结果</text>')
L.append(f'<text x="{w/2}" y="{acc_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#166534">10.302867</text>')
for i in range(3):
    x = x_positions[i] + BOX_W / 2
    y1 = TOP + BOX_H + 32
    L.append(f'<line x1="{x}" y1="{y1}" x2="{w/2 + (i-1)*40}" y2="{acc_y-2}" '
              f'stroke="#16a34a" stroke-width="1.6" marker-end="url(#ag)"/>')

# 底部:误差对比条
cmp_y = acc_y + 90
L.append(f'<text x="{w/2}" y="{cmp_y-14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">相对误差(相对 fp32 参考值 10.302868)</text>')

CMP = [
    ("单次 tf32", "0.00044", "#fca5a5", "#b91c1c"),
    ("tf32x3(三次累加)", "0.00000011", "#86efac", "#166534"),
]
cmp_box_w, cmp_box_h = 340, 60
cmp_gap = 120
cmp_total = cmp_box_w * 2 + cmp_gap
cmp_x0 = (w - cmp_total) / 2
for i, (name, val, fill, stroke) in enumerate(CMP):
    x = cmp_x0 + i * (cmp_box_w + cmp_gap)
    L.append(f'<rect x="{x}" y="{cmp_y}" width="{cmp_box_w}" height="{cmp_box_h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+cmp_box_w/2}" y="{cmp_y+24}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{stroke}">{esc(name)}</text>')
    L.append(f'<text x="{x+cmp_box_w/2}" y="{cmp_y+46}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="16" font-weight="bold" '
              f'fill="{stroke}">{esc(val)}</text>')
mid_y = cmp_y + cmp_box_h / 2
L.append(f'<line x1="{cmp_x0+cmp_box_w+10}" y1="{mid_y}" x2="{cmp_x0+cmp_box_w+cmp_gap-10}" '
          f'y2="{mid_y}" stroke="#166534" stroke-width="2.5" marker-end="url(#ag)"/>')
L.append(f'<text x="{cmp_x0+cmp_box_w+cmp_gap/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="#166534">改善 3901 倍*</text>')
# 小字注:3901 倍来自完整精度,非图中两约数直接相除
L.append(f'<text x="{w/2}" y="{cmp_y+cmp_box_h+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#64748b">'
          f'* 3901 倍基于未舍入的完整精度计算，非图中两个约数直接相除(约 4000 倍)。</text>')

CAPTION_LINES = [
    "四项交叉积里,零头×零头(aSmall·bSmall)小到 fp32 都读不出(占结果 0.000000047),丢它无损;",
    "剩下三项每项至少一边是高精度,三次 tf32 dot 就把误差从单次的 0.00044 逼到 0.00000011,约 3901 倍改善。",
]
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{w/2}" y="{h-30+i*17}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#475569">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-tf32x3-decompose.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
