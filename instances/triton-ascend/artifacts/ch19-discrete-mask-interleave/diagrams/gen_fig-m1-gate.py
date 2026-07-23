#!/usr/bin/env python3
"""fig-m1-gate: 总闸分流图（flow 模板）。
isDiscreteMask() 复用 MaskState::parse 判连续 vs 离散：
parse 成功→撤销副作用后放行给结构化 DMA 路径（回指 MaskState 所在章）；
parse 失败→本 pass 接管，走离散访存改写。
全坐标由常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W = 900
BOX_W, BOX_H = 300, 46
PAD, TOP = 40, 92
GAP_V = 34
CX = W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 640">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="640" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc("总闸：isDiscreteMask() 复用 MaskState::parse 判连续 vs 离散")}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("DiscreteMaskAccessConversionPass.cpp:L59-L72")}</text>')

# entry box
y0 = TOP
L.append(f'<rect x="{CX-BOX_W/2}" y="{y0}" width="{BOX_W}" height="{BOX_H}" rx="8" '
         'fill="#e2e8f0" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{CX}" y="{y0+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#0f172a">{esc("op 上是否已打 is_discrete_mask 标记？")}</text>')

# side box: already tagged -> failure (guard against reentry)
guard_x = CX + BOX_W/2 + 60
guard_w = 200
L.append(f'<rect x="{guard_x}" y="{y0}" width="{guard_w}" height="{BOX_H}" rx="8" '
         'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4,3"/>')
L.append(f'<text x="{guard_x+guard_w/2}" y="{y0+BOX_H/2-4}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#475569">{esc("已标记 →")}</text>')
L.append(f'<text x="{guard_x+guard_w/2}" y="{y0+BOX_H/2+13}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12" fill="#475569">{esc("return failure()（防重入）")}</text>')
L.append(f'<line x1="{CX+BOX_W/2}" y1="{y0+BOX_H/2}" x2="{guard_x}" y2="{y0+BOX_H/2}" '
         'stroke="#94a3b8" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+BOX_W/2+8}" y="{y0+BOX_H/2-6}" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("是")}</text>')

# arrow down to parse box
y1 = y0 + BOX_H + GAP_V
L.append(f'<line x1="{CX}" y1="{y0+BOX_H}" x2="{CX}" y2="{y1}" '
         'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+10}" y="{y0+BOX_H+GAP_V/2}" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">{esc("否")}</text>')

parse_w = 380
L.append(f'<rect x="{CX-parse_w/2}" y="{y1}" width="{parse_w}" height="{BOX_H}" rx="8" '
         'fill="#dbeafe" stroke="#1e3a8a" stroke-width="2"/>')
L.append(f'<text x="{CX}" y="{y1+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#1e3a8a">{esc("MaskState::parse(mask)")}</text>')

# branch down into two columns
y2 = y1 + BOX_H + 56
left_x = PAD + 30
right_x = W - PAD - 30 - BOX_W
left_cx = left_x + BOX_W / 2
right_cx = right_x + BOX_W / 2

for cx, label in [(left_cx, "parse 成功（还原成矩形连续区间）"),
                  (right_cx, "parse 失败（非连续 / 含离散因子）")]:
    L.append(f'<line x1="{CX}" y1="{y1+BOX_H}" x2="{cx}" y2="{y2-14}" '
             'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{cx}" y="{y2-20}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12" fill="#334155">{esc(label)}</text>')

# left branch box 1: eraseInsertedOps
L.append(f'<rect x="{left_x}" y="{y2}" width="{BOX_W}" height="{BOX_H}" rx="8" '
         'fill="#dcfce7" stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<text x="{left_cx}" y="{y2+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#14532d">{esc("eraseInsertedOps 撤销副作用 op")}</text>')

# left branch box 2: return failure
y3 = y2 + BOX_H + GAP_V
L.append(f'<line x1="{left_cx}" y1="{y2+BOX_H}" x2="{left_cx}" y2="{y3}" '
         'stroke="#15803d" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{left_x}" y="{y3}" width="{BOX_W}" height="{BOX_H}" rx="8" '
         'fill="#dcfce7" stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<text x="{left_cx}" y="{y3+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#14532d">{esc("return failure()")}</text>')

# left final output
y4 = y3 + BOX_H + GAP_V
L.append(f'<line x1="{left_cx}" y1="{y3+BOX_H}" x2="{left_cx}" y2="{y4}" '
         'stroke="#15803d" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<rect x="{left_x-10}" y="{y4}" width="{BOX_W+20}" height="{BOX_H+14}" rx="8" '
         'fill="#22c55e" stroke="#14532d" stroke-width="2"/>')
L.append(f'<text x="{left_cx}" y="{y4+(BOX_H+14)/2-2}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="white">{esc("放行")}</text>')
L.append(f'<text x="{left_cx}" y="{y4+(BOX_H+14)/2+16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="white">{esc("走结构化 DMA 下降路径")}</text>')

# right branch box: return success
L.append(f'<rect x="{right_x}" y="{y2}" width="{BOX_W}" height="{BOX_H}" rx="8" '
         'fill="#ffedd5" stroke="#c2410c" stroke-width="1.5"/>')
L.append(f'<text x="{right_cx}" y="{y2+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" fill="#7c2d12">{esc("return success()")}</text>')

y3b = y3
L.append(f'<line x1="{right_cx}" y1="{y2+BOX_H}" x2="{right_cx}" y2="{y3b}" '
         'stroke="#c2410c" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{right_x-10}" y="{y3b}" width="{BOX_W+20}" height="{BOX_H+14}" rx="8" '
         'fill="#f97316" stroke="#7c2d12" stroke-width="2"/>')
L.append(f'<text x="{right_cx}" y="{y3b+(BOX_H+14)/2-2}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="white">{esc("本 pass 接管")}</text>')
L.append(f'<text x="{right_cx}" y="{y3b+(BOX_H+14)/2+16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="white">{esc("离散 load/store/atomic 改写")}</text>')

# example fixture note at bottom
foot_y = y4 + BOX_H + 60
L.append(f'<rect x="{PAD}" y="{foot_y-28}" width="{W-2*PAD}" height="54" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-6}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("示例：mask = (idx < 200) ∨ (idx > 400)——两段区间的「或」，画不成一个矩形连续区间")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+14}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("→ parse 失败 → 判离散，走右支")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-gate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
