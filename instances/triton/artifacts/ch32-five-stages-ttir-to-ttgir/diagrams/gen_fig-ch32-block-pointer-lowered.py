#!/usr/bin/env python3
"""fig-ch32-block-pointer-lowered: before-after 模板。
block pointer 只活到 TTIR 级:追踪期的 3 个 tt.make_tensor_ptr,经 make_ttir 里的
RewriteTensorPointer 被降解成显式指针张量算术,TTGIR 里再也见不到——回扣 ch07。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PANEL_W, PAD, TOP = 340, 44, 130
BOX_H, VGAP = 40, 14
w = PAD * 2 + PANEL_W * 2 + 130
h = TOP + 5 * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker></defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(
    f'<text x="{w/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("block pointer 只活到 TTIR 级")}</text>'
)
L.append(
    f'<text x="{w/2}" y="54" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
    f'fill="#475569">{esc("lib/Dialect/Triton/Transforms/RewriteTensorPointer.cpp:L227-L250 rewriteMakeTensorPtrOp")}</text>'
)

x_left = PAD
x_right = PAD + PANEL_W + 130

L.append(f'<text x="{x_left+PANEL_W/2}" y="{TOP-18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#334155">{esc("追踪期(任何 pass 前)")}</text>')
L.append(f'<text x="{x_right+PANEL_W/2}" y="{TOP-18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#1d4ed8">{esc("make_ttir 之后")}</text>')

# 左面板:3 个 tt.make_tensor_ptr
LEFT_ROWS = [
    "%2 = tt.make_tensor_ptr %arg0 (A)",
    "%5 = tt.make_tensor_ptr %arg1 (B)",
    "%11 = tt.make_tensor_ptr %arg2 (C)",
]
for i, line in enumerate(LEFT_ROWS):
    y = TOP + i * (BOX_H + VGAP)
    L.append(f'<rect x="{x_left}" y="{y}" width="{PANEL_W}" height="{BOX_H}" rx="8" '
             f'fill="#fef3c7" stroke="#d97706" stroke-width="1.6"/>')
    L.append(f'<text x="{x_left+PANEL_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
             f'font-family="monospace" font-size="12" fill="#92400e">{esc(line)}</text>')
count_y = TOP + len(LEFT_ROWS) * (BOX_H + VGAP) + 4
L.append(f'<rect x="{x_left}" y="{count_y}" width="{PANEL_W}" height="30" rx="8" '
         f'fill="#fffbeb" stroke="#d97706" stroke-width="1.2"/>')
L.append(f'<text x="{x_left+PANEL_W/2}" y="{count_y+20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#92400e">{esc("3 个 tt.make_tensor_ptr(A/B/C 各一)")}</text>')

# 右面板:降解后的指针张量算术链
RIGHT_ROWS = [
    "tt.splat(base 广播)",
    "tt.make_range + expand_dims + broadcast",
    "arith.addi(拼出偏移张量)",
    "tt.addptr(base + 偏移)",
]
for i, line in enumerate(RIGHT_ROWS):
    y = TOP + i * (BOX_H + VGAP)
    L.append(f'<rect x="{x_right}" y="{y}" width="{PANEL_W}" height="{BOX_H}" rx="8" '
             f'fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/>')
    L.append(f'<text x="{x_right+PANEL_W/2}" y="{y+BOX_H/2+5}" text-anchor="middle" '
             f'font-family="monospace" font-size="11.5" fill="#1e3a8a">{esc(line)}</text>')
    if i < len(RIGHT_ROWS) - 1:
        cx = x_right + PANEL_W / 2
        y1 = y + BOX_H
        y2 = y1 + VGAP
        L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2-3}" stroke="#93c5fd" '
                 f'stroke-width="1.8" marker-end="url(#a)"/>')
count_y2 = TOP + len(RIGHT_ROWS) * (BOX_H + VGAP) + 4
L.append(f'<rect x="{x_right}" y="{count_y2}" width="{PANEL_W}" height="30" rx="8" '
         f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.4"/>')
L.append(f'<text x="{x_right+PANEL_W/2}" y="{count_y2+20}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc("0 个 tt.make_tensor_ptr")}</text>')

# 中间大箭头
midy = TOP + (3 * (BOX_H + VGAP)) / 2 - VGAP / 2
ax1 = x_left + PANEL_W + 10
ax2 = x_right - 10
L.append(f'<line x1="{ax1}" y1="{midy}" x2="{ax2}" y2="{midy}" stroke="#2563eb" '
         f'stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{midy-14}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc("RewriteTensorPointer")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{midy+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#475569">{esc("拆 base+strides+offsets")}</text>')

# 底部断言
assert_y = max(count_y, count_y2) + 66
L.append(f'<rect x="{PAD}" y="{assert_y}" width="{w-2*PAD}" height="32" rx="8" '
         f'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.4"/>')
assert_text = "driver 断言: ‘make_tensor_ptr’ not in ttir_after_make_ttir == True"
L.append(
    f'<text x="{w/2}" y="{assert_y+21}" text-anchor="middle" font-family="monospace" '
    f'font-size="12" fill="#15803d">'
    f'{esc(assert_text)}</text>'
)

concl_y = assert_y + 60
L.append(f'<text x="{PAD}" y="{concl_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'{esc("block pointer 是给前端/边界检查用的高层抽象;降到 TTGIR 前就被拆成显式指针算术——回扣第 7 章讲的 tt.make_tensor_ptr,只活到 TTIR 这一级")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-block-pointer-lowered.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
