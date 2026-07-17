#!/usr/bin/env python3
"""fig-ch32-default-blocked: layout 模板。
TypeConverter 见张量无 encoding 就贴默认 #blocked1(getDefaultBlockedEncoding)。
只展示"贴上什么属性"这一事实,不重讲 Blocked 编码内部含义(回指 ch21)。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


ATTRS = [
    ("sizePerThread", "[1, 1]"),
    ("threadsPerWarp", "[2, 16]"),
    ("warpsPerCTA", "[4, 1]"),
    ("order", "[1, 0]"),
]

PANEL_W, PAD, TOP = 300, 44, 118
ROW_H = 34
CARD_H = len(ATTRS) * ROW_H + 46
BOX_H = 74
GAP = 130
w = PAD * 2 + PANEL_W * 2 + GAP
h = TOP + max(BOX_H, CARD_H) + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker></defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(
    f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("无布局张量 → 默认贴上 #blocked1")}</text>'
)
L.append(
    f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
    f'fill="#475569">{esc("lib/Dialect/TritonGPU/IR/Dialect.cpp:L520-L532 getDefaultBlockedEncoding")}</text>'
)

# 左面板:无 encoding 的张量
lx = PAD
ly = TOP
L.append(f'<rect x="{lx}" y="{ly}" width="{PANEL_W}" height="{BOX_H}" rx="10" '
         f'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.6"/>')
L.append(
    f'<text x="{lx+PANEL_W/2}" y="{ly+30}" text-anchor="middle" font-family="monospace" '
    f'font-size="14" fill="#334155">{esc("tensor<16x16xf16>")}</text>'
)
L.append(
    f'<text x="{lx+PANEL_W/2}" y="{ly+52}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#64748b">{esc("encoding = (无)")}</text>'
)

# 右面板:#blocked1 属性卡
rx = PAD + PANEL_W + GAP
ry = TOP
card_h = CARD_H
L.append(f'<rect x="{rx}" y="{ry}" width="{PANEL_W}" height="{card_h}" rx="10" '
         f'fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
L.append(
    f'<text x="{rx+PANEL_W/2}" y="{ry+26}" text-anchor="middle" font-family="monospace" '
    f'font-size="14" font-weight="bold" fill="#1d4ed8">{esc("#blocked1")}</text>'
)
for i, (k, v) in enumerate(ATTRS):
    yrow = ry + 46 + i * ROW_H
    L.append(
        f'<text x="{rx+18}" y="{yrow}" font-family="monospace" font-size="12.5" '
        f'fill="#1e40af">{esc(k)}</text>'
    )
    L.append(
        f'<text x="{rx+PANEL_W-18}" y="{yrow}" text-anchor="end" font-family="monospace" '
        f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(v)}</text>'
    )

# 箭头:左面板 -> 右面板(从框边缘算)
mid_y = TOP + max(BOX_H, card_h) / 2
ax1 = lx + PANEL_W
ax2 = rx
L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2-6}" y2="{mid_y}" stroke="#2563eb" '
         f'stroke-width="2.4" marker-end="url(#a)"/>')
L.append(
    f'<text x="{(ax1+ax2)/2}" y="{mid_y-14}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc("TypeConverter")}</text>'
)
L.append(
    f'<text x="{(ax1+ax2)/2}" y="{mid_y+22}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#475569">{esc("getEncoding()")}</text>'
)
L.append(
    f'<text x="{(ax1+ax2)/2}" y="{mid_y+38}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#475569">{esc("为空 → 贴默认")}</text>'
)

# 底部触发条件说明
cond_y = TOP + max(BOX_H, card_h) + 46
L.append(
    f'<rect x="{PAD}" y="{cond_y}" width="{w-2*PAD}" height="32" rx="8" '
    f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.2"/>'
)
L.append(
    f'<text x="{w/2}" y="{cond_y+20}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#1d4ed8">'
    f'{esc("触发点:张量已有 encoding → 原样返回;无 encoding → 调 getDefaultBlockedEncoding 贴上 Blocked")}</text>'
)

# 底部实测来源
src_y = cond_y + 56
L.append(
    f'<text x="{PAD}" y="{src_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
    f'{esc("实测:16×16 kernel(num_warps=4, threads_per_warp=32, num_ctas=1) · Triton v3.2.0 headless 编译 · #blocked1")}</text>'
)

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-default-blocked.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
