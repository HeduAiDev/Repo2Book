#!/usr/bin/env python3
"""flow 模板:extra/ 目录的 pkgutil 动态发现接缝。import 时遍历子模块,只收 is_pkg 的
子包(跳过 libdevice.py),逐个 import 进 sys.modules。数据来自
python/triton/language/extra/__init__.py。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, H, PAD, TOP = 1100, 560, 40, 110
BOX_W, BOX_H, GAP = 420, 54, 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="40" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">extra/ 是后端插座:pkgutil 动态发现子包,零硬编码</text>',
     f'<text x="{PAD}" y="60" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'python/triton/language/extra/__init__.py:L11,L15</text>']

cx = W / 2
steps = [
    ("import triton.language.extra", "#e2e8f0", "#64748b", False),
    ("pkgutil.iter_modules(__path__) 逐个子模块", "#dbeafe", "#1e40af", False),
]
y = TOP
for text, fill, stroke, bold in steps:
    L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    wt = 'font-weight="bold" ' if bold else ''
    L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" {wt}fill="#0f172a">{esc(text)}</text>')
    ny = y + BOX_H + GAP
    L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{ny-4}" stroke="#64748b" '
              f'stroke-width="1.5" marker-end="url(#a)"/>')
    y = ny

# 分支判定 is_pkg
diamond_y = y
dw, dh = 200, 70
L.append(f'<polygon points="{cx},{diamond_y} {cx+dw/2},{diamond_y+dh/2} {cx},{diamond_y+dh} '
          f'{cx-dw/2},{diamond_y+dh/2}" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
L.append(f'<text x="{cx}" y="{diamond_y+dh/2-4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#0f172a">is_pkg?</text>')
L.append(f'<text x="{cx}" y="{diamond_y+dh/2+14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">(子包 vs .py 文件)</text>')

by = diamond_y + dh
left_x, right_x = cx - 260, cx + 260
box_y = by + GAP
BOX2_W = 320
L.append(f'<rect x="{left_x-BOX2_W/2}" y="{box_y}" width="{BOX2_W}" height="{BOX_H+16}" rx="8" '
          f'fill="#ecfdf5" stroke="#047857" stroke-width="1.5"/>')
L.append(f'<text x="{left_x}" y="{box_y+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#047857">是(如 cuda/hip)</text>')
L.append(f'<text x="{left_x}" y="{box_y+50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#0f172a">import 并塞进 sys.modules</text>')

L.append(f'<rect x="{right_x-BOX2_W/2}" y="{box_y}" width="{BOX2_W}" height="{BOX_H+16}" rx="8" '
          f'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
L.append(f'<text x="{right_x}" y="{box_y+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#b91c1c">否(如 libdevice.py)</text>')
L.append(f'<text x="{right_x}" y="{box_y+50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#0f172a">跳过,不收进模块表</text>')

L.append(f'<path d="M{cx-dw/2},{diamond_y+dh/2} L{left_x},{box_y-4}" stroke="#047857" '
          f'stroke-width="1.5" fill="none" marker-end="url(#a)"/>')
L.append(f'<path d="M{cx+dw/2},{diamond_y+dh/2} L{right_x},{box_y-4}" stroke="#b91c1c" '
          f'stroke-width="1.5" fill="none" marker-end="url(#a)"/>')

foot_y = box_y + BOX_H + 16 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'上游零硬编码后端列表——姊妹篇 triton-ascend 就在这个 extra/ 接缝挂自己的 libdevice 入口</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-extra-seam.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
