#!/usr/bin/env python3
"""fig-m4-add-stages-skeleton：add_stages 五段骨架跨后端不变——前三段
ttir/ttgir/llir 完全同名，只有末两段分叉（AMD amdgcn/hsaco，NVIDIA ptx/cubin）。
改自 before-after 模板，支持多个高亮下标（分叉段）。全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PANELS = [
    ("NVIDIA add_stages（5 段，第 37 章）",
     ["ttir", "ttgir", "llir", "ptx", "cubin"], {3, 4}, "cubin"),
    ("AMD add_stages（5 段，本章）",
     ["ttir", "ttgir", "llir", "amdgcn", "hsaco"], {3, 4}, "hsaco"),
]
BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 220, 42, 24, 300, 40, 110
FOOT_LINES = [
    "stages 段数：两后端均为 5；同名共享前缀段：3；分叉末段数：2。",
    "产物标识 NVIDIA=cubin（third_party/nvidia/backend/compiler.py:L384-L389）",
    "产物标识 AMD=hsaco（third_party/amd/backend/compiler.py:L358-L363）",
]
FOOT_LINE_H = 18

n_steps = len(PANELS[0][1])
w = PAD * 2 + PANEL_W * 2 + 80
stack_bottom = TOP + n_steps * (BOX_H + VGAP)  # 最后一个箭头 gap 后的 y（略多余，留白用）
foot_block_h = len(FOOT_LINES) * FOOT_LINE_H + 24
h = stack_bottom + foot_block_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD}" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">add_stages：同一个五段骨架，位置不变、末端换内容</text>',
     f'<text x="{w/2}" y="{PAD+22}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">前 3 段（ttir/ttgir/llir）两后端同名共享；末 2 段分叉产出不同格式的最终产物</text>']

for p, (title, steps, hot_set, final_label) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 80)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = i in hot_set
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1}"/>')
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+5}" text-anchor="middle" '
                  f'font-family="monospace" font-size="13" font-weight="{"bold" if hl else "normal"}" '
                  f'fill="#0f172a">{esc(step)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    # 分叉起点标注（第 3 段之后，即 llir -> 末两段）
    fork_y = TOP + 2 * (BOX_H + VGAP) + BOX_H
    L.append(f'<text x="{cx+BOX_W/2+10}" y="{fork_y+VGAP/2+4}" font-family="sans-serif" '
              f'font-size="10.5" fill="#d97706">← 分叉</text>')

# 中间横向对照箭头（同名前缀段）说明——竖直虚线框住前3段，跨两面板
prefix_bottom = TOP - 8
prefix_top = TOP + 3 * (BOX_H + VGAP) - VGAP + 4
L.append(f'<rect x="{PAD-14}" y="{prefix_bottom}" width="{PANEL_W+28}" height="{prefix_top-prefix_bottom}" '
          'fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,4" rx="10"/>')
px2 = PAD + 1 * (PANEL_W + 80)
L.append(f'<rect x="{px2-14}" y="{prefix_bottom}" width="{PANEL_W+28}" height="{prefix_top-prefix_bottom}" '
          'fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="5,4" rx="10"/>')
L.append(f'<text x="{PAD-14}" y="{prefix_bottom-8}" font-family="sans-serif" font-size="11" '
          f'fill="#1d4ed8" font-weight="bold">同名前缀段（共 3 段，两后端完全同构）</text>')

foot_y = h - foot_block_h + 12
for k, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y+k*FOOT_LINE_H}" font-family="sans-serif" font-size="11.5" '
              f'fill="#64748b">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-add-stages-skeleton.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
