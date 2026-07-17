#!/usr/bin/env python3
"""fig-optimize-dot-operands (before-after 模板)
HoistLayoutConversion 把 convert 上移贴近 load,elementwise 直接在 dot_operand 布局上做,
省掉一次 shmem 往返(cap>=80 才开)。底部列出 OptimizeDotOperands 的四个 pattern。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PANELS = [
    ("改前(未 hoist)", [
        "load(blocked)",
        "convert -> blocked\n(shmem 往返 #1)",
        "elementwise(blocked)",
        "convert -> dot_operand\n(shmem 往返 #2 起点)",
        "tt.dot",
    ], [1, 3], "1 次 shmem 往返"),
    ("改后(HoistLayoutConversion)", [
        "load(blocked)",
        "convert -> dot_operand\n(直接贴 load)",
        "elementwise(dot_operand)",
        "tt.dot",
    ], [1], "0 次 shmem 往返"),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 58, 30, 380, 60, 150
w = PAD * 2 + PANEL_W * 2 + 90
max_steps = max(len(p[1]) for p in PANELS)
h = TOP + max_steps * (BOX_H + VGAP) + 230

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="44" font-family="sans-serif" font-size="18.5" '
          f'font-weight="bold" fill="#0f172a">{esc("HoistLayoutConversion:把 convert 移到 load 边上,省一次 shmem 往返")}</text>')
L.append(f'<text x="{PAD}" y="68" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("同构双面板,高亮处为差异步骤——代价是 elementwise 在更多线程上重复计算,换来少一次 blocked<->dot_operand 的共享内存搬运")}</text>')

for p, (title, steps, hot_list, sub) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{cx}" y="{TOP-18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11.5" fill="#64748b">{esc(sub)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = i in hot_list
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                  f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2.4 if hl else 1}"/>')
        lines = step.split("\n")
        y0 = y + BOX_H / 2 - (len(lines) - 1) * 8
        for li, sl in enumerate(lines):
            L.append(f'<text x="{cx}" y="{y0+li*16+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" '
                      f'fill="{"#78350f" if hl else "#0f172a"}">{esc(sl)}</text>')
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

# 中间对比箭头
mid_y = TOP + 1.6 * (BOX_H + VGAP)
mid_x1 = PAD + PANEL_W + 14
mid_x2 = PAD + PANEL_W + 76
L.append(f'<line x1="{mid_x1}" y1="{mid_y}" x2="{mid_x2}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(mid_x1+mid_x2)/2}" y="{mid_y-12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#92400e">{esc("hoist")}</text>')
L.append(f'<text x="{(mid_x1+mid_x2)/2}" y="{mid_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#92400e">{esc("cap>=80")}</text>')

foot_y = TOP + max_steps * (BOX_H + VGAP) - VGAP + 46
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="130" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{foot_y+26}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">{esc("OptimizeDotOperands 的四个 pattern(OptimizeDotOperands.cpp:L323-L328)")}</text>')
PATTERNS = [
    "SwizzleShmemConvert —— 把 tt.trans 融进 swizzled 共享编码(回指 swizzle/tt.trans)",
    "HoistLayoutConversion —— 本图讲的提速主力,convert 上移贴近 load(cap>=80)",
    "FuseTransHopper —— Hopper 专属的 trans 融合变体",
    "MMAV3UseRegOperand —— MMAv3 场景下让操作数走寄存器而非共享内存的特判",
]
for i, ptxt in enumerate(PATTERNS):
    L.append(f'<text x="{PAD+20}" y="{foot_y+50+i*22}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc("- "+ptxt)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-optimize-dot-operands.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
