#!/usr/bin/env python3
"""fig-mmav3-operand-to-shared (before-after 模板)
MMAv2 把 A/B 转 DotOperandEncodingAttr(寄存器)发 DotOp;
MMAv3(Hopper WGMMA)把 A/B 过 getSharedMemoryMMAOperand 搬进 SharedEncodingAttr(共享内存)
发 WarpGroupDotOp,warp 最小单元变 (4,1) = 一个 warpgroup。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PANELS = [
    ("MMAv2(Ampere/Turing)", [
        "A / B(tt.dot 操作数)",
        "DotOperandEncodingAttr\n(寄存器 fragment,逐 lane)",
        "DotOp",
    ], None, "versionMajor == 2"),
    ("MMAv3(Hopper WGMMA)", [
        "A / B(tt.dot 操作数)",
        "getSharedMemoryMMAOperand\n-> SharedEncodingAttr(共享内存)",
        "WarpGroupDotOp",
    ], 1, "versionMajor == 3  (AccelerateMatmul.cpp:L313)"),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 56, 40, 380, 60, 130
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

L.append(f'<text x="{PAD}" y="44" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="#0f172a">{esc("Hopper 换了 fragment 契约:操作数从寄存器搬进共享内存")}</text>')
L.append(f'<text x="{PAD}" y="66" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc("同构双面板,仅差异处高亮——WGMMA 是异步 warpgroup(4 warps)级指令,直接从 shared 读操作数")}</text>')

for p, (title, steps, hot, sub) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-38}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{cx}" y="{TOP-18}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" fill="#64748b">{esc(sub)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        hl = (i == hot)
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
last_step_i = len(PANELS[0][1]) - 1
midy = TOP + last_step_i * (BOX_H + VGAP) / 2
mid_x1 = PAD + PANEL_W + 14
mid_x2 = PAD + PANEL_W + 76
L.append(f'<line x1="{mid_x1}" y1="{midy}" x2="{mid_x2}" y2="{midy}" '
         'stroke="#d97706" stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(mid_x1+mid_x2)/2}" y="{midy-12}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#92400e">{esc("代际升级")}</text>')

# 底部:warp 最小单元对比 + 回指
foot_y = TOP + len(PANELS[0][1]) * (BOX_H + VGAP) - VGAP + 50
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="72" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+20}" y="{foot_y+26}" font-family="sans-serif" font-size="12.5" '
         f'fill="#0f172a">{esc("warp 最小不可分单元:MMAv2 可到单 warp;MMAv3 是 (4,1) = 一个 warpgroup(4 warps)")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+46}" font-family="sans-serif" font-size="12.5" '
         f'fill="#0f172a">{esc("(warpsPerTileV3, AccelerateMatmul.cpp:L119-L120;操作数几乎总在 shared, .td:L1312-1313)")}</text>')
L.append(f'<text x="{PAD+20}" y="{foot_y+66}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("本图只讲『为什么操作数在 shared』的契约根因;WarpGroupDotOp 的异步流水回指第 24 章")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-mmav3-operand-to-shared.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w}x{h}")
