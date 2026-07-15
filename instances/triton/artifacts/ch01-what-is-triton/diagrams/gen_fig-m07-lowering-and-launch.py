#!/usr/bin/env python3
"""fig-m07-lowering-and-launch: flow 模板——编译段(上排)与发射段(下排)是同一条
链的前后半;缓存命中则从入口短路直达发射段首站。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

COMPILE_ROW = [
    ["JITFunction.run", "缓存未命中"],
    ["ASTSource.make_ir(L273)", "产追踪期 TTIR", "dump 循环(L278-292)之外—从不落盘"],
    ["五级降级", "ttir→ttgir→llir→ptx→cubin", "首 pass=add_inliner(tt.call 在此消失)"],
]
LAUNCH_ROW = [
    ["CompiledKernel.__getattribute__('run')", "劫持首次访问", "→ _init_handles → launcher_cls"],
    ["make_launcher", "现场生成 C 源码"],
    ["compile_module_from_src", "现编 .so → dlopen"],
    ["发射 cubin"],
]

BOX_W, BOX_H, HGAP, ROW_GAP, PAD, TOP = 250, 74, 46, 96, 40, 90

def row_positions(n, box_w, hgap, pad):
    xs_ = [pad + i * (box_w + hgap) for i in range(n)]
    return xs_

compile_xs = row_positions(len(COMPILE_ROW), BOX_W, HGAP, PAD)
launch_xs = row_positions(len(LAUNCH_ROW), BOX_W, HGAP, PAD)
w = max(compile_xs[-1] + BOX_W, launch_xs[-1] + BOX_W) + PAD

y_compile = TOP
y_launch = TOP + BOX_H + ROW_GAP
h = y_launch + BOX_H + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
          '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#1e293b">@triton.jit 到 GPU 发射:编译段与发射段是同一条链的前后半</text>')

L.append(f'<text x="{PAD}" y="{y_compile-14}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#1d4ed8">编译段</text>')
L.append(f'<text x="{PAD}" y="{y_launch-14}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#047857">发射段</text>')

def draw_box(x, y, w_, h_, lines, fill, stroke):
    out = [f'<rect x="{x}" y="{y}" width="{w_}" height="{h_}" rx="8" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    n = len(lines)
    cy = y + h_ / 2
    y0 = cy - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if k == 0 else ''
        fs = 12 if k == 0 else 10.5
        out.append(f'<text x="{x+w_/2}" y="{y0+k*15}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{fs}" fill="#0f172a" {fw}>{esc(line)}</text>')
    return out

for i, lines in enumerate(COMPILE_ROW):
    L += draw_box(compile_xs[i], y_compile, BOX_W, BOX_H, lines, "#eff6ff", "#1d4ed8")
    if i < len(COMPILE_ROW) - 1:
        x1 = compile_xs[i] + BOX_W
        x2 = compile_xs[i+1]
        cy = y_compile + BOX_H / 2
        L.append(f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" '
                  'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>')

for i, lines in enumerate(LAUNCH_ROW):
    L += draw_box(launch_xs[i], y_launch, BOX_W, BOX_H, lines, "#ecfdf5", "#047857")
    if i < len(LAUNCH_ROW) - 1:
        x1 = launch_xs[i] + BOX_W
        x2 = launch_xs[i+1]
        cy = y_launch + BOX_H / 2
        L.append(f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" '
                  'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>')

# 编译段末尾 -> 发射段首(常规链路)
lx = compile_xs[-1] + BOX_W / 2
ly1 = y_compile + BOX_H
ly2 = y_launch
L.append(f'<line x1="{lx}" y1="{ly1}" x2="{launch_xs[0]+BOX_W/2}" y2="{ly2}" '
          'stroke="#475569" stroke-width="1.5" marker-end="url(#a)"/>')

# 缓存命中短路:入口直达发射段首站
sx = compile_xs[0] + BOX_W / 2
L.append(f'<path d="M {sx} {y_compile} C {sx-140} {y_compile+40}, '
          f'{launch_xs[0]+BOX_W/2-90} {y_launch-40}, {launch_xs[0]+BOX_W/2-20} {y_launch}" '
          'fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="7,5" marker-end="url(#b)"/>')
L.append(f'<text x="{sx-150}" y="{(y_compile+y_launch)/2}" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#a16207">缓存命中:短路</text>')

# 底部命题条
foot_y = y_launch + BOX_H + 40
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{w-2*PAD}" height="40" rx="6" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y+25}" font-family="sans-serif" font-size="11.5" '
          f'fill="#1e293b">命题(a):整条链从不调用用户的 Python 函数体(run 只 compile + kernel.run,jit.py:L638-L655)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m07-lowering-and-launch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
