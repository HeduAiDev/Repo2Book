#!/usr/bin/env python3
"""fig-ch01-three-stage-lowering — flow 模板。
AscendBackend.add_stages 只登记 3 段：ttir(941)→ttadapter(949)→npubin(959，默认
A2_A3 分支)。另画一条 force_simt_only 快路径虚线旁路（ttir 直出 npubin）与
910_95 芯片条件变体注记。坐标全部由循环/常量计算。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STAGES = [
    ("ttir", ["stages['ttir'] = make_ttir"], "compiler.py:L941"),
    ("ttadapter", ["stages['ttadapter'] = ttir_to_linalg"], "compiler.py:L949"),
    ("npubin", ["stages['npubin'] =", "linalg_to_bin_enable_npu_compile_A2_A3"], "compiler.py:L959（默认 else 分支）"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 320, 108, 60, 40, 150
n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = TOP + BOX_H + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("三段结构化下降链：AscendBackend.add_stages 只挂 3 段")}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(n)]
cy = TOP + BOX_H / 2

for i, (name, fn_lines, loc) in enumerate(STAGES):
    x = xs_[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
             f'fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+26}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="15" font-weight="bold" fill="#1e3a8a">{esc(f"段 {i+1}：{name}")}</text>')
    for k, ln in enumerate(fn_lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+48+k*16}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="11" fill="#1e3a8a">{esc(ln)}</text>')
    loc_y = TOP + 48 + len(fn_lines) * 16 + 12
    L.append(f'<text x="{x+BOX_W/2}" y="{loc_y}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#2563eb" font-weight="bold">{esc(loc)}</text>')
    if i < n - 1:
        x1 = x + BOX_W
        x2 = xs_[i+1]
        L.append(f'<line x1="{x1+4}" y1="{cy}" x2="{x2-4}" y2="{cy}" '
                 f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# force_simt_only 快路径：ttir 直出 npubin（虚线弧线旁路，走在盒子上方，与标题错开）
arc_apex = TOP - 74
x_start = xs_[0] + BOX_W / 2
x_end = xs_[2] + BOX_W / 2
L.append(f'<path d="M {x_start} {TOP} Q {(x_start+x_end)/2} {arc_apex} {x_end} {TOP}" '
         f'fill="none" stroke="#d97706" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#b)"/>')
L.append(f'<text x="{(x_start+x_end)/2}" y="{arc_apex-10}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#b45309">{esc("force_simt_only=True 快路径：ttir_to_npubin 直通（compiler.py:L943）")}</text>')

# 底部注记：910_95 变体 + 对照基座
note_y = TOP + BOX_H + 50
L.append(f'<rect x="{PAD}" y="{note_y}" width="{w-2*PAD}" height="70" rx="8" '
         f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{note_y+22}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("默认走 A2_A3 分支（else，L959）；compile_on_910_95=True 时走 910_95 芯片条件变体")}</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+40}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("linalg_to_bin_enable_npu_compile_910_95（L953）。npubin 一段吞掉基座 llir/ptx/cubin 三段。")}</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+58}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc("对照基座 GPU 路：5 段 ttir/ttgir/llir/ptx/cubin（nvidia compiler.py:L385-L389）——详见后一张图。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-three-stage-lowering.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
