#!/usr/bin/env python3
"""layout 模板(定制):driver 边界一跳。上层 Python 运行时四步横排,
中间粗虚线标『driver 边界』,下层 driver 子系统(灰显,标见第十二章)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TOP_STEPS = [
    ("get_current_device", "device=0", True),
    ("get_current_stream", "stream", True),
    ("get_current_target", "target=cuda sm=80\nwarp=32", True),
    ("make_backend(target)", "-> CUDABackend\n(headless 可跑)", False),
]
BOT_STEPS = [
    "GPUDriver.active",
    "torch.cuda\n.current_device",
    "raw CUDA\nstream",
    "GPUTarget",
]

CELL_W, CELL_H, GAP, PAD = 220, 92, 30, 40
n = len(TOP_STEPS)
w = PAD * 2 + n * CELL_W + (n - 1) * GAP
h = 500

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="38" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("driver 边界一跳：run() 开场四问，只有一道门后是真 GPU")}</text>')

TOP_Y = 96
L.append(f'<rect x="{PAD-16}" y="{TOP_Y-40}" width="{w-2*(PAD-16)}" height="{CELL_H+70}" rx="10" '
          'fill="#eff6ff" stroke="#93c5fd" stroke-dasharray="3,3"/>')
L.append(f'<text x="{PAD}" y="{TOP_Y-16}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1d4ed8">{esc("Python 运行时（jit.py run 开场）")}</text>')

xs_top = []
for i, (name, detail, needs_real) in enumerate(TOP_STEPS):
    x = PAD + i * (CELL_W + GAP)
    xs_top.append(x)
    stroke = "#b91c1c" if needs_real else "#1d4ed8"
    fill = "#fee2e2" if needs_real else "#dbeafe"
    L.append(f'<rect x="{x}" y="{TOP_Y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{x+CELL_W/2}" y="{TOP_Y+24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    for k, line in enumerate(detail.split("\n")):
        L.append(f'<text x="{x+CELL_W/2}" y="{TOP_Y+46+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#334155">{esc(line)}</text>')
    if needs_real:
        L.append(f'<text x="{x+CELL_W-10}" y="{TOP_Y+16}" text-anchor="end" font-family="sans-serif" '
                  f'font-size="13" fill="#b91c1c">{esc("⚡")}</text>')
    if i < n - 1:
        mx = x + CELL_W + GAP / 2
        L.append(f'<line x1="{x+CELL_W}" y1="{TOP_Y+CELL_H/2}" x2="{x+CELL_W+GAP-4}" y2="{TOP_Y+CELL_H/2}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

boundary_y = TOP_Y + CELL_H + 66
L.append(f'<line x1="{PAD-20}" y1="{boundary_y}" x2="{w-PAD+20}" y2="{boundary_y}" '
          'stroke="#b91c1c" stroke-width="3" stroke-dasharray="10,6"/>')
L.append(f'<text x="{w/2}" y="{boundary_y-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#b91c1c">'
          f'{esc("driver 边界（host 无 GPU 在此断裂 ⚡）")}</text>')

BOT_Y = boundary_y + 46
L.append(f'<rect x="{PAD-16}" y="{BOT_Y-36}" width="{w-2*(PAD-16)}" height="{CELL_H+56}" rx="10" '
          'fill="#f1f5f9" stroke="#94a3b8"/>')
L.append(f'<text x="{PAD}" y="{BOT_Y-12}" font-family="sans-serif" font-size="13" font-weight="bold" '
          f'fill="#64748b">{esc("driver 子系统（见第十二章，本章不展开）")}</text>')

for i, name in enumerate(BOT_STEPS):
    x = xs_top[i]
    L.append(f'<rect x="{x}" y="{BOT_Y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
              'fill="#e2e8f0" stroke="#94a3b8" stroke-dasharray="3,3"/>')
    for k, line in enumerate(name.split("\n")):
        cy = BOT_Y + CELL_H/2 - (len(name.split("\n"))-1)*8 + k*16
        L.append(f'<text x="{x+CELL_W/2}" y="{cy}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" fill="#64748b">{esc(line)}</text>')
    if i < 3:
        L.append(f'<line x1="{x+CELL_W}" y1="{BOT_Y+CELL_H/2}" x2="{x+CELL_W+GAP-4}" y2="{BOT_Y+CELL_H/2}" '
                  'stroke="#94a3b8" stroke-width="1.4" stroke-dasharray="3,3" marker-end="url(#a)"/>')
    # 竖向连接线(上层 -> 下层),仅前三列(需真设备的三问)
    if i < 3:
        L.append(f'<line x1="{x+CELL_W/2}" y1="{TOP_Y+CELL_H}" x2="{x+CELL_W/2}" y2="{BOT_Y}" '
                  'stroke="#b91c1c" stroke-width="1.4" stroke-dasharray="4,3" marker-end="url(#a)"/>')

foot_y = h - 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("make_backend(target) 只按 target 选后端，headless 可跑；三次 get_current_* 桥到 torch.cuda，需真设备。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("本书用 stub（FakeDriver）顶替这道门，才让后面真编译在 host 上跑起来。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-driver-boundary.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
