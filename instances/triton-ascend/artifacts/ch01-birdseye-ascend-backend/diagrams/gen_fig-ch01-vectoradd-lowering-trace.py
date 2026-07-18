#!/usr/bin/env python3
"""fig-ch01-vectoradd-lowering-trace — tensor-flow 模板（flow 骨架 + 每条边标标量）。
同一份 add_kernel（256 元素 / BLOCK_SIZE=64 → 4 个 program）从前端 JIT 走三段下降，
最终落在达芬奇 vector 核。各阶段需真机才能真正跑出 IR dump，本图按 pin 源码结构
推演，标注"需真机"角标。全部坐标由循环/常量计算。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STAGES = [
    ("前端 JIT", ["add_kernel", "pid/offsets/mask"], "01-vector-add.py:L50-L75", True),
    ("段 1：ttir", ["make_ttir", "通用 TTIR pass"], "compiler.py:L941", True),
    ("段 2：ttadapter", ["ttir_to_linalg", "指针 → memref"], "compiler.py:L949", True),
    ("段 3：npubin", ["bishengir-compile", "落 VECTOR 核"], "compiler.py:L959", True),
]
EDGE_LABELS = [
    "device='npu'\n（唯一改动）",
    "256/64=4\nprogram",
    "3 段下降\n（结构化）",
]

BOX_W, BOX_H, GAP, PAD, TOP = 250, 100, 120, 40, 138
n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = TOP + BOX_H + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("同一份 add_kernel：256 元素 / BLOCK_SIZE=64 → 4 个 program，三段下降到 vector 核")}</text>']

xs_ = [PAD + i * (BOX_W + GAP) for i in range(n)]
cy = TOP + BOX_H / 2

# ── 第一遍：全部方框（先画完，箭头/标签留到第二遍画在上层，避免被后画的方框盖住）──
for i, (name, lines, loc, need_hw) in enumerate(STAGES):
    x = xs_[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{BOX_W}" height="{BOX_H}" rx="10" '
             f'fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14.5" font-weight="bold" fill="#1e3a8a">{esc(name)}</text>')
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+BOX_W/2}" y="{TOP+46+k*17}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12" fill="#1e3a8a">{esc(ln)}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{TOP+BOX_H-10}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#2563eb" font-weight="bold">{esc(loc)}</text>')
    if need_hw:
        badge_w = 62
        bx = x + BOX_W/2 - badge_w/2
        by = TOP - 32
        L.append(f'<rect x="{bx}" y="{by}" width="{badge_w}" height="18" rx="9" fill="#fee2e2" '
                 f'stroke="#dc2626" stroke-width="1.2"/>')
        L.append(f'<text x="{bx+badge_w/2}" y="{by+13}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="10" font-weight="bold" fill="#b91c1c">{esc("需真机")}</text>')

# ── 第二遍：箭头 + 边标签（画在方框之上，且标签宽度已按 GAP 预留够）──
for i in range(n - 1):
    x1, x2 = xs_[i] + BOX_W, xs_[i+1]
    L.append(f'<line x1="{x1+4}" y1="{cy}" x2="{x2-4}" y2="{cy}" '
             f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    lbl = EDGE_LABELS[i].split("\n")
    ly0 = cy + 22
    for k, ln in enumerate(lbl):
        L.append(f'<text x="{(x1+x2)/2}" y="{ly0+k*14}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="11" fill="#0369a1" font-weight="bold">{esc(ln)}</text>')

# 底部：核体不变声明 + 基座对照
note_y = TOP + BOX_H + 46
L.append(f'<rect x="{PAD}" y="{note_y}" width="{w-2*PAD}" height="86" rx="8" '
         f'fill="#f0fdf4" stroke="#16a34a" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+16}" y="{note_y+24}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc("add_kernel 核体逐字不变（tl.program_id/tl.arange/tl.load/tl.store 一个不少）；")}</text>')
DEVICE_NOTE = "host 侧唯一改动是张量 device='npu'，编译分叉发生在 supports_target 认领 npu 之后的 add_stages 内部。"
L.append(f'<text x="{PAD+16}" y="{note_y+44}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc(DEVICE_NOTE)}</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+64}" font-family="sans-serif" font-size="12" '
         f'fill="#166534">{esc("对照基座：同一核在 GPU 路走 5 段落到 SIMT 线程；本例走 3 段落到达芬奇 VECTOR 核（非 CUBE，因是 elementwise）。")}</text>')

foot_y = note_y + 86 + 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc("host 无 NPU/CANN/bishengir-compile，各级 IR dump 标『需真机』，由 illustrator/tester 在真机或容器补抓。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch01-vectoradd-lowering-trace.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
