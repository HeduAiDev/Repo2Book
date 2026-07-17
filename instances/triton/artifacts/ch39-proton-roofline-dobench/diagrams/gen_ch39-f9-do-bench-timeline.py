#!/usr/bin/env python3
"""ch39-f9-do-bench-timeline: do_bench 五段计时协议(swimlane 改,单泳道时间轴)。
claim: 触发编译→估时→定 warmup/repeat→预热→逐轮 zero L2 + CUDA event 打点。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

STAGES = [
    ("① 触发编译", ["fn() + di.synchronize()", "JIT 编译 + 懒初始化(不计时)"]),
    ("② 估时", ["跑 5 次(elapsed/5)", "得 estimate_ms(testing.py:L131)"]),
    ("③ 定次数", ["n_warmup=max(1,⌊25/est⌋)", "n_repeat=max(1,⌊100/est⌋)"]),
    ("④ 预热", ["跑 n_warmup 次", "进稳态,不计时"]),
    ("⑤ 正式计时", ["每轮 cache.zero_() 冲 L2", "→ 该轮专属 CUDA event 打点"]),
]
STAGE_W, STAGE_H, GAP, PAD, TOP = 250, 84, 30, 40, 190
w = PAD * 2 + len(STAGES) * STAGE_W + (len(STAGES) - 1) * GAP
h = TOP + STAGE_H + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">do_bench 五段计时协议</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'python/triton/testing.py —— 默认 warmup=25ms / rep=100ms(L95)</text>']

# top annotation box: default budgets + L2 buffer + estimate loop count
L.append(f'<rect x="{PAD}" y="72" width="{w-2*PAD}" height="66" rx="8" fill="#fef9c3" stroke="#ca8a04"/>')
L.append(f'<text x="{PAD+16}" y="96" font-family="sans-serif" font-size="12" font-weight="bold" '
         f'fill="#854d0e">默认 warmup/rep 预算 = 25ms / 100ms;估时循环 = 5 次;'
         f'L2 冲刷缓冲 = 256MB int 数组(device=cuda)</text>')
L.append(f'<text x="{PAD+16}" y="118" font-family="sans-serif" font-size="11" '
         f'fill="#a16207">estimate=0.2ms 时:n_warmup=125,n_repeat=500(python/triton/testing.py:L134-L135)——'
         f'快 kernel 自动多测摊平抖动。</text>')

x = PAD
y = TOP
box_xs = []
for i, (title, lines) in enumerate(STAGES):
    fill, stroke, tc = "#e0f2fe", "#0369a1", "#0c4a6e"
    if i == 4:
        fill, stroke, tc = "#dcfce7", "#15803d", "#14532d"
    L.append(f'<rect x="{x}" y="{y}" width="{STAGE_W}" height="{STAGE_H}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{x+STAGE_W/2}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13.5" font-weight="bold" fill="{tc}">{esc(title)}</text>')
    y0 = y + 42
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+STAGE_W/2}" y="{y0+k*18}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="10.8" fill="{tc}">{esc(line)}</text>')
    box_xs.append(x)
    x += STAGE_W + GAP

for i in range(len(STAGES) - 1):
    x1 = box_xs[i] + STAGE_W
    x2 = box_xs[i+1]
    ay = y + STAGE_H/2
    L.append(f'<line x1="{x1}" y1="{ay}" x2="{x2}" y2="{ay}" stroke="#334155" '
             f'stroke-width="2" marker-end="url(#a)"/>')

# loop annotation on stage 5 (repeat n_repeat times): self-loop arc, both ends on box's bottom edge
p1x, p2x = box_xs[4] + STAGE_W*0.32, box_xs[4] + STAGE_W*0.68
p1y = y + STAGE_H
L.append(f'<path d="M {p1x},{p1y} C {p1x-10},{p1y+55} {p2x+10},{p1y+55} {p2x},{p1y}" '
         f'fill="none" stroke="#15803d" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{box_xs[4]+STAGE_W/2}" y="{p1y+75}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#15803d">循环 n_repeat 次</text>')

foot_y = h - 60
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'循环结束后一次 synchronize 批量读回各段耗时;CUDA event 是 GPU 端时间戳,避开 '
         f'host-device 同步误差(driver.py:L480-L481 给出 L2 冲刷缓冲大小)。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'第一段(触发编译)与第二段(估时)不进最终统计;真正的计时只发生在第五段。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f9-do-bench-timeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
