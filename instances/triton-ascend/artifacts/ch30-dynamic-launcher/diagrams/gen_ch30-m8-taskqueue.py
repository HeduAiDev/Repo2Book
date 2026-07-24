#!/usr/bin/env python3
"""ch30-m8-taskqueue：enable_taskqueue(默认真) 决定发射收尾——真则包成 lambda 异步入队，
host 不阻塞；假则末尾 rtStreamSynchronize 同步等待。before-after 模板，同构双面板。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "enable_taskqueue 决定发射收尾：异步入队 vs 同步阻塞"
SUBTITLE = "driver.py:L512-513 —— TRITON_ENABLE_TASKQUEUE 默认 'true'"

PANELS = [
    ("enable_taskqueue = False（同步）", [
        "发射逻辑直接内联执行", "（不包 lambda）",
        "末尾 rtStreamSynchronize(stream)",
        "host 阻塞，等 kernel 跑完才返回",
    ], [2]),
    ("enable_taskqueue = True（默认，异步）", [
        "整段发射逻辑包成",
        "auto launch_call = [=]() -> rtError_t",
        "async_launch(launch_call) 入队",
        "host 不阻塞，立即返回",
    ], [1, 2]),
]

BOX_W, VGAP, PANEL_W, PAD, TOP = 420, 22, 460, 40, 116
LINE_H = 20

def box_height(lines):
    return 26 + LINE_H * len(lines)

w = PAD * 2 + PANEL_W * 2 + 90
elems = []
def add(s): elems.append(s)

add('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')

panel_tops = []
panel_bottoms = []
for p, (title, steps, hot_idxs) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 90)
    cx = px + PANEL_W / 2
    add(f'<text x="{cx:.0f}" y="{TOP-38:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="14.5" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    y = TOP
    ys = []
    # 把每个 step 当作一行文本框（部分是多行合并的语义块）
    STEP_BLOCKS = {
        0: [["发射逻辑直接内联执行", "（不包 lambda）"], ["rtStreamSynchronize(stream)", "driver.py:L839"],
            ["host 阻塞：等 kernel 跑完才返回"]],
        1: [["整段发射逻辑包成 lambda：", "auto launch_call = [=]() -> rtError_t", "driver.py:L777"],
            ["async_launch(launch_call) 入队", "driver.py:L841"],
            ["host 不阻塞：立即返回，不等 kernel"]],
    }
    blocks = STEP_BLOCKS[p]
    HOT = {0: [1], 1: [0, 1]}[p]
    for i, block_lines in enumerate(blocks):
        bh = box_height(block_lines)
        ys.append((y, bh))
        hl = i in HOT
        fill = "#fef3c7" if hl else "#e2e8f0"
        stroke = "#d97706" if hl else "#64748b"
        add(f'<rect x="{cx-BOX_W/2:.0f}" y="{y:.0f}" width="{BOX_W}" height="{bh}" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hl else 1}"/>')
        y0 = y + 22
        for k, ln in enumerate(block_lines):
            fw = 'font-weight="bold" ' if k == 0 else ''
            fam = "monospace" if k > 0 and ("(" in ln or "L7" in ln or "L8" in ln) else "sans-serif"
            fs = 12.5 if k == 0 else 11
            add(f'<text x="{cx:.0f}" y="{y0+k*LINE_H:.0f}" text-anchor="middle" '
                f'font-family="{fam}" font-size="{fs}" {fw}fill="#0f172a">{esc(ln)}</text>')
        y += bh + VGAP
        if i < len(blocks) - 1:
            add(f'<line x1="{cx:.0f}" y1="{y-VGAP:.0f}" x2="{cx:.0f}" y2="{y-4:.0f}" '
                'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    panel_tops.append(TOP)
    panel_bottoms.append(y - VGAP)

content_bottom = max(panel_bottoms)

# 中间分隔箭头(双向对比提示)
midy = TOP + (content_bottom - TOP) / 2
gap_x1 = PAD + PANEL_W + 15
gap_x2 = PAD + PANEL_W + 75
add(f'<line x1="{gap_x1}" y1="{midy:.0f}" x2="{gap_x2}" y2="{midy:.0f}" '
    'stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,4"/>')
add(f'<text x="{(gap_x1+gap_x2)/2:.0f}" y="{midy-10:.0f}" text-anchor="middle" font-family="sans-serif" '
    'font-size="11" fill="#64748b">默认值切换</text>')

note_lines = [
    "torch_npu 后端的 async_launch 实现为 OpCommand.SetCustomHandler(func).Run()（backend_register.py:L335-336）；",
    "两种收尾方式共享同一段发射逻辑（rtKernelLaunch 等），仅是否包 lambda 异步入队不同——这是昇腾相对基座 CUDA launcher 多出的第二样。",
]
note_top = content_bottom + 30
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h:.0f}">',
     f'<rect width="{w}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch30-m8-taskqueue.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w} h={h:.0f}")
