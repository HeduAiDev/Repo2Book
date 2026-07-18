#!/usr/bin/env python3
"""fig-m1-two-line-diff: before-after 模板变体——三行(改动点位),
左列=基座 triton(GPU),右列=triton-ascend(NPU)。每行按 kind 高亮
(新增/改字符串/逐字节相同),行间箭头带徽标标注改动量。
全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "GPU → NPU 最小改写：全部改动只有两处"
COL_TITLES = ("基座 triton（GPU）", "triton-ascend（NPU）")

# 每行: site 标签(读者向,不印行号——精确行号由正文代码块引用承担),
# before/after 代码行, 徽标文字, is_new(新增行加粗), 高亮色三元组(fill, stroke, text)
ROWS = [
    {
        "site": "import 段",
        "before": ["import torch"],
        "after": ["import torch", "import torch_npu"],
        "badge": "+1 行 import",
        "is_new": True,
        "fill": "#dcfce7", "stroke": "#16a34a", "text": "#166534",
    },
    {
        "site": "调用侧 device",
        "before": [
            "x = torch.rand(size, device='cuda')",
            "y = torch.rand(size, device='cuda')",
        ],
        "after": [
            "x = torch.rand(size, device='npu')",
            "y = torch.rand(size, device='npu')",
        ],
        "badge": "改 2 处 device",
        "is_new": False,
        "fill": "#ffedd5", "stroke": "#ea580c", "text": "#9a3412",
    },
    {
        "site": "add_kernel 内核体",
        "before": [
            "@triton.jit",
            "def add_kernel(...):",
            "    pid = tl.program_id(axis=0)",
            "    ... load / add / store ...",
        ],
        "after": [
            "@triton.jit",
            "def add_kernel(...):",
            "    pid = tl.program_id(axis=0)",
            "    ... load / add / store ...",
        ],
        "badge": "0 处改动 · 逐字节相同",
        "is_new": False,
        "fill": "#dbeafe", "stroke": "#3b82f6", "text": "#1e40af",
    },
]

CAPTION_LINES = [
    "add_kernel 内核体逐字节不变；GPU→NPU 只新增 1 行 import、改 2 处 device 字符串。",
    "三级 tiling / 物理核绑定 / compile_hint 都是「跑通之后的优化」——vector-add 里根本没有。",
]

BOX_W = 320
LINE_H = 17
CODE_FS = 12.5
PAD_IN = 12          # 框内上下左右留白
SITE_TAG_H = 20      # 行首 site 标签高度
ROW_VGAP = 30
GAP_MID = 190        # 两列之间留给箭头+徽标
PAD = 40
TOP = 96             # 标题 + 列标题所占高度
BEFORE_FILL, BEFORE_STROKE, BEFORE_TEXT = "#f1f5f9", "#94a3b8", "#334155"

def row_height(row):
    n = max(len(row["before"]), len(row["after"]))
    return SITE_TAG_H + PAD_IN * 2 + n * LINE_H

row_heights = [row_height(r) for r in ROWS]
row_ys = []
y = TOP
for rh in row_heights:
    row_ys.append(y)
    y += rh + ROW_VGAP
content_bottom = y - ROW_VGAP

W = PAD * 2 + BOX_W * 2 + GAP_MID
H = content_bottom + 20 + LINE_H * len(CAPTION_LINES) + 16

col_x = (PAD, PAD + BOX_W + GAP_MID)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="arrow" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>']

# 标题
L.append(f'<text x="{W/2}" y="{PAD-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="18" font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')

# 列标题
for cx, title in zip(col_x, COL_TITLES):
    L.append(f'<text x="{cx+BOX_W/2}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e40af">{esc(title)}</text>')

for row, ry, rh in zip(ROWS, row_ys, row_heights):
    # site 标签(挂在左列框上方)
    L.append(f'<text x="{col_x[0]}" y="{ry-6}" font-family="sans-serif" font-size="11.5" '
              f'fill="#64748b">{esc(row["site"])}</text>')
    box_h = rh - SITE_TAG_H
    box_y = ry + SITE_TAG_H - 8
    # 左列(基座,恒定中性色)
    L.append(f'<rect x="{col_x[0]}" y="{box_y}" width="{BOX_W}" height="{box_h}" rx="8" '
              f'fill="{BEFORE_FILL}" stroke="{BEFORE_STROKE}" stroke-width="1.5"/>')
    for i, line in enumerate(row["before"]):
        ly = box_y + PAD_IN + (i + 0.8) * LINE_H
        L.append(f'<text x="{col_x[0]+PAD_IN}" y="{ly}" font-family="monospace" '
                  f'font-size="{CODE_FS}" fill="{BEFORE_TEXT}">{esc(line)}</text>')
    # 右列(按 kind 高亮)
    L.append(f'<rect x="{col_x[1]}" y="{box_y}" width="{BOX_W}" height="{box_h}" rx="8" '
              f'fill="{row["fill"]}" stroke="{row["stroke"]}" stroke-width="2"/>')
    for i, line in enumerate(row["after"]):
        is_new = row["is_new"] and i == 1  # 新增的那一行加粗强调
        weight = "bold" if is_new else "normal"
        ly = box_y + PAD_IN + (i + 0.8) * LINE_H
        L.append(f'<text x="{col_x[1]+PAD_IN}" y="{ly}" font-family="monospace" '
                  f'font-size="{CODE_FS}" font-weight="{weight}" '
                  f'fill="{row["text"]}">{esc(line)}</text>')
    # 行间箭头 + 徽标
    mid_y = box_y + box_h / 2
    ax1, ax2 = col_x[0] + BOX_W + 6, col_x[1] - 6
    L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2-4}" y2="{mid_y}" '
              f'stroke="{row["stroke"]}" stroke-width="2" marker-end="url(#arrow)"/>')
    L.append(f'<rect x="{ax1+4}" y="{mid_y-24}" width="{GAP_MID-20}" height="18" rx="9" '
              f'fill="white" stroke="{row["stroke"]}" stroke-width="1.2"/>')
    L.append(f'<text x="{ax1+4+(GAP_MID-20)/2}" y="{mid_y-11}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="{row["text"]}">{esc(row["badge"])}</text>')

# 图注
cap_y0 = content_bottom + 24
for i, line in enumerate(CAPTION_LINES):
    L.append(f'<text x="{W/2}" y="{cap_y0 + i*LINE_H}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#475569">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-two-line-diff.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
