#!/usr/bin/env python3
"""fig-ch27-pass-pipeline：ttir_to_linalg 把散落 ch10-24 的 add_* pass 按真实顺序串成一条
流水线，每个 pass 恰对应全书某一章——本章是这条流水线的『边框』，各专章是『拼图块』。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


TITLE = "ttir_to_linalg：把散落全书的 pass 拼成一条真实流水线"
SUBTITLE = "third_party/ascend/backend/compiler.py:L118-L157（ttadapter 段主线，前接共享的 make_ttir）"

# (short_lines[list], line_no, chapter_tag or None)
FRONT = (["make_ttir", "8 个 TTIR pass"], "L79-L86", "与基座共享")
NODES = [
    (["add_auto_blockify"], "L118", "ch15"),
    (["add_triton_to_structure", "(第 1 遍)"], "L131", "ch10-13"),
    (["add_discrete_mask_", "access_conversion"], "L136", "ch19"),
    (["add_triton_to_", "annotation"], "L142", None),
    (["add_triton_to_", "unstructure"], "L143", "ch14"),
    (["add_triton_to_hivm"], "L148", "ch23"),
    (["add_triton_to_hfusion"], "L149", "ch21"),
    (["add_triton_to_llvm"], "L150", None),
    (["add_bubble_up_", "operation"], "L151", None),
    (["add_triton_to_structure", "(第 2 遍)"], "L152", "ch10-13"),
    (["add_triton_to_linalg", "(收口)"], "L157", "ch10"),
]
ALL = [FRONT] + NODES  # 12 个节点

PER_ROW = 4
BOX_W, BOX_H = 210, 100
COL_GAP = 26
ROW_VGAP = 78
PAD = 44
TOP = 130

n_rows = (len(ALL) + PER_ROW - 1) // PER_ROW
row_w = PER_ROW * BOX_W + (PER_ROW - 1) * COL_GAP
w = PAD * 2 + row_w

elems = []


def add(s):
    elems.append(s)


def node_xy(idx):
    row = idx // PER_ROW
    col = idx % PER_ROW
    # 蛇形：偶数行从左到右，奇数行从右到左
    if row % 2 == 1:
        col = PER_ROW - 1 - col
    x = PAD + col * (BOX_W + COL_GAP)
    y = TOP + row * (BOX_H + ROW_VGAP)
    return x, y, row, col


def draw_node(idx, lines, lineno, tag, front=False):
    x, y, row, col = node_xy(idx)
    if front:
        fill, stroke, tf = "#e2e8f0", "#475569", "#334155"
    elif tag and tag.startswith("ch"):
        fill, stroke, tf = "#e0e7ff", "#4338ca", "#312e81"
    else:
        fill, stroke, tf = "#f1f5f9", "#64748b", "#334155"
    add(f'<rect x="{x:.0f}" y="{y:.0f}" width="{BOX_W}" height="{BOX_H}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    cy = y + 16
    for line in lines:
        add(f'<text x="{x+BOX_W/2:.0f}" y="{cy:.0f}" text-anchor="middle" font-family="monospace" '
            f'font-size="12.5" font-weight="bold" fill="{tf}">{esc(line)}</text>')
        cy += 18
    add(f'<text x="{x+BOX_W/2:.0f}" y="{cy+2:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11.5" fill="#475569">{esc(lineno)}</text>')
    cy += 22
    if tag:
        chip_w = cjk_w(tag, 11.5) + 20
        chip_fill = "#c7d2fe" if tag.startswith("ch") else "#e2e8f0"
        chip_stroke = "#4338ca" if tag.startswith("ch") else "#64748b"
        chip_tf = "#312e81" if tag.startswith("ch") else "#334155"
        add(f'<rect x="{x+BOX_W/2-chip_w/2:.0f}" y="{cy-14:.0f}" width="{chip_w:.0f}" height="20" rx="10" '
            f'fill="{chip_fill}" stroke="{chip_stroke}" stroke-width="1"/>')
        add(f'<text x="{x+BOX_W/2:.0f}" y="{cy:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11" font-weight="bold" fill="{chip_tf}">{esc(tag)}</text>')
    return x, y


for i, (lines, lineno, tag) in enumerate(ALL):
    draw_node(i, lines, lineno, tag, front=(i == 0))

# 连接箭头：同行内左右相邻；行末→下一行首（竖直转折）
for i in range(len(ALL) - 1):
    x1, y1, row1, col1 = node_xy(i)
    x2, y2, row2, col2 = node_xy(i + 1)
    cy = y1 + BOX_H / 2
    if row1 == row2:
        if col2 > col1:  # 从左到右
            add(f'<line x1="{x1+BOX_W:.0f}" y1="{cy:.0f}" x2="{x2:.0f}" y2="{cy:.0f}" '
                'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
        else:  # 从右到左（蛇形折返行）
            add(f'<line x1="{x1:.0f}" y1="{cy:.0f}" x2="{x2+BOX_W:.0f}" y2="{cy:.0f}" '
                'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
    else:
        # 蛇形折返：行末列与下一行首列在同一 x（node_xy 已按 col 镜像对齐），直接竖直连
        xc1 = x1 + BOX_W / 2
        xc2 = x2 + BOX_W / 2
        add(f'<line x1="{xc1:.0f}" y1="{y1+BOX_H:.0f}" x2="{xc2:.0f}" y2="{y2:.0f}" '
            'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

content_bottom = TOP + n_rows * BOX_H + (n_rows - 1) * ROW_VGAP

# 头部计数横幅
count_y = TOP - 42
add(f'<rect x="{PAD}" y="{count_y:.0f}" width="{row_w:.0f}" height="30" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
count_text = "第一段 make_ttir：8 pass（L79-L86，与基座共享） ｜ 第二段 ttadapter 主线：11 个 add_*（L118-L157）"
add(f'<text x="{PAD+row_w/2:.0f}" y="{count_y+20:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#1e3a5f">{esc(count_text)}</text>')

note_lines = [
    "同一 pass（add_triton_to_structure）在链中运行两遍：第 1 遍先把能结构化的指针访存变结构化，",
    "中间经 unstructure/hivm/hfusion/llvm 等 pass 暴露新结构后，第 2 遍再收一轮，最后 triton_to_linalg 收口。",
]
note_top = content_bottom + 30
note_h = 22 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{row_w:.0f}" height="{note_h}" rx="8" '
    'fill="#f0fdf4" stroke="#86efac"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#14532d">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-ch27-pass-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f} rows={n_rows}")
