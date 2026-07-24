#!/usr/bin/env python3
"""ch30-m1-inject：generate_npu_wrapper_src 是一段约 560 行的 C++ 模板 f-string，
按本次 kernel 的 metadata 三个开关现开现关六处注入点——命中的拼进去，跳过的整段留空。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "generate_npu_wrapper_src：同一模板据 metadata 开关现开现关注入点"
SUBTITLE = ("driver.py:L403-965（约 560 行 C++ 模板）—— 本例 metadata:"
            "workspace_size=256, lock_num=2, enable_taskqueue=True")

PAD = 40
W = 1180

elems = []


def add(s):
    elems.append(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---- 顶部输入框 ----
top_y = 84
top_h = 40
top_w = 560
top_x = W / 2 - top_w / 2
add(f'<rect x="{top_x:.0f}" y="{top_y}" width="{top_w}" height="{top_h}" rx="10" '
    'fill="#e0e7ff" stroke="#4338ca" stroke-width="1.5"/>')
add(f'<text x="{W/2:.0f}" y="{top_y+top_h/2+5:.0f}" text-anchor="middle" '
    'font-family="sans-serif" font-size="13.5" font-weight="bold" '
    f'fill="#312e81">{esc("generate_npu_wrapper_src(constants, signature, metadata)")}</text>')

fork_top = top_y + top_h
fork_y = fork_top + 26
add(f'<line x1="{W/2:.0f}" y1="{fork_top}" x2="{W/2:.0f}" y2="{fork_y}" '
    'stroke="#334155" stroke-width="2"/>')

# ---- 6 个条件注入点：2 行 x 3 列 ----
COLS = 3
COL_W = 340
COL_GAP = 30
ROWS_GAP = 30
total_w = COL_W * COLS + COL_GAP * (COLS - 1)
grid_x0 = W / 2 - total_w / 2
col_cx = [grid_x0 + COL_W / 2 + i * (COL_W + COL_GAP) for i in range(COLS)]

# 横向分流线
add(f'<line x1="{col_cx[0]:.0f}" y1="{fork_y}" x2="{col_cx[2]:.0f}" y2="{fork_y}" '
    'stroke="#334155" stroke-width="2"/>')

ITEMS = [
    # (row, col, 注入点, 条件, 本例取值, 命中?, 源码行)
    (0, 0, "workspace 分配段", "workspace_size > 0", "256 > 0", True, "driver.py:L773-776"),
    (0, 1, "syncBlockLock 段", "lock_num > 0", "2 > 0", True, "driver.py:L800-815"),
    (0, 2, "发射逻辑包成 lambda\n(异步)", "enable_taskqueue", "True", True, "driver.py:L777,841"),
    (1, 0, "ffts_addr 字段/取址", "target_support_ffts", "False", False, "driver.py:L795,818"),
    (1, 1, "device_print(DTData)", "enable_device_print", "False", False, "driver.py:L752,823"),
    (1, 2, "rtKernelLaunchWithFlagV2\n变体", "compile_on_910_95 and\nenable_simt", "False", False, "driver.py:L736-744"),
]

HIT_FILL, HIT_STROKE, HIT_TEXT = "#dcfce7", "#15803d", "#14532d"
SKIP_FILL, SKIP_STROKE, SKIP_TEXT = "#f1f5f9", "#94a3b8", "#475569"

# 每格高度按内容行数自算：name 行 * 17 + cond 行 * 15 + verdict(1*16) + srcline(1*14) + 上下留白
BOX_TOP_PAD = 22
GAP_NAME_COND = 6
GAP_COND_VERDICT = 8
GAP_VERDICT_SRC = 16
BOX_BOTTOM_PAD = 14


def box_height(name, cond):
    n_name = len(name.split("\n"))
    n_cond = len(cond.split("\n"))
    return (BOX_TOP_PAD + n_name * 17 + GAP_NAME_COND + n_cond * 15 +
            GAP_COND_VERDICT + 16 + GAP_VERDICT_SRC + BOX_BOTTOM_PAD)


row_items = {0: [], 1: []}
for it in ITEMS:
    row_items[it[0]].append(it)
row_h = {r: max(box_height(it[2], it[3]) for it in row_items[r]) for r in (0, 1)}

ROW1_TOP = fork_y + 24
ROW2_TOP = ROW1_TOP + row_h[0] + ROWS_GAP

box_bottoms = {}
for row, col, name, cond, val, hit, srcline in ITEMS:
    cx = col_cx[col]
    by = ROW1_TOP if row == 0 else ROW2_TOP
    bh = row_h[row]
    bx = cx - COL_W / 2
    fill, stroke, tcolor = (HIT_FILL, HIT_STROKE, HIT_TEXT) if hit else (SKIP_FILL, SKIP_STROKE, SKIP_TEXT)
    if row == 0:
        add(f'<line x1="{cx:.0f}" y1="{fork_y}" x2="{cx:.0f}" y2="{by}" '
            'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    add(f'<rect x="{bx:.0f}" y="{by}" width="{COL_W}" height="{bh}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    name_lines = name.split("\n")
    y0 = by + BOX_TOP_PAD
    for k, ln in enumerate(name_lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*17:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="13" font-weight="bold" fill="{tcolor}">{esc(ln)}</text>')
    y1 = y0 + len(name_lines) * 17 - 17 + 17 + GAP_NAME_COND
    cond_lines = cond.split("\n")
    for k, ln in enumerate(cond_lines):
        add(f'<text x="{cx:.0f}" y="{y1+k*15:.0f}" text-anchor="middle" font-family="monospace" '
            f'font-size="11" fill="#334155">{esc(ln)}</text>')
    y2 = y1 + len(cond_lines) * 15 - 15 + 15 + GAP_COND_VERDICT
    verdict = f"{val} → {'命中' if hit else '跳过'}"
    add(f'<text x="{cx:.0f}" y="{y2:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" font-weight="bold" fill="{stroke}">{esc(verdict)}</text>')
    y3 = y2 + GAP_VERDICT_SRC
    add(f'<text x="{cx:.0f}" y="{y3:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="10" fill="#64748b">{esc(srcline)}</text>')
    box_bottoms[(row, col)] = by + bh

# 行 0 -> 行 1 连线（同列直下）
for col in range(COLS):
    cx = col_cx[col]
    y_from = box_bottoms[(0, col)]
    y_to = ROW2_TOP
    add(f'<line x1="{cx:.0f}" y1="{y_from}" x2="{cx:.0f}" y2="{y_to}" '
        'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

merge_top = ROW2_TOP + row_h[1]
# 汇入下方输出框
merge_y = merge_top + 30
for col in range(COLS):
    cx = col_cx[col]
    add(f'<line x1="{cx:.0f}" y1="{merge_top}" x2="{cx:.0f}" y2="{merge_y}" '
        'stroke="#334155" stroke-width="1.5"/>')
add(f'<line x1="{col_cx[0]:.0f}" y1="{merge_y}" x2="{col_cx[2]:.0f}" y2="{merge_y}" '
    'stroke="#334155" stroke-width="2"/>')
out_top = merge_y + 26
add(f'<line x1="{W/2:.0f}" y1="{merge_y}" x2="{W/2:.0f}" y2="{out_top}" '
    'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

out_h = 56
out_w = 640
out_x = W / 2 - out_w / 2
add(f'<rect x="{out_x:.0f}" y="{out_top}" width="{out_w}" height="{out_h}" rx="10" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{W/2:.0f}" y="{out_top+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    'font-size="13" font-weight="bold" fill="#78350f">'
    f'{esc("wrapper_src —— 本 kernel 专属的 C++ 发射器源码")}</text>')
add(f'<text x="{W/2:.0f}" y="{out_top+42:.0f}" text-anchor="middle" font-family="sans-serif" '
    'font-size="11.5" fill="#92400e">'
    f'{esc("命中 3 处 / 跳过 3 处 —— 换一个 kernel 命中集不同,就是另一份专属源码")}</text>')

content_bottom = out_top + out_h

note_lines = [
    "六处注入点由 metadata 的三个开关独立决定，互不影响；本例命中 workspace/syncBlockLock/异步 lambda 三段、",
    "跳过 ffts/device_print/rtKernelLaunchWithFlagV2 变体三段——模板主体本身逐字节不变，变的只是哪几段被拼进去。",
]
note_top = content_bottom + 30
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top}" width="{W-2*PAD}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

H = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="{H:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch30-m1-inject.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={W} h={H:.0f}")
