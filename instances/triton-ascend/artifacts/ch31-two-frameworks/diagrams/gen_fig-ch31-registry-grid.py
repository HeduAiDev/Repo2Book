#!/usr/bin/env python3
"""fig-ch31-registry-grid — 两级策略注册表网格(state-table 变体)。
行 = method(15 能力,取自 traces/registry_trace.json: m1_methods_mindspore,字面量
逐字复用),列 = category(2 框架:mindspore / torch_npu)+ 1 个 ghost 列(不存在的
jax 框架)。高亮 cxx_abi 行,标注命中路径 execute_func(mindspore, cxx_abi) -> 0；
网格外挂三个 raise 出口(缺列/缺行/重复登记)+ 一张命中成功卡片,共 4 张卡片决定
画布宽度,网格居中摆放在画布内。全坐标由循环/常量计算,文本全 esc()。
(方案说明:方法名作横排行标签而非旋转列头,避免几何 linter 对 rotate() 文字宽度
估算失真导致的假阳性碰撞/越界。)
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "策略注册表:2 框架 × 15 能力 = 30 格"
SUBTITLE = "execute_func(category, method) 两次 O(1) 查表命中唯一一格；缺列/缺行/重复登记三处 fail-fast"

CATEGORIES = ["mindspore", "torch_npu"]
METHODS = [
    "version_hash", "cxx_abi", "type_convert", "get_device_interface",
    "get_empty_tensor", "get_tensor_params_shape", "get_cc_cmd",
    "get_current_device", "set_current_device", "get_current_stream",
    "header_file", "allocate_memory", "allocate_sync_block_lock",
    "pre_launch", "async_launch",
]
HIT_ROW = METHODS.index("cxx_abi")
HIT_COL = CATEGORIES.index("mindspore")

PAD = 44
LABEL_W = 190
COL_W = 150
ROW_H = 32
HEADER_H = 40
TOP = 96
GHOST_GAP = 30
GHOST_W = 130

n_rows = len(METHODS)
n_cols = len(CATEGORIES)
grid_top = TOP + HEADER_H

# 四张卡片(命中 1 + raise 3)决定画布宽度;网格另居中摆放,不必与卡片行等宽。
CARD_W = 300
CARD_GAP = 24
CALLOUT_H = 118
n_cards = 4
w = PAD * 2 + n_cards * CARD_W + (n_cards - 1) * CARD_GAP

grid_total_w = LABEL_W + n_cols * COL_W + GHOST_GAP + GHOST_W
grid_x0 = PAD + (w - 2 * PAD - grid_total_w) / 2 + LABEL_W  # 数据列起点(label 右侧)

col_x = [grid_x0 + j * COL_W for j in range(n_cols)]
row_y = [grid_top + i * ROW_H for i in range(n_rows)]
ghost_row_y = row_y[-1] + ROW_H  # 额外一行:不存在的方法 nonexistent_cap(缺行样例)

grid_right = col_x[-1] + COL_W
grid_bottom = ghost_row_y + ROW_H
ghost_x = grid_right + GHOST_GAP
ghost_cx = ghost_x + GHOST_W / 2

callout_top = grid_bottom + 56
h = callout_top + CALLOUT_H + 66

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头(category)
for j, cat in enumerate(CATEGORIES):
    cx = col_x[j] + COL_W / 2
    L.append(f'<rect x="{col_x[j] + 4}" y="{TOP + HEADER_H - 30}" width="{COL_W - 8}" '
              f'height="26" rx="4" fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.2"/>')
    L.append(f'<text x="{cx}" y="{TOP + HEADER_H - 12}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="white">{esc(cat)}</text>')

# ghost 列头(jax,虚线,不在真实注册表内)
L.append(f'<rect x="{ghost_x + 4}" y="{TOP + HEADER_H - 30}" width="{GHOST_W - 8}" '
          f'height="26" rx="4" fill="none" stroke="#b91c1c" stroke-width="1.4" '
          f'stroke-dasharray="5,4"/>')
L.append(f'<text x="{ghost_cx}" y="{TOP + HEADER_H - 12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" font-style="italic" '
          f'fill="#b91c1c">{esc("jax(不存在)")}</text>')

# 行标签 + 网格单元
for i, method in enumerate(METHODS):
    ry = row_y[i]
    hit_row = (i == HIT_ROW)
    L.append(f'<text x="{grid_x0 - 14}" y="{ry + ROW_H / 2 + 4}" text-anchor="end" '
              f'font-family="monospace" font-size="11" '
              f'font-weight="{"bold" if hit_row else "normal"}" '
              f'fill="{"#b45309" if hit_row else "#0f172a"}">{esc(method)}</text>')
    for j in range(n_cols):
        cx = col_x[j]
        hot = hit_row and (j == HIT_COL)
        also_hit_row = hit_row and not hot
        if hot:
            fill, stroke, sw = "#dcfce7", "#15803d", 2.4
        elif also_hit_row:
            fill, stroke, sw = "#fef3c7", "#b45309", 1.6
        else:
            fill, stroke, sw = "#e0f2fe", "#0369a1", 1.0
        L.append(f'<rect x="{cx + 4}" y="{ry + 3}" width="{COL_W - 8}" height="{ROW_H - 6}" '
                  f'rx="3" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        if hot:
            L.append(f'<text x="{cx + COL_W / 2}" y="{ry + ROW_H / 2 + 4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                      f'fill="#166534">return 0</text>')
    # ghost 列单元格(每行都是虚线空格,只在命中行标 ✗)
    L.append(f'<rect x="{ghost_x + 4}" y="{ry + 3}" width="{GHOST_W - 8}" height="{ROW_H - 6}" '
              f'rx="3" fill="none" stroke="#b91c1c" stroke-width="1.1" '
              f'stroke-dasharray="4,3"/>')
    if hit_row:
        L.append(f'<text x="{ghost_cx}" y="{ry + ROW_H / 2 + 4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#b91c1c">✗</text>')

# ghost 行(nonexistent_cap,缺行样例:该方法根本不存在,两框架列 + ghost 列同一行全虚线)
L.append(f'<text x="{grid_x0 - 14}" y="{ghost_row_y + ROW_H / 2 + 4}" text-anchor="end" '
          f'font-family="monospace" font-size="11" font-style="italic" font-weight="bold" '
          f'fill="#b91c1c">{esc("nonexistent_cap")}</text>')
for j in range(n_cols):
    cx = col_x[j]
    L.append(f'<rect x="{cx + 4}" y="{ghost_row_y + 3}" width="{COL_W - 8}" height="{ROW_H - 6}" '
              f'rx="3" fill="none" stroke="#b91c1c" stroke-width="1.1" stroke-dasharray="4,3"/>')
    if j == HIT_COL:
        L.append(f'<text x="{cx + COL_W / 2}" y="{ghost_row_y + ROW_H / 2 + 4}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="12" '
                  f'font-weight="bold" fill="#b91c1c">✗</text>')
L.append(f'<rect x="{ghost_x + 4}" y="{ghost_row_y + 3}" width="{GHOST_W - 8}" height="{ROW_H - 6}" '
          f'rx="3" fill="none" stroke="#b91c1c" stroke-width="1.1" stroke-dasharray="4,3"/>')

# 命中路径:入口箭头(从行标签指向命中格)
hit_cx = col_x[HIT_COL] + COL_W / 2
hit_cy = row_y[HIT_ROW] + ROW_H / 2
L.append(f'<line x1="{grid_x0 - 6}" y1="{hit_cy}" x2="{hit_cx - (COL_W / 2 - 8)}" y2="{hit_cy}" '
          f'stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')

# 四张卡片(命中成功 + 三个 raise 出口),两行文本的标题以压缩宽度
card_y = callout_top


def card(idx, color_fill, color_stroke, color_text, lines, target_xy):
    cx0 = PAD + idx * (CARD_W + CARD_GAP)
    L.append(f'<rect x="{cx0}" y="{card_y}" width="{CARD_W}" height="{CALLOUT_H}" rx="8" '
              f'fill="{color_fill}" stroke="{color_stroke}" stroke-width="1.6"/>')
    ly = card_y + 20
    for text, size, weight, color in lines:
        wattr = 'font-weight="bold" ' if weight else ''
        L.append(f'<text x="{cx0 + 12}" y="{ly}" font-family="sans-serif" font-size="{size}" '
                  f'{wattr}fill="{color}">{esc(text)}</text>')
        ly += 21
    tx, ty = target_xy
    L.append(f'<line x1="{cx0 + CARD_W / 2}" y1="{card_y}" x2="{tx}" y2="{ty}" '
              f'stroke="{color_stroke}" stroke-width="1.6" stroke-dasharray="4,3" '
              f'marker-end="url(#{"g" if color_stroke == "#15803d" else "r"})"/>')


card(0, "#f0fdf4", "#15803d", "#14532d", [
    ("命中:execute_func(mindspore, cxx_abi)", 11.5, True, "#14532d"),
    ("两级查表命中唯一一格", 11, False, "#334155"),
    ("调用 get_mindspore_cxx_abi()", 11, False, "#334155"),
    ("→ return 0", 12, True, "#15803d"),
], (hit_cx, hit_cy))

card(1, "#fef2f2", "#b91c1c", "#7f1d1d", [
    ("① execute_func(jax, cxx_abi)", 11.5, True, "#7f1d1d"),
    ("缺列(category) → raise", 11, False, "#334155"),
    ("Strategy jax not", 10.5, True, "#b91c1c"),
    ("registered", 10.5, True, "#b91c1c"),
], (ghost_cx, row_y[HIT_ROW] + ROW_H))

card(2, "#fef2f2", "#b91c1c", "#7f1d1d", [
    ("② execute_func(mindspore,", 11.5, True, "#7f1d1d"),
    ("nonexistent_cap)", 11.5, True, "#7f1d1d"),
    ("缺行(method) → raise", 11, False, "#334155"),
    ("Strategy nonexistent_cap not registered", 9.5, True, "#b91c1c"),
], (col_x[HIT_COL] + COL_W / 2, ghost_row_y + ROW_H))

card(3, "#fef2f2", "#b91c1c", "#7f1d1d", [
    ("③ 二次 @register(mindspore,", 11.5, True, "#7f1d1d"),
    ("cxx_abi)", 11.5, True, "#7f1d1d"),
    ("重复登记 → raise", 11, False, "#334155"),
    ("Strategy cxx_abi already registered", 9.5, True, "#b91c1c"),
], (hit_cx, hit_cy))

foot_y = h - 20
FOOT_TEXT = "新增框架 = 加一整列；新增能力 = 每列加一行；driver.py/utils.py 的消费点 get_backend_func(...) 一律不动。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(FOOT_TEXT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch31-registry-grid.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w:.0f}x{h:.0f})')
