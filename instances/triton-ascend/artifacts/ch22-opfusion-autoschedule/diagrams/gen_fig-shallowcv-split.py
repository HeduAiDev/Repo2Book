#!/usr/bin/env python3
"""before-after 模板：ShallowCVScheduler 把一个 ShallowCV 融合核（@forward，3 层 MLP）
的 cube 段（matmul）留原核、vector 段外提为 LastAxisPBR 子核（转交 AnyPBRScheduler），
blockDim 由 40 减半为 20。只画 IR 可数的结构量——数字：3(cube段)/3(vector链)/8(vector算子总数)/
40(blockDim输入)/20(减半后)，全部来自 explainer numbers。精确外提子核个数需实跑 bishengir-opt
点名（host 无工具链），本图不断言。全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    total = 0.0
    for ch in s:
        total += size if ord(ch) > 0x2e80 else size * 0.58
    return total

TITLE = "ShallowCVScheduler：cube 段留原核，vector 段外提为子核"
SUBTITLE = "Step1 对 ShallowCV 核再跑一遍 LastAxisPBR 融合，外提纯 vector 段；cube 段留原核（ShallowCVSchedule.cpp:L40-L65）"

PAD, TOP = 46, 96
PANEL_W = 430
GAP_BETWEEN = 130

# 左面板：ShallowCV 融合核（交替 matmul / vector 链），3 层
LAYERS = [
    ("mm1", "matmul_transpose_b", ["bcast", "add", "max"]),
    ("mm2", "matmul_transpose_b", ["bcast", "add", "max"]),
    ("mm3", "matmul_transpose_b", ["bcast", "add"]),
]
CUBE_FILL, CUBE_STROKE, CUBE_TF = "#fef3c7", "#b45309", "#78350f"
VEC_FILL, VEC_STROKE, VEC_TF = "#dbeafe", "#1d4ed8", "#1e3a8a"

CUBE_H = 40
CHIP_W, CHIP_H, CHIP_GAP = 66, 32, 8
LAYER_GAP = 30

def layer_height(chips):
    return CUBE_H + 10 + CHIP_H

left_x = PAD
left_top = TOP + 40
layer_ys = []
y = left_top
for _, _, chips in LAYERS:
    layer_ys.append(y)
    y += layer_height(chips) + LAYER_GAP
left_bottom = y - LAYER_GAP

right_x = left_x + PANEL_W + GAP_BETWEEN
# 右面板：cube 子核 x3 (左列) + vector 子核 x3 (右列)，各自指向调度器
sub_h = 60
sub_gap = 22
right_top = left_top + 6
CUBE_COL_W, VEC_COL_W, COL_GAP = 180, 270, 30
cube_col_x = right_x
vec_col_x = right_x + CUBE_COL_W + COL_GAP
cube_ys = [right_top + i * (sub_h + sub_gap) for i in range(3)]
vec_ys = list(cube_ys)
right_bottom = cube_ys[-1] + sub_h

panel_bottom = max(left_bottom, right_bottom)

w = vec_col_x + VEC_COL_W + PAD
subtitle_w = PAD + text_w(SUBTITLE, 12) + PAD
w = max(w, subtitle_w)

blockdim_top = panel_bottom + 60
blockdim_h = 70
disc_y = blockdim_top + blockdim_h + 34
foot_y = disc_y + 28
h = foot_y + 26

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 面板标题
L.append(f'<text x="{left_x+PANEL_W/2}" y="{TOP+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13.5" font-weight="bold" fill="#1e293b">拆分前：ShallowCV 融合核 @forward'
          f'（blockDim 输入 = 40）</text>')
L.append(f'<text x="{(right_x+vec_col_x+VEC_COL_W)/2}" y="{TOP+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#78350f">'
          f'拆分后：3 个 cube 段留原核 + 3 条 vector 链外提（blockDim = 20）</text>')

# 左面板边框（整体虚线大框，标出"一个融合核"）
left_box_pad = 14
L.append(f'<rect x="{left_x-left_box_pad}" y="{left_top-left_box_pad}" '
          f'width="{PANEL_W+left_box_pad*2}" height="{left_bottom-left_top+left_box_pad*2}" '
          f'rx="10" fill="#fafafa" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')

CUBE_LEFT_W = 190
for i, (mid, mm_label, chips) in enumerate(LAYERS):
    ly = layer_ys[i]
    # cube 块
    L.append(f'<rect x="{left_x}" y="{ly}" width="{CUBE_LEFT_W}" height="{CUBE_H}" rx="7" '
              f'fill="{CUBE_FILL}" stroke="{CUBE_STROKE}" stroke-width="2"/>')
    L.append(f'<text x="{left_x+CUBE_LEFT_W/2}" y="{ly+CUBE_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{CUBE_TF}" '
              f'font-weight="bold">{esc(mid)}: {esc(mm_label)}</text>')
    # 箭头 cube -> 第一个 vector chip
    chain_x0 = left_x + CUBE_LEFT_W + 18
    L.append(f'<line x1="{left_x+CUBE_LEFT_W}" y1="{ly+CUBE_H/2}" x2="{chain_x0-4}" y2="{ly+CUBE_H/2}" '
              'stroke="#64748b" stroke-width="1.4" marker-end="url(#a)"/>')
    # vector 链（同一行，紧随其后）
    cy = ly + CUBE_H / 2
    for j, chip in enumerate(chips):
        cx0 = chain_x0 + j * (CHIP_W + CHIP_GAP)
        L.append(f'<rect x="{cx0}" y="{cy-CHIP_H/2}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
                  f'fill="{VEC_FILL}" stroke="{VEC_STROKE}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx0+CHIP_W/2}" y="{cy+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{VEC_TF}">{esc(chip)}</text>')
        if j < len(chips) - 1:
            nx = chain_x0 + (j + 1) * (CHIP_W + CHIP_GAP)
            L.append(f'<line x1="{cx0+CHIP_W}" y1="{cy}" x2="{nx-4}" y2="{cy}" '
                      'stroke="#64748b" stroke-width="1.2" marker-end="url(#a)"/>')
    # 该行 vector 链号标注
    L.append(f'<text x="{left_x}" y="{ly+CUBE_H+18}" font-family="sans-serif" font-size="10" '
              f'fill="#94a3b8">vector 链 {i+1}（{len(chips)} 算子）</text>')

# 右面板：cube 子核列
L.append(f'<text x="{cube_col_x+CUBE_COL_W/2}" y="{right_top-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#78350f">cube 段 ×3（留原核）</text>')
L.append(f'<text x="{vec_col_x+VEC_COL_W/2}" y="{right_top-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#1e3a8a">'
          f'vector 链 ×3（外提为 LastAxisPBR 子核）</text>')

VEC_SIZES = [3, 3, 2]
for i in range(3):
    cy0 = cube_ys[i]
    L.append(f'<rect x="{cube_col_x}" y="{cy0}" width="{CUBE_COL_W}" height="{sub_h}" rx="8" '
              f'fill="{CUBE_FILL}" stroke="{CUBE_STROKE}" stroke-width="2"/>')
    L.append(f'<text x="{cube_col_x+CUBE_COL_W/2}" y="{cy0+sub_h/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{CUBE_TF}">mm{i+1}: matmul</text>')
    L.append(f'<text x="{cube_col_x+CUBE_COL_W/2}" y="{cy0+sub_h/2+14}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#92400e">留原核，cube 路径</text>')

    vy0 = vec_ys[i]
    L.append(f'<rect x="{vec_col_x}" y="{vy0}" width="{VEC_COL_W}" height="{sub_h}" rx="8" '
              f'fill="{VEC_FILL}" stroke="{VEC_STROKE}" stroke-width="2"/>')
    L.append(f'<text x="{vec_col_x+VEC_COL_W/2}" y="{vy0+sub_h/2-4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="{VEC_TF}">vector 链{i+1}（{VEC_SIZES[i]} 算子）→ AnyPBRScheduler</text>')
    L.append(f'<text x="{vec_col_x+VEC_COL_W/2}" y="{vy0+sub_h/2+14}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#1e40af">外提子核，真正切 tile</text>')

# 中间大箭头：左面板 -> 右面板
mid_y = (left_top + left_bottom) / 2
arrow_x1 = left_x + PANEL_W + left_box_pad + 6
arrow_x2 = right_x - 10
L.append(f'<line x1="{arrow_x1}" y1="{mid_y}" x2="{arrow_x2}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{mid_y-10}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#92400e" font-weight="bold">Step1 外提</text>')

# 底部 blockDim 说明卡
bd_x0, bd_w = PAD, w - PAD * 2
L.append(f'<rect x="{bd_x0}" y="{blockdim_top}" width="{bd_w}" height="{blockdim_h}" rx="10" '
          'fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{bd_x0+bd_w/2}" y="{blockdim_top+26}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12.5" font-weight="bold" fill="#1e293b">'
          'blockDim：40 → 20（cube : vector = 1 : 2，setOptionsForFunc 对 ShallowCV 减半）</text>')
L.append(f'<text x="{bd_x0+bd_w/2}" y="{blockdim_top+48}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" fill="#475569">'
          '@forward 共 3 个 cube 段（matmul_transpose_b）+ 3 条 vector 链（8 个 vector 算子：bcast×3、add×3、max×2）</text>')

DISCLAIMER = ("免责：精确外提的 vector 子核个数需实跑 bishengir-opt 才能逐一点名（host 无工具链），"
              "本图不断言；只画 IR 里可数的结构量——3 个 cube 段 / 3 条 vector 链。")
L.append(f'<text x="{PAD}" y="{disc_y}" font-family="sans-serif" font-size="11" '
          f'fill="#b45309">{esc(DISCLAIMER)}</text>')

FOOTER = ("ShallowCVScheduler 自己不切 tile：cube 段留原核、vector 段外提转交 AnyPBRScheduler 各自调度，"
          "'双核分工' 在 pass 层的落点。")
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOTER)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-shallowcv-split.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
