#!/usr/bin/env python3
"""fig-m2-support-census — layout 模板:支持面正面清单。
顶部总览条(323 全部 .py -> 317 三子目录 + 6 其余),下方四个内容分区泳道
(tutorials/逐算子/昇腾扩展/custom_op)各挂代表性测试文件名,底部对拍容差脚注。
全部坐标由循环/常量计算,文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


TITLE = "支持面正面清单:317 个测试文件,四大内容分区"
SUBTITLE = "unittest 下共 323 个 .py;本章聚焦的三个子目录合计 317 个——每个测试都用同一 torch 表达式真机对拍过"

# 顶部总览:总数 -> 三子目录细分 + 其余
OVERVIEW_TOTAL = ("323", "unittest 下全部 .py")
OVERVIEW_SPLIT = [
    ("pytest_ut", "297"),
    ("autotune_ut", "13"),
    ("custom_op", "7"),
]
OVERVIEW_SUBTOTAL = ("317", "三子目录合计(本章聚焦)")
OVERVIEW_REST = ("6", "其余:affine_map 5 + 顶层 conftest.py 1")

# 四个内容分区(节选代表性文件名,非逐一穷举)
LANES = [
    ("tutorials 01-18", "教程范例,序号 0N/1N 即教程顺序",
     ["vector_add", "fused_softmax", "matmul", "dropout", "layer_norm",
      "fused_attention", "…共 18 篇教程"]),
    ("逐算子(按类别)", "语言层内建算子逐个对拍",
     ["math(abs/acos/exp/log…)", "reduce/scan", "dot/matmul",
      "attention", "atomic", "block_ptr/advance"]),
    ("昇腾专属扩展", "extension.* 下 NPU 专属能力逐项测",
     ["compile_hint", "sync_block(_all)", "multibuffer",
      "npu_indexing(×2)", "fixpipe", "paged_kvcache_krope",
      "barrier · makeblockptr(padding/permute)"]),
    ("custom_op 子套件(7 个文件)", "自定义算子注册全流程",
     ["custom_op_demo", "builtin_ops_demo", "…register 全流程 7 个文件"]),
]

FOOTNOTE = "判据统一:同一 torch 表达式作真值,validate_cmp/assert_close 对拍 —— 容差按 dtype 分档:fp16/bf16 rtol=atol=1e-3,fp32 rtol=atol=1e-4(test_common.py)"

# ---- 版式常量 ----
PAD = 40
W = 1360
TOP = 96

# 总览条
OV_Y = TOP
OV_BOX_H = 56
OV_GAP = 26

# lane 区
LANE_TITLE_H = 24
LANE_SUB_H = 18
CHIP_H = 30
CHIP_GAP = 10
LANE_PAD = 14
LANE_GAP = 18

L = []


def chip_row_width(chips, size=12):
    xs_ = []
    x = 0
    for c in chips:
        cw = cjk_text_width(c, size) + 24
        xs_.append((x, cw))
        x += cw + CHIP_GAP
    return xs_, x - CHIP_GAP if chips else 0


# 预计算每个 lane 的高度(chip 可能换行,这里每行放 4 个)
CHIPS_PER_ROW = 4
lane_heights = []
for _, _, chips in LANES:
    rows = (len(chips) + CHIPS_PER_ROW - 1) // CHIPS_PER_ROW
    h = LANE_PAD * 2 + LANE_TITLE_H + LANE_SUB_H + rows * (CHIP_H + CHIP_GAP) - CHIP_GAP
    lane_heights.append(h)

lanes_top = OV_Y + OV_BOX_H + 90
lanes_y = []
y = lanes_top
for h in lane_heights:
    lanes_y.append(y)
    y += h + LANE_GAP

H = y - LANE_GAP + 70  # 底部脚注留白

L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">')
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="19" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# ---- 总览条:323 -> [297,13,7]=317 + 6 ----
ov_box_w = 168
x0 = PAD
L.append(f'<rect x="{x0}" y="{OV_Y}" width="{ov_box_w}" height="{OV_BOX_H}" rx="10" '
          f'fill="#1e3a8a" stroke="#1e3a8a"/>')
L.append(f'<text x="{x0+ov_box_w/2}" y="{OV_Y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="20" font-weight="bold" fill="white">{esc(OVERVIEW_TOTAL[0])}</text>')
L.append(f'<text x="{x0+ov_box_w/2}" y="{OV_Y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#dbeafe">{esc(OVERVIEW_TOTAL[1])}</text>')

arrow_x1 = x0 + ov_box_w
arrow_x2 = arrow_x1 + 44
ay = OV_Y + OV_BOX_H / 2
L.append(f'<line x1="{arrow_x1}" y1="{ay}" x2="{arrow_x2}" y2="{ay}" stroke="#475569" '
          f'stroke-width="1.5" marker-end="url(#a)"/>')

# 三子目录小盒子(297/13/7),再合并成 317 大盒子,右侧再加 "其余 6"
sub_x = arrow_x2 + 6
sub_box_w = 118
sub_box_h = (OV_BOX_H - 12) / 3
for i, (name, val) in enumerate(OVERVIEW_SPLIT):
    sy = OV_Y + i * sub_box_h
    L.append(f'<rect x="{sub_x}" y="{sy}" width="{sub_box_w}" height="{sub_box_h-3}" rx="4" '
              f'fill="#dbeafe" stroke="#3b82f6"/>')
    L.append(f'<text x="{sub_x+8}" y="{sy+sub_box_h/2-3+2}" font-family="sans-serif" '
              f'font-size="11" fill="#1e3a8a">{esc(name)}</text>')
    L.append(f'<text x="{sub_x+sub_box_w-10}" y="{sy+sub_box_h/2-3+2}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#1e3a8a">{esc(val)}</text>')

sum_x = sub_x + sub_box_w + 20
L.append(f'<text x="{sum_x}" y="{ay-2}" font-family="sans-serif" font-size="14" '
          f'fill="#334155">=</text>')
sum2_x = sum_x + 20
L.append(f'<rect x="{sum2_x}" y="{OV_Y}" width="146" height="{OV_BOX_H}" rx="10" '
          f'fill="#eff6ff" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{sum2_x+73}" y="{OV_Y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="19" font-weight="bold" fill="#1d4ed8">{esc(OVERVIEW_SUBTOTAL[0])}</text>')
L.append(f'<text x="{sum2_x+73}" y="{OV_Y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1e40af">{esc(OVERVIEW_SUBTOTAL[1])}</text>')

plus_x = sum2_x + 146 + 16
L.append(f'<text x="{plus_x}" y="{ay-2}" font-family="sans-serif" font-size="14" '
          f'fill="#334155">+</text>')
rest_x = plus_x + 20
rest_w = W - PAD - rest_x
L.append(f'<rect x="{rest_x}" y="{OV_Y}" width="{rest_w}" height="{OV_BOX_H}" rx="10" '
          f'fill="#f8fafc" stroke="#94a3b8" stroke-dasharray="4,3"/>')
L.append(f'<text x="{rest_x+rest_w/2}" y="{OV_Y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="19" font-weight="bold" fill="#475569">{esc(OVERVIEW_REST[0])}</text>')
L.append(f'<text x="{rest_x+rest_w/2}" y="{OV_Y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc(OVERVIEW_REST[1])}</text>')

L.append(f'<text x="{PAD}" y="{lanes_top-20}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#0f172a">{esc("317 个测试落在四个内容分区(节选代表性文件名)")}</text>')

LANE_COLORS = ["#e0f2fe", "#dcfce7", "#fef3c7", "#ede9fe"]
LANE_STROKES = ["#0284c7", "#16a34a", "#d97706", "#7c3aed"]

for i, ((name, sub, chips), ly, lh, color, stroke) in enumerate(
        zip(LANES, lanes_y, lane_heights, LANE_COLORS, LANE_STROKES)):
    L.append(f'<rect x="{PAD}" y="{ly}" width="{W-2*PAD}" height="{lh}" rx="10" '
              f'fill="{color}" fill-opacity="0.35" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD+LANE_PAD}" y="{ly+LANE_PAD+14}" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{PAD+LANE_PAD}" y="{ly+LANE_PAD+14+LANE_SUB_H}" font-family="sans-serif" '
              f'font-size="11" fill="#475569">{esc(sub)}</text>')
    chip_y0 = ly + LANE_PAD + LANE_TITLE_H + LANE_SUB_H
    for j, chip in enumerate(chips):
        row = j // CHIPS_PER_ROW
        col = j % CHIPS_PER_ROW
        cw = cjk_text_width(chip, 11.5) + 26
        cx = PAD + LANE_PAD + col * ((W - 2*PAD - 2*LANE_PAD) / CHIPS_PER_ROW)
        cy = chip_y0 + row * (CHIP_H + CHIP_GAP)
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{CHIP_H}" rx="15" '
                  f'fill="white" stroke="{stroke}"/>')
        L.append(f'<text x="{cx+cw/2}" y="{cy+CHIP_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="#1e293b">{esc(chip)}</text>')

foot_y = H - 26
L.append(f'<line x1="{PAD}" y1="{foot_y-18}" x2="{W-PAD}" y2="{foot_y-18}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc(FOOTNOTE)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m2-support-census.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
