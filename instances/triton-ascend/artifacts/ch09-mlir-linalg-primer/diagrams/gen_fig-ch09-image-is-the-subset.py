#!/usr/bin/env python3
"""fig-ch09-image-is-the-subset — m15 求像即子集。
上:迭代域 w 轴,高亮当前 tile[0,8) 与相邻 tile[8,16)。
下:三条张量带 O/I/K 各自的像;I 的像比 tile 宽出 2(halo,斜纹标注,与相邻 tile 的像 [8,18) 重叠)；
K 的像与空间 tile 无关,单独一块不随 x 轴缩放。
底部新增「halo 代价」标注(数字逐字取自 explainer m15 的 quantified 字段 / chapter.md 同段落)。

[FIX-ROUND-2](2026-07-21,盲审+Lead 复核后修复):
  1. lint_diagrams.py 报 overflow BLOCKING——根因不是标题真的出界,是标题里为规避
     rsvg-convert 把「量」字加粗渲成实心块而插入的 `<tspan font-weight="normal">`
     补丁,被 linter 的正则当成普通文字字符计入估算宽度(标签字符也按 0.55×fs 计),
     把估算宽度撑高了整整一个 tspan 标签的长度。不改 linter(exp-0713-3 的规矩),
     改用与 linter **完全相同的保守公式**(CJK 正则+权重)在生成脚本里对
     esc_bold(TITLE)/esc(SUBTITLE) 反向算出所需宽度,取 max 后再加安全边界定宽——
     以后标题再改,宽度会自动跟着算,不会重蹈覆辙。
  2. 画布下方约 35% 空白——h 的计算沿用了旧版「3 条张量带」的公式常数
     (BAND_H+BAND_GAP)*3,但 bands 列表早已只剩 O/I 两条(K 带另画在右侧),
     多算的一整条带的高度全变成了空白。改用 `(BAND_H+BAND_GAP)*len(bands)` 按
     实际条数算,省下的高度用来放新增的「halo 代价」标注,不再单纯裁掉留白。
"""
import re
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def esc_bold(s):
    # rsvg-convert/Droid Sans Fallback 在 font-weight="bold" 下把「量」(U+91CF)
    # 误渲染成实心方块(逐次重渲复现,与字号/字体族无关)；用 tspan 把该字降回
    # normal 权重规避——此字体的中文本就不随 bold 变粗,视觉零回归。
    return esc(s).replace('量', '<tspan font-weight="normal">量</tspan>')


# 与 scripts/lint_diagrams.py 的 overflow 估算算法逐字对齐(CJK 正则+权重),
# 用来反向解出「这行文字(含 esc_bold 插入的 tspan 标签字符)至少需要多宽的画布」——
# 不改 linter,而是让生成脚本对齐 linter 的度量口径,一次性根治、非一次性加宽。
_LINT_CJK = re.compile(r'[㐀-䶿一-鿿]')


def lint_conservative_width(content_as_rendered: str, font_size: float) -> float:
    stripped = re.sub(r'\s', '', content_as_rendered)
    return sum(1.0 if _LINT_CJK.match(c) else 0.55 for c in stripped) * font_size


TITLE = "求像即子集:把切过的迭代域代进索引表达式,就得到每个张量要碰的那一片"
SUBTITLE = "tile 宽 8、核宽 3 时,输入 I 的像比 tile 宽出 2——这 2 列与下一块 tile 的像 [8,18) 重叠(halo)"
TITLE_FS, SUBTITLE_FS = 16.5, 12

UNIT = 30          # 1 个 w 单位 = 30px(示意,非等比于论文 988)
PAD = 44
DOMAIN_MAX = 18     # 轴画到 18 够放下 [8,18)
AXIS_TOP = 110
BAND_H = 46
BAND_GAP = 34

bands = [
    ("O", "n,w,f", 0, 8, None, "O 的像 = tile 本身  [0,8)"),
    ("I", "n,w+kw,c", 0, 10, (8, 10), "I 的像宽 10 = tile 宽 8 + (核宽 3 − 1)"),
]

# halo 代价标注——数字逐字取自 explainer.json m15.quantified 与 chapter.md 同段落
# (「每块 I 侧多读 2 列 = 25% 的额外读入（10 对 8）；若 tile 宽取 4，多读比例升到
#  50%（6 对 4）」),不是本脚本新算的数。
HALO_COST_TITLE = "halo 这笔账"
HALO_COST_LINES = [
    "tile 宽 8:I 侧多读 2 列 = 10 对 8,多读 25%",
    "tile 宽 4:多读比例升到 50%(6 对 4)——tile 越小,halo 占比越大",
]


def X(v):
    return PAD + v * UNIT


# ---- 宽度:两路取最大——版面本身的几何宽度,与「标题/副标题按 linter 口径至少需要
# 多宽」——保证 overflow 检查(right > vbw+2)不会因为 tspan 补丁的标签字符被误计而误报。
w_geometry = PAD * 2 + DOMAIN_MAX * UNIT + 260   # 右侧留给 K 带
_title_need = PAD + lint_conservative_width(esc_bold(TITLE), TITLE_FS) + PAD
_subtitle_need = PAD + lint_conservative_width(esc(SUBTITLE), SUBTITLE_FS) + PAD
SAFETY_MARGIN = 24  # 额外缓冲,覆盖 linter "> vbw+2" 的 2px 容差与浮点误差
w = max(w_geometry, _title_need + SAFETY_MARGIN, _subtitle_need + SAFETY_MARGIN)

# ---- 高度:按 bands 实际条数算(不再沿用「3 条带」的旧常数),省下的高度分给
# 新增的 halo 代价标注区,而不是单纯留白。
axis_y = AXIS_TOP
band_y0 = axis_y + 70
last_band_bottom = band_y0 + (len(bands) - 1) * (BAND_H + BAND_GAP) + BAND_H
halo_box_bottom = last_band_bottom + 6  # halo 虚线框比带本身上下各多探 6px

ANNOT_GAP_ABOVE = 22
ANNOT_PAD = 12
ANNOT_TITLE_H = 18
ANNOT_LINE_H = 16
ANNOT_H = ANNOT_PAD * 2 + ANNOT_TITLE_H + ANNOT_LINE_H * len(HALO_COST_LINES)
annot_y0 = halo_box_bottom + ANNOT_GAP_ABOVE

FOOT_GAP = 22
FOOT_H = 20
h = annot_y0 + ANNOT_H + FOOT_GAP + FOOT_H

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0f172a"/></marker>'
          '<pattern id="halo" width="7" height="7" patternTransform="rotate(45)" '
          'patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#fed7aa"/>'
          '<line x1="0" y1="0" x2="0" y2="7" stroke="#c2410c" stroke-width="2.4"/></pattern>'
          '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD - 6}" font-family="sans-serif" font-size="{TITLE_FS}" '
          f'font-weight="bold" fill="#0f172a">{esc_bold(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD + 16}" font-family="sans-serif" font-size="{SUBTITLE_FS}" '
          f'fill="#475569">{esc(SUBTITLE)}</text>')

# ---- 迭代域 w 轴 ----
L.append(f'<text x="{PAD}" y="{axis_y - 10}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#334155">{esc("迭代域(w 方向)")}</text>')
L.append(f'<line x1="{X(0)}" y1="{axis_y + 20}" x2="{X(16)}" y2="{axis_y + 20}" '
          f'stroke="#94a3b8" stroke-width="1.5"/>')
# tile0 [0,8) 当前块
L.append(f'<rect x="{X(0)}" y="{axis_y}" width="{X(8) - X(0)}" height="30" rx="4" '
          f'fill="#3b82f6" stroke="#1e3a8a" stroke-width="1.5"/>')
L.append(f'<text x="{(X(0) + X(8)) / 2}" y="{axis_y + 20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
          f'fill="white">{esc("tile 0  [0,8)")}</text>')
# tile1 [8,16) 相邻块
L.append(f'<rect x="{X(8)}" y="{axis_y}" width="{X(16) - X(8)}" height="30" rx="4" '
          f'fill="#dbeafe" stroke="#93c5fd" stroke-width="1.5" stroke-dasharray="4,3"/>')
L.append(f'<text x="{(X(8) + X(16)) / 2}" y="{axis_y + 20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="#1e40af">{esc("tile 1  [8,16)")}</text>')
for tick in [0, 8, 16]:
    L.append(f'<line x1="{X(tick)}" y1="{axis_y + 20}" x2="{X(tick)}" y2="{axis_y + 26}" '
              f'stroke="#64748b" stroke-width="1.2"/>')
    L.append(f'<text x="{X(tick)}" y="{axis_y + 40}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#64748b">{tick}</text>')

# ---- 三条张量带 ----
for bi, (name, idxexpr, lo, hi, halo, note) in enumerate(bands):
    by = band_y0 + bi * (BAND_H + BAND_GAP)
    # 张量底带(示意全长,浅灰)
    L.append(f'<rect x="{X(0)}" y="{by}" width="{X(DOMAIN_MAX) - X(0)}" height="{BAND_H}" rx="4" '
              f'fill="#f1f5f9" stroke="#cbd5e1"/>')
    # 高亮的像区间
    fill = "#93c5fd" if name == "O" else "#fdba74"
    stroke = "#1d4ed8" if name == "O" else "#c2410c"
    core_hi = halo[0] if halo else hi
    L.append(f'<rect x="{X(lo)}" y="{by}" width="{X(core_hi) - X(lo)}" height="{BAND_H}" rx="4" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    if halo:
        L.append(f'<rect x="{X(halo[0])}" y="{by}" width="{X(halo[1]) - X(halo[0])}" height="{BAND_H}" '
                  f'fill="url(#halo)" stroke="{stroke}" stroke-width="1.8"/>')
        # 相邻 tile 的像 [8,18) 虚线框,示意重叠
        L.append(f'<rect x="{X(8)}" y="{by - 6}" width="{X(18) - X(8)}" height="{BAND_H + 12}" rx="4" '
                  f'fill="none" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="4,3"/>')
        L.append(f'<text x="{X(18) + 6}" y="{by + BAND_H / 2 + 4}" font-family="sans-serif" font-size="10.5" '
                  f'fill="#64748b">{esc("← 相邻 tile 的像 [8,18)")}</text>')
    L.append(f'<text x="{X(lo) + 8}" y="{by + BAND_H / 2 + 5}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}   '
              f'{esc("索引: " + idxexpr)}</text>')
    L.append(f'<text x="{PAD}" y="{by - 8}" font-family="sans-serif" font-size="11.5" '
              f'fill="#475569">{esc(note)}</text>')
    # 从迭代域指向本带的箭头(纯视觉引导,无文字,避免跨带穿插造成遮挡)
    # O 走带内左侧短箭头;I 走左侧留白通道下探,不穿过 O 带矩形
    ax = X(0) + 14 if name == "O" else X(0) - 20
    ay_top = axis_y + 50
    ay_bot = by - 4
    L.append(f'<line x1="{ax}" y1="{ay_top}" x2="{ax}" y2="{ay_bot}" '
              f'stroke="#0f172a" stroke-width="1.3" marker-end="url(#a)" opacity="0.5"/>')

# ---- K 带(不随空间 tile 变,单独放右侧) ----
k_x = X(DOMAIN_MAX) + 30
k_y = band_y0 + BAND_H / 2 - 20
k_w, k_h = 190, 40
L.append(f'<rect x="{k_x}" y="{k_y}" width="{k_w}" height="{k_h}" rx="6" '
          f'fill="#bbf7d0" stroke="#15803d" stroke-width="1.8"/>')
L.append(f'<text x="{k_x + k_w / 2}" y="{k_y + 18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#14532d">{esc("K 的像 = 完整 K")}</text>')
L.append(f'<text x="{k_x + k_w / 2}" y="{k_y + 34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#166534">{esc("索引: kw,c,f  —— 不随空间 tile 变")}</text>')
# 起点选在 tile 0 色块内部(靠右上,但明显落在 [0,8) 区间中段,不贴在 x=8 的
# tile0/tile1 分界线上),避免看起来像从 tile 1 发出
k_arrow_x0 = X(6)
L.append(f'<line x1="{k_arrow_x0}" y1="{axis_y}" x2="{k_x + k_w / 2}" y2="{k_y}" '
          f'stroke="#0f172a" stroke-width="1.1" marker-end="url(#a)" opacity="0.4" stroke-dasharray="3,3"/>')

# ---- halo 代价标注(填补此前的下方留白;数字逐字取自正文/explainer m15) ----
annot_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{annot_y0:.1f}" width="{annot_w:.1f}" height="{ANNOT_H:.1f}" rx="8" '
          f'fill="#fff7ed" stroke="#c2410c" stroke-width="1.4"/>')
L.append(f'<text x="{PAD + ANNOT_PAD}" y="{annot_y0 + ANNOT_PAD + 12:.1f}" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#9a3412">{esc(HALO_COST_TITLE)}</text>')
for li, line in enumerate(HALO_COST_LINES):
    ly = annot_y0 + ANNOT_PAD + ANNOT_TITLE_H + 11 + li * ANNOT_LINE_H
    L.append(f'<text x="{PAD + ANNOT_PAD}" y="{ly:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="#7c2d12">{esc(line)}</text>')

foot_y = h - 16
L.append(f'<text x="{PAD}" y="{foot_y:.1f}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("核宽 kw=3;halo 多出 2 列;数据:本章参考实现实测(tile 宽 8 的论文形状用例)")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-image-is-the-subset.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f})")
