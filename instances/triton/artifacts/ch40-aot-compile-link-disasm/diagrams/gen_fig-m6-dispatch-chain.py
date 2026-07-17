#!/usr/bin/env python3
"""fig-m6-dispatch-chain: flow 模板。
link.py 按 num_specs 降序生成整除性分派链:先试约束最强的特化,退而恒真兜底,
皆不中报 CUDA_ERROR_INVALID_VALUE——ch14 compute_spec_key 的 C 化身。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
BOX_W = 640
BRANCH_W = 300
GAP = 30
TOP = 108
LEFT_MARGIN = 40  # 主干左移,给右侧终止分支留空间

TITLE = "整除性分派链:link.py 按 num_specs 降序生成 C 的 if 链(ch14 compute_spec_key 的 C 化身)"
SUBTITLE = "python/triton/tools/link.py —— 2 份特化归入一组 add_1024_warps4xstages3"

x_main = PAD + LEFT_MARGIN
x_center = x_main + BOX_W / 2
branch_x = x_main + BOX_W + 170
branch_cx = branch_x + BRANCH_W / 2

elems = []


def add(s):
    elems.append(s)


def main_box(y, lines, fill, stroke, text_fill, bold_last=False):
    n = len(lines)
    box_h = 26 + 22 * (n - 1) + 34
    add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    for k, line in enumerate(lines):
        add(f'<text x="{x_center:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="12.5" fill="{text_fill}">{esc(line)}</text>')
    return box_h


def side_box(y, lines, fill, stroke, text_fill):
    n = len(lines)
    box_h = 26 + 20 * (n - 1) + 30
    add(f'<rect x="{branch_x:.0f}" y="{y:.0f}" width="{BRANCH_W}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    for k, line in enumerate(lines):
        yy = y + 22 + k * 20
        add(f'<text x="{branch_cx:.0f}" y="{yy:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="11.5" fill="{text_fill}">{esc(line)}</text>')
    return box_h


def vline(x, y1, y2, color="#334155", dash=False):
    dash_attr = ' stroke-dasharray="4,3"' if dash else ''
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2"{dash_attr} marker-end="url(#a)"/>')


y = TOP
# 起点
bh = main_box(y, ["运行期调用:add(stream, X, Y, N)"], "#e2e8f0", "#64748b", "#0f172a")
y += bh

# --- 判定 1:最特化(012d, num_specs=1) ---
vline(x_center, y, y + GAP)
y += GAP
judge1_h = main_box(y, ['if ((N % 16 == 0))  —  特化 012d, num_specs=1(约束最强,最先测)'],
                     "#fef3c7", "#b45309", "#78350f")
judge1_bottom = y + judge1_h
judge1_mid = y + judge1_h / 2

# 右分支(是 → 命中,终止)
add(f'<line x1="{x_main+BOX_W:.0f}" y1="{judge1_mid:.0f}" x2="{branch_x:.0f}" y2="{judge1_mid:.0f}" '
    'stroke="#15803d" stroke-width="2" marker-end="url(#a)"/>')
add(f'<text x="{(x_main+BOX_W+branch_x)/2:.0f}" y="{judge1_mid-10:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">是 → 命中</text>')
side_box(judge1_mid - 27, ["return add_deadbeef_012d(", "  stream, X, Y, N);"],
         "#dcfce7", "#15803d", "#166534")

# 下分支(否 → 继续)
y = judge1_bottom
vline(x_center, y, y + GAP)
add(f'<text x="{x_center+16:.0f}" y="{y+GAP-8:.0f}" font-family="sans-serif" font-size="11.5" '
    f'font-weight="bold" fill="#64748b">否(fallthrough)</text>')
y += GAP

# --- 判定 2:恒真兜底(012, num_specs=0) ---
judge2_h = main_box(y, ['if (1)  —  恒真兜底 · 特化 012, num_specs=0'],
                     "#fef3c7", "#b45309", "#78350f")
judge2_bottom = y + judge2_h
judge2_mid = y + judge2_h / 2

add(f'<line x1="{x_main+BOX_W:.0f}" y1="{judge2_mid:.0f}" x2="{branch_x:.0f}" y2="{judge2_mid:.0f}" '
    'stroke="#15803d" stroke-width="2" marker-end="url(#a)"/>')
add(f'<text x="{(x_main+BOX_W+branch_x)/2:.0f}" y="{judge2_mid-10:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#15803d">是(恒真)→ 命中</text>')
side_box(judge2_mid - 27, ["return add_cafef00d_012(", "  stream, X, Y, N);"],
         "#dcfce7", "#15803d", "#166534")

# 下分支(理论否 → 兜底耗尽)
y = judge2_bottom
vline(x_center, y, y + GAP, color="#94a3b8", dash=True)
add(f'<text x="{x_center+16:.0f}" y="{y+GAP-8:.0f}" font-family="sans-serif" font-size="11" '
    f'fill="#94a3b8">否(仅无兜底组时可达)</text>')
y += GAP

# --- 兜底耗尽 ---
final_h = main_box(y, ["return CUDA_ERROR_INVALID_VALUE;"], "#fee2e2", "#b91c1c", "#7f1d1d")
bottom_y = y + final_h

w = branch_x + BRANCH_W + PAD

note_lines = [
    "2 份特化(012d / 012)归入同一组 add_1024_warps4xstages3;metas 按 -num_specs 降序排列,",
    "保证约束最强(通常最快)的先测;每条路径必 return,链尾兜底耗尽才报 CUDA_ERROR_INVALID_VALUE。",
]
note_top = bottom_y + 34
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m6-dispatch-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
