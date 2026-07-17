#!/usr/bin/env python3
"""fig-ch19-type-system: state-table 模板改造——tt 层四类类型对照表。
核心论点：tt 层张量只带 shape+dtype，encoding 栏通常为空（硬件无关的类型层证据）。
四列：类型/dump 里长成/带 encoding 吗/从哪来+定义行。宽度用 max(sans,mono) 双估算防溢出。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def char_w(c):
    o = ord(c)
    if o == 0x20:
        return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if c.isascii() and c.isalnum():
        return 0.58
    return 0.5


def text_w(s, size):
    return size * sum(char_w(c) for c in s)


def mono_w(s, size):
    return len(s) * size * 0.62


def cell_w(s, size):
    return max(text_w(s, size), mono_w(s, size))


TITLE = "tt 层类型系统 —— 四类类型都只带 shape+dtype，encoding 栏通常为空"
SUBTITLE = "这正是 tt.* 硬件无关（任何后端都认）的类型层证据；encoding 栏要到 ttg 层才被填上（前瞻 ch21/ch24）"

COLS = ["类型", "dump 里长成", "带 encoding 吗", "从哪来 / 定义"]
FONT = 12

ROWS = [
    ["标量指针 PointerType", "!tt.ptr<f32>（addressSpace=1 时省略数字）",
     "不适用（标量）", "TritonTypes.td:L53；print Types.cpp:L45-L51"],
    ["指针张量 tensor<ptr<>>", "tensor<4x!tt.ptr<f32>>",
     "否（getEncoding() 返回空）", "addptr 逐元素算术；TritonTypes.td:L80"],
    ["块指针 ptr<tensor<>>（TensorPtr）", "!tt.ptr<tensor<8x8xf16>>",
     "否", "make_tensor_ptr；isTensorPointerType Types.cpp:L178"],
    ["内存描述符 MemDescType", "memdesc<128x64xf16, #enc, #smem>",
     "有 encoding/memorySpace 栏（tt 层仍常空）", "TritonTypes.td:L96；ttg 层填共享内存布局（前瞻 ch24）"],
]
# 每行 encoding 栏的语义色：绿=空（放行）/黄=有栏但常空
ENC_STATUS = ["na", "empty", "empty", "hasfield"]
ENC_COLOR = {
    "na": ("#f1f5f9", "#64748b"),
    "empty": ("#ecfdf5", "#047857"),
    "hasfield": ("#fffbeb", "#b45309"),
}

PAD = 30
TOP = 96
HEADER_H = 34
ROW_H = 44
COL_GAP = 22
CELL_PAD = 12

col_w = []
for j in range(len(COLS)):
    max_content = max(cell_w(r[j], FONT) for r in ROWS)
    max_header = text_w(COLS[j], 12.5) + 4
    col_w.append(max(max_content, max_header) + CELL_PAD * 2)

col_x = [PAD]
for j in range(1, len(COLS)):
    col_x.append(col_x[j - 1] + col_w[j - 1] + COL_GAP)

w = col_x[-1] + col_w[-1] + PAD
h = TOP + HEADER_H + ROW_H * len(ROWS) + 56

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

for j, name in enumerate(COLS):
    cx = col_x[j]
    L.append(f'<rect x="{cx:.0f}" y="{TOP}" width="{col_w[j]:.0f}" height="{HEADER_H}" '
              'fill="#3b82f6"/>')
    L.append(f'<text x="{cx+CELL_PAD:.0f}" y="{TOP+HEADER_H/2+4.5:.0f}" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="white">{esc(name)}</text>')

table_left = col_x[0]
table_w = col_x[-1] + col_w[-1] - table_left
for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    if i % 2 == 1:
        L.append(f'<rect x="{table_left:.0f}" y="{ry:.0f}" width="{table_w:.0f}" '
                  f'height="{ROW_H}" fill="#f8fafc"/>')
    L.append(f'<line x1="{table_left:.0f}" y1="{ry:.0f}" x2="{table_left+table_w:.0f}" '
              f'y2="{ry:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
    ty = ry + ROW_H / 2 + 4.5
    # 列 0：类型名，sans 加粗深蓝
    L.append(f'<text x="{col_x[0]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#1e40af">{esc(row[0])}</text>')
    # 列 1：dump 长相，monospace
    L.append(f'<text x="{col_x[1]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="monospace" '
              f'font-size="12" fill="#0f172a">{esc(row[1])}</text>')
    # 列 2：encoding 栏，按语义色底纹
    fill, stroke = ENC_COLOR[ENC_STATUS[i]]
    L.append(f'<rect x="{col_x[2]:.0f}" y="{ry+6:.0f}" width="{col_w[2]:.0f}" '
              f'height="{ROW_H-12}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{col_x[2]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{stroke}">{esc(row[2])}</text>')
    # 列 3：从哪来/定义
    L.append(f'<text x="{col_x[3]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="sans-serif" '
              f'font-size="12" fill="#374151">{esc(row[3])}</text>')

bottom_y = TOP + HEADER_H + ROW_H * len(ROWS)
L.append(f'<line x1="{table_left:.0f}" y1="{bottom_y:.0f}" x2="{table_left+table_w:.0f}" '
          f'y2="{bottom_y:.0f}" stroke="#94a3b8" stroke-width="1.5"/>')

legend_y = bottom_y + 26
legend_items = [("empty", "encoding 为空（getEncoding() 返回空，恒放行）"),
                ("hasfield", "encoding 有栏但 tt 层仍常空（ttg 层才填）"),
                ("na", "不适用（标量，非张量）")]
lx = PAD
for tag, label in legend_items:
    fill, stroke = ENC_COLOR[tag]
    L.append(f'<rect x="{lx:.0f}" y="{legend_y-11:.0f}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{lx+22:.0f}" y="{legend_y+1:.0f}" font-family="sans-serif" '
              f'font-size="11" fill="#475569">{esc(label)}</text>')
    lx += 22 + text_w(label, 11) + 30

L.append('</svg>')
out = Path(__file__).with_name("fig-ch19-type-system.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
