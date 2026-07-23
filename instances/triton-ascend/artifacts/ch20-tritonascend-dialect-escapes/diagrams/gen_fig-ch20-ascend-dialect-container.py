#!/usr/bin/env python3
"""fig-ch20-ascend-dialect-container: state-table 模板（用作数据目录表）。
ascend 方言定义 11 个 op（TritonAscendOps.td），本章三条逃生舱只消费其中 2 个
（ascend.mod → HFusion 舱；ascend.custom → HIVM 舱），其余 9 个在别处被消费。
表格 4 列 x 11 行，命中本章的 2 行高亮为绿色，其余灰色。
全坐标由循环/常量计算，列宽按内容自动估算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def text_width(s, font_size=12):
    """粗略估算文本像素宽：CJK 字符按 font_size*1.05，ASCII 按 font_size*0.58。"""
    w = 0.0
    for ch in s:
        if ord(ch) > 0x2E80:
            w += font_size * 1.05
        else:
            w += font_size * 0.58
    return w


TITLE = "TritonAscend 方言：11 个 op 的语义容器（TritonAscendOps.td）"
SUBTITLE = "本章三条逃生舱只消费其中 2 个（ascend.mod → HFusion 舱；ascend.custom → HIVM 舱），其余 9 个在别处被消费"

COLUMNS = ["IR 名（ascend.<助记符>）", "C++ 类（triton::ascend::）", "承载的 NPU 语义", "本章三舱是否消费 / 消费方"]

ROWS = [
    ["ascend.annotation", "AnnotationOp", "给张量挂 key-value 编译期标注", "否（主链 annotation 相关 pass）"],
    ["ascend.mod", "ModOp", "逐元素取余（%）硬件二元算子", "是 → HFusion 舱"],
    ["ascend.index_put", "IndexPutOp", "embedding 语义的 scatter 写回 GM", "否"],
    ["ascend.gather_out_to_ub", "GatherOutToUbOp", "GM→UB 按维 gather（带越界处理）", "否"],
    ["ascend.scatter_ub_to_out", "ScatterUbToOutOp", "UB→GM 按维 scatter", "否"],
    ["ascend.index_select_simd", "IndexSelectSimdOp", "GM 按索引 SIMD 选取直落 UB", "否"],
    ["ascend.indirect_load", "IndirectLoadOp", "编译器内建：逐元素偏移的离散 load（ch19 发射）", "否（ch19 离散掩码）"],
    ["ascend.indirect_store", "IndirectStoreOp", "编译器内建：逐元素偏移的离散 store（ch19 发射）", "否（ch19 离散掩码）"],
    ["ascend.custom", "CustomOp", "自定义 op 载体（本章承载 sync_block_* 双核同步）", "是 → HIVM 舱"],
    ["ascend.flip", "FlipOp", "沿维反转张量", "否"],
    ["ascend.sort", "SortOp", "沿维排序张量", "否"],
]

CONSUMED_ROWS = {1, 8}  # ascend.mod, ascend.custom（0-indexed）

FONT_SIZE = 12
CELL_PAD_X = 14
COL_PAD = 26  # 列间额外留白

col_w = []
for j, header in enumerate(COLUMNS):
    max_w = text_width(header, 12.5)
    for row in ROWS:
        max_w = max(max_w, text_width(row[j], FONT_SIZE))
    col_w.append(max_w + CELL_PAD_X * 2 + COL_PAD)

PAD, TOP, HEADER_H, ROW_H = 36, 96, 40, 34
w = PAD * 2 + sum(col_w)
h = TOP + HEADER_H + ROW_H * len(ROWS) + 96

col_x = [PAD]
for cw in col_w[:-1]:
    col_x.append(col_x[-1] + cw)

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 表头
for j, header in enumerate(COLUMNS):
    x, cw = col_x[j], col_w[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{cw-4}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(cw-4)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(header)}</text>')

# 数据行
for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    consumed = i in CONSUMED_ROWS
    row_fill = "#dcfce7" if consumed else ("#f8fafc" if i % 2 == 0 else "#ffffff")
    row_stroke = "#15803d" if consumed else "#e2e8f0"
    L.append(f'<rect x="{PAD}" y="{ry+2}" width="{sum(col_w)-4}" height="{ROW_H-4}" '
              f'fill="{row_fill}" stroke="{row_stroke}" stroke-width="{2 if consumed else 1}"/>')
    for j, cell in enumerate(row):
        x, cw = col_x[j], col_w[j]
        text_fill = "#14532d" if consumed else "#334155"
        weight = 'font-weight="bold" ' if (consumed and j in (0, 3)) else ''
        L.append(f'<text x="{x+CELL_PAD_X}" y="{ry+ROW_H/2+4}" '
                  f'font-family="sans-serif" font-size="{FONT_SIZE}" fill="{text_fill}" '
                  f'{weight}>{esc(cell)}</text>')

# 图例
legend_y = TOP + HEADER_H + ROW_H * len(ROWS) + 26
L.append(f'<rect x="{PAD}" y="{legend_y-13}" width="16" height="16" rx="3" '
          'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{PAD+22}" y="{legend_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">绿色 = 本章三条逃生舱消费的 op（2/11）；其余灰白行在别处（主链其它 pass / ch18 / ch19）被消费</text>')

# 底部数字小结（逐条对 spec.numbers）
foot_y = legend_y + 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#0f172a" font-weight="bold">'
          f'{esc("方言前缀 = “ascend”（TritonAscendDialect.td:L15 let name）；方言 op 总数 = 11；本章三舱消费的方言 op 数 = 2")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch20-ascend-dialect-container.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size={w}x{h}")
