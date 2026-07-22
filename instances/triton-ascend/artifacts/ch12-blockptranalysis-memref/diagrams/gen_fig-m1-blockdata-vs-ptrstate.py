#!/usr/bin/env python3
"""fig-m1-blockdata-vs-ptrstate: BlockData 是 ch11 PtrState 在 TritonToLinalg
侧的镜像状态类（state-table 模板改写为 2 列对照表）。
左列=ch11 PtrState（背景，仅作对比，不重复讲）；右列=本章 BlockData（高亮，spec.numbers 逐条落地）。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "BlockData 是 PtrState 在 TritonToLinalg 侧的镜像状态类"
SUBTITLE = "同样的 offsets/sizes/strides 三元组，载体从分析期整型升级为 OpFoldResult，source 从 tt.ptr 升级为 memref，并新增 MemAccType 决策位"

COL_LEFT = "PtrState（ch11，TritonToStructured）"
COL_RIGHT = "BlockData（ch12，TritonToLinalg）"

ROWS = [
    ("offsets/sizes/strides\n载体",
     "分析期整型 Value\n（标量代数）",
     "OpFoldResult\n（可静可动）",
     "third_party/ascend/include/TritonToLinalg/BlockPtrAnalysis.h BlockData 定义"),
    ("source 类型",
     "tt.ptr\n（指针，未物化）",
     "已是 memref\n（非 tt.ptr）",
     "BlockPtrAnalysis.cpp:L157 getResultMemrefType 取 source 的 BaseMemRefType"),
    ("新增字段",
     "（无此字段）",
     "scalar / resElemTy\n/ memAccTy",
     "dossier key_classes BlockData responsibility"),
]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 400, 84, 40, 118, 34
n_rows = len(ROWS)
w = PAD * 2 + LABEL_W + COL_W * 2
h = TOP + HEADER_H + ROW_H * n_rows + PAD + 34
col_x = [PAD + LABEL_W, PAD + LABEL_W + COL_W]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-4}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+18}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
L.append(f'<rect x="{col_x[0]}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="4" '
          'fill="#94a3b8" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{col_x[0]+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">{esc(COL_LEFT)}</text>')
L.append(f'<rect x="{col_x[1]}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="4" '
          'fill="#3b82f6" stroke="#1e3a8a" stroke-width="1.5"/>')
L.append(f'<text x="{col_x[1]+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="white" font-weight="bold">{esc(COL_RIGHT)}</text>')

row_y0 = TOP + HEADER_H
for i, (label, left_val, right_val, prov) in enumerate(ROWS):
    ry = row_y0 + i * ROW_H
    # 行标签
    lines = label.split("\n")
    n = len(lines)
    ly0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{ly0+k*16}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="#374151">{esc(line)}</text>')
    # 左列（背景灰，仅对比）
    lx = col_x[0]
    L.append(f'<rect x="{lx}" y="{ry+6}" width="{COL_W-10}" height="{ROW_H-12}" rx="6" '
              'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5"/>')
    llines = left_val.split("\n")
    ln = len(llines)
    ly = ry + ROW_H / 2 - (ln - 1) * 9 + 4
    for k, line in enumerate(llines):
        L.append(f'<text x="{lx+(COL_W-10)/2}" y="{ly+k*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#475569">{esc(line)}</text>')
    # 右列（高亮蓝，本章内容）
    rx = col_x[1]
    L.append(f'<rect x="{rx}" y="{ry+6}" width="{COL_W-10}" height="{ROW_H-12}" rx="6" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="2"/>')
    rlines = right_val.split("\n")
    rn = len(rlines)
    ryy = ry + ROW_H / 2 - (rn - 1) * 9 + 4
    for k, line in enumerate(rlines):
        L.append(f'<text x="{rx+(COL_W-10)/2}" y="{ryy+k*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#1e3a8a">{esc(line)}</text>')
    # 迁移箭头（左→右）
    amy = ry + ROW_H / 2
    L.append(f'<line x1="{lx+COL_W-10}" y1="{amy}" x2="{rx-4}" y2="{amy}" '
              'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

foot_y = h - PAD + 10
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
          f'fill="#94a3b8">右列数字出处：BlockPtrAnalysis.h/.cpp 类定义与字段声明（逐条见图注 provenance）</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-blockdata-vs-ptrstate.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
