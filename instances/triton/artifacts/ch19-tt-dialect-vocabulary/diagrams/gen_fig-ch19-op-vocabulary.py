#!/usr/bin/env python3
"""fig-ch19-op-vocabulary: state-table 模板改造——16 个常用 tt.* 算子的
识字词汇表。三列：算子(带 .td 行号)/结构要点(读脸关键)/dump 里长成(样例)。
左对齐、单行一算子、斑马纹分隔，列宽按内容实测宽度算，零手写魔数。"""
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
    # 等宽字体：均匀步进（char_w 对符号权重低估，monospace 下所有字符等宽）
    return len(s) * size * 0.62


TITLE = "tt.* 算子词汇表 —— 16 个常用算子在 dump 里的固定长相"
SUBTITLE = "认脸不查义：语义回指 ch07（访存）/ ch08（dot-reduce-scan）/ ch04（tl 两层结构）"

COLS = ["算子（.td 定义行）", "结构要点（读脸的关键）", "dump 里长成（样例，BLOCK=4）"]
FONT = [12.5, 12, 12]

ROWS = [
    ["tt.make_range (L803)", "无操作数，两个 I32Attr", "tt.make_range {start=0, end=4} : tensor<4xi32>"],
    ["tt.splat (L422)", "标量→张量广播，带 `->` 箭头", "tt.splat %p : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>"],
    ["tt.addptr (L199)", "指针+偏移，两操作数双类型", "tt.addptr %p, %o : tensor<4x!tt.ptr<f32>>, tensor<4xi32>"],
    ["tt.load (L231)", "可选 mask/other + 默认值属性", "tt.load %p, %m : tensor<4x!tt.ptr<f32>>"],
    ["tt.store (L306)", "有 MemWrite 副作用，无结果", "tt.store %p, %v, %m : tensor<4x!tt.ptr<f32>>"],
    ["tt.dot (L635)", "三输入 a,b,c，`*`/`->` 形状式", "tt.dot %a, %b, %c : tensor<..> * tensor<..> -> tensor<..>"],
    ["tt.reduce (L711)", "带 region 的 combineOp 内联块", "tt.reduce (%x) ({ ^bb0(..): tt.reduce.return .. }) {axis=1}"],
    ["tt.scan (L743)", "同 reduce，多 reverse 属性，形状不缩", "tt.scan (%x) ({ .. }) {axis=1, reverse=false}"],
    ["tt.expand_dims (L436)", "插一维，带 axis 属性", "tt.expand_dims %x {axis=0} : tensor<4xi32> -> tensor<1x4xi32>"],
    ["tt.reshape (L451)", "改形状，allow_reorder 属性", "tt.reshape %x : tensor<4x4xf32> -> tensor<16xf32>"],
    ["tt.broadcast (L471)", "沿大小=1 维扩张", "tt.broadcast %x : tensor<1x4xf32> -> tensor<8x4xf32>"],
    ["tt.join + tt.split (L506/L524，2 算子合 1 行)", "沿新末维拼/拆，2 的幂约束", "tt.join %a, %b : tensor<4xf32> -> tensor<4x2xf32>"],
    ["tt.trans (L544)", "按 order 转置（tt 层=重命名）", "tt.trans %x {order=[1,0]} : tensor<4x8xf32> -> tensor<8x4xf32>"],
    ["tt.make_tensor_ptr (L908)", "块指针构造，base+shape/strides/offsets", "tt.make_tensor_ptr %b, [%s..],[%t..],[%o..] : !tt.ptr<tensor<8x8xf16>>"],
    ["tt.atomic_rmw (L357)", "读-改-写，带 rmw_op/sem/scope 属性", "tt.atomic_rmw fadd, %p, %v, %m : tensor<..>"],
    ["tt.atomic_cas (L390)", "比较交换，MemRead+MemWrite", "tt.atomic_cas %p, %cmp, %val : tensor<..>"],
]

PAD = 30
TOP = 84
HEADER_H = 34
ROW_H = 32
COL_GAP = 22
CELL_PAD = 12

WFN = [mono_w, text_w, mono_w]  # 列 0/2 monospace, 列 1 sans
col_w = []
for j in range(3):
    max_content = max(WFN[j](r[j], FONT[j]) for r in ROWS)
    max_header = text_w(COLS[j], 12.5) + 4
    col_w.append(max(max_content, max_header) + CELL_PAD * 2)

col_x = [PAD]
for j in range(1, 3):
    col_x.append(col_x[j - 1] + col_w[j - 1] + COL_GAP)

w = col_x[-1] + col_w[-1] + PAD
h = TOP + HEADER_H + ROW_H * len(ROWS) + 46

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>']

L.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# 表头
for j, name in enumerate(COLS):
    cx = col_x[j]
    L.append(f'<rect x="{cx:.0f}" y="{TOP}" width="{col_w[j]:.0f}" height="{HEADER_H}" '
              'fill="#3b82f6"/>')
    L.append(f'<text x="{cx+CELL_PAD:.0f}" y="{TOP+HEADER_H/2+4.5:.0f}" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="white">{esc(name)}</text>')

# 表体：斑马纹 + 每行三列
table_left = col_x[0]
table_w = col_x[-1] + col_w[-1] - table_left
for i, row in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    if i % 2 == 1:
        L.append(f'<rect x="{table_left:.0f}" y="{ry:.0f}" width="{table_w:.0f}" '
                  f'height="{ROW_H}" fill="#f1f5f9"/>')
    L.append(f'<line x1="{table_left:.0f}" y1="{ry:.0f}" x2="{table_left+table_w:.0f}" '
              f'y2="{ry:.0f}" stroke="#e2e8f0" stroke-width="1"/>')
    ty = ry + ROW_H / 2 + 4.5
    # 列 0：算子名，monospace 蓝加粗
    L.append(f'<text x="{col_x[0]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="monospace" '
              f'font-size="{FONT[0]}" font-weight="bold" fill="#1e40af">{esc(row[0])}</text>')
    # 列 1：结构要点，sans 深灰
    L.append(f'<text x="{col_x[1]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="sans-serif" '
              f'font-size="{FONT[1]}" fill="#374151">{esc(row[1])}</text>')
    # 列 2：dump 样例，monospace 深绿（代码感）
    L.append(f'<text x="{col_x[2]+CELL_PAD:.0f}" y="{ty:.0f}" font-family="monospace" '
              f'font-size="{FONT[2]}" fill="#065f46">{esc(row[2])}</text>')

bottom_y = TOP + HEADER_H + ROW_H * len(ROWS)
L.append(f'<line x1="{table_left:.0f}" y1="{bottom_y:.0f}" x2="{table_left+table_w:.0f}" '
          f'y2="{bottom_y:.0f}" stroke="#94a3b8" stroke-width="1.5"/>')
L.append(f'<text x="{PAD}" y="{bottom_y+28:.0f}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">共 15 行 / 16 个算子（tt.join、tt.split 合并展示为一行）；'
          f'掌握 make_range/splat/addptr 三张模板即可读懂 add_kernel '
          f'一段 TTIR 中约 69% 的 tt.* 行，再补 load/store 两张覆盖近全部（见正文量化）。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch19-op-vocabulary.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
