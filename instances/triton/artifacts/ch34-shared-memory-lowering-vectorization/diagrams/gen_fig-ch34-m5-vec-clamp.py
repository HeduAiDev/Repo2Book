#!/usr/bin/env python3
"""state-table 模板:getVectorSize = min(128/bitwidth, contiguity) 再被 mask
alignment 夹。fp16 满宽 8 元素(128 位)沿 contiguity->maskAlign 逐级掉到 4/2/1,
i8 满宽 16 元素同样是 128 位。数据取自 explainer/traces/vector_size.out。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_text_width(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "getVectorSize = min(128/bitwidth, contiguity, maskAlign) —— 三把夹子定向量宽"
SUBTITLE = "fp16(128/16=8)沿 contiguity→maskAlign 逐级收紧到 4/2/1;vec==1 且 numElems>1 时编译器发 remark;i8(128/8=16)位宽小仍可满宽"

COLS = ["场景", "dtype", "128/bw\n硬件上限", "contiguity", "maskAlign", "vec", "字节", "位", "remark?"]
# (场景, dtype, 128/bw, contiguity, maskAlign, vec, bytes, bits, remark, highlight)
ROWS = [
    ("A 满宽",     "fp16", "8",  "16", "无", "8", "16", "128", "False", "chain"),
    ("B 连续度夹", "fp16", "8",  "4",  "无", "4", "8",  "64",  "False", "chain"),
    ("C mask 夹",  "fp16", "8",  "16", "2",  "2", "4",  "32",  "False", "chain"),
    ("D 塌成标量", "fp16", "8",  "16", "1",  "1", "2",  "16",  "True",  "warn"),
    ("E fp32 满宽", "fp32", "4",  "8",  "无", "4", "16", "128", "False", "ok"),
    ("F i8 满宽",  "i8",   "16", "32", "无", "16","16", "128", "False", "ok"),
]
HL_COLOR = {
    "chain": ("#eff6ff", "#3b82f6", "#1e3a5f"),
    "warn":  ("#fef2f2", "#ef4444", "#7f1d1d"),
    "ok":    ("#ecfdf5", "#16a34a", "#065f46"),
}

COL_W = [96, 62, 76, 84, 78, 56, 60, 60, 76]
LABEL_W = 0
ROW_H, HEADER_H, TOP, PAD = 48, 56, 116, 32
NUM_COLS = len(COLS)
min_w_table = PAD * 2 + sum(COL_W)
FOOT = "调优 = 把 contiguity(第 25 章 AxisInfo)与 mask 对齐都顶到 128/bitwidth,才能拿到满宽向量访存;任一收紧就从合并访存掉向标量。"
min_w_text = PAD * 2 + max(cjk_text_width(SUBTITLE, 12), cjk_text_width(FOOT, 12))
w = max(min_w_table, min_w_text)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 56

col_x = [PAD + sum(COL_W[:i]) for i in range(NUM_COLS)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W[j]-6}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    ny0 = TOP + (HEADER_H-6)/2 - (len(lines)-1)*7 + 4
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x+(COL_W[j]-6)/2}" y="{ny0+k*14}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="white" '
                  f'font-weight="bold">{esc(ln)}</text>')

for i, row in enumerate(ROWS):
    ry = row_y[i]
    kind = row[-1]
    vals = row[:-1]
    fill, stroke, tcolor = HL_COLOR[kind]
    for j, val in enumerate(vals):
        cx = col_x[j]
        use_fill = fill if kind != "chain" or j in (0,) else ("#f8fafc" if False else fill)
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W[j]-6}" height="{ROW_H-8}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        fsz = 12 if len(val) <= 6 else 11
        weight = 'font-weight="bold" ' if j in (0, 5, 8) else ''
        L.append(f'<text x="{cx+(COL_W[j]-6)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="monospace" font-size="{fsz}" fill="{tcolor}" '
                  f'{weight}>{esc(val)}</text>')

# 图例
legend_y = row_y[-1] + ROW_H + 22
legend_items = [("chain", "fp16 主链(A→D 逐级收紧)"), ("warn", "标量退化 + remark"), ("ok", "不同位宽各自满宽(128 位)")]
lx = PAD
for key, label in legend_items:
    fill, stroke, tcolor = HL_COLOR[key]
    L.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{lx+22}" y="{legend_y+13}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 22 + cjk_text_width(label, 11) + 30

foot_y = legend_y + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(FOOT)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m5-vec-clamp.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
