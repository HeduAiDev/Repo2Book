#!/usr/bin/env python3
"""state-table 模板改造:_parse_linalg_metadata 用 6 条正则一次性从 Linalg IR
文本抠出 metadata 字段(compiler.py:L183-L233)。行=6 条正则,列=命中文本/方法/
结果/落点。底部脚注补三个不在主表里的关键数字(shared/required_ub_bits/tensor_kinds)。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "_parse_linalg_metadata：6 条正则一次性抠出 metadata"
SUBTITLE = "示例 IR：add_kernel(2 输入 + 1 输出 + 1 标量 %arg3) — compiler.py:L197-L233"
COLS = ["命中的 IR 文本", "re 方法", "抽取结果", "metadata 落点"]
ROW_LABELS = [
    "MIX_MODE_REGEX",
    "PARALLEL_MODE_REGEX",
    "KERNEL_NAME_REGEX",
    "TENSOR_KIND_REGEX",
    "BITCODES_REGEX",
    "DISABLE_AUTO_TILE…REGEX",
]
CELLS = {
    "MIX_MODE_REGEX": [
        'mix_mode = "aiv"',
        "re.search().group(1)",
        "aiv",
        "metadata['mix_mode']",
    ],
    "PARALLEL_MODE_REGEX": [
        'parallel_mode = "mix_simd_simt"',
        "re.search().group(1)",
        "mix_simd_simt",
        "metadata['parallel_mode']",
    ],
    "KERNEL_NAME_REGEX": [
        "func.func @add_kernel",
        "re.search().group(1)",
        "add_kernel",
        "metadata['kernel_name']",
    ],
    "TENSOR_KIND_REGEX": [
        "3 个 %argN 的\n{tt.tensor_kind=k}",
        "re.findall → list",
        "[0, 0, 1]",
        "metadata['tensor_kinds']",
    ],
    "BITCODES_REGEX": [
        'bitcode = "libdevice.bc"',
        "re.findall 展平",
        "['libdevice.bc']",
        "metadata['bitcodes']",
    ],
    "DISABLE_AUTO_TILE…REGEX": [
        "(本例不含该属性)",
        "not re.search\n(None → True)",
        "True",
        "metadata[\n'auto_tile_and_bind_subblock']",
    ],
}
# 6 行全部标 matched 语义色(都命中/推导出值),无 changed/stable 二分,统一一种强调色
STATUS = {r: ["matched"] * len(COLS) for r in ROW_LABELS}
COLOR = {"matched": ("#eff6ff", "#1e40af")}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 250, 60, 34, 100, 32
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 96
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签(源码常量名) + 单元格
    ry = row_y[i]
    L.append(f'<rect x="{PAD}" y="{ry+4}" width="{LABEL_W-8}" height="{ROW_H-8}" rx="4" '
              'fill="#f1f5f9" stroke="#94a3b8"/>')
    L.append(f'<text x="{PAD+(LABEL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
              f'font-family="monospace" font-size="11.5" font-weight="bold" '
              f'fill="#0f172a">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="monospace" font-size="11" fill="{text_fill}">{esc(line)}</text>')

# 脚注:三个不在主表里的关键落地数字(带出处)
foot_top = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
foot_box_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{foot_top}" width="{foot_box_w}" height="72" rx="6" '
          'fill="#fefce8" stroke="#ca8a04"/>')
foot_lines = [
    "另有两处硬编码初值(不经正则,同一函数内直接赋值)：",
    "metadata['shared'] = 1（compiler.py:L216，硬编码）；"
    "metadata['required_ub_bits'] = 0（compiler.py:L229，初值——后续从闭源编译器 stdout 正则回填）",
]
for k, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD+16}" y="{foot_top+24+k*22}" font-family="sans-serif" '
              f'font-size="12.5" fill="#854d0e">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch28-regex-extract.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
