#!/usr/bin/env python3
"""ch37-fig-launcher-codegen-map: state-table 模板。
make_launcher 把 add_kernel 的签名 {0:*fp32,1:*fp32,2:*fp32,3:i32} 逐参映射成
生成 C 扩展里的 PyArg 格式字符/C 类型/取址方式；底部注解拼出完整 format 串。
全部坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "make_launcher：签名 → 生成 C 扩展的逐参映射（add_kernel）"
SUBTITLE = "third_party/nvidia/backend/driver.py:L117-L239 —— signature = {0:*fp32, 1:*fp32, 2:*fp32, 3:i32}"

COLS = ["签名类型", "ty_to_cpp", "PyArg 格式字符", "取址方式 (internal_arg)"]
ROW_LABELS = ["arg0 (x_ptr)", "arg1 (y_ptr)", "arg2 (out_ptr)", "arg3 (n_elements)"]
CELLS = {
    "arg0 (x_ptr)":      ["*fp32", "CUdeviceptr", "字符 'O'", "ptr_info0.dev_ptr ←\ngetPointer(_arg0, 0)"],
    "arg1 (y_ptr)":      ["*fp32", "CUdeviceptr", "字符 'O'", "ptr_info1.dev_ptr ←\ngetPointer(_arg1, 1)"],
    "arg2 (out_ptr)":    ["*fp32", "CUdeviceptr", "字符 'O'", "ptr_info2.dev_ptr ←\ngetPointer(_arg2, 2)"],
    "arg3 (n_elements)": ["i32", "int32_t", "字符 'i'", "_arg3（标量直传，\n无 getPointer）"],
}
PTR_ROWS = {"arg0 (x_ptr)", "arg1 (y_ptr)", "arg2 (out_ptr)"}
COLOR_PTR = ("#eff6ff", "#2563eb")
COLOR_SCALAR = ("#fff7ed", "#c2410c")

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 210, 60, 34, 96, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

elems = []


def add(s):
    elems.append(s)


add(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
    f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
add(f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="11.5" '
    f'fill="#64748b">{esc(SUBTITLE)}</text>')

for j, name in enumerate(COLS):  # 列头
    x = col_x[j]
    add(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
        'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    add(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="white" '
        f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):  # 行标签 + 单元格
    ry = row_y[i]
    is_ptr = row in PTR_ROWS
    fill, stroke = COLOR_PTR if is_ptr else COLOR_SCALAR
    add(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
        f'font-family="monospace" font-size="12.5" font-weight="bold" '
        f'fill="{stroke}">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        add(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            add(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*16}" text-anchor="middle" '
                f'font-family="monospace" font-size="11.5" fill="{stroke}">{esc(line)}</text>')

table_bottom = row_y[-1] + ROW_H

# --- 底部：完整 format 串拼出 ---
note_top = table_bottom + 34
note_lines = [
    "固定前缀 'iiiKKOOOO'（9 字符：gridX/Y/Z + stream/function + metadata/launch_metadata/两 hook）",
    "+ 本例 4 个签名实参的 args_format 'OOOi' → 完整 format = 'iiiKKOOOOOOOi'（9+4 = 13 字符）",
    "3 个指针参走 getPointer 取 dev_ptr；生成 C 源 9027 字节 → 按 sha256 编成缓存 .so。",
]
note_h = 22 * len(note_lines) + 24
w_note = w - 2 * PAD
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w_note}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h:.0f}" fill="white"/>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch37-fig-launcher-codegen-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} w={w} h={h:.0f}")
