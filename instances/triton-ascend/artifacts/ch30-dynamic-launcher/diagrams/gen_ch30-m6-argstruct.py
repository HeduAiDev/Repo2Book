#!/usr/bin/env python3
"""ch30-m6-argstruct：打给设备的 packed 参数块共 56 字节，字段顺序/对齐固定，
argsSize = sizeof(args) 与该布局同源。layout 模板：按字节数等比例画宽度。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "packed 参数块：56 字节，字段顺序与对齐固定"
SUBTITLE = "driver.py:L817-833 —— 指针/64 位按 8 对齐、其余按 4（L821）；argsSize = sizeof(args) 传入 rtKernelLaunch（L734）"

# (字段名, 字节数, 对齐, 偏移, 类别)
FIELDS = [
    ("syncBlockLock", 8, 8, 0, "ptr"),
    ("workspace_addr", 8, 8, 8, "ptr"),
    ("arg0 (x_ptr)", 8, 8, 16, "ptr"),
    ("arg1 (y_ptr)", 8, 8, 24, "ptr"),
    ("arg2 (out_ptr)", 8, 8, 32, "ptr"),
    ("arg3", 4, 4, 40, "i32"),
    ("gridX", 4, 4, 44, "i32"),
    ("gridY", 4, 4, 48, "i32"),
    ("gridZ", 4, 4, 52, "i32"),
]
COLORS = {"ptr": ("#dbeafe", "#1d4ed8", "#1e3a8a"), "i32": ("#fef3c7", "#b45309", "#78350f")}
LEGEND = [("ptr", "指针字段（8 字节对齐）"), ("i32", "int32_t 字段（4 字节对齐）")]

SCALE = 14  # px / 字节 —— 保证 8B 字段(112px)能放下 13~14 字符的字段名
PAD = 50
TOP = 130
BOX_H = 92

x = PAD
xs_left = []
for name, size, align, offset, kind in FIELDS:
    xs_left.append(x)
    x += size * SCALE
total_struct_w = x - PAD
w = max(1180, PAD * 2 + total_struct_w + 40)

elems = []
def add(s): elems.append(s)

# 结构体格子
for (name, size, align, offset, kind), bx in zip(FIELDS, xs_left):
    bw = size * SCALE
    fill, stroke, tcolor = COLORS[kind]
    add(f'<rect x="{bx:.0f}" y="{TOP}" width="{bw:.0f}" height="{BOX_H}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    base, paren = (name.split(" (", 1) if " (" in name else (name, None))
    paren = ("(" + paren) if paren else None
    cx = bx + bw / 2
    if paren:
        add(f'<text x="{cx:.0f}" y="{TOP+BOX_H/2-6:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" font-weight="bold" fill="{tcolor}">{esc(base)}</text>')
        add(f'<text x="{cx:.0f}" y="{TOP+BOX_H/2+12:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="10" fill="{tcolor}">{esc(paren)}</text>')
    else:
        add(f'<text x="{cx:.0f}" y="{TOP+BOX_H/2+4:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" font-weight="bold" fill="{tcolor}">{esc(base)}</text>')
    add(f'<text x="{cx:.0f}" y="{TOP+BOX_H+18:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="10.5" fill="#475569">{size}B</text>')

# 偏移刻度（每个字段边界一个数字）
tick_y = TOP + BOX_H + 4
offsets_display = [f[3] for f in FIELDS] + [FIELDS[-1][3] + FIELDS[-1][1]]
tick_xs = xs_left + [xs_left[-1] + FIELDS[-1][1] * SCALE]
for off, tx in zip(offsets_display, tick_xs):
    add(f'<line x1="{tx:.0f}" y1="{TOP}" x2="{tx:.0f}" y2="{tick_y}" stroke="#94a3b8" stroke-width="1"/>')
    add(f'<text x="{tx:.0f}" y="{tick_y+30:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="10" fill="#64748b">{off}</text>')

struct_bottom = TOP + BOX_H + 44

# 总大小括注
brace_y = TOP - 14
add(f'<line x1="{PAD}" y1="{brace_y}" x2="{xs_left[-1]+FIELDS[-1][1]*SCALE:.0f}" y2="{brace_y}" '
    'stroke="#334155" stroke-width="1.5"/>')
add(f'<line x1="{PAD}" y1="{brace_y-6}" x2="{PAD}" y2="{brace_y}" stroke="#334155" stroke-width="1.5"/>')
end_x = xs_left[-1] + FIELDS[-1][1] * SCALE
add(f'<line x1="{end_x:.0f}" y1="{brace_y-6}" x2="{end_x:.0f}" y2="{brace_y}" stroke="#334155" stroke-width="1.5"/>')
add(f'<text x="{(PAD+end_x)/2:.0f}" y="{brace_y-12:.0f}" text-anchor="middle" font-family="sans-serif" '
    'font-size="13" font-weight="bold" fill="#0f172a">sizeof(args) = 56 字节</text>')

# 图例
ly = struct_bottom + 8
for j, (key, label) in enumerate(LEGEND):
    fill, stroke, _ = COLORS[key]
    lx = PAD + j * 260
    add(f'<rect x="{lx}" y="{ly}" width="16" height="16" rx="3" fill="{fill}" stroke="{stroke}"/>')
    add(f'<text x="{lx+24}" y="{ly+13}" font-family="sans-serif" font-size="12" '
        f'fill="#334155">{esc(label)}</text>')

call_y = ly + 46
call_text = "rtKernelLaunch(func, blockNum=4, static_cast<void*>(&args), sizeof(args)=56, NULL, stream)"
add(f'<rect x="{PAD}" y="{call_y}" width="{w-2*PAD}" height="40" rx="8" '
    'fill="#ecfdf5" stroke="#059669" stroke-width="1.5"/>')
add(f'<text x="{PAD+16}" y="{call_y+25}" font-family="monospace" font-size="12.5" '
    f'fill="#065f46">{esc(call_text)}</text>')
add(f'<text x="{w-PAD-16}" y="{call_y+25}" text-anchor="end" font-family="sans-serif" font-size="11" '
    f'fill="#059669">driver.py:L734</text>')

content_bottom = call_y + 40

note_lines = [
    "arg0/arg1/arg2 对应 kernel 实参 x_ptr/y_ptr/out_ptr；arg3 对应 n_elements —— 均来自 signature.items() 顺序。",
    "字段声明顺序与初始化列表顺序一致（同一循环生成，L818-823/L825-832）；constants 不入 struct；"
    "__attribute__((packed)) 无编译器补齐 —— 偏移即前缀宽度和，argsSize 随字段增减自动同步。",
]
note_top = content_bottom + 26
note_h = 24 * len(note_lines) + 20
add(f'<rect x="{PAD}" y="{note_top}" width="{w-2*PAD}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*24}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

H = note_top + note_h + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {H:.0f}">',
     f'<rect width="{w:.0f}" height="{H:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']
out = Path(__file__).with_name("ch30-m6-argstruct.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={H:.0f}")
