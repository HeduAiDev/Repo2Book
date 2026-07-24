#!/usr/bin/env python3
"""fig-m2-crosslang-chain：一个二进制 blob 从 triton 核心到设备句柄，跨「核心→Python
驱动层→C++ 扩展→CANN 运行时」四道栏；中途 driver.py 用 rsplit 拆出双核 mix_mode，
C++ 侧用 PyArg_ParseTuple 解包。swimlane 模板改造：call 箭头(实线,向右/下一层)配
return 箭头(虚线,向左/回上一层)，同一泳道内的处理步骤用小号侧注文字标在生命线右侧。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "跨语言接力：一个二进制 blob 从 triton 核心装到 CANN 运行时的四道栏"
SUBTITLE = "compiler.py → driver.py → npu_utils.cpp → CANN rt* —— call 实线 / return 虚线"

LANES = ["triton 核心\n(compiler.py)", "Python 驱动层\n(driver.py)",
         "C++ 扩展\n(npu_utils.cpp)", "CANN 运行时\n(rt*)"]

# (kind, src_lane_idx, dst_lane_idx, label_lines, side_note_lane_idx_or_None, side_note_lines)
# 每行长度控制在相邻泳道间距内(≤~40 monospace 字符@fs12/~45@fs11)，避免文字压过生命线。
EVENTS = [
    ("call", 0, 1,
     ["load_binary(self.name,",
      "self.kernel=blob, shared, device)"],
     None, None),
    ("note", 1, 1, None, 1,
     ["name.rsplit(\"_\", 1) → (fnname, mix_mode)", "（从右切一刀，driver.py:L78）"]),
    ("call", 1, 2,
     ["load_kernel_binary(fnname, kernel,",
      "shared, device, mix_mode)"],
     None, None),
    ("note", 2, 2, None, 2,
     ["PyArg_ParseTuple(\"ss#iis\", ...) —— 6 个入参",
      "解出 name/data/data_size/shared/device/kernel_mode（npu_utils.cpp:L92-L93）"]),
    ("call", 2, 3,
     ["registerKernel(...)：rtSetDevice →",
      "rtDevBinaryRegister → rtFunctionRegister"],
     None, None),
    ("ret", 3, 2,
     ["(devbinHandle,", "func_stub_handle)"],
     None, None),
    ("ret", 2, 1,
     ["Py_BuildValue(\"(KKii)\")",
      "→ (module, function, 0, 0)",
      "（npu_utils.cpp:L106）"],
     None, None),
    ("ret", 1, 0,
     ["(module, function,", "n_regs=0, n_spills=0)"],
     None, None),
    ("note", 0, 0, None, 0,
     ["self.module, self.function,", "self.n_regs, self.n_spills = 该四元组"]),
]

LANE_W = 380
TOP = 118
PAD = 40
STEP_H_CALL = 66
STEP_H_NOTE = 58

w = PAD * 2 + LANE_W * (len(LANES) - 1) + 200
X = [PAD + 100 + i * LANE_W for i in range(len(LANES))]

elems = []
def add(s): elems.append(s)

# 先算总高度：逐事件累加
y_positions = []
y = TOP
for kind, s, d, label, note_lane, note_lines in EVENTS:
    y_positions.append(y)
    if kind == "note":
        n = len(note_lines)
        y += 20 + 16 * n + 14
    else:
        n = len(label)
        y += 18 + 16 * n + 20
h = y + PAD + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0369a1"/></marker>'
     '</defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 泳道头 + 生命线
for i, name in enumerate(LANES):
    x = X[i]
    lines = name.split("\n")
    add(f'<rect x="{x-85:.0f}" y="{TOP-58:.0f}" width="170" height="42" rx="8" '
        'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
    for k, ln in enumerate(lines):
        fw = 'font-weight="bold" ' if k == 0 else ''
        fs = 13 if k == 0 else 11
        add(f'<text x="{x:.0f}" y="{TOP-40+k*17:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="{fs}" {fw}fill="#0f172a">{esc(ln)}</text>')
    add(f'<line x1="{x:.0f}" y1="{TOP-14:.0f}" x2="{x:.0f}" y2="{h-PAD-10:.0f}" '
        'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for (kind, s, d, label, note_lane, note_lines), y0 in zip(EVENTS, y_positions):
    if kind == "note":
        x = X[note_lane]
        n = len(note_lines)
        box_w = 340
        box_h = 16 * n + 20
        add(f'<rect x="{x+18:.0f}" y="{y0:.0f}" width="{box_w}" height="{box_h}" rx="6" '
            'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
        for k, ln in enumerate(note_lines):
            fs = 11.5 if k == 0 else 10.5
            fw = 'font-weight="bold" ' if k == 0 else ''
            add(f'<text x="{x+30:.0f}" y="{y0+18+k*16:.0f}" font-family="sans-serif" '
                f'font-size="{fs}" {fw}fill="#334155">{esc(ln)}</text>')
        continue
    x1, x2 = X[s], X[d]
    n = len(label)
    label_top = y0
    label_h = 16 * n
    liney = label_top + label_h + 12
    color = "#334155" if kind == "call" else "#0369a1"
    marker = "url(#a)" if kind == "call" else "url(#b)"
    dash = "" if kind == "call" else ' stroke-dasharray="7,5"'
    add(f'<line x1="{x1:.0f}" y1="{liney:.0f}" x2="{x2:.0f}" y2="{liney:.0f}" '
        f'stroke="{color}" stroke-width="1.8"{dash} marker-end="{marker}"/>')
    for k, ln in enumerate(label):
        fs = 11.5 if k == 0 else 10.5
        add(f'<text x="{(x1+x2)/2:.0f}" y="{label_top+8+k*16:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{fs}" fill="{color}">{esc(ln)}</text>')

# 底部数字小结
def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

note_lines2 = [
    "跨语言边界数 = 4 层（核心 / Python / C++ / CANN）；rsplit 从右切一刀拆出 mix_mode；",
    "PyArg 格式串 ss#iis 对应 6 个入参；C++ 回传固定 (KKii) 四元组，末两位 n_regs/n_spills 恒为 0（占位，NPU 无此概念）。",
]
note_top = h - PAD - 10
note_h = 24 * len(note_lines2) + 20
note_w_needed = max(cjk_w(s, 12) for s in note_lines2) + 32
w = max(w, note_w_needed + 2 * PAD)
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines2):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')
h = note_top + note_h + 20

L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">'
L[2] = f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>'
L = L + elems + ['</svg>']

out = Path(__file__).with_name("fig-m2-crosslang-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
