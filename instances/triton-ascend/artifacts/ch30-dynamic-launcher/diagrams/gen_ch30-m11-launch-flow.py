#!/usr/bin/env python3
"""ch30-m11-launch-flow：launch() C 入口一次发射的定序流水——解包→enter hook→getPointer
解指针→_launch 转调→exit hook→返回。swimlane 模板：Python 调用方 / launch() C 入口 两道栏，
call 实线向右、return 虚线向左，栏内处理步骤用侧注标出。坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "launch() C 入口：一次发射的定序流水"
SUBTITLE = "driver.py —— PyArg_ParseTuple 解包 → enter hook → getPointer → _launch → exit hook → 返回"

LANES = ["Python 调用方\n(NPULauncher.__call__)", "launch() C 入口\n(driver.py:L876-942)"]

# (kind, src_lane_idx, dst_lane_idx, label_lines, note_lane_idx_or_None, note_lines)
EVENTS = [
    ("call", 0, 1, ["self.launch(*args, **kwargs)", "driver.py:L140"], None, None),
    ("note", 1, 1, None, 1,
     ["① PyArg_ParseTuple(args, format, &gridX.., ...)",
      "按 format 解出 grid/stream/function/metadata/hooks/kernel 实参（L886-895）"]),
    ("note", 1, 1, None, 1,
     ["② launch_enter_hook 执行",
      "if (launch_enter_hook != Py_None) 调 PyObject_CallObject（L906-911）"]),
    ("note", 1, 1, None, 1,
     ["③ getPointer 把每个 tensor 实参解成设备指针",
      "DevicePtrInfo ptr_info_i = getPointer(_arg_i, i)（L929）"]),
    ("note", 1, 1, None, 1,
     ["④ _launch(kernelName, function, stream, gridX.., ..., ptr_info...)",
      "转调发射（driver.py:L930）"]),
    ("note", 1, 1, None, 1,
     ["launch_exit_hook 执行 + Py_RETURN_NONE",
      "if (launch_exit_hook != Py_None) 调 PyObject_CallObject（L934-941）"]),
    ("ret", 1, 0, ["⑤ 返回 profiler_registered", "driver.py:L139-142"], None, None),
]

LANE_W = 560
TOP = 128
PAD = 40
STEP_H_CALL = 60
STEP_H_NOTE_BASE = 20
NOTE_BOX_W = 560

X = [PAD + 130 + i * LANE_W for i in range(len(LANES))]
w = X[-1] + 22 + NOTE_BOX_W + PAD

elems = []
def add(s): elems.append(s)

y_positions = []
y = TOP
for kind, s, d, label, note_lane, note_lines in EVENTS:
    y_positions.append(y)
    if kind == "note":
        n = len(note_lines)
        y += 22 + 16 * n + 16
    else:
        n = len(label)
        y += 18 + 16 * n + 22
h = y + PAD + 46

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

for i, name in enumerate(LANES):
    x = X[i]
    lines = name.split("\n")
    add(f'<rect x="{x-135:.0f}" y="{TOP-64:.0f}" width="270" height="46" rx="8" '
        'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
    for k, ln in enumerate(lines):
        fw = 'font-weight="bold" ' if k == 0 else ''
        fs = 13.5 if k == 0 else 11.5
        add(f'<text x="{x:.0f}" y="{TOP-44+k*18:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="{fs}" {fw}fill="#0f172a">{esc(ln)}</text>')
    add(f'<line x1="{x:.0f}" y1="{TOP-16:.0f}" x2="{x:.0f}" y2="{h-PAD-16:.0f}" '
        'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for (kind, s, d, label, note_lane, note_lines), y0 in zip(EVENTS, y_positions):
    if kind == "note":
        x = X[note_lane]
        n = len(note_lines)
        box_w = NOTE_BOX_W
        box_h = 16 * n + 22
        add(f'<rect x="{x+22:.0f}" y="{y0:.0f}" width="{box_w}" height="{box_h}" rx="6" '
            'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1"/>')
        for k, ln in enumerate(note_lines):
            fs = 11.5 if k == 0 else 10.5
            fw = 'font-weight="bold" ' if k == 0 else ''
            fam = "monospace" if k == 0 else "sans-serif"
            add(f'<text x="{x+34:.0f}" y="{y0+20+k*16:.0f}" font-family="{fam}" '
                f'font-size="{fs}" {fw}fill="#334155">{esc(ln)}</text>')
        continue
    x1, x2 = X[s], X[d]
    n = len(label)
    label_top = y0
    label_h = 16 * n
    liney = label_top + label_h + 14
    color = "#334155" if kind == "call" else "#0369a1"
    marker = "url(#a)" if kind == "call" else "url(#b)"
    dash = "" if kind == "call" else ' stroke-dasharray="7,5"'
    add(f'<line x1="{x1:.0f}" y1="{liney:.0f}" x2="{x2:.0f}" y2="{liney:.0f}" '
        f'stroke="{color}" stroke-width="1.8"{dash} marker-end="{marker}"/>')
    for k, ln in enumerate(label):
        fs = 12 if k == 0 else 11
        fam = "monospace" if k > 0 else "sans-serif"
        fw = 'font-weight="bold" ' if k == 0 else ''
        add(f'<text x="{(x1+x2)/2:.0f}" y="{label_top+8+k*16:.0f}" text-anchor="middle" '
            f'font-family="{fam}" font-size="{fs}" {fw}fill="{color}">{esc(ln)}</text>')

note_lines2 = [
    "五步在同一个 C 函数 launch() 内顺序执行，中途任一步失败（解包失败/hook 抛异常）即 return NULL，不会跑到 _launch；",
    "getPointer 只对指针类实参调用（*fp32 等），i32 等标量实参由 PyArg_ParseTuple 直接收进 _arg_i,不经 getPointer。",
]
note_top = h - PAD - 20
note_h = 24 * len(note_lines2) + 20
note_w_needed = max(len(s) for s in note_lines2) * 8 + 32
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

out = Path(__file__).with_name("ch30-m11-launch-flow.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
