#!/usr/bin/env python3
"""figure m1-lazy-state: LazyProxy 两态状态机——首次属性访问才真正构造 driver。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "LazyProxy：首次属性访问才构造 driver"
SUBTITLE = "python/triton/runtime/driver.py —— import triton 时只登记 init_fn，不碰 CUDA"

TRIGGER_LINES = [
    "首次访问触发",
    "__getattr__ / __setattr__ / __delattr__ / __str__",
    "若 self._obj is None: self._obj = self._init_fn()",
]
NOTE_LINES = [
    "import triton 时只把 default = LazyProxy(_create_driver) 登记好（driver.py:L50）——",
    "停在左态，headless，不触碰 torch.cuda；真正需要设备时才向右跳一次转移，",
    "这正是上一章标出的『真设备断裂点』被推迟发生的位置。",
]

# 每个状态框内三行文案 + 各自字号，用于反推框宽——不手写魔数。
STATE1_LINES = [("未初始化", 15), ("self._obj = None", 13),
                ("driver.py:L14-L16（LazyProxy.__init__）", 11)]
STATE2_LINES = [("已初始化", 15), ("self._obj = 真 driver 实例", 13),
                ("driver.py:L18-L24（_initialize_obj → __getattr__）", 11)]

BOX_H = 116
box_w_needed = max(cjk_w(t, sz) for t, sz in STATE1_LINES + STATE2_LINES) + 24
BOX_W = max(260, box_w_needed)
trigger_w = max(cjk_w(s, 12) for s in TRIGGER_LINES) + 20
GAP = max(340, trigger_w)
PAD = 40
TOP = 110
w_content = BOX_W * 2 + GAP
note_w_needed = max(cjk_w(s, 12.5) for s in NOTE_LINES) + 32
w = PAD * 2 + max(w_content, note_w_needed)
h = TOP + BOX_H + 200

x1 = PAD + (w - 2 * PAD - w_content) / 2
x2 = x1 + BOX_W + GAP
y = TOP

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# --- 状态1：未初始化 / 状态2：已初始化（同一份 STATE*_LINES 数据驱动，不重复写字符串） ---
STATE_BOXES = [
    (x1, "#fef9c3", "#ca8a04", "#854d0e", "#a16207", STATE1_LINES),
    (x2, "#dcfce7", "#16a34a", "#14532d", "#15803d", STATE2_LINES),
]
LINE_Y = [32, 60, 90]
LINE_FILL_IDX = [3, 3, 4]  # 前两行用标题色，第三行(源码行号)用稍浅色
for bx, fill, stroke, title_fill, code_fill, lines in STATE_BOXES:
    L.append(f'<rect x="{bx:.0f}" y="{y}" width="{BOX_W:.0f}" height="{BOX_H}" rx="14" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    for (text, size), ly, weight in zip(lines, LINE_Y,
                                          ['font-weight="bold" ', '', '']):
        line_fill = title_fill if ly != 90 else code_fill
        L.append(f'<text x="{bx+BOX_W/2:.0f}" y="{y+ly}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{size}" {weight}'
                  f'fill="{line_fill}">{esc(text)}</text>')

# --- 转移箭头 + 触发标签（三行，垂直居中排列在箭头上方/下方） ---
ay = y + BOX_H / 2
mx = (x1 + BOX_W + x2) / 2
L.append(f'<line x1="{x1+BOX_W:.0f}" y1="{ay:.0f}" x2="{x2:.0f}" y2="{ay:.0f}" '
          'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{mx:.0f}" y="{ay-34:.0f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#334155">{esc(TRIGGER_LINES[0])}</text>')
L.append(f'<text x="{mx:.0f}" y="{ay-14:.0f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">{esc(TRIGGER_LINES[1])}</text>')
L.append(f'<text x="{mx:.0f}" y="{ay+22:.0f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">{esc(TRIGGER_LINES[2])}</text>')

# --- 底部注解 ---
note_top = y + BOX_H + 46
note_h = 26 * len(NOTE_LINES) + 24
L.append(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
          'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(NOTE_LINES):
    L.append(f'<text x="{PAD+16}" y="{note_top+26+i*24:.0f}" font-family="sans-serif" '
              f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("m1-lazy-state.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
