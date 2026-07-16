#!/usr/bin/env python3
"""figure m2-discover-flow: _discover_backends 扫 backends/<name>/ 目录、
各取唯一 concrete 子类的流程 + 0/1/>1 三分支判定。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

TITLE = "_discover_backends：扫目录、各取唯一 concrete 子类"
SUBTITLE = "python/triton/backends/__init__.py —— 新后端只需照此结构放两份文件即入表"

PAD = 40
BOX_W = 560
STEP_H = 74
GAP = 30
TOP = 96

STEPS = [
    "os.listdir(backends/)  —— 逐个非 __ 开头的子目录 <name>",
    "_load_module(<name>, backends/<name>/compiler.py)\n"
    "_load_module(<name>, backends/<name>/driver.py)",
    "_find_concrete_subclasses(module, BaseBackend / DriverBase)\n（各自筛出 concrete 子类）",
]

BRANCH_LABELS = [
    ("0", "RuntimeError\n（一个后端都没实现）", "#fee2e2", "#b91c1c"),
    ("1", "唯一 —— ret[0]\n继续组装", "#dcfce7", "#15803d"),
    (">1", "RuntimeError\n（目录里混了多个 concrete 类）", "#fee2e2", "#b91c1c"),
]

TAIL_STEPS = [
    "Backend(compiler=..., driver=...)  —— 组装",
    "backends[<name>] = Backend(...)  —— 存入表",
]

BRANCH_W = 280
branch_gap = 30
branch_total_w = BRANCH_W * 3 + branch_gap * 2
content_w = max(BOX_W, branch_total_w)
w = PAD * 2 + content_w

x_center = PAD + content_w / 2
x_main = x_center - BOX_W / 2

L = [None]  # placeholder for svg header, filled later once h computed
elems = []

def add(s): elems.append(s)

y = TOP

def step_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e"):
    n = len(lines)
    box_h = 34 + 22 * (n - 1) + 40
    add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    for k, line in enumerate(lines):
        add(f'<text x="{x_center:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="{text_fill}">{esc(line)}</text>')
    return box_h

def arrow(x, y1, y2):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

# --- 前两个主线步骤 ---
box_ys = []
for i, s in enumerate(STEPS[:2]):
    if i > 0:
        arrow(x_center, y - GAP, y)
    box_ys.append(y)
    bh = step_box(y, s.split("\n"))
    y += bh
    last_bh = bh
y_step2_bottom = y

# --- 第三步：判定框（菱形样式用圆角矩形替代，居中） ---
arrow(x_center, y, y + GAP)
y += GAP
judge_h = 78
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
lines3 = STEPS[2].split("\n")
y0 = y + judge_h / 2 - (len(lines3) - 1) * 11 + 4
for k, line in enumerate(lines3):
    add(f'<text x="{x_center:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13" font-weight="bold" '
        f'fill="#78350f">{esc(line)}</text>')
judge_bottom = y + judge_h
judge_mid_y = y + judge_h

# --- 三分支 ---
y_branch = judge_bottom + 60
branch_h = 70
bx0 = x_center - branch_total_w / 2
branch_xs = [bx0 + i * (BRANCH_W + branch_gap) for i in range(3)]
branch_cx = [bx + BRANCH_W / 2 for bx in branch_xs]

# 判定框底边到三个分支框顶边的折线（从中点分叉）
fork_y = judge_bottom + 20
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{fork_y:.0f}" x2="{branch_cx[2]:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{fork_y:.0f}" x2="{cx:.0f}" y2="{y_branch:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

for (label, text, fill, stroke_c), bx, cx in zip(BRANCH_LABELS, branch_xs, branch_cx):
    add(f'<text x="{cx:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" font-weight="bold" fill="#334155">len(ret) = {esc(label)}</text>')
    add(f'<rect x="{bx:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{branch_h}" rx="10" '
        f'fill="{fill}" stroke="{stroke_c}" stroke-width="1.5"/>')
    lines = text.split("\n")
    y0 = y_branch + branch_h / 2 - (len(lines) - 1) * 10 + 4
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*18:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" fill="{stroke_c}">{esc(line)}</text>')

# --- 只有中间分支（唯一）汇入后续组装步骤 ---
y = y_branch + branch_h
merge_y = y + 34
add(f'<line x1="{branch_cx[1]:.0f}" y1="{y:.0f}" x2="{branch_cx[1]:.0f}" y2="{merge_y:.0f}" '
    'stroke="#16a34a" stroke-width="2"/>')
add(f'<line x1="{branch_cx[1]:.0f}" y1="{merge_y:.0f}" x2="{x_center:.0f}" y2="{merge_y:.0f}" '
    'stroke="#16a34a" stroke-width="2"/>')
tail_top = merge_y + GAP
add(f'<line x1="{x_center:.0f}" y1="{merge_y:.0f}" x2="{x_center:.0f}" y2="{tail_top:.0f}" '
    'stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>')

y = tail_top
for i, s in enumerate(TAIL_STEPS):
    if i > 0:
        arrow(x_center, y - GAP, y)
    bh = step_box(y, [s], fill="#dbeafe", stroke="#1d4ed8", text_fill="#1e3a5f")
    y += bh
    if i == 0:
        y += GAP

tail_bottom = y

# --- 底部注解 ---
note_lines = [
    "compiler.py / driver.py 各恰好 1 个 concrete 子类是硬契约——姊妹篇只需在 backends/ 下",
    "新增一个 ascend/ 目录、照此结构放两份文件，主库代码零改动即被纳入 {name: Backend} 表。",
]
note_top = tail_bottom + 36
note_w_needed = max(cjk_w(s, 12.5) for s in note_lines) + 32
content_w = max(content_w, note_w_needed)
w = PAD * 2 + content_w
note_h = 24 * len(note_lines) + 24
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+26+i*24:.0f}" font-family="sans-serif" '
        f'font-size="12.5" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("m2-discover-flow.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
