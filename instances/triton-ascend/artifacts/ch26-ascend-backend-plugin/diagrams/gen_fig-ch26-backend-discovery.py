#!/usr/bin/env python3
"""fig-ch26-backend-discovery：_discover_backends 扫 triton/backends/ 每个子目录，
跳过 nvidia/amd（本包不发行其加载文件），其余目录各取唯一非抽象子类装配成 Backend。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "_discover_backends：跳过 nvidia/amd，其余目录各取唯一 concrete 子类"
SUBTITLE = "python/triton/backends/__init__.py —— ascend 目录放对 compiler.py/driver.py 即被自动装配，无需注册表"

PAD = 40
BOX_W = 620
STEP_H = 74
GAP = 28
TOP = 96

STEP1 = "os.listdir(triton/backends/)  —— 逐个子目录 name"

BRANCH_W = 300
branch_gap = 30
branch_total_w = BRANCH_W * 2 + branch_gap

content_w = max(BOX_W, branch_total_w)
w = PAD * 2 + content_w
x_center = PAD + content_w / 2
x_main = x_center - BOX_W / 2

elems = []


def add(s):
    elems.append(s)


def step_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e", bold=False, box_w=None):
    bw = box_w if box_w is not None else BOX_W
    bx = x_center - bw / 2
    n = len(lines)
    box_h = 34 + 22 * (n - 1) + 40
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{bw}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    fw = 'font-weight="bold" ' if bold else ''
    for k, line in enumerate(lines):
        add(f'<text x="{x_center:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" {fw}fill="{text_fill}">{esc(line)}</text>')
    return box_h


def arrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


y = TOP
bh = step_box(y, [STEP1])
y += bh
y_step1_bottom = y

# --- 判定框：name in ignored_dirs = {'nvidia', 'amd'} ？ ---
arrow(x_center, y, y + GAP)
y += GAP
judge_h = 78
judge_lines = ["name ∈ ignored_dirs = {\"nvidia\", \"amd\"}？", "（本包不发行其加载文件，__init__.py:L40-41）"]
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
y0 = y + judge_h / 2 - (len(judge_lines) - 1) * 11 + 4
for k, line in enumerate(judge_lines):
    fs = 13 if k == 0 else 11.5
    fw = 'font-weight="bold" ' if k == 0 else ''
    add(f'<text x="{x_center:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="{fs}" {fw}fill="#78350f">{esc(line)}</text>')
judge_bottom = y + judge_h

# --- 两分支 ---
y_branch = judge_bottom + 56
branch_h = 92
bx0 = x_center - branch_total_w / 2
branch_xs = [bx0, bx0 + BRANCH_W + branch_gap]
branch_cx = [bx + BRANCH_W / 2 for bx in branch_xs]

fork_y = judge_bottom + 18
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{fork_y:.0f}" x2="{branch_cx[1]:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{fork_y:.0f}" x2="{cx:.0f}" y2="{y_branch:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

BRANCH_DATA = [
    ("是（2 个目录：nvidia、amd）", ["continue —— 跳过", "该目录不参与装配"], "#fee2e2", "#b91c1c"),
    ("否（如 ascend）", ["继续往下走装配流程"], "#dcfce7", "#15803d"),
]
for (label, lines, fill, stroke_c), cx in zip(BRANCH_DATA, branch_cx):
    add(f'<text x="{cx:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" font-weight="bold" fill="#334155">{esc(label)}</text>')
    add(f'<rect x="{cx-BRANCH_W/2:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{branch_h}" rx="10" '
        f'fill="{fill}" stroke="{stroke_c}" stroke-width="1.5"/>')
    y0 = y_branch + branch_h / 2 - (len(lines) - 1) * 10 + 4
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*18:.0f}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="12" fill="{stroke_c}">{esc(line)}</text>')

# --- 只有「否」分支（右）汇入后续装配步骤 ---
y = y_branch + branch_h
merge_y = y + 34
right_cx = branch_cx[1]
add(f'<line x1="{right_cx:.0f}" y1="{y:.0f}" x2="{right_cx:.0f}" y2="{merge_y:.0f}" '
    'stroke="#16a34a" stroke-width="2"/>')
add(f'<line x1="{right_cx:.0f}" y1="{merge_y:.0f}" x2="{x_center:.0f}" y2="{merge_y:.0f}" '
    'stroke="#16a34a" stroke-width="2"/>')
tail_top = merge_y + GAP
add(f'<line x1="{x_center:.0f}" y1="{merge_y:.0f}" x2="{x_center:.0f}" y2="{tail_top:.0f}" '
    'stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>')

TAIL_STEPS = [
    "_load_module(name, backends/ascend/compiler.py)\n_load_module(name, backends/ascend/driver.py)",
    "_find_concrete_subclasses(compiler_mod, BaseBackend)\n_find_concrete_subclasses(driver_mod, DriverBase)\n"
    "—— 各自要求恰好 1 个非抽象子类（0 个或 >1 个都 raise）",
    "backends[\"ascend\"] = Backend(compiler=AscendBackend, driver=NPUDriver)",
]

y = tail_top
for i, s in enumerate(TAIL_STEPS):
    if i > 0:
        arrow(x_center, y - GAP, y)
    fill, stroke_c, text_fill = ("#dbeafe", "#1d4ed8", "#1e3a5f") if i != 1 else ("#ede9fe", "#6d28d9", "#3730a3")
    bh = step_box(y, s.split("\n"), fill=fill, stroke=stroke_c, text_fill=text_fill)
    y += bh
    if i < len(TAIL_STEPS) - 1:
        y += GAP

tail_bottom = y

# --- 底部注解 ---
def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


note_lines = [
    "「恰好 1 个非抽象子类」是硬契约（0 个/>1 个都 raise）——新后端只需在 triton/backends/",
    "下新增一个目录、照此结构放两份文件，主库代码零改动即被纳入 backends 表。",
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

out = Path(__file__).with_name("fig-ch26-backend-discovery.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
