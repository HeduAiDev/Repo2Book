#!/usr/bin/env python3
"""fig-m1-jit-cache：NPUUtils 首次实例化即以 npu_utils.cpp 源码 md5 为缓存 key，
决定「复用已编 .so」还是「就地 _build_npu_ext 重编」。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "NPUUtils：首次实例化即时编译 npu_utils.so，md5 做缓存 key"
SUBTITLE = "third_party/ascend/backend/driver.py —— 源不变复用旧 .so，改一行 C++ 即换 key 重编"

PAD = 40
BOX_W = 640
GAP = 26
TOP = 96

elems = []


def add(s):
    elems.append(s)


def step_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e",
             bold_first=True, box_w=None, cx=None):
    bw = box_w if box_w is not None else BOX_W
    ccx = cx if cx is not None else x_center
    bx = ccx - bw / 2
    n = len(lines)
    box_h = 34 + 22 * (n - 1) + 34
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{bw}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if (bold_first and k == 0) else ''
        fs = 13 if k == 0 else 11.5
        fill_c = text_fill if k == 0 else "#334155"
        add(f'<text x="{ccx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="{fs}" {fw}fill="{fill_c}">{esc(line)}</text>')
    return box_h


def arrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


x_center = PAD + BOX_W / 2
w = PAD * 2 + BOX_W

y = TOP
bh = step_box(y, [
    "NPUUtils() 首次调用 —— __new__",
    "hasattr(cls,'instance')？是→复用旧单例／否→新建实例",
    "(driver.py:L48-L51；单例 instance 数恒为 1)",
])
y += bh + GAP

arrow(x_center, y - GAP, y)
bh = step_box(y, [
    "__init__：读 npu_utils.cpp 源文本 src",
    "(driver.py:L53-L56)",
])
y += bh + GAP

arrow(x_center, y - GAP, y)
bh = step_box(y, [
    "key = md5(src).hexdigest() —— 32 位十六进制",
    "cache = get_cache_manager(key)；cache_path = cache.get_file(\"npu_utils.so\")",
    "(driver.py:L57-L60)",
])
y += bh + GAP

arrow(x_center, y - GAP, y)
judge_h = 66
judge_lines = ["cache_path is None？（缓存命中判定，driver.py:L61）"]
add(f'<rect x="{x_center-BOX_W/2:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judge_h/2+5:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
    f'fill="#78350f">{esc(judge_lines[0])}</text>')
judge_bottom = y + judge_h

# 两分支
y_branch = judge_bottom + 56
branch_w = 300
branch_gap = 30
branch_total_w = branch_w * 2 + branch_gap
bx0 = x_center - branch_total_w / 2
branch_xs = [bx0, bx0 + branch_w + branch_gap]
branch_cx = [bx + branch_w / 2 for bx in branch_xs]

fork_y = judge_bottom + 18
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{fork_y:.0f}" x2="{branch_cx[1]:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{fork_y:.0f}" x2="{cx:.0f}" y2="{y_branch:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

BRANCH_DATA = [
    ("否（命中）", ["直接用 cache_path 里", "已编好的 .so"], "#dcfce7", "#15803d"),
    ("是（未命中）", ["_build_npu_ext 就地把 .cpp 编成 .so", "→ cache.put 落盘 (driver.py:L62-L68)"], "#fee2e2", "#b91c1c"),
]
branch_hs = []
for (label, lines, fill, stroke_c), cx in zip(BRANCH_DATA, branch_cx):
    add(f'<text x="{cx:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="12" font-weight="bold" fill="#334155">{esc(label)}</text>')
    bh_i = step_box(y_branch, lines, fill=fill, stroke=stroke_c, text_fill=stroke_c,
                     bold_first=False, box_w=branch_w, cx=cx)
    branch_hs.append(bh_i)
branch_h = max(branch_hs)

# 两分支汇入 importlib 步骤
y_after_branch = y_branch + branch_h
merge_y = y_after_branch + 34
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{y_after_branch:.0f}" x2="{cx:.0f}" y2="{merge_y:.0f}" '
        'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{merge_y:.0f}" x2="{branch_cx[1]:.0f}" y2="{merge_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
tail_top = merge_y + GAP
add(f'<line x1="{x_center:.0f}" y1="{merge_y:.0f}" x2="{x_center:.0f}" y2="{tail_top:.0f}" '
    'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

y = tail_top
bh = step_box(y, [
    "importlib 动态加载 .so → self.npu_utils_mod",
    "(driver.py:L69-L73)",
])
y += bh + GAP

arrow(x_center, y - GAP, y)
bh = step_box(y, [
    "对外暴露：load_binary / get_arch / get_aicore_num",
    "(driver.py:L77-L79, L90-L92, L94-L97)",
], fill="#ede9fe", stroke="#6d28d9", text_fill="#3730a3")
y += bh

tail_bottom = y


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


note_lines = [
    "同一 CANN/ABI 环境下第一次付编译代价、之后复用；源码一改 md5 变、",
    "key 变、自动重编——无需为每个环境预打包 .so。",
]
note_top = tail_bottom + 30
note_h = 24 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+24+i*24:.0f}" font-family="sans-serif" '
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

out = Path(__file__).with_name("fig-m1-jit-cache.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
