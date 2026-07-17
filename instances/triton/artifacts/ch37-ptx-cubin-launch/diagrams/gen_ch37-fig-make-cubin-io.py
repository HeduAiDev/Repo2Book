#!/usr/bin/env python3
"""ch37-fig-make-cubin-io: flow 模板。
make_cubin 起 ptxas 子进程,把 PTX 文本汇编成 cubin 字节;-v 让 ptxas 把每核
寄存器/spill 占用回执打到 stderr;返回码非零则按分类映射成可读 RuntimeError。
全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
BOX_W = 620
STEP_H = 60
GAP = 32
TOP = 92

TITLE = "make_cubin：起 ptxas 子进程，把 PTX 文本汇编成 cubin 字节"
SUBTITLE = "third_party/nvidia/backend/compiler.py:L340-L380 —— 编译链唯一起外部工具进程的一段"

STEPS = [
    "定位 ptxas —— _path_to_binary('ptxas')：TRITON_PTXAS_PATH env 或 backend bin/ptxas",
    "拼命令：ptxas -lineinfo -v --gpu-name=sm_120a add_kernel.ptx -o add_kernel.o",
    "subprocess.run(ptxas_cmd, check=True) —— 起子进程",
]

JUDGE_LABEL = "返回码 == 0 ?"

x_center = PAD + BOX_W / 2
x_main = PAD

L = [None]
elems = []


def add(s):
    elems.append(s)


def step_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e", w=BOX_W, cx=None, x=None):
    cx = x_center if cx is None else cx
    x = x_main if x is None else x
    n = len(lines)
    box_h = 30 + 22 * (n - 1) + 30
    add(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 11 + 5
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*22:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" fill="{text_fill}">{esc(line)}</text>')
    return box_h


def arrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


y = TOP
for i, s in enumerate(STEPS):
    if i > 0:
        arrow(x_center, y, y + GAP)
        y += GAP
    bh = step_box(y, [s])
    y += bh

# --- 判定框 ---
arrow(x_center, y, y + GAP)
y += GAP
judge_h = 50
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judge_h/2+5:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" '
    f'fill="#78350f">{esc(JUDGE_LABEL)}</text>')
judge_bottom = y + judge_h

# --- 两分支 ---
BRANCH_W = 300
branch_gap = 40
branch_total_w = BRANCH_W * 2 + branch_gap
bx0 = x_center - branch_total_w / 2
branch_xs = [bx0, bx0 + BRANCH_W + branch_gap]
branch_cx = [bx + BRANCH_W / 2 for bx in branch_xs]

fork_y = judge_bottom + 20
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{fork_y:.0f}" x2="{branch_cx[1]:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{fork_y:.0f}" x2="{cx:.0f}" y2="{fork_y+30:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

BRANCH_LABELS = ["是（成功）", "否（非零返回码）"]
y_branch = fork_y + 30

add(f'<text x="{branch_cx[0]:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#15803d">{esc(BRANCH_LABELS[0])}</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#b91c1c">{esc(BRANCH_LABELS[1])}</text>')

# --- 成功分支 ---
succ_lines_top = [
    'stderr: "Used 28 registers,',
    '0 bytes spill"',
]
succ_h1 = 60
add(f'<rect x="{branch_xs[0]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{succ_h1}" rx="10" '
    'fill="#dcfce7" stroke="#15803d" stroke-width="1.5"/>')
for k, line in enumerate(succ_lines_top):
    add(f'<text x="{branch_cx[0]:.0f}" y="{y_branch+24+k*18:.0f}" text-anchor="middle" '
        f'font-family="monospace" font-size="11.5" fill="#166534">{esc(line)}</text>')
y2 = y_branch + succ_h1
arrow(branch_cx[0], y2, y2 + 26, color="#15803d")
y2 += 26
succ_h2 = 54
add(f'<rect x="{branch_xs[0]:.0f}" y="{y2:.0f}" width="{BRANCH_W}" height="{succ_h2}" rx="10" '
    'fill="#ecfdf5" stroke="#22c55e" stroke-width="1.6"/>')
add(f'<text x="{branch_cx[0]:.0f}" y="{y2+22:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" fill="#166534">读回 cubin 字节：</text>')
add(f'<text x="{branch_cx[0]:.0f}" y="{y2+42:.0f}" text-anchor="middle" '
    f'font-family="monospace" font-size="14" font-weight="bold" '
    f'fill="#15803d">add_kernel.o = 10544 字节</text>')
succ_bottom = y2 + succ_h2

# --- 失败分支 ---
fail_lines = [
    "255 → Internal Triton PTX codegen error",
    "128+SIGSEGV → ptxas raised SIGSEGV",
    "其它 → ptxas failed with error code N",
]
fail_h1 = 30 + 18 * len(fail_lines)
add(f'<rect x="{branch_xs[1]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{fail_h1}" rx="10" '
    'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
for k, line in enumerate(fail_lines):
    add(f'<text x="{branch_cx[1]:.0f}" y="{y_branch+22+k*18:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11" fill="#7f1d1d">{esc(line)}</text>')
yf = y_branch + fail_h1
arrow(branch_cx[1], yf, yf + 26, color="#b91c1c")
yf += 26
fail_h2 = 54
add(f'<rect x="{branch_xs[1]:.0f}" y="{yf:.0f}" width="{BRANCH_W}" height="{fail_h2}" rx="10" '
    'fill="#fef2f2" stroke="#dc2626" stroke-width="1.6"/>')
add(f'<text x="{branch_cx[1]:.0f}" y="{yf+22:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#b91c1c">RuntimeError</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{yf+42:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11" fill="#7f1d1d">附 stderr + 可复现命令</text>')
fail_bottom = yf + fail_h2

bottom_y = max(succ_bottom, fail_bottom)

# --- 底部注解:编译链段数 ---
note_lines = [
    "编译链共 5 段（ttir/ttgir/llir/ptx/cubin，add_stages 注册）——make_cubin 是",
    "唯一起外部进程的一段（5/1）；-v 拿到的寄存器/spill 回执是判断 kernel 性能的第一手数据。",
]
note_top = bottom_y + 34
note_h = 24 * len(note_lines) + 20
w = PAD * 2 + BOX_W
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

out = Path(__file__).with_name("ch37-fig-make-cubin-io.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
