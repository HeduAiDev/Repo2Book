#!/usr/bin/env python3
"""ch37-fig-launch-dispatch-chain: flow 模板。
kernel[grid](*args) 到 cuLaunchKernel/cuLaunchKernelEx 的发射调用链：
_init_handles 懒装载 -> runner -> 生成的 launch -> _launch 按 grid/num_ctas 两级判定分派。
全部坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


PAD = 40
BOX_W = 640
GAP = 30
TOP = 92
LEFT_W = 220
LEFT_MARGIN = LEFT_W + 60  # 给判定 A 的"否"侧支线留出的左侧空间

TITLE = "发射调用链：kernel[grid](*args) → _init_handles 懒装载 → cuLaunchKernel(Ex)"
SUBTITLE = "third_party/nvidia/backend/driver.py:L117-L239 —— CudaLauncher.__call__"

STEPS = [
    "kernel[grid](*args) —— 用户调用",
    "首调触发 _init_handles()：load_binary 懒装载\n（module/function/n_regs/n_spills）、建 launcher",
    "runner：self.run(grid, stream, function, metadata, hooks, *args)",
    "生成的 launch(...)：PyArg_ParseTuple 解析实参、getPointer 取指针\n→ 调 _launch(...)",
]

x_main = PAD + LEFT_MARGIN
x_center = x_main + BOX_W / 2

elems = []


def add(s):
    elems.append(s)


def step_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e", w=BOX_W, cx=None, x=None):
    cx = x_center if cx is None else cx
    x = x_main if x is None else x
    n = len(lines)
    box_h = 26 + 22 * (n - 1) + 34
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
    bh = step_box(y, s.split("\n"))
    y += bh

# --- 判定 A：grid 是否非空 ---
arrow(x_center, y, y + GAP)
y += GAP
judgeA_h = 50
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judgeA_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judgeA_h/2+5:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
    f'fill="#78350f">gridX × gridY × gridZ &gt; 0 ？（driver.py:L210）</text>')
judgeA_bottom = y + judgeA_h

# 分支：左=否(不发射，短支线)；右=继续主干
left_x = PAD
left_cx = left_x + LEFT_W / 2
forkA_y = judgeA_bottom + 40

# 左分支线（从判定框左侧引出）
add(f'<line x1="{x_main:.0f}" y1="{judgeA_bottom-judgeA_h/2:.0f}" x2="{left_cx:.0f}" '
    f'y2="{judgeA_bottom-judgeA_h/2:.0f}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3"/>')
add(f'<text x="{left_cx:.0f}" y="{judgeA_bottom-judgeA_h/2-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#64748b">否</text>')
add(f'<line x1="{left_cx:.0f}" y1="{judgeA_bottom-judgeA_h/2:.0f}" x2="{left_cx:.0f}" '
    f'y2="{forkA_y:.0f}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#a)"/>')
left_h = 50
add(f'<rect x="{left_x:.0f}" y="{forkA_y:.0f}" width="{LEFT_W}" height="{left_h}" rx="8" '
    'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')
add(f'<text x="{left_cx:.0f}" y="{forkA_y+30:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#334155">空 grid，不发射</text>')

# 右（主干继续）
arrow(x_center, judgeA_bottom, judgeA_bottom + GAP)
add(f'<text x="{x_center+16:.0f}" y="{judgeA_bottom+GAP-6:.0f}" text-anchor="start" '
    f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#78350f">是</text>')
y = judgeA_bottom + GAP

# --- 判定 B：num_ctas == 1 ---
judgeB_h = 50
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judgeB_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judgeB_h/2+5:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
    f'fill="#78350f">num_ctas == 1 ？（driver.py:L211）</text>')
judgeB_bottom = y + judgeB_h

# 两分支
BRANCH_W = 300
branch_gap = 40
branch_total_w = BRANCH_W * 2 + branch_gap
bx0 = x_center - branch_total_w / 2
branch_xs = [bx0, bx0 + BRANCH_W + branch_gap]
branch_cx = [bx + BRANCH_W / 2 for bx in branch_xs]

forkB_y = judgeB_bottom + 20
add(f'<line x1="{x_center:.0f}" y1="{judgeB_bottom:.0f}" x2="{x_center:.0f}" y2="{forkB_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{branch_cx[0]:.0f}" y1="{forkB_y:.0f}" x2="{branch_cx[1]:.0f}" y2="{forkB_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
for cx in branch_cx:
    add(f'<line x1="{cx:.0f}" y1="{forkB_y:.0f}" x2="{cx:.0f}" y2="{forkB_y+28:.0f}" '
        'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

add(f'<text x="{branch_cx[0]:.0f}" y="{forkB_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#15803d">是（经典 API）</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{forkB_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#c2410c">否（Hopper cluster）</text>')

y_branch = forkB_y + 28

# 左：cuLaunchKernel
left_lines = [
    "cuLaunchKernel(function,",
    "gridX, gridY, gridZ,",
    "32 × num_warps, 1, 1, ...)",
]
lh = 26 + 20 * (len(left_lines) - 1) + 30
add(f'<rect x="{branch_xs[0]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{lh}" rx="10" '
    'fill="#dcfce7" stroke="#15803d" stroke-width="1.8"/>')
for k, line in enumerate(left_lines):
    yy = y_branch + 24 + k * 20
    add(f'<text x="{branch_cx[0]:.0f}" y="{yy:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="11.5" fill="#166534">{esc(line)}</text>')
left_bottom = y_branch + lh

# 右：cuLaunchKernelEx
right_lines = [
    "组 CUlaunchConfig +",
    "2 个 launchAttr：",
    "CLUSTER_DIMENSION /",
    "CLUSTER_SCHEDULING_POLICY",
    "→ dlsym 出 cuLaunchKernelEx(...)",
]
rh = 26 + 20 * (len(right_lines) - 1) + 30
add(f'<rect x="{branch_xs[1]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{rh}" rx="10" '
    'fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/>')
for k, line in enumerate(right_lines):
    yy = y_branch + 24 + k * 20
    add(f'<text x="{branch_cx[1]:.0f}" y="{yy:.0f}" text-anchor="middle" font-family="monospace" '
        f'font-size="11" fill="#9a3412">{esc(line)}</text>')
right_bottom = y_branch + rh

bottom_y = max(left_bottom, right_bottom, forkA_y + left_h)

# --- 底部注解 ---
note_lines = [
    "cluster 场景才会走 cuLaunchKernelEx——普通核（num_ctas==1）恒走 cuLaunchKernel，",
    "blockDim 恒为 32×num_warps（每 warp 32 线程）；空 grid（三维之积为 0）在判定 A 处提前拦下、不发射。",
]
note_top = bottom_y + 34
note_h = 24 * len(note_lines) + 20
w = PAD + LEFT_MARGIN + BOX_W + PAD
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
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch37-fig-launch-dispatch-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
