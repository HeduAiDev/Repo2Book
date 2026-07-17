#!/usr/bin/env python3
"""ch37-fig-loadbinary-flow: flow 模板。
loadBinary 把 cubin 装载进显存、拿函数句柄、读回 n_regs/n_spills；
若 shared>48KB 且设备支持，显式 cuFuncSetAttribute 开 opt-in 动态共享内存。
全部坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


PAD = 40
BOX_W = 640
GAP = 32
TOP = 92

TITLE = "loadBinary：装载 cubin、读回 n_regs/n_spills，按 48KB 阈值 opt-in 动态共享内存"
SUBTITLE = "third_party/nvidia/backend/driver.c:L93-L140"
TITLE_FONT = 16.5
BOX_W = max(BOX_W, int(cjk_w(TITLE, TITLE_FONT)) + 20)

STEPS = [
    "cuModuleLoadData(cubin) —— 把 cubin 字节装载进显存",
    "cuModuleGetFunction(name) —— 拿函数句柄",
]

x_center = PAD + BOX_W / 2
x_main = PAD

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

# --- 读回属性 + 两个真实 kernel 的回执callout ---
arrow(x_center, y, y + GAP)
y += GAP
attr_h = 54
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{attr_h}" rx="10" '
    'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.8"/>')
add(f'<text x="{x_center:.0f}" y="{y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="13" font-weight="bold" fill="#5b21b6">cuFuncGetAttribute 读回两个 perf 命门数</text>')
add(f'<text x="{x_center:.0f}" y="{y+42:.0f}" text-anchor="middle" font-family="monospace" '
    f'font-size="11.5" fill="#6d28d9">NUM_REGS → n_regs；LOCAL_SIZE_BYTES ÷ 4 → n_spills</text>')
attr_bottom = y + attr_h

# 两个 callout：n_regs=212(mm_kernel) 与 n_spills=8(heavy_kernel)
callout_y = attr_bottom + 28
callout_w = (BOX_W - 24) / 2
c1_x = x_main
c2_x = x_main + callout_w + 24
for cx0, label1, label2, color, stroke in [
    (c1_x, "mm_kernel 读回", "n_regs = 212", "#eff6ff", "#2563eb"),
    (c2_x, "heavy_kernel 读回", "n_spills = 32 bytes ÷ 4 = 8", "#fff7ed", "#c2410c"),
]:
    add(f'<line x1="{cx0+callout_w/2:.0f}" y1="{attr_bottom:.0f}" x2="{cx0+callout_w/2:.0f}" '
        f'y2="{callout_y:.0f}" stroke="{stroke}" stroke-width="1.6" stroke-dasharray="4,3" '
        f'marker-end="url(#a)"/>')
    ch = 54
    add(f'<rect x="{cx0:.0f}" y="{callout_y:.0f}" width="{callout_w:.0f}" height="{ch}" rx="8" '
        f'fill="{color}" stroke="{stroke}" stroke-width="1.4"/>')
    add(f'<text x="{cx0+callout_w/2:.0f}" y="{callout_y+22:.0f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11.5" fill="{stroke}">{esc(label1)}</text>')
    add(f'<text x="{cx0+callout_w/2:.0f}" y="{callout_y+42:.0f}" text-anchor="middle" '
        f'font-family="monospace" font-size="12.5" font-weight="bold" '
        f'fill="{stroke}">{esc(label2)}</text>')
callout_bottom = callout_y + 54

# --- 判定框 ---
# 两示例框只是并列展示 cuFuncGetAttribute 的两种读数，与下方 shared opt-in 判定
# 是同一段代码里的两个独立步骤、彼此无因果——在汇聚箭头旁明说，避免顺序控制流误读。
gap2 = 60
y = callout_bottom + gap2
arrow(x_center, callout_bottom, y)
note_mid = callout_bottom + gap2 / 2
add(f'<text x="{x_center+14:.0f}" y="{note_mid-2:.0f}" text-anchor="start" '
    f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">示例读数，与下方判定无因果</text>')
add(f'<text x="{x_center+14:.0f}" y="{note_mid+12:.0f}" text-anchor="start" '
    f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">——仅示意 cuFuncGetAttribute 两种读法</text>')
judge_h = 54
add(f'<rect x="{x_main:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="13.5" font-weight="bold" fill="#78350f">shared &gt; 49152（48KB 静态共享内存硬上限）？</text>')
add(f'<text x="{x_center:.0f}" y="{y+42:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#92400e">driver.c:L132 —— shared &gt; 49152 且 shared_optin &gt; 49152</text>')
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

add(f'<text x="{branch_cx[0]:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#334155">否（add_kernel 等）</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#c2410c">是（mm_kernel：shared=65536）</text>')

y_branch = fork_y + 30

# 左分支：无需 opt-in
noop_h = 60
add(f'<rect x="{branch_xs[0]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{noop_h}" rx="10" '
    'fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>')
add(f'<text x="{branch_cx[0]:.0f}" y="{y_branch+26:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" fill="#334155">不申请动态共享内存，</text>')
add(f'<text x="{branch_cx[0]:.0f}" y="{y_branch+46:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" fill="#334155">句柄直接可用</text>')
left_bottom = y_branch + noop_h

# 右分支：opt-in
optin_h = 78
add(f'<rect x="{branch_xs[1]:.0f}" y="{y_branch:.0f}" width="{BRANCH_W}" height="{optin_h}" rx="10" '
    'fill="#ffedd5" stroke="#c2410c" stroke-width="1.8"/>')
add(f'<text x="{branch_cx[1]:.0f}" y="{y_branch+22:.0f}" text-anchor="middle" font-family="monospace" '
    f'font-size="11" font-weight="bold" fill="#9a3412">cuFuncSetAttribute(</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{y_branch+40:.0f}" text-anchor="middle" font-family="monospace" '
    f'font-size="10.5" fill="#9a3412">MAX_DYNAMIC_SHARED_SIZE_BYTES</text>')
add(f'<text x="{branch_cx[1]:.0f}" y="{y_branch+58:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11" fill="#7c2d12">= 设备 opt-in 上限 101376 − shared_static（未实测）)</text>')
right_bottom = y_branch + optin_h

bottom_y = max(left_bottom, right_bottom)

# --- 底部注解 ---
note_lines = [
    "不 opt-in 就编不出大 block ——共享内存超 48KB 硬上限的 kernel（如本例 mm_kernel 的",
    "64KB）必须在装载期显式申请动态额度，本例设备 opt-in 上限 101376 字节仍够用。",
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
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("ch37-fig-loadbinary-flow.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
