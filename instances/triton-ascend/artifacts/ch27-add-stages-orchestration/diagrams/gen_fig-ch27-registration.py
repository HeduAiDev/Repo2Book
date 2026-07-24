#!/usr/bin/env python3
"""fig-ch27-registration：add_stages 只做『登记』不做『编译』——
按名把三段可调用体塞进 stages dict，两个 if 只改『登记哪个实现』，不在 pass 链内部加分支。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "add_stages：只『登记』，不『编译』"
SUBTITLE = "third_party/ascend/backend/compiler.py:L939-968"

PAD = 40
CENTER_W = 560
TOP = 92
GAP = 24

elems = []


def add(s):
    elems.append(s)


def box(cx, y, lines, w=CENTER_W, fill="#e0f2fe", stroke="#0369a1",
        text_fill="#0c4a6e", bold=False, fs=13):
    n = len(lines)
    box_h = 30 + 20 * (n - 1) + 30
    bx = cx - w / 2
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{w:.0f}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    y0 = y + box_h / 2 - (n - 1) * 10 + 5
    fw = 'font-weight="bold" ' if bold else ''
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{fs}" {fw}fill="{text_fill}">{esc(line)}</text>')
    return box_h


def varrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


LEFT_W = 380
RIGHT_W = 560
GUTTER = 56
total_w = LEFT_W + GUTTER + RIGHT_W
w = PAD * 2 + total_w
x_center = PAD + total_w / 2
x_left = PAD + LEFT_W / 2
x_right = PAD + LEFT_W + GUTTER + RIGHT_W / 2

y = TOP
bh = box(x_center, y, ["add_stages(self, stages, options)"], w=460,
         fill="#dcfce7", stroke="#15803d", text_fill="#14532d", bold=True, fs=13.5)
y += bh
varrow(x_center, y, y + GAP)
y += GAP

bh = box(x_center, y, ["stages[\"ttir\"] = make_ttir", "无条件登记 · L941"], w=460)
y += bh
varrow(x_center, y, y + GAP)
y += GAP

judge_h = 58
add(f'<rect x="{x_center-260:.0f}" y="{y:.0f}" width="520" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judge_h/2+5:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="14" font-weight="bold" fill="#78350f">{esc("分叉①：force_simt_only ？ · L942")}</text>')
judge_bottom = y + judge_h

fork_y = judge_bottom + 22
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{x_left:.0f}" y1="{fork_y:.0f}" x2="{x_right:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
varrow(x_left, fork_y, fork_y + GAP)
varrow(x_right, fork_y, fork_y + GAP)
add(f'<text x="{x_left:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#334155">True</text>')
add(f'<text x="{x_right:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#334155">False</text>')

y_branch = fork_y + GAP

# --- 左分支：force_simt_only=True，快路径 2 段 ---
ly = y_branch
lbh = box(x_left, ly, ["stages[\"npubin\"] = ttir_to_npubin", "L943-948"], w=LEFT_W,
          fill="#fee2e2", stroke="#b91c1c", text_fill="#7f1d1d", fs=12.5)
ly += lbh
varrow(x_left, ly, ly + GAP)
ly += GAP
lbh2 = box(x_left, ly, ["return"], w=LEFT_W,
           fill="#fef2f2", stroke="#b91c1c", text_fill="#7f1d1d", bold=True)
ly += lbh2
tag_y = ly + 16
add(f'<rect x="{x_left-LEFT_W/2:.0f}" y="{tag_y:.0f}" width="{LEFT_W}" height="34" rx="8" '
    'fill="#fecaca" stroke="#b91c1c" stroke-width="1"/>')
add(f'<text x="{x_left:.0f}" y="{tag_y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#7f1d1d">快路径：登记 2 段 stage</text>')
left_bottom = tag_y + 34

# --- 右分支：force_simt_only=False，继续登记 ttadapter，再第二次判定 ---
ry = y_branch
rbh = box(x_right, ry, ["stages[\"ttadapter\"] = ttir_to_linalg", "(named_ops=True) · L949-951"],
          w=RIGHT_W, fill="#dcfce7", stroke="#15803d", text_fill="#14532d", fs=12.5)
ry += rbh
varrow(x_right, ry, ry + GAP)
ry += GAP

judge2_h = 54
add(f'<rect x="{x_right-RIGHT_W/2:.0f}" y="{ry:.0f}" width="{RIGHT_W}" height="{judge2_h}" rx="12" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_right:.0f}" y="{ry+judge2_h/2+5:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#78350f">{esc("分叉②：compile_on_910_95 ？ · L952-963")}</text>')
judge2_bottom = ry + judge2_h

fork2_y = judge2_bottom + 20
x_sub_l = x_right - RIGHT_W / 2 + RIGHT_W / 4
x_sub_r = x_right + RIGHT_W / 2 - RIGHT_W / 4
add(f'<line x1="{x_right:.0f}" y1="{judge2_bottom:.0f}" x2="{x_right:.0f}" y2="{fork2_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{x_sub_l:.0f}" y1="{fork2_y:.0f}" x2="{x_sub_r:.0f}" y2="{fork2_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
varrow(x_sub_l, fork2_y, fork2_y + GAP)
varrow(x_sub_r, fork2_y, fork2_y + GAP)
add(f'<text x="{x_sub_l:.0f}" y="{fork2_y-6:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11.5" font-weight="bold" fill="#334155">True</text>')
add(f'<text x="{x_sub_r:.0f}" y="{fork2_y-6:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="11.5" font-weight="bold" fill="#334155">False</text>')

y_sub = fork2_y + GAP
SUB_W = RIGHT_W / 2 - 20
sub_l_h = box(x_sub_l, y_sub, ["npubin =", "..._910_95"], w=SUB_W,
              fill="#e0e7ff", stroke="#4338ca", text_fill="#312e81", fs=11)
sub_r_h = box(x_sub_r, y_sub, ["npubin =", "..._A2_A3"], w=SUB_W,
              fill="#e0e7ff", stroke="#4338ca", text_fill="#312e81", fs=11)
sub_bottom = y_sub + max(sub_l_h, sub_r_h)

tag2_y = sub_bottom + 14
add(f'<rect x="{x_right-RIGHT_W/2:.0f}" y="{tag2_y:.0f}" width="{RIGHT_W}" height="34" rx="8" '
    'fill="#bbf7d0" stroke="#15803d" stroke-width="1"/>')
add(f'<text x="{x_right:.0f}" y="{tag2_y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#14532d">常规路径：登记 3 段 stage</text>')
right_bottom = tag2_y + 34

content_bottom = max(left_bottom, right_bottom)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


note_lines = [
    "两处 if 只改『登记哪个实现』——force_simt_only 为真时 ttadapter 段整段不注册（少登记而非加分支）；",
    "compile_on_910_95 只切换 npubin 段末两个候选实现，二者细节归下一章 bishengir 边界。",
]
note_top = content_bottom + 32
note_w_needed = max(cjk_w(s, 12.5) for s in note_lines) + 32
total_w = max(total_w, note_w_needed)
w = PAD * 2 + total_w
note_h = 22 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-ch27-registration.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
