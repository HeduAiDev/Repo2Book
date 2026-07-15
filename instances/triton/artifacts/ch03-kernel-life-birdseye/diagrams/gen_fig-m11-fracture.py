#!/usr/bin/env python3
"""fig-m11-fracture: 无 GPU 断裂线 —— make_ir..make_cubin 六级 headless 可跑
（实测 CUDA_VISIBLE_DEVICES 置空仍产出全套），断裂线落在 _init_handles 的
load_binary。before-after 变体：左区(绿·headless 可跑) / 右区(红·需真 GPU)，
中间一道断裂线。数字取自 explainer figure_specs['fig-m11-fracture'].numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


LEFT_TITLE = "headless 可跑（无 GPU）"
LEFT_STEPS = [
    ("make_ir → … → make_ptx", "纯编译：Python 前端 + libtriton.so（C++/MLIR）",
     "产出 56/38/39/150/377 行文本 · 未建 CUDA context"),
    ("make_cubin", "ptxas 子进程（CPU 程序，triton wheel 自带 · L341）",
     "9488 字节 cubin · 编译目标 sm_90a ≠ 本机卡"),
]
RIGHT_TITLE = "需要真 GPU（断裂线起）"
RIGHT_STEPS = [
    ("_init_handles → load_binary", "CUDA driver 把 cubin 灌进显存（compiler.py:L390）",
     "不能（断裂线，本 run 刻意未跨越）"),
]

PAD, COL_W, BOX_H, VGAP, TOP = 46, 560, 74, 22, 118
GUTTER = 70

w = PAD * 2 + COL_W * 2 + GUTTER
n_rows = max(len(LEFT_STEPS), len(RIGHT_STEPS))
h = TOP + n_rows * (BOX_H + VGAP) + 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">无 GPU 断裂线：能编译不等于能跑</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("headless 可跑级数=6（make_ir + 五级 stages）· 断裂线唯一且位置固定")}</text>']

left_x = PAD
right_x = PAD + COL_W + GUTTER

L.append(f'<text x="{left_x+COL_W/2}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#166534">{esc(LEFT_TITLE)}</text>')
L.append(f'<text x="{right_x+COL_W/2}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#991b1b">{esc(RIGHT_TITLE)}</text>')


def draw_col(x, steps, fill, stroke, text_fill, marker):
    for i, (title, mid, ev) in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        L.append(f'<rect x="{x}" y="{y}" width="{COL_W}" height="{BOX_H}" rx="10" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{x+16}" y="{y+22}" font-family="sans-serif" font-size="13.5" '
                  f'font-weight="bold" fill="{text_fill}">{esc(title)}</text>')
        L.append(f'<text x="{x+16}" y="{y+40}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#334155">{esc(mid)}</text>')
        L.append(f'<text x="{x+16}" y="{y+58}" font-family="sans-serif" font-size="11.5" '
                  f'font-weight="bold" fill="{stroke}">{esc(ev)}</text>')
        if i < len(steps) - 1:
            cx = x + COL_W / 2
            L.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                      f'stroke="{stroke}" stroke-width="1.5" marker-end="url(#{marker})"/>')


draw_col(left_x, LEFT_STEPS, "#dcfce7", "#16a34a", "#14532d", "g")
draw_col(right_x, RIGHT_STEPS, "#fee2e2", "#b91c1c", "#7f1d1d", "r")

# 断裂线：竖直锯齿虚线，位于两栏之间
fx = PAD + COL_W + GUTTER / 2
fy0, fy1 = TOP - 30, TOP + n_rows * (BOX_H + VGAP) - VGAP + 10
L.append(f'<line x1="{fx}" y1="{fy0}" x2="{fx}" y2="{fy1}" stroke="#b91c1c" '
          'stroke-width="2.5" stroke-dasharray="10,6"/>')
L.append(f'<text x="{fx}" y="{fy1+22}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" font-weight="bold" fill="#b91c1c" transform="rotate(0)">断裂线</text>')

# 跨栏箭头：从左栏最后一格 -> 断裂线 -> 右栏第一格
last_left_y = TOP + (len(LEFT_STEPS) - 1) * (BOX_H + VGAP) + BOX_H / 2
first_right_y = TOP + BOX_H / 2
L.append(f'<line x1="{left_x+COL_W}" y1="{last_left_y}" x2="{right_x}" y2="{first_right_y}" '
          'stroke="#b91c1c" stroke-width="2" stroke-dasharray="4,4" marker-end="url(#r)"/>')

foot_y = TOP + n_rows * (BOX_H + VGAP) + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("TRITON_KERNEL_DUMP=1（compiler.py:L237）即可在无卡机上复现左侧全部六级产物")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("绿=headless 可跑 红=需真 GPU · 断裂线右侧才第一次触碰设备")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m11-fracture.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
