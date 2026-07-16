#!/usr/bin/env python3
"""flow 模板:counter-based 无状态 RNG。两条输入(seed,offset)各自独立流过 10 轮 Philox
混合,直接产出随机数;同参数重跑得到逐位相同结果(可复现),换 offset 结果全变(可并行独立)。
数据来自 traces/algorithms.json philox。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def wrap_nums(nums, per_line=2):
    lines = []
    for i in range(0, len(nums), per_line):
        lines.append(", ".join(str(n) for n in nums[i:i+per_line]))
    return lines

OUT_A = [2632642643, 2012563771, 314527917, 1463989207]
OUT_A_RERUN = OUT_A  # 逐位相同
OUT_B = [4242219303, 1404726525, 2207210094, 1951270651]

W, PAD, TOP = 1180, 40, 130
BOX_W, BOX_H = 260, 60
MID_W, MID_H = 300, 70
GAP_Y = 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 620">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{W}" height="620" fill="white"/>',
     f'<text x="{PAD}" y="40" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">Philox:counter-based 无状态 RNG,同参必复现、换计数器即独立并行</text>',
     f'<text x="{PAD}" y="60" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'random.py:L5 N_ROUNDS_DEFAULT=10;seed=42(0x2A);offset 即计数器,无跨调用可变状态</text>']

# 左侧两个输入框(offset0 首次 / offset0 重跑),右侧一个(offset1)
cx = W / 2
in1_x, in1_y = cx - 420, TOP
in2_x, in2_y = cx - 420, TOP + GAP_Y + BOX_H
in3_x, in3_y = cx - 420, TOP + 2*(GAP_Y + BOX_H)

def box(x, y, w, h, text_lines, fill, stroke, bold=False):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
              f'stroke="{stroke}" stroke-width="1.5"/>')
    n = len(text_lines)
    y0 = y + h/2 - (n-1)*8
    for i, t in enumerate(text_lines):
        wt = 'font-weight="bold" ' if bold else ''
        L.append(f'<text x="{x+w/2}" y="{y0+i*16+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" {wt}fill="#0f172a">{esc(t)}</text>')

box(in1_x, in1_y, BOX_W, BOX_H, ["首次 launch", "seed=42, offset=0"], "#e2e8f0", "#64748b")
box(in2_x, in2_y, BOX_W, BOX_H, ["同参重跑", "seed=42, offset=0"], "#e2e8f0", "#64748b")
box(in3_x, in3_y, BOX_W, BOX_H, ["换计数器", "seed=42, offset=1"], "#e2e8f0", "#64748b")

mid_x, mid_y = cx - MID_W/2, TOP + GAP_Y + BOX_H/2 - MID_H/2
box(mid_x, mid_y, MID_W, MID_H, ["Philox 混合", "10 轮 umulhi + 异或", "(无共享状态)"],
    "#dbeafe", "#1e40af", bold=True)

out1_x, out1_y = cx + 420 - BOX_W, in1_y
out2_x, out2_y = cx + 420 - BOX_W, in2_y
out3_x, out3_y = cx + 420 - BOX_W, in3_y

box(out1_x, out1_y, BOX_W, BOX_H, wrap_nums(OUT_A), "#ecfdf5", "#047857")
box(out2_x, out2_y, BOX_W, BOX_H, wrap_nums(OUT_A_RERUN), "#ecfdf5", "#047857")
box(out3_x, out3_y, BOX_W, BOX_H, wrap_nums(OUT_B), "#fee2e2", "#b91c1c")

# 箭头:input -> mid(左边缘 3 个不同高度落点,避免重叠成一条线) / mid -> output(右边缘同理)
mid_left_ys = [mid_y + MID_H*0.22, mid_y + MID_H*0.5, mid_y + MID_H*0.78]
for (ix, iy), ty in zip([(in1_x, in1_y), (in2_x, in2_y), (in3_x, in3_y)], mid_left_ys):
    x1, y1 = ix + BOX_W, iy + BOX_H/2
    x2 = mid_x
    L.append(f'<path d="M{x1},{y1} L{x2-10},{ty}" stroke="#334155" stroke-width="1.5" '
              f'fill="none" marker-end="url(#a)"/>')
mid_right_ys = mid_left_ys
for (ox, oy), sy in zip([(out1_x, out1_y), (out2_x, out2_y), (out3_x, out3_y)], mid_right_ys):
    x1 = mid_x + MID_W
    x2, y2 = ox, oy + BOX_H/2
    L.append(f'<path d="M{x1},{sy} L{x2-10},{y2}" stroke="#334155" stroke-width="1.5" '
              f'fill="none" marker-end="url(#a)"/>')

# 标注:重跑结果逐位相同 / 换 offset 结果全变
tag_y1 = in2_y + BOX_H/2 + 5
L.append(f'<text x="{out1_x+BOX_W/2}" y="{out1_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#047857">randint4x(seed=42,offset=0)</text>')
L.append(f'<text x="{out2_x+BOX_W/2}" y="{out2_y+BOX_H+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#047857">'
          f'逐位相同 = 可复现</text>')
L.append(f'<text x="{out3_x+BOX_W/2}" y="{out3_y+BOX_H+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#b91c1c">'
          f'四元组全变 = 相互独立</text>')

foot_y = TOP + 3*(GAP_Y+BOX_H) + 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'无状态:第 offset 个数只依赖 (seed,offset),不依赖前 offset-1 个 -&gt; 千万线程各算各的、零同步</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch09-philox-stateless.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
