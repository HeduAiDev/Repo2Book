#!/usr/bin/env python3
"""fig-ch07-02-custom-semantic-swimlane — swimlane 模板（al.custom 一次调用的六步）。
三条泳道：Python 侧(al.custom) / 注册表(_custom_op_registry) / IR 侧(builder)。
同泳道自述步骤画成生命线上的小圆点+右侧文字（不是跨道箭头）；跨道步骤画横向箭头。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


LANES = ["Python 侧\n(al.custom)", "注册表\n(_custom_op_registry)", "IR 侧\n(builder)"]

# (from_lane_idx, to_lane_idx, 标签行(list))；from==to 表示同泳道自述步骤
STEPS = [
    (0, 0, ["① al.custom(name, src, index, ...) → custom(@core.builtin) → custom_semantic"]),
    (0, 1, ["② _get_op_class('__builtin_index_select')  查注册表"]),
    (1, 0, ["命中：返回真实注册类（非 __builtin_ 哑类兜底）"]),
    (0, 0, ["③ _init_op → __init__ 断言 + arg_type 定型（src_rank=2, idx_rank=1）"]),
    (0, 2, ["④ _to_operands/_args_to_operands → outputs=1，inputs=10",
            "_make_attrs/_make_arg_attrs → attrs=4，arg_attrs=9 槽"]),
    (0, 2, ["⑤ _builder.create_custom_op → emit 1 条 hivm.custom（ttadapter 阶段）"]),
    (2, 0, ["⑥ _to_result 按 out 类型把结果包回 tl.tensor（1 个）"]),
]

LANE_W, TOP, PAD = 300, 96, 40
STEP_H_1LINE = 62
STEP_H_2LINE = 82

n_lanes = len(LANES)
w = PAD * 2 + LANE_W * (n_lanes - 1) + 200
X = {i: PAD + 100 + i * LANE_W for i in range(n_lanes)}

step_h = [STEP_H_2LINE if len(lbl) > 1 else STEP_H_1LINE for _, _, lbl in STEPS]
step_y = []
y_cur = TOP + 60
for sh in step_h:
    step_y.append(y_cur)
    y_cur += sh
h = y_cur + PAD + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("al.custom 一次调用的六步——跨三条泳道")}</text>',
     f'<text x="{w/2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("注册期声明的算子类，在调用期被翻成一条 IR")}</text>']

for i, name in LANES if False else enumerate(LANES):
    x = X[i]
    lines = name.split("\n")
    L.append(f'<rect x="{x-95}" y="{TOP-38}" width="190" height="{34 if len(lines)>1 else 26}" rx="6" '
              'fill="#e2e8f0" stroke="#64748b"/>')
    for k, ln in enumerate(lines):
        L.append(f'<text x="{x}" y="{TOP-38+16+k*15}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(ln)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-2}" x2="{x}" y2="{h-PAD}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (fr, to, labels) in enumerate(STEPS):
    y = step_y[i]
    sh = step_h[i]
    cy = y + sh / 2
    if fr == to:
        x = X[fr]
        L.append(f'<circle cx="{x}" cy="{cy}" r="5" fill="#0369a1"/>')
        anchor = "start" if x < w / 2 else "end"
        tx = x + 16 if anchor == "start" else x - 16
        n = len(labels)
        y0 = cy - (n - 1) * 8
        for k, ln in enumerate(labels):
            L.append(f'<text x="{tx}" y="{y0+k*16}" text-anchor="{anchor}" font-family="sans-serif" '
                      f'font-size="12" fill="#1e3a5f">{esc(ln)}</text>')
    else:
        x1, x2 = X[fr], X[to]
        color = "#15803d" if to == 0 and fr != 0 else "#334155"
        marker = "url(#g)" if color == "#15803d" else "url(#a)"
        L.append(f'<line x1="{x1}" y1="{cy}" x2="{x2}" y2="{cy}" stroke="{color}" '
                  f'stroke-width="1.8" marker-end="{marker}"/>')
        n = len(labels)
        y0 = cy - 14 - (n - 1) * 16
        for k, ln in enumerate(labels):
            L.append(f'<text x="{(x1+x2)/2}" y="{y0+k*16}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="12" fill="{color}">{esc(ln)}</text>')

foot_y = h - 12
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("三条泳道走完：10 个输入操作数、1 个输出、4 条属性（另 9 个 arg_attrs 槽=signature 形参数）落成 1 条 hivm.custom。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch07-02-custom-semantic-swimlane.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
