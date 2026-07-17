#!/usr/bin/env python3
"""fig-m10-iterarg-expansion: before-after — 循环签名从 3 个 iter_arg 扩到 7 个:
+2 环形游标、+2 跨 stage 搬运单副本,后者正是模变量扩展的产物。
数字来自 explainer.json m10.figure_specs[0].numbers:
  3 = make_ttir 之后原循环 iter_args=3(acc,a_ptrs,b_ptrs)
  7 = matmul_sm90_ns3.ttgir.mlir:L83 %64:7
  2 = PipelineExpander.cpp:L430-438 跨 2-stage token 各补 2 份中的新增部分(2 个 token)
  1 = matmul_sm90_ns3.ttgir.mlir:L83 insertIdx 初值 1
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LEFT_TITLE = "优化前:3 个 iter_arg"
LEFT_BADGE = "iter_args = 3"
LEFT_STEPS = [
    "%arg13 acc(累加器)",
    "%arg14 a_ptrs(指针)",
    "%arg15 b_ptrs(指针)",
]

RIGHT_TITLE = "优化后:7 个 iter_arg"
RIGHT_BADGE = "iter_args = 7"
RIGHT_STEPS = [
    "%arg13 acc(累加器)",
    "%arg14 a_ptrs(指针)",
    "%arg15 b_ptrs(指针)",
    "%arg16 insertIdx(初值=1)",
    "%arg17 extractIdx(初值=-1)",
    "%arg18 async.token A(跨2-stage副本)",
    "%arg19 async.token B(跨2-stage副本)",
]
RIGHT_HOT = {3, 4, 5, 6}  # 新增的 4 个

BOX_W, BOX_H, VGAP = 300, 48, 16
PAD, TOP, PANEL_GAP = 44, 128, 110

left_h = len(LEFT_STEPS) * (BOX_H + VGAP) - VGAP
right_h = len(RIGHT_STEPS) * (BOX_H + VGAP) - VGAP
body_h = max(left_h, right_h)

w = PAD * 2 + BOX_W * 2 + PANEL_GAP
h = TOP + body_h + 116

lx = PAD
rx = PAD + BOX_W + PANEL_GAP
lcx = lx + BOX_W / 2
rcx = rx + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">『展开』的字面含义:循环签名 iter_arg 从 3 涨到 7</text>',
     f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">matmul_sm90_ns3.ttgir.mlir:L83 %64:7(make_ttgir 之后)</text>']


def panel(cx, x0, title, badge, steps, hot):
    L.append(f'<text x="{cx}" y="{TOP-56}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    L.append(f'<rect x="{cx-90}" y="{TOP-48}" width="180" height="26" rx="13" '
              'fill="#eef2ff" stroke="#6366f1"/>')
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#4338ca">{esc(badge)}</text>')
    for i, step in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        is_hot = (hot is not None and i in hot)
        fill = "#fef3c7" if is_hot else "#e2e8f0"
        stroke = "#d97706" if is_hot else "#64748b"
        sw = 2.2 if is_hot else 1
        L.append(f'<rect x="{x0}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        fw = 'font-weight="bold" ' if is_hot else ''
        L.append(f'<text x="{cx}" y="{y+BOX_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="#0f172a" {fw}>'
                  f'{esc(step)}</text>')
        if i < len(steps) - 1:
            y2 = y + BOX_H
            L.append(f'<line x1="{cx}" y1="{y2}" x2="{cx}" y2="{y2+VGAP-3}" '
                      'stroke="#94a3b8" stroke-width="1.2"/>')


panel(lcx, lx, LEFT_TITLE, LEFT_BADGE, LEFT_STEPS, None)
panel(rcx, rx, RIGHT_TITLE, RIGHT_BADGE, RIGHT_STEPS, RIGHT_HOT)

mid_y = TOP + body_h / 2 - 40
L.append(f'<line x1="{lx+BOX_W+10}" y1="{mid_y}" x2="{rx-10}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="3" marker-end="url(#ah)"/>')
L.append(f'<text x="{(lx+BOX_W+rx)/2}" y="{mid_y-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#d97706">'
          f'+4(模变量扩展)</text>')

foot_y = TOP + body_h + 44
L.append(f'<line x1="{PAD}" y1="{foot_y-22}" x2="{w-PAD}" y2="{foot_y-22}" stroke="#e2e8f0"/>')
FOOT_LINES = [
    "新增 4 个:+2 环形游标(createAsyncOps 建模期添加)、",
    "+2 跨 2-stage 的 async.token 副本(PipelineExpander 模变量扩展添加)。",
]
for k, line in enumerate(FOOT_LINES):
    L.append(f'<text x="{PAD}" y="{foot_y+k*20}" font-family="sans-serif" font-size="12.5" '
              f'fill="#374151">{esc(line)}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m10-iterarg-expansion.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
