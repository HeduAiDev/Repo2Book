#!/usr/bin/env python3
"""fig-ch06-06-three-ways — swimlane 模板(改造:3 条独立垂直泳道,
而非跨泳道消息 —— 三条路各自跑完落到各自的 IR 算子,再在底部汇入同一个
结果,并显式标注"无因果·仅示意")。数据取自 traces/three_ways.json(m13)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


BLUE, ORANGE, GRAY, RED, GREEN, PURPLE = "#1d4ed8", "#c2410c", "#94a3b8", "#b91c1c", "#15803d", "#7c3aed"

W = 1420
PAD = 50
TOP = 150
LANE_W, LANE_GAP = 380, 60
STEP_H, STEP_GAP = 40, 10

LANES = [
    dict(title="A 手写基线", color=BLUE, fill="#eff6ff",
         steps=["get_element(index[i])(Python 标量读,不计入 IR 算子)",
                "tl.load(第 i 行)", "insert_slice(拼进结果第 i 行)",
                "tl.load(第 i+1 行)", "insert_slice(拼进结果第 i+1 行)", "tl.store(写回)"],
         op_count=5, read="2 次请求,各 4 个连续元素",
         sink="上游 tensor 方言", sink_op="tensor::ExtractOp / InsertSliceOp", sink_stage="ttir(结构化)"),
    dict(title="B 内建算子", color=GREEN, fill="#ecfdf5",
         steps=["index_select_simd(src, index)", "tl.store(写回)"],
         op_count=2, read="2 条 tile 读(算子内部的事)",
         sink="ascend.index_select_simd", sink_op="TritonAscendOps.td:L249", sink_stage="ttir"),
    dict(title="C 交给编译器", color=ORANGE, fill="#fff7ed",
         steps=["tl.load(逐元素偏移张量)", "tl.store(写回)"],
         op_count=2, read="1 次请求覆盖 8 个离散地址",
         sink="ascend.indirect_load", sink_op="UnstructureConversionPass.cpp:L367", sink_stage="ttadapter(编译器改写)"),
]

max_steps = max(len(lane["steps"]) for lane in LANES)
H = TOP + max_steps * (STEP_H + STEP_GAP) + 420

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc("同一个 index_select 的三条泳道")}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("kernel:out[i,j] = src[index[i],j] · indices=[2,0] · 4x4 表 · 期望结果 [[8,9,10,11],[0,1,2,3]]")}</text>']

n = len(LANES)
total_w = n * LANE_W + (n - 1) * LANE_GAP
LX0 = (W - total_w) / 2

for li, lane in enumerate(LANES):
    lx = LX0 + li * (LANE_W + LANE_GAP)
    cx = lx + LANE_W / 2
    color = lane["color"]
    L.append(f'<rect x="{lx}" y="{TOP-42}" width="{LANE_W}" height="30" rx="7" '
             f'fill="{color}" />')
    L.append(f'<text x="{cx}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13.5" font-weight="bold" fill="white">{esc(lane["title"])}</text>')
    n_steps = len(lane["steps"])
    for si, step in enumerate(lane["steps"]):
        y = TOP + si * (STEP_H + STEP_GAP)
        L.append(f'<rect x="{lx}" y="{y}" width="{LANE_W}" height="{STEP_H}" rx="7" '
                 f'fill="{lane["fill"]}" stroke="{color}" stroke-width="1.6"/>')
        L.append(f'<text x="{cx}" y="{y+STEP_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12" fill="#1e293b">{esc(step)}</text>')
        if si < n_steps - 1:
            L.append(f'<line x1="{cx}" y1="{y+STEP_H}" x2="{cx}" y2="{y+STEP_H+STEP_GAP-2}" '
                     f'stroke="{color}" stroke-width="1.4" marker-end="url(#a)"/>')
    steps_end_y = TOP + n_steps * (STEP_H + STEP_GAP)
    op_summary = f'共 {lane["op_count"]} 个算子 · ' + lane["read"]
    L.append(f'<text x="{cx}" y="{steps_end_y+8}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11.5" font-weight="bold" fill="{color}">'
             f'{esc(op_summary)}</text>')
    # 落点 badge
    badge_y = TOP + max_steps * (STEP_H + STEP_GAP) + 30
    L.append(f'<rect x="{lx}" y="{badge_y}" width="{LANE_W}" height="70" rx="9" '
             f'fill="{lane["fill"]}" stroke="{color}" stroke-width="2"/>')
    sink_summary = "落点:" + lane["sink"]
    L.append(f'<text x="{cx}" y="{badge_y+22}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{color}">'
             f'{esc(sink_summary)}</text>')
    L.append(f'<text x="{cx}" y="{badge_y+40}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" fill="#475569">{esc(lane["sink_op"])}</text>')
    stage_summary = "阶段:" + lane["sink_stage"]
    L.append(f'<text x="{cx}" y="{badge_y+58}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10.5" font-weight="bold" fill="#334155">'
             f'{esc(stage_summary)}</text>')
    # 三条泳道 -> 汇聚箭头
    conv_y0 = badge_y + 70
    L.append(f'<line x1="{cx}" y1="{conv_y0}" x2="{cx}" y2="{conv_y0+30}" stroke="{color}" '
             f'stroke-width="1.6" marker-end="url(#a)"/>')

# ── 汇聚:结果一致(显式标注"无因果·仅示意") ────────────────────────
res_y = TOP + max_steps * (STEP_H + STEP_GAP) + 30 + 70 + 30
res_box = (PAD, res_y, W - 2 * PAD, 90)
rbx, rby, rbw, rbh = res_box
L.append(f'<rect x="{rbx}" y="{rby}" width="{rbw}" height="{rbh}" rx="10" fill="#f8fafc" '
         f'stroke="#334155" stroke-width="2"/>')
L.append(f'<text x="{rbx+rbw/2}" y="{rby+26}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#0f172a">'
         f'{esc("结果 tile 逐元素相等:[[8.0,9.0,10.0,11.0],[0.0,1.0,2.0,3.0]]")}</text>')
L.append(f'<text x="{rbx+rbw/2}" y="{rby+50}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11.5" fill="#475569">'
         f'{esc("差别只在前端发出的算子数与访存形状,不在语义 —— 三条路可以写成同一个 out[i,j]=src[index[i],j]")}</text>')
L.append(f'<text x="{rbx+rbw/2}" y="{rby+74}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="{PURPLE}">'
         f'{esc("⚠ 无因果·仅示意:三条路各自独立跑,同一个箭头汇聚不代表执行顺序或依赖关系")}</text>')

H_ACTUAL = res_y + rbh + 24
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-06-three-ways.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
