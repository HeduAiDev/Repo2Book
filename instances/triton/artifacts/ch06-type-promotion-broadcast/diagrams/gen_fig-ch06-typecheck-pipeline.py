#!/usr/bin/env python3
"""fig-ch06-typecheck-pipeline: binary_op_type_checking_impl 五步流水（swimlane 变体：
lhs/rhs 两条生命线横向铺开，5 个时间站）。全部坐标按顺序累加区块高度算出，各区块
垂直互不重叠；文本全 esc()。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# 5 步：(函数名, 简述, lhs 变化, rhs 变化)
STEPS = [
    ("to_tensor", "裹标量成 0 维 tensor",
     "标量→tensor(若是标量)", "标量→tensor(若是标量)"),
    ("check_ptr_type_impl", "查指针合法性（双向各查一次）",
     "check(lhs, rhs)", "check(rhs, lhs)"),
    ("computation_type_impl", "算统一结果类型 ret_sca_ty",
     "（只产出 1 个 ret，双侧共用）", "（只产出 1 个 ret，双侧共用）"),
    ("full / cast", "标量→full(ret)；张量→cast(x, ret)",
     "full((), 值, dtype=ret)", "cast(x, ret)"),
    ("broadcast_impl_value", "对齐形状",
     "→ 广播后 shape", "→ 广播后 shape"),
]

PAD = 40
LEFT_LABEL_W = 150
LABEL_GAP = 24
STEP_W, STEP_GAP = 250, 26
N = len(STEPS)

STEP0_X = PAD + LEFT_LABEL_W + LABEL_GAP
step_x = [STEP0_X + i * (STEP_W + STEP_GAP) for i in range(N)]
STEP_AREA_R = step_x[-1] + STEP_W
w = STEP_AREA_R + PAD

# ---- vertical bands, each computed from the previous band's bottom (no overlap by construction) ----
TITLE_Y = 30
SUB_Y = 54
BAND_GAP = 22

HEADER_TOP = SUB_Y + BAND_GAP           # 76
HEADER_H = 46
HEADER_BOTTOM = HEADER_TOP + HEADER_H   # 122

NOTE_H = 34
LANE1_TOP = HEADER_BOTTOM + 46
LANE1_BOTTOM = LANE1_TOP + NOTE_H
LIFELINE1_Y = LANE1_BOTTOM + 14

LANE2_TOP = LIFELINE1_Y + 46
LANE2_BOTTOM = LANE2_TOP + NOTE_H
LIFELINE2_Y = LANE2_BOTTOM + 14

FORK_TOP = LIFELINE2_Y + 50
FORK_H = 58

OUT_TOP = FORK_TOP + FORK_H + 40
OUT_H = 56

h = OUT_TOP + OUT_H + PAD

LANES = [("lhs", "lhs 生命线", LANE1_TOP, LANE1_BOTTOM, LIFELINE1_Y),
         ("rhs", "rhs 生命线", LANE2_TOP, LANE2_BOTTOM, LIFELINE2_Y)]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="aRed" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# Title + subtitle (own band, above everything else)
L.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="19" font-weight="bold" '
          f'fill="#0f172a">{esc("binary_op_type_checking_impl：一次 x + y 背后的 5 步流水")}</text>')
L.append(f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#475569">{esc("流水步数 = 5 —— semantic.py:L166-L196（回收上一章的伏笔：tensor 只转发，决策全在这条流水里）")}</text>')

# lane life-lines (span full step area, drawn before lane label boxes so label sits on top)
for _, _, top, bottom, ly in LANES:
    L.append(f'<line x1="{PAD}" y1="{ly}" x2="{STEP_AREA_R}" y2="{ly}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

# lane label chips (left column, own x-range, never overlaps step columns)
for _, label, top, bottom, ly in LANES:
    cy = (top + bottom) / 2
    L.append(f'<rect x="{PAD}" y="{cy-16}" width="{LEFT_LABEL_W}" height="32" rx="7" '
              'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{PAD+LEFT_LABEL_W/2}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#0f172a">{esc(label)}</text>')

# step header boxes + inter-step arrows
for i, (fn, desc, lhs_note, rhs_note) in enumerate(STEPS):
    x = step_x[i]
    L.append(f'<rect x="{x}" y="{HEADER_TOP}" width="{STEP_W}" height="{HEADER_H}" rx="8" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    L.append(f'<text x="{x+STEP_W/2}" y="{HEADER_TOP+19}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc(f"{i+1}. {fn}")}</text>')
    L.append(f'<text x="{x+STEP_W/2}" y="{HEADER_TOP+36}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#1e40af">{esc(desc)}</text>')
    if i < N - 1:
        nx = step_x[i + 1]
        ay = HEADER_TOP + HEADER_H / 2
        L.append(f'<line x1="{x+STEP_W}" y1="{ay}" x2="{nx}" y2="{ay}" '
                  'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

    for lane_idx, (name, label, top, bottom, ly) in enumerate(LANES):
        note = lhs_note if lane_idx == 0 else rhs_note
        cy = (top + bottom) / 2
        L.append(f'<rect x="{x+8}" y="{top}" width="{STEP_W-16}" height="{NOTE_H}" rx="6" '
                  'fill="white" stroke="#94a3b8" stroke-width="1"/>')
        L.append(f'<text x="{x+STEP_W/2}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="#0f172a">{esc(note)}</text>')

# fork callout under step 4 (index 3): scalar path vs tensor path — recovers ch05 foreshadow
fork_i = 3
fx = step_x[fork_i]
L.append(f'<rect x="{fx}" y="{FORK_TOP}" width="{STEP_W}" height="{FORK_H}" rx="8" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.4" stroke-dasharray="5,3"/>')
L.append(f'<text x="{fx+STEP_W/2}" y="{FORK_TOP+21}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#991b1b">{esc("关键分叉（回收上一章的伏笔）")}</text>')
L.append(f'<text x="{fx+STEP_W/2}" y="{FORK_TOP+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#991b1b">{esc("标量:full((),值,dtype=ret)｜张量:cast(x,ret)")}</text>')
L.append(f'<line x1="{fx+STEP_W/2}" y1="{LIFELINE2_Y}" x2="{fx+STEP_W/2}" y2="{FORK_TOP}" '
          'stroke="#dc2626" stroke-width="1.4" marker-end="url(#aRed)"/>')

# final outcome bar spanning full width
L.append(f'<rect x="{PAD}" y="{OUT_TOP}" width="{STEP_AREA_R-PAD}" height="{OUT_H}" rx="10" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="1.6"/>')
L.append(f'<text x="{PAD+(STEP_AREA_R-PAD)/2}" y="{OUT_TOP+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#14532d">'
          f'{esc("流水终点：lhs、rhs 同 dtype、同 shape —— add() 等算子只需按 kind 选 IR builder")}</text>')
L.append(f'<text x="{PAD+(STEP_AREA_R-PAD)/2}" y="{OUT_TOP+41}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#166534">'
          f'{esc("ch05：tensor 的 dunder『只转发不决策』——决策全在这 5 步里")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch06-typecheck-pipeline.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}  size={w}x{h}')
