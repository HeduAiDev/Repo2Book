#!/usr/bin/env python3
"""flow 模板：SchedulerBase::applySchedule 读回 FusionKindAttr，switch 选具体调度器；
凡含 cube 的 kind，blockDim 减半（cube:vector=1:2）。
数字：4(PBR家族共用调度器的kind数)/0(ShallowVV no-op,不实例化调度器)/2(cube类blockDim因子)/
3(触发减半的kind数),全部来自 explainer numbers。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text_w(s, size):
    total = 0.0
    for ch in s:
        total += size if ord(ch) > 0x2e80 else size * 0.58
    return total

TITLE = "AutoSchedule：按 FusionKind 分派调度器"
SUBTITLE = "SchedulerBase::applySchedule 读回 OpFusion 写下的 FusionKindAttr，switch 选调度器（AutoScheduleBase.cpp:L579-L611）"

PAD = 46
BOX_W, BOX_H = 560, 56
GAP_V = 46

top = 96
src_y = top
switch_y = src_y + BOX_H + GAP_V

# 四条分支，纵向排开
branches = [
    ("PureElemwise / AnyPB / LastAxisPBR / AnyPBR（4 种，PBR 家族）",
     "AnyPBRScheduler", "#3b82f6", "#dbeafe", "#1e3a8a"),
    ("SingleCube", "SingleCubeScheduler", "#b45309", "#fef3c7", "#78350f"),
    ("ShallowCV（本章 worked example）", "ShallowCVScheduler", "#b45309", "#fef3c7", "#78350f"),
    ("ShallowVV", "return success()（no-op，0 个 scheduler 被实例化）", "#64748b", "#f1f5f9", "#334155"),
]
branch_h = 46
branch_gap = 18
branches_top = switch_y + BOX_H + 70

TRUNK_MARGIN = 70  # 总线与画布左边界、总线与分支框之间的留白
cx = PAD + TRUNK_MARGIN + BOX_W / 2
left_x = cx - BOX_W / 2

# 分支的左标签框 + 右调度器框
LBL_W, RES_W = 430, 320
row_gap_x = 70

w = left_x + LBL_W + row_gap_x + RES_W + PAD + 60
rows_y = [branches_top + i * (branch_h + branch_gap) for i in range(len(branches))]
branches_bottom = rows_y[-1] + branch_h

# 底部：cube 减半说明卡
note_top = branches_bottom + 70
note_h = 92
h = note_top + note_h + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 源框
L.append(f'<rect x="{left_x}" y="{src_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          'fill="#f1f5f9" stroke="#475569" stroke-width="2"/>')
L.append(f'<text x="{cx}" y="{src_y+24}" text-anchor="middle" font-family="sans-serif" '
          'font-size="14" font-weight="bold" fill="#1e293b">func 上的 #hfusion.fusion_kind&lt;...&gt;</text>')
L.append(f'<text x="{cx}" y="{src_y+42}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11" fill="#475569">由 OpFusion 阶段的 FusibleBlockOutliner 写入（回指 ch21）</text>')

# 箭头 源->switch
L.append(f'<line x1="{cx}" y1="{src_y+BOX_H}" x2="{cx}" y2="{switch_y}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')

# switch 框
L.append(f'<rect x="{left_x}" y="{switch_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{cx}" y="{switch_y+24}" text-anchor="middle" font-family="sans-serif" '
          'font-size="14" font-weight="bold" fill="#1e3a8a">switch (fusionKind)</text>')
L.append(f'<text x="{cx}" y="{switch_y+42}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11" fill="#1e40af">applySchedule 内的分派枢纽</text>')

# 分支线：从 switch 底部引出一条总线（画在标签框左侧之外），再各自水平接入每个分支框——
# 总线不得穿过任何文字框内部，故总线 x 取 left_x 左侧留白处。
switch_bottom = switch_y + BOX_H
trunk_x = left_x - TRUNK_MARGIN + 20
first_ly = rows_y[0] + branch_h / 2
last_ly = rows_y[-1] + branch_h / 2
L.append(f'<path d="M {cx} {switch_bottom} L {cx} {first_ly-24} L {trunk_x} {first_ly-24} '
          f'L {trunk_x} {last_ly}" fill="none" stroke="#94a3b8" stroke-width="1.6"/>')
for i, (kind_label, sched_label, edge_c, fill_c, tf_c) in enumerate(branches):
    ry = rows_y[i]
    ly = ry + branch_h / 2
    # 总线在该行的分支点 -> 水平接入标签框左边（不再穿过任何框体内部）
    L.append(f'<line x1="{trunk_x}" y1="{ly}" x2="{left_x-6}" y2="{ly}" '
              f'stroke="{edge_c}" stroke-width="1.6" marker-end="url(#a)"/>')
    # 标签框
    L.append(f'<rect x="{left_x}" y="{ry}" width="{LBL_W}" height="{branch_h}" rx="8" '
              f'fill="{fill_c}" stroke="{edge_c}" stroke-width="2"/>')
    L.append(f'<text x="{left_x+LBL_W/2}" y="{ly+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="{tf_c}" '
              f'font-weight="bold">{esc(kind_label)}</text>')
    # 箭头 标签->结果
    res_x = left_x + LBL_W + row_gap_x
    L.append(f'<line x1="{left_x+LBL_W}" y1="{ly}" x2="{res_x-6}" y2="{ly}" '
              f'stroke="{edge_c}" stroke-width="1.6" marker-end="url(#a)"/>')
    # 结果框
    is_noop = (i == 3)
    res_fill, res_stroke = ("#f1f5f9", "#64748b") if is_noop else (fill_c, edge_c)
    dash = ' stroke-dasharray="5,3"' if is_noop else ''
    L.append(f'<rect x="{res_x}" y="{ry}" width="{RES_W}" height="{branch_h}" rx="8" '
              f'fill="{res_fill}" stroke="{res_stroke}" stroke-width="2"{dash}/>')
    L.append(f'<text x="{res_x+RES_W/2}" y="{ly+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" fill="{tf_c if not is_noop else "#334155"}" '
              f'font-weight="bold">{esc(sched_label)}</text>')

# 底部脚注：MixCV/MixC2/Unknown -> default
foot1_y = branches_bottom + 34
L.append(f'<text x="{left_x}" y="{foot1_y}" font-family="sans-serif" font-size="10.5" '
          'fill="#94a3b8">（MixCV / MixC2 / Unknown 未在此 switch 显式列支，落入 default → emitError，'
          '与本章脊梁无关，不展开）</text>')

# cube 减半说明卡
note_y = note_top
L.append(f'<rect x="{left_x}" y="{note_y}" width="{left_x+LBL_W+row_gap_x+RES_W-left_x}" '
          f'height="{note_h}" rx="10" fill="#fffbeb" stroke="#d97706" stroke-width="2"/>')
note_cx = left_x + (LBL_W + row_gap_x + RES_W) / 2
L.append(f'<text x="{note_cx}" y="{note_y+26}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12.5" font-weight="bold" fill="#92400e">'
          '含 cube 的 3 种 kind（MixCV / SingleCube / ShallowCV）blockDim 减半</text>')
L.append(f'<text x="{note_cx}" y="{note_y+50}" text-anchor="middle" font-family="sans-serif" '
          'font-size="12" fill="#78350f">'
          'options.blockDim = max(blockDim / 2, 1) —— cube : vector = 1 : 2（AutoScheduleBase.cpp:L1228）</text>')
L.append(f'<text x="{note_cx}" y="{note_y+72}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11" fill="#92400e">举例：blockDim 输入 40 → 减半后 20（本章 ShallowCV worked example 沿用）</text>')

foot_y = h - 24
FOOTER = ("同一枚 FusionKind 印章、两处消费：OpFusion 写下它决定融合边界，AutoSchedule 读回它选调度器——"
          "无需二次推断融合意图。")
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOTER)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-schedule-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
