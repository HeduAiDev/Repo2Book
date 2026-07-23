#!/usr/bin/env python3
"""flow 变体（分层）：FusionKind 是打在 func 上的融合意图标签。
四层自顶向下：func 算子模式 -> InferFuncFusionKind pass -> 10 种 FusionKind（择一）
-> AutoSchedule 按 kind 分派对应调度器（调度器具体实现留后续章，不在图上编具体类名）。
数值：10 个枚举值/枚举定位/属性打印形态/推断 pass 名，全部来自 explainer numbers。
全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "FusionKind：func 级融合意图标签"
SUBTITLE = "一个 func 的算子模式 -> 推断出 10 种 FusionKind 之一 -> 调度器按 kind 分派（HFusionEnums.td:L184-L206）"

KINDS = ["PureElemwise", "AnyPB", "LastAxisPBR", "AnyPBR", "SingleCube",
         "ShallowCV", "ShallowVV", "MixCV", "MixC2", "Unknown"]
KIND_ROW1, KIND_ROW2 = KINDS[:5], KINDS[5:]

PAD, TOP = 50, 100
BOX_W, BOX_H = 620, 56
GAP_V = 54
CHIP_W, CHIP_H, CHIP_GAP = 118, 36, 10

n_row = 5
chips_w = n_row * CHIP_W + (n_row - 1) * CHIP_GAP
content_w = max(BOX_W, chips_w)
w = content_w + PAD * 2

func_y = TOP
attr_y = func_y + BOX_H + GAP_V
chip_top = attr_y + BOX_H + GAP_V + 22
row2_top = chip_top + CHIP_H + CHIP_GAP
sched_y = row2_top + CHIP_H + GAP_V

h = sched_y + BOX_H + PAD + 24

cx = w / 2
box_x = cx - BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-24}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 第一层：func 算子模式
L.append(f'<rect x="{box_x}" y="{func_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          'fill="#f1f5f9" stroke="#475569" stroke-width="2"/>')
L.append(f'<text x="{cx}" y="{func_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#1e293b">func 内核（一段算子模式）</text>')
L.append(f'<text x="{cx}" y="{func_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">融合决策的粒度是整个 func，不是单个 op</text>')

# 箭头 1：func -> attr（标签居中拆两行，避免单行超出画布右边界）
L.append(f'<line x1="{cx}" y1="{func_y+BOX_H}" x2="{cx}" y2="{attr_y}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
arrow1_mid = (func_y + BOX_H + attr_y) / 2
L.append(f'<text x="{cx}" y="{arrow1_mid-6}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">InferFuncFusionKind pass（hfusion-infer-func-fusion-kind）</text>')
L.append(f'<text x="{cx}" y="{arrow1_mid+10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#94a3b8">Transforms/Passes.td:L345-L350</text>')

# 第二层：属性
L.append(f'<rect x="{box_x}" y="{attr_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2.5"/>')
L.append(f'<text x="{cx}" y="{attr_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#1e3a8a">#hfusion.fusion_kind&lt;...&gt;</text>')
L.append(f'<text x="{cx}" y="{attr_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#1e40af">打在 func 上的属性（HFusionAttrs.td:L39-L47）</text>')

# 箭头 2：attr -> 10 kind
arrow2_y1 = attr_y + BOX_H
arrow2_y2 = chip_top - 8
L.append(f'<line x1="{cx}" y1="{arrow2_y1}" x2="{cx}" y2="{arrow2_y2}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{cx}" y="{(arrow2_y1+arrow2_y2)/2+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">'
          f'择一（枚举 10 值，从 1 起无 0）</text>')

# 第三层：10 个 kind 芯片（2 行 x 5），Unknown 单独标灰=兜底
chips_x0 = cx - chips_w / 2
L.append(f'<text x="{cx}" y="{chip_top-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">10 种 FusionKind（HFusionEnums.td:L184-L206）</text>')
for i, name in enumerate(KIND_ROW1):
    x = chips_x0 + i * (CHIP_W + CHIP_GAP)
    fill, stroke, tf = "#eff6ff", "#1d4ed8", "#1e3a8a"
    L.append(f'<rect x="{x}" y="{chip_top}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{chip_top+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{tf}">{esc(name)}</text>')
for i, name in enumerate(KIND_ROW2):
    x = chips_x0 + i * (CHIP_W + CHIP_GAP)
    is_unknown = (name == "Unknown")
    fill, stroke, tf = ("#f1f5f9", "#64748b", "#475569") if is_unknown else ("#eff6ff", "#1d4ed8", "#1e3a8a")
    dash = ' stroke-dasharray="4,3"' if is_unknown else ''
    L.append(f'<rect x="{x}" y="{row2_top}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    L.append(f'<text x="{x+CHIP_W/2}" y="{row2_top+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{tf}">{esc(name)}</text>')
L.append(f'<text x="{chips_x0+chips_w}" y="{row2_top+CHIP_H+16}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10" fill="#94a3b8">Unknown＝兜底（虚线）</text>')

# 箭头 3：kind -> AutoSchedule
arrow3_y1 = row2_top + CHIP_H
arrow3_y2 = sched_y
L.append(f'<line x1="{cx}" y1="{arrow3_y1}" x2="{cx}" y2="{arrow3_y2}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{cx}" y="{(arrow3_y1+arrow3_y2)/2+16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">AutoSchedule 按 kind 分派</text>')

# 第四层：调度器（泛化，不编具体类名）
L.append(f'<rect x="{box_x}" y="{sched_y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
          'fill="#fef9c3" stroke="#a16207" stroke-width="2"/>')
L.append(f'<text x="{cx}" y="{sched_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#78350f">对应调度器（按 kind 择一分派）</text>')
L.append(f'<text x="{cx}" y="{sched_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#92400e">调度器内部实现留后续深调度章</text>')

foot_y = h - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">『该怎么融合』在 IR 里显式决策一次（func 级属性），后续 pass 无需重算</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch21-m6-fusionkind.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
