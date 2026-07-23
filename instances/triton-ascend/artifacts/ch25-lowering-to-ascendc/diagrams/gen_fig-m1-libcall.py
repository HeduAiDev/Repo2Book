#!/usr/bin/env python3
"""fig-m1-libcall — before-after 模板改造：
左：illegal 的 hivm.hir.vadd（HIVM 硬件 op）。
右：createLibCall 落地的两样东西——module 末尾一条去重的 func.func 外部声明
（打 3 个属性）+ 原地一条 func.call；原 op 被 replaceOp 抹去。
坐标全部由常量/循环计算，箭头端点取自元素边缘。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# ---- 内容常量 ----
LEFT_TITLE = "转换前：IR 里的 hivm.hir.vadd（illegal）"
LEFT_OP_L1 = 'hivm.hir.vadd(%a, %b, %c)'
LEFT_OP_L2 = ': memref<16xf16, #hivm.address_space<gm>>'

RIGHT_TITLE = "转换后：createLibCall 落地的两样东西"
GATE_LABEL = "mod.lookupSymbol(“vadd_1d_half”) 命中？"
GATE_NUM = "L114"
GATE_NO = "否（首次遇到）→ 插声明"
GATE_YES = "是（已插过）→ 跳过声明，直接点单"

DECL_TITLE = "module 末尾（仅首次插入）"
DECL_SIG = "func.func private @vadd_1d_half(...)"
DECL_ATTRS = [
    ("emit_c_wrapper", "L123"),
    ("hacc.always_inline", "L126"),
    ("private（无 body，外部符号）", "L130"),
]

CALL_LABEL = "func.call @vadd_1d_half(%a, %b, %c)"
CALL_NUM = "L165"
REPLACE_LABEL = "rewriter.replaceOp(原 hivm.hir.vadd, call) → HIVM op 消失"
REPLACE_NUM = "L179"

CAPTION = ("createLibCall：去重开关(L114)决定要不要现插一条打 3 属性的 func.func 声明"
           "(L123/126/130)，两条路径都汇到同一条 func.call(L165)，"
           "replaceOp(L179) 把原 hivm.hir.vadd 换掉——HIVM op 就此消失。")

# ---- 版式常量 ----
FONT = "sans-serif"
NUM_FILL = "#b91c1c"          # 行号标注色
BOX_STROKE = "#1e3a5f"
GATE_FILL, GATE_STROKE = "#fef9c3", "#a16207"
DECL_FILL, DECL_STROKE = "#e0f2fe", "#0369a1"
CALL_FILL, CALL_STROKE = "#dcfce7", "#15803d"
LEFT_FILL = "#fee2e2"
LEFT_STROKE = "#b91c1c"

PAD = 44
TOP = 100
COL_GAP = 190
LEFT_W = 300
RIGHT_W = 460
BYPASS_OUT = 130   # gate 右边缘向外绕行的水平距离（是-分支）

LEFT_OP_H = 68

GATE_W, GATE_H = RIGHT_W, 56
DECL_W = RIGHT_W
DECL_HEAD_H = 30
DECL_SIG_H = 26
DECL_ATTR_H = 22
DECL_H = DECL_HEAD_H + DECL_SIG_H + DECL_ATTR_H * len(DECL_ATTRS) + 10
CALL_W, CALL_H = RIGHT_W, 44
REPLACE_H = 40

VGAP = 26

gate_y = TOP + 26
decl_y = gate_y + GATE_H + VGAP + 18   # +18 留给分支标签
call_y = decl_y + DECL_H + VGAP
replace_y = call_y + CALL_H + VGAP - 6

left_x = PAD
right_x = PAD + LEFT_W + COL_GAP
left_op_y = TOP + (call_y - TOP) / 2 - LEFT_OP_H / 2  # 竖直居中对齐右列主体

gy_x = right_x + RIGHT_W + BYPASS_OUT   # 是-分支绕行线的竖直段 x 坐标

w = gy_x + 100
h = replace_y + REPLACE_H + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '<marker id="ay" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
          '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{PAD}" y="40" font-family="{FONT}" font-size="18" font-weight="bold" '
          f'fill="#0f172a">HIVMToStandard：一个 hivm.hir.vadd 被 createLibCall 降成什么</text>')

# 列标题
L.append(f'<text x="{left_x+LEFT_W/2}" y="{TOP-14}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="14" font-weight="bold" fill="{LEFT_STROKE}">{esc(LEFT_TITLE)}</text>')
L.append(f'<text x="{right_x+RIGHT_W/2}" y="{TOP-14}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="14" font-weight="bold" fill="#0369a1">{esc(RIGHT_TITLE)}</text>')

# ---- 左：illegal op ----
L.append(f'<rect x="{left_x}" y="{left_op_y}" width="{LEFT_W}" height="{LEFT_OP_H}" rx="8" '
          f'fill="{LEFT_FILL}" stroke="{LEFT_STROKE}" stroke-width="2"/>')
L.append(f'<text x="{left_x+LEFT_W/2}" y="{left_op_y+20}" text-anchor="middle" '
          f'font-family="{FONT}" font-size="12" font-weight="bold" fill="#7f1d1d">'
          f'{esc("illegal（addIllegalOp）")}</text>')
L.append(f'<text x="{left_x+LEFT_W/2}" y="{left_op_y+38}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#7f1d1d">{esc(LEFT_OP_L1)}</text>')
L.append(f'<text x="{left_x+LEFT_W/2}" y="{left_op_y+54}" text-anchor="middle" '
          f'font-family="monospace" font-size="11" fill="#7f1d1d">{esc(LEFT_OP_L2)}</text>')

# 长箭头：左 -> 右（贯穿到 call 框，虚线，标 pattern 重写）
mid_left_y = left_op_y + LEFT_OP_H / 2
arrow_x1 = left_x + LEFT_W
arrow_x2 = right_x
L.append(f'<line x1="{arrow_x1}" y1="{mid_left_y}" x2="{arrow_x2-6}" y2="{mid_left_y}" '
          f'stroke="#334155" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#a)"/>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{mid_left_y-22}" text-anchor="middle" '
          f'font-family="{FONT}" font-size="11" fill="#334155">{esc("OpRewritePattern")}</text>')
L.append(f'<text x="{(arrow_x1+arrow_x2)/2}" y="{mid_left_y-8}" text-anchor="middle" '
          f'font-family="{FONT}" font-size="11" fill="#334155">{esc("匹配 → 重写")}</text>')

# ---- 右：gate ----
L.append(f'<rect x="{right_x}" y="{gate_y}" width="{GATE_W}" height="{GATE_H}" rx="10" '
          f'fill="{GATE_FILL}" stroke="{GATE_STROKE}" stroke-width="2"/>')
L.append(f'<text x="{right_x+GATE_W/2}" y="{gate_y+GATE_H/2+5}" text-anchor="middle" '
          f'font-family="{FONT}" font-size="12.5" fill="#713f12">{esc(GATE_LABEL)}</text>')
L.append(f'<text x="{right_x+GATE_W-10}" y="{gate_y+16}" text-anchor="end" '
          f'font-family="{FONT}" font-size="11" font-weight="bold" fill="{NUM_FILL}">{esc(GATE_NUM)}</text>')

# gate -> decl（否）
gd_x1 = right_x + GATE_W * 0.28
L.append(f'<line x1="{gd_x1}" y1="{gate_y+GATE_H}" x2="{gd_x1}" y2="{decl_y}" '
          f'stroke="{DECL_STROKE}" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{gd_x1+8}" y="{gate_y+GATE_H+15}" font-family="{FONT}" font-size="11" '
          f'fill="{DECL_STROKE}">{esc(GATE_NO)}</text>')

# gate -> call 直达（是，跳过声明）：右侧绕行虚线
bypass_mid_y = (gate_y + GATE_H / 2 + call_y + CALL_H / 2) / 2
L.append(f'<line x1="{right_x+GATE_W}" y1="{gate_y+GATE_H/2}" x2="{gy_x}" y2="{gate_y+GATE_H/2}" '
          f'stroke="#15803d" stroke-width="1.8" stroke-dasharray="4,3"/>')
L.append(f'<line x1="{gy_x}" y1="{gate_y+GATE_H/2}" x2="{gy_x}" y2="{call_y+CALL_H/2}" '
          f'stroke="#15803d" stroke-width="1.8" stroke-dasharray="4,3"/>')
L.append(f'<line x1="{gy_x}" y1="{call_y+CALL_H/2}" x2="{right_x+CALL_W}" y2="{call_y+CALL_H/2}" '
          f'stroke="#15803d" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#ay)"/>')
L.append(f'<text x="{gy_x+10}" y="{gate_y+GATE_H/2-8}" font-family="{FONT}" font-size="10.5" '
          f'fill="#15803d">{esc("是（已插过）")}</text>')
L.append(f'<text x="{gy_x+10}" y="{bypass_mid_y}" font-family="{FONT}" font-size="10.5" '
          f'fill="#15803d">{esc("↓跳过声明")}</text>')
L.append(f'<text x="{gy_x+10}" y="{call_y+CALL_H/2+16}" font-family="{FONT}" font-size="10.5" '
          f'fill="#15803d">{esc("直接点单")}</text>')

# ---- 右：declaration box ----
L.append(f'<rect x="{right_x}" y="{decl_y}" width="{DECL_W}" height="{DECL_H}" rx="10" '
          f'fill="{DECL_FILL}" stroke="{DECL_STROKE}" stroke-width="2"/>')
L.append(f'<text x="{right_x+12}" y="{decl_y+20}" font-family="{FONT}" font-size="12" '
          f'font-weight="bold" fill="#0c4a6e">{esc(DECL_TITLE)}</text>')
sig_y = decl_y + DECL_HEAD_H + 14
L.append(f'<text x="{right_x+12}" y="{sig_y}" font-family="monospace" font-size="12" '
          f'fill="#0c4a6e">{esc(DECL_SIG)}</text>')
for i, (attr, num) in enumerate(DECL_ATTRS):
    ay = sig_y + DECL_SIG_H - 8 + i * DECL_ATTR_H
    L.append(f'<text x="{right_x+28}" y="{ay}" font-family="{FONT}" font-size="11.5" '
              f'fill="#0c4a6e">{esc("· " + attr)}</text>')
    L.append(f'<text x="{right_x+DECL_W-10}" y="{ay}" text-anchor="end" font-family="{FONT}" '
              f'font-size="11" font-weight="bold" fill="{NUM_FILL}">{esc(num)}</text>')

# decl -> call
dc_x = right_x + DECL_W * 0.28
L.append(f'<line x1="{dc_x}" y1="{decl_y+DECL_H}" x2="{dc_x}" y2="{call_y}" '
          f'stroke="{CALL_STROKE}" stroke-width="2" marker-end="url(#a)"/>')

# ---- 右：call box ----
L.append(f'<rect x="{right_x}" y="{call_y}" width="{CALL_W}" height="{CALL_H}" rx="10" '
          f'fill="{CALL_FILL}" stroke="{CALL_STROKE}" stroke-width="2"/>')
L.append(f'<text x="{right_x+16}" y="{call_y+CALL_H/2+5}" font-family="monospace" '
          f'font-size="12.5" fill="#14532d">{esc(CALL_LABEL)}</text>')
L.append(f'<text x="{right_x+CALL_W-10}" y="{call_y-6}" text-anchor="end" font-family="{FONT}" '
          f'font-size="11" font-weight="bold" fill="{NUM_FILL}">{esc(CALL_NUM)}</text>')

# call -> replace note
cn_x = right_x + CALL_W / 2
L.append(f'<line x1="{cn_x}" y1="{call_y+CALL_H}" x2="{cn_x}" y2="{replace_y}" '
          f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{right_x+CALL_W-10}" y="{replace_y-8}" text-anchor="end" font-family="{FONT}" '
          f'font-size="11" font-weight="bold" fill="{NUM_FILL}">{esc(REPLACE_NUM)}</text>')
L.append(f'<text x="{right_x+CALL_W/2}" y="{replace_y+18}" text-anchor="middle" font-family="{FONT}" '
          f'font-size="12" fill="#334155">{esc(REPLACE_LABEL)}</text>')

# 图注
L.append(f'<text x="{w/2}" y="{h-16}" text-anchor="middle" font-family="{FONT}" font-size="12" '
          f'fill="#475569">{esc(CAPTION)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m1-libcall.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w} h={h}")
