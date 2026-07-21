#!/usr/bin/env python3
"""fig-ch09-op-region-block-recursion — m03 IR 的三层递归:Op -> region -> block -> Op。
重绘自 arXiv:2002.11054 Fig.3。中心画一个 Op 的四槽位(operand/result、attribute、region、location);
region 里两个 block 组成小 CFG,block 里再嵌一个 Op(回环箭头示意递归);isolated-from-above 标注作用域屏障;
右侧栏给类型系统三条 + module/function 落点。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "MLIR 的 IR 是一次三层递归:Op 挂 region,region 装 block,block 里又是 Op"
SUBTITLE = "重绘自 arXiv:2002.11054 Fig.3——递归怎么闭合,类型系统怎么落地"

PAD = 40
W = 1560
H = 980

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
         '<marker id="a2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
         'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12.5" '
         f'fill="#475569">{esc(SUBTITLE)}</text>')

# ================= 左侧主体:Op 外框(四槽位) =================
OP_X, OP_Y = PAD, 100
OP_W, OP_H = 1000, 800

L.append(f'<rect x="{OP_X}" y="{OP_Y}" width="{OP_W}" height="{OP_H}" rx="14" '
         f'fill="#f8fafc" stroke="#1d4ed8" stroke-width="2.4"/>')
L.append(f'<rect x="{OP_X+18}" y="{OP_Y-16}" width="150" height="32" rx="6" '
         f'fill="#1d4ed8"/>')
L.append(f'<text x="{OP_X+93}" y="{OP_Y+6}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="white">{esc("Op")}</text>')
L.append(f'<text x="{OP_X+190}" y="{OP_Y+6}" font-family="sans-serif" font-size="12" '
         f'fill="#1e3a8a">{esc("例:affine.for  ——  指令 / 函数 / module 皆是 Op")}</text>')

# --- 槽位 1: operand / result ---
slot1_y = OP_Y + 34
slot1_h = 60
L.append(f'<rect x="{OP_X+24}" y="{slot1_y}" width="{OP_W-48}" height="{slot1_h}" rx="8" '
         f'fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5"/>')
L.append(f'<text x="{OP_X+40}" y="{slot1_y+24}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#1e3a8a">{esc("operand / result(SSA,均带类型)")}</text>')
L.append(f'<text x="{OP_X+40}" y="{slot1_y+44}" font-family="sans-serif" font-size="11" '
         f'fill="#1e40af">{esc("%i = affine.for %arg = 0 to 8 { ... }  —— %arg 的类型来自其 def")}</text>')

# --- 槽位 2: attribute 字典 ---
slot2_y = slot1_y + slot1_h + 14
slot2_h = 90
L.append(f'<rect x="{OP_X+24}" y="{slot2_y}" width="{OP_W-48}" height="{slot2_h}" rx="8" '
         f'fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
L.append(f'<text x="{OP_X+40}" y="{slot2_y+22}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#78350f">{esc("attribute 字典(编译期静态信息,open key-value)")}</text>')
L.append(f'<text x="{OP_X+40}" y="{slot2_y+46}" font-family="monospace" font-size="11.5" '
         f'fill="#78350f">'
         f'{esc("{ lower_bound = ()->(0), step = 1:index, upper_bound = #map3 }")}</text>')
L.append(f'<text x="{OP_X+40}" y="{slot2_y+68}" font-family="sans-serif" font-size="11" '
         f'fill="#92400e">{esc("仿射映射(#map3)是 attribute,不是特殊语法——这正是索引映射能被")}</text>')
L.append(f'<text x="{OP_X+40}" y="{slot2_y+84}" font-family="sans-serif" font-size="11" '
         f'fill="#92400e">{esc("编译器直接读来推理的根本原因")}</text>')

# --- 槽位 3: region(内含 block CFG,递归发生处) ---
slot3_y = slot2_y + slot2_h + 18
slot3_h = 470
L.append(f'<rect x="{OP_X+24}" y="{slot3_y}" width="{OP_W-48}" height="{slot3_h}" rx="8" '
         f'fill="#ecfdf5" stroke="#15803d" stroke-width="1.5"/>')
L.append(f'<text x="{OP_X+40}" y="{slot3_y+22}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#14532d">{esc("region(嵌套结构机制:region 装 block,block 装 Op)")}</text>')

# isolated-from-above 虚线包裹整个 region
iso_pad = 10
L.append(f'<rect x="{OP_X+24-iso_pad}" y="{slot3_y+34-iso_pad}" '
         f'width="{OP_W-48+2*iso_pad}" height="{slot3_h-44+2*iso_pad}" rx="10" '
         f'fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="7,5"/>')
L.append(f'<text x="{OP_X+OP_W-48-10}" y="{slot3_y+34-iso_pad-8}" text-anchor="end" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" '
         f'fill="#7c3aed">{esc("isolated-from-above —— 作用域屏障:可并行编译,代价是无全模块 use-def 链")}</text>')

# 两个 block
block_top = slot3_y + 50
block_h = 190
block_w = (OP_W - 48 - 3*20) / 2
b0_x = OP_X + 24 + 20
b1_x = b0_x + block_w + 20

def draw_block(bx, by, bw, bh, name, arg_label, has_inner_op, term_label):
    L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" '
              f'fill="white" stroke="#16a34a" stroke-width="1.6"/>')
    L.append(f'<text x="{bx+10}" y="{by+18}" font-family="sans-serif" font-size="11.5" '
              f'font-weight="bold" fill="#166534">{esc(name)}</text>')
    # block argument
    L.append(f'<rect x="{bx+10}" y="{by+26}" width="{bw-20}" height="26" rx="5" '
              f'fill="#dcfce7" stroke="#22c55e" stroke-width="1.2"/>')
    L.append(f'<text x="{bx+bw/2}" y="{by+44}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#166534">{esc(arg_label)}</text>')
    # 内嵌 Op(递归)
    if has_inner_op:
        iy = by + 60
        L.append(f'<rect x="{bx+16}" y="{iy}" width="{bw-32}" height="70" rx="7" '
                  f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.8"/>')
        L.append(f'<text x="{bx+bw/2}" y="{iy+28}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="bold" fill="#1e3a8a">{esc("Op")}</text>')
        L.append(f'<text x="{bx+bw/2}" y="{iy+46}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="9.5" fill="#1e40af">{esc("(可再挂 region —— 递归)")}</text>')
        # 回环箭头:从内嵌 Op 指回外层 "Op" 标签,示意递归闭合
        loop_y = iy + 70 + 8
    # terminator
    ty = by + bh - 34
    L.append(f'<rect x="{bx+10}" y="{ty}" width="{bw-20}" height="24" rx="5" '
              f'fill="#fee2e2" stroke="#dc2626" stroke-width="1.2"/>')
    L.append(f'<text x="{bx+bw/2}" y="{ty+17}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="#991b1b">{esc(term_label)}</text>')

draw_block(b0_x, block_top, block_w, block_h, "block ^bb0", "block argument %arg(取代 φ 节点)",
           True, "terminator: cf.cond_br")
draw_block(b1_x, block_top, block_w, block_h, "block ^bb1", "block argument(无)",
           False, "terminator: affine.yield")

# block0 -> block1 CFG 边
mid_y = block_top + block_h / 2
L.append(f'<line x1="{b0_x+block_w}" y1="{mid_y}" x2="{b1_x}" y2="{mid_y}" '
         f'stroke="#16a34a" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(b0_x+block_w+b1_x)/2}" y="{mid_y-8}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" fill="#166534">{esc("CFG 边")}</text>')

# 递归回环箭头:从 block0 内嵌 Op 弯回顶部 "Op" 标签
inner_op_top_y = block_top + 60
loop_start_x = b0_x + 16
loop_start_y = inner_op_top_y + 35
L.append(f'<path d="M {loop_start_x} {loop_start_y} '
         f'C {OP_X-6} {loop_start_y}, {OP_X-6} {OP_Y+10}, {OP_X+18} {OP_Y+10}" '
         f'fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="2,3" '
         f'marker-end="url(#a2)"/>')
# 说明文字改放进 inner-Op 小盒自身的注记("可再挂 region —— 递归"),
# 此处窄边距不够放文字,只留纯视觉回环箭头,避免与 attribute/operand 盒文字重叠。

# --- 槽位 4: location ---
slot4_y = slot3_y + slot3_h + 14
slot4_h = 40
L.append(f'<rect x="{OP_X+24}" y="{slot4_y}" width="{OP_W-48}" height="{slot4_h}" rx="8" '
         f'fill="#f1f5f9" stroke="#64748b" stroke-width="1.3"/>')
L.append(f'<text x="{OP_X+40}" y="{slot4_y+25}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("location —— 每个 Op 实例都带的源位置信息")}</text>')

# ================= 右侧栏:类型系统 + module/function 落点 =================
side_x = OP_X + OP_W + 40
side_w = W - side_x - PAD
L.append(f'<text x="{side_x}" y="{OP_Y+6}" font-family="sans-serif" font-size="14.5" '
         f'font-weight="bold" fill="#0f172a">{esc("类型系统(下游满眼 tensor<…>/memref<…> 落于此)")}</text>')

bullets = [
    "严格类型相等,不提供隐式转换规则",
    "只支持非依赖类型",
    "函数与 module 就是 builtin 方言里的 Op",
]
by0 = OP_Y + 30
bh = 56
for i, txt in enumerate(bullets):
    by = by0 + i * (bh + 12)
    L.append(f'<rect x="{side_x}" y="{by}" width="{side_w}" height="{bh}" rx="7" '
              f'fill="#eef2ff" stroke="#6366f1" stroke-width="1.4"/>')
    L.append(f'<text x="{side_x+12}" y="{by+bh/2+5}" font-family="sans-serif" font-size="12" '
              f'fill="#3730a3">{esc("• " + txt)}</text>')

mini_y = by0 + 3 * (bh + 12) + 20
L.append(f'<text x="{side_x}" y="{mini_y}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#334155">{esc("module / function 也只是 Op")}</text>')

mini_box_h = 74
m1_y = mini_y + 16
L.append(f'<rect x="{side_x}" y="{m1_y}" width="{side_w}" height="{mini_box_h}" rx="7" '
         f'fill="#f8fafc" stroke="#334155" stroke-width="1.3"/>')
L.append(f'<text x="{side_x+12}" y="{m1_y+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#0f172a">{esc("module")}</text>')
L.append(f'<text x="{side_x+12}" y="{m1_y+42}" font-family="sans-serif" font-size="11" '
         f'fill="#334155">{esc("= 1 个 region、1 个 block")}</text>')
L.append(f'<text x="{side_x+12}" y="{m1_y+60}" font-family="sans-serif" font-size="11" '
         f'fill="#334155">{esc("(不是与 Op 并列的新概念)")}</text>')

m2_y = m1_y + mini_box_h + 14
L.append(f'<rect x="{side_x}" y="{m2_y}" width="{side_w}" height="{mini_box_h}" rx="7" '
         f'fill="#f8fafc" stroke="#334155" stroke-width="1.3"/>')
L.append(f'<text x="{side_x+12}" y="{m2_y+22}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#0f172a">{esc("function")}</text>')
L.append(f'<text x="{side_x+12}" y="{m2_y+42}" font-family="sans-serif" font-size="11" '
         f'fill="#334155">{esc("region 的入口 block 参数")}</text>')
L.append(f'<text x="{side_x+12}" y="{m2_y+60}" font-family="sans-serif" font-size="11" '
         f'fill="#334155">{esc("即函数参数")}</text>')

# 方言共存 note
dialect_y = m2_y + mini_box_h + 34
L.append(f'<rect x="{side_x}" y="{dialect_y-24}" width="{side_w}" height="70" rx="7" '
         f'fill="#fdf4ff" stroke="#a21caf" stroke-width="1.3"/>')
L.append(f'<text x="{side_x+12}" y="{dialect_y-4}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="#701a75">{esc("dialect = 分组机制")}</text>')
L.append(f'<text x="{side_x+12}" y="{dialect_y+16}" font-family="sans-serif" font-size="10.5" '
         f'fill="#701a75">{esc("不同方言的 Op 可在任意层级、任意时刻共存")}</text>')
L.append(f'<text x="{side_x+12}" y="{dialect_y+34}" font-family="sans-serif" font-size="10.5" '
         f'fill="#701a75">{esc("——渐进式下降的物理基础")}</text>')

foot_y = H - 14
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc("依据:arXiv:2002.11054 §3;算子名仅示意上游 affine.for,非本仓自研方言")}</text>')

L.append('</svg>')

out = Path(__file__).with_name("fig-ch09-op-region-block-recursion.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
