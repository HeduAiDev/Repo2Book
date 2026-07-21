#!/usr/bin/env python3
"""fig-ch06-05-cast-decision-tree — flow 模板。
cast 决策树:bf16/fp16 想去 fp32 以外的目标一律拆成两跳(先转 fp32);
saturate 整型收窄按芯片分叉:910_95 挂 2 条 compile_hint,
非 910_95 绕道 fp32(2 条算子,0 条 hint)。数据取自 traces/builder_calls.json(m11)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


BLUE, ORANGE, GRAY, RED, GREEN = "#1d4ed8", "#c2410c", "#94a3b8", "#b91c1c", "#15803d"

W, H = 1420, 900
PAD = 50
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{BLUE}"/></marker>'
     '<marker id="o" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ORANGE}"/></marker>'
     '<marker id="rd" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{RED}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc("ascend_cast_impl 决策树:两条粗线是本章要记住的")}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("递归深度上界 2:内层目标恒为 float32,float32 不再满足任何触发递归的条件")}</text>']

ROOT = (W / 2 - 130, 82, 260, 44)
rx, ry, rw, rh = ROOT
L.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" rx="9" fill="#e2e8f0" '
         f'stroke="#334155" stroke-width="2"/>')
L.append(f'<text x="{rx+rw/2}" y="{ry+rh/2+5}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="#0f172a">{esc("ascend_cast(src, dst, builder)")}</text>')
root_cx = rx + rw / 2
root_by = ry + rh

# 三个分支问句
Q_Y = 170
Q = [
    (240, "src∈{fp16,bf16}\ndst≠fp32?"),
    (W / 2, "saturate\n整型收窄?"),
    (W - 240, "其余(直发)"),
]
for cx, label in Q:
    lines = label.split("\n")
    qh = 44
    L.append(f'<path d="M {root_cx} {root_by} L {cx} {Q_Y}" fill="none" stroke="{GRAY}" '
             f'stroke-width="1.6" marker-end="url(#a)"/>')
    L.append(f'<rect x="{cx-95}" y="{Q_Y}" width="190" height="{qh}" rx="8" fill="#f8fafc" '
             f'stroke="{GRAY}" stroke-width="1.6"/>')
    for i, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{Q_Y+18+i*18}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                 f'fill="#334155">{esc(ln)}</text>')

# ── 分支 1(左):bf16/fp16 → 非 fp32,两跳 ────────────────────────────
LX = 240
LEAF_Y = 300
L.append(f'<path d="M {LX} {Q_Y+44} L {LX} {LEAF_Y}" fill="none" stroke="{BLUE}" '
         f'stroke-width="2.4" marker-end="url(#b)"/>')
L.append(f'<text x="{LX+8}" y="{(Q_Y+44+LEAF_Y)/2}" font-family="sans-serif" font-size="10.5" '
         f'fill="{BLUE}">{esc("是")}</text>')
leaf1 = (LX - 175, LEAF_Y, 350, 92)
lx1, ly1, lw1, lh1 = leaf1
L.append(f'<rect x="{lx1}" y="{ly1}" width="{lw1}" height="{lh1}" rx="9" fill="#eff6ff" '
         f'stroke="{BLUE}" stroke-width="2.4"/>')
for i, (txt, bold) in enumerate([
        ("第一跳:先转 float32(内层递归)", True),
        ("例 bf16 → fp16:2 条算子", False),
        ("create_fp_ext(→fp32) + create_fp_trunc(→fp16)", False)]):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{lx1+lw1/2}" y="{ly1+22+i*24}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{12 if bold else 11}"{b} '
             f'fill="{"#1e3a8a" if bold else "#1e40af"}">{esc(txt)}</text>')

# fp32→int32 对照(直发 1 条),放在分支1下方,呼应"其余直发"
LEAF1B_Y = LEAF_Y + lh1 + 22
leaf1b = (lx1, LEAF1B_Y, lw1, 56)
lb_x, lb_y, lb_w, lb_h = leaf1b
L.append(f'<rect x="{lb_x}" y="{lb_y}" width="{lb_w}" height="{lb_h}" rx="8" fill="#f1f5f9" '
         f'stroke="{GRAY}" stroke-width="1.4"/>')
L.append(f'<text x="{lb_x+lb_w/2}" y="{lb_y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#334155">{esc("对照:fp32 → int32(dst 已是判据外)")}</text>')
L.append(f'<text x="{lb_x+lb_w/2}" y="{lb_y+42}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#475569">{esc("1 条算子:create_fp_to_si,不递归")}</text>')

# ── 分支 2(中):saturate 整型收窄 —— 芯片分叉 ──────────────────────
MX = W / 2
L.append(f'<path d="M {MX} {Q_Y+44} L {MX} {LEAF_Y}" fill="none" stroke="{ORANGE}" '
         f'stroke-width="2.4" marker-end="url(#o)"/>')
L.append(f'<text x="{MX+8}" y="{(Q_Y+44+LEAF_Y)/2}" font-family="sans-serif" font-size="10.5" '
         f'fill="{ORANGE}">{esc("是")}</text>')
chip_label_y = LEAF_Y
L.append(f'<text x="{MX+70}" y="{(Q_Y+44+LEAF_Y)/2+14}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" '
         f'fill="#334155">{esc("(按芯片分叉)")}</text>')
CHIP_Y = LEAF_Y + 6
CW, CH, CGAP = 210, 100, 30
c1x = MX - CW - CGAP / 2
c2x = MX + CGAP / 2
L.append(f'<rect x="{c1x}" y="{CHIP_Y}" width="{CW}" height="{CH}" rx="9" fill="#fff7ed" '
         f'stroke="{ORANGE}" stroke-width="2.2"/>')
for i, (txt, bold) in enumerate([
        ("910_95", True),
        ("uint32→int16 saturate:", False),
        ("1 条 create_int_cast", False),
        ("+ 2 条 compile_hint", False)]):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{c1x+CW/2}" y="{CHIP_Y+20+i*20}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{12 if bold else 10.5}"{b} '
             f'fill="{"#7c2d12" if bold else "#9a3412"}">{esc(txt)}</text>')
L.append(f'<rect x="{c2x}" y="{CHIP_Y}" width="{CW}" height="{CH}" rx="9" fill="#fef2f2" '
         f'stroke="{RED}" stroke-width="2"/>')
for i, (txt, bold) in enumerate([
        ("非 910_95", True),
        ("同一收窄:绕道 fp32", False),
        ("2 条算子(ui_to_fp+fp_to_si)", False),
        ("0 条 compile_hint", False)]):
    b = ' font-weight="bold"' if bold else ''
    L.append(f'<text x="{c2x+CW/2}" y="{CHIP_Y+20+i*20}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{12 if bold else 10.5}"{b} '
             f'fill="{"#7f1d1d" if bold else "#991b1b"}">{esc(txt)}</text>')

# ── 分支 3(右):其余 → 直发 1 条;非 910_95 硬拒绝 fp8/fp64 ─────────
RX = W - 240
L.append(f'<path d="M {RX} {Q_Y+44} L {RX} {LEAF_Y}" fill="none" stroke="{GRAY}" '
         f'stroke-width="2" marker-end="url(#a)"/>')
leaf3 = (RX - 155, LEAF_Y, 310, 70)
l3x, l3y, l3w, l3h = leaf3
L.append(f'<rect x="{l3x}" y="{l3y}" width="{l3w}" height="{l3h}" rx="9" fill="#f1f5f9" '
         f'stroke="{GRAY}" stroke-width="1.6"/>')
L.append(f'<text x="{l3x+l3w/2}" y="{l3y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#334155">{esc("直接发一条转换算子")}</text>')
L.append(f'<text x="{l3x+l3w/2}" y="{l3y+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="#475569">{esc("例 int32→int1:create_splat + create_icmpNE")}</text>')

leaf3b_y = leaf3[1] + l3h + 24
leaf3b = (l3x, leaf3b_y, l3w, 70)
lb3x, lb3y, lb3w, lb3h = leaf3b
L.append(f'<rect x="{lb3x}" y="{lb3y}" width="{lb3w}" height="{lb3h}" rx="9" fill="#fee2e2" '
         f'stroke="{RED}" stroke-width="2"/>')
L.append(f'<text x="{lb3x+lb3w/2}" y="{lb3y+22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="{RED}">{esc("非 910_95 硬拒绝")}</text>')
L.append(f'<text x="{lb3x+lb3w/2}" y="{lb3y+44}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="{RED}">{esc("fp8e4nv/fp64 → ValueError")}</text>')
L.append(f'<text x="{lb3x+lb3w/2}" y="{lb3y+62}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="10.5" fill="{RED}">{esc("(unsupported on Ascend for now)")}</text>')

# ── 底部脚注 ──────────────────────────────────────────────────────────
FOOT_Y = leaf3b_y + lb3h + 40
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="76" rx="9" fill="#f8fafc" '
         f'stroke="{GRAY}" stroke-width="1.4"/>')
FOOT = [
    "8 组用例里 7 组被接受、1 组被拒;转换算子条数只有 1 或 2 两种取值。",
    "递归深度上界 2:两处触发递归的写法都是 ascend_cast_impl(ascend_cast_impl(input, tl.float32, builder), dst, builder),"
    "内层目标恒为 float32(vec_ops.py:L445-447、L484-485)。",
]
for i, ln in enumerate(FOOT):
    L.append(f'<text x="{PAD+16}" y="{FOOT_Y+26+i*24}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(ln)}</text>')

H_ACTUAL = FOOT_Y + 96
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-05-cast-decision-tree.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
