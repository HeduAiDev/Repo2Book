#!/usr/bin/env python3
"""fig-ch07-04-libdevice-three-paths — layout 模板（libdevice.py 的四类形态拼装）。
libdevice.py 的 37 个顶层函数按「一张菜单 14 / 两张菜单 2 / 一半菜单一半自己算 18 /
从不碰菜单 3」划开（互斥且穷尽），math_ops.py 另有 3 个 @jit 组合函数（路径③）；
四者都汇进同一个 al.libdevice.* 命名空间。全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "libdevice 不是一层薄壳，是四类形态的拼装"
SUBTITLE = "37 个顶层函数按「一张菜单 14 / 两张菜单 2 / 一半菜单一半自己算 18 / 从不碰菜单 3」划开——互斥且穷尽"

PAD = 40
TOP = 108
w = 1160

# libdevice.py 大框，内部四段（竖排 A/B/C/D）
BIG_X, BIG_Y, BIG_W = PAD, TOP, 700
SEG_H = 78
SEG_GAP = 14
SEGS = [
    ("A · 一张菜单，直接点符号", "代表 reciprocal（14 个函数只走这条）", "14", "#dbeafe", "#1d4ed8", False),
    ("B · 两张菜单，按开关/架构换一张符号表", "代表 tanh、pow（两端都还是符号表，仅 2 个）", "2", "#ede9fe", "#6d28d9", False),
    ("C · 一半菜单一半自己算——最常见", "代表 acos（一端点符号、另一端改用纯 IR，18 个）", "18", "#fef3c7", "#b45309", True),
    ("D · 从不碰菜单，全程纯 IR", "gamma / fast_dividef / fast_expf（3 个）", "3", "#dcfce7", "#15803d", False),
]
big_header_h = 36
big_h = big_header_h + len(SEGS) * SEG_H + (len(SEGS) - 1) * SEG_GAP + 20

# math_ops.py 侧框（路径③，@jit 组合已有原语，不动）
SIDE_X = BIG_X + BIG_W + 50
SIDE_W = w - PAD - SIDE_X
side_h = big_h

merge_y = BIG_Y + big_h + 70
MERGE_W, MERGE_H = BIG_W + 50 + SIDE_W, 66
merge_x = BIG_X

callout_y = merge_y + MERGE_H + 46
CALLOUT_H = 96

h = callout_y + CALLOUT_H + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# libdevice.py 外框
L.append(f'<rect x="{BIG_X}" y="{BIG_Y}" width="{BIG_W}" height="{big_h}" rx="12" '
          f'fill="#f8fafc" stroke="#334155" stroke-width="2" stroke-dasharray="7,5"/>')
L.append(f'<text x="{BIG_X+16}" y="{BIG_Y+24}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#334155">{esc("libdevice.py — 1032 行 / 37 个顶层函数")}</text>')

y_cur = BIG_Y + big_header_h
for i, (name, detail, cnt, fill, stroke, is_main) in enumerate(SEGS):
    sx = BIG_X + 16
    sw = BIG_W - 32
    L.append(f'<rect x="{sx}" y="{y_cur}" width="{sw}" height="{SEG_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{3 if is_main else 2}"/>')
    L.append(f'<text x="{sx+16}" y="{y_cur+22}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{sx+16}" y="{y_cur+42}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(detail)}</text>')
    if is_main:
        L.append(f'<text x="{sx+16}" y="{y_cur+SEG_H-10}" font-family="sans-serif" font-size="11" '
                  f'font-weight="bold" fill="{stroke}">{esc("最常见")}</text>')
    L.append(f'<text x="{sx+sw-16}" y="{y_cur+SEG_H-14}" text-anchor="end" font-family="sans-serif" '
              f'font-size="20" font-weight="bold" fill="{stroke}">{esc(cnt)}</text>')
    y_cur += SEG_H + SEG_GAP

# math_ops.py 侧框（路径③）
L.append(f'<rect x="{SIDE_X}" y="{BIG_Y}" width="{SIDE_W}" height="{side_h}" rx="12" '
          f'fill="#f8fafc" stroke="#334155" stroke-width="2" stroke-dasharray="7,5"/>')
L.append(f'<text x="{SIDE_X+16}" y="{BIG_Y+24}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#334155">{esc("math_ops.py — 53 行")}</text>')
sx3 = SIDE_X + 16
sw3 = SIDE_W - 32
sy3 = BIG_Y + big_header_h
sh3 = big_h - big_header_h - 20
L.append(f'<rect x="{sx3}" y="{sy3}" width="{sw3}" height="{sh3}" rx="9" '
          f'fill="#e0e7ff" stroke="#4338ca" stroke-width="2"/>')
L.append(f'<text x="{sx3+16}" y="{sy3+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc("路径③ · @jit 组合已有原语")}</text>')
L.append(f'<text x="{sx3+16}" y="{sy3+50}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("代表 isfinited =")}</text>')
L.append(f'<text x="{sx3+16}" y="{sy3+68}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">{esc("~isnan & ~isinf")}</text>')
L.append(f'<text x="{sx3+sw3-16}" y="{sy3+sh3-14}" text-anchor="end" font-family="sans-serif" '
          f'font-size="20" font-weight="bold" fill="#4338ca">{esc("3 个")}</text>')

# 汇合箭头：libdevice.py 与 math_ops.py 都指向下方命名空间
big_cx = BIG_X + BIG_W / 2
side_cx = SIDE_X + SIDE_W / 2
L.append(f'<line x1="{big_cx}" y1="{BIG_Y+big_h}" x2="{big_cx}" y2="{merge_y}" '
          f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{side_cx}" y1="{BIG_Y+big_h}" x2="{side_cx}" y2="{merge_y}" '
          f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

L.append(f'<rect x="{merge_x}" y="{merge_y}" width="{MERGE_W}" height="{MERGE_H}" rx="10" '
          f'fill="#e2e8f0" stroke="#475569" stroke-width="1.8"/>')
L.append(f'<text x="{merge_x+MERGE_W/2}" y="{merge_y+27}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e293b">'
          f'{esc("四类形态都汇进同一个 al.libdevice.* 命名空间")}</text>')
L.append(f'<text x="{merge_x+MERGE_W/2}" y="{merge_y+47}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">'
          f'{esc("cann/__init__.py 的 import 期覆盖挂载")}</text>')

# 底部：数字核对 + __hmf_ 规模 + acos 实测
L.append(f'<rect x="{PAD}" y="{callout_y}" width="{w-PAD*2}" height="{CALLOUT_H}" rx="10" '
          f'fill="#fefce8" stroke="#ca8a04" stroke-width="1.6"/>')
L.append(f'<text x="{PAD+16}" y="{callout_y+24}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#713f12">'
          f'{esc("__hmf_ 符号：66 处引用 / 60 个不同符号——昇腾自带的数学库菜单（A、B、C 都点这份菜单）")}</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+46}" font-family="sans-serif" font-size="11.5" '
          f'fill="#854d0e">'
          f'{esc("acos 实测（CPU 精简版）：纯 IR 逼近 vs math.acos 最大绝对误差（8 个采样点）= 1e-05；")}</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+66}" font-family="sans-serif" font-size="11.5" '
          f'fill="#854d0e">'
          f'{esc("走 extern 分支时执行的浮点算术条数 = 0（直接点 __hmf_acos_fp32，不算多项式）。")}</text>')
L.append(f'<text x="{PAD+16}" y="{callout_y+86}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#713f12">'
          f'{esc("14 + 2 + 18 + 3 = 37 = libdevice.py 顶层函数总数——分类互斥且穷尽。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch07-04-libdevice-three-paths.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
