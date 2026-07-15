#!/usr/bin/env python3
"""fig-tl-namespace: flow 模板 —— language/__init__.py 按固定顺序把
core/standard/math/random 四个子模块的符号 re-export 到 tl.* 顶层，
拼成 __all__ 门面（131 项，其中 3 项在册未导入）。
四个源模块方框 -> 汇入 tl.* 方框 -> 侧注 __all__ 门面计数。
全坐标由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "tl.* 这张表面怎么铺出来"
SUBTITLE = "python/triton/language/__init__.py 依序 re-export 四个子模块符号（Triton v3.2.0，AST 统计）"

SOURCES = [
    ("standard.py", "18", "组合函数（被追踪，@jit）"),
    ("core.py", "81", "原语（多为 @builtin）"),
    ("math.py", "17", "外部数学函数"),
    ("random.py", "10", "philox 随机数"),
]

SRC_W, SRC_H, SRC_GAP = 230, 78, 26
LEFT_PAD, TOP = 40, 96
srcs_x = LEFT_PAD
total_src_h = len(SOURCES) * SRC_H + (len(SOURCES) - 1) * SRC_GAP

NS_W, NS_H = 260, total_src_h
NS_X = srcs_x + SRC_W + 110
NS_Y = TOP

ALL_W, ALL_H = 300, 132
ALL_X = NS_X + NS_W + 110
ALL_Y = TOP + (total_src_h - ALL_H) / 2

w = ALL_X + ALL_W + 40
h = TOP + total_src_h + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
          'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{LEFT_PAD}" y="34" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>')
L.append(f'<text x="{LEFT_PAD}" y="58" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# 四个源模块方框
src_ys = []
y = TOP
for i, (name, cnt, desc) in enumerate(SOURCES):
    src_ys.append(y)
    L.append(f'<rect x="{srcs_x}" y="{y}" width="{SRC_W}" height="{SRC_H}" rx="8" '
              'fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>')
    L.append(f'<text x="{srcs_x+16}" y="{y+26}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="#1e3a5f">{esc(name)}</text>')
    L.append(f'<text x="{srcs_x+16}" y="{y+46}" font-family="sans-serif" font-size="12" '
              f'fill="#374151">{esc(desc)}</text>')
    L.append(f'<text x="{srcs_x+SRC_W-16}" y="{y+SRC_H-14}" text-anchor="end" '
              f'font-family="sans-serif" font-size="20" font-weight="bold" '
              f'fill="#1d4ed8">{esc(cnt)}</text>')
    L.append(f'<text x="{srcs_x+16}" y="{y+SRC_H-14}" font-family="sans-serif" font-size="11" '
              f'fill="#64748b">from .{name.split(".")[0]} 提到顶层</text>')
    y += SRC_H + SRC_GAP

# 汇入箭头：每个源框右边缘 -> tl.* 框左边缘
ns_cy = NS_Y + NS_H / 2
for sy in src_ys:
    scy = sy + SRC_H / 2
    L.append(f'<path d="M {srcs_x+SRC_W} {scy} C {srcs_x+SRC_W+55} {scy}, '
              f'{NS_X-55} {ns_cy}, {NS_X} {ns_cy}" fill="none" stroke="#94a3b8" '
              'stroke-width="1.6" marker-end="url(#a)"/>')

# tl.* 命名空间方框
L.append(f'<rect x="{NS_X}" y="{NS_Y}" width="{NS_W}" height="{NS_H}" rx="10" '
          'fill="#1e40af" stroke="#1e3a5f" stroke-width="2"/>')
L.append(f'<text x="{NS_X+NS_W/2}" y="{NS_Y+NS_H/2-8}" text-anchor="middle" '
          'font-family="sans-serif" font-size="20" font-weight="bold" '
          'fill="white">tl.*</text>')
L.append(f'<text x="{NS_X+NS_W/2}" y="{NS_Y+NS_H/2+16}" text-anchor="middle" '
          'font-family="sans-serif" font-size="12" '
          'fill="#dbeafe">18+81+17+10 = 126 个符号</text>')
L.append(f'<text x="{NS_X+NS_W/2}" y="{NS_Y+NS_H/2+34}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11.5" '
          'fill="#bfdbfe">汇聚到顶层命名空间</text>')

# tl.* -> __all__ 箭头
L.append(f'<line x1="{NS_X+NS_W}" y1="{ns_cy}" x2="{ALL_X-6}" y2="{ns_cy}" '
          'stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

# __all__ 门面方框
L.append(f'<rect x="{ALL_X}" y="{ALL_Y}" width="{ALL_W}" height="{ALL_H}" rx="10" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{ALL_X+ALL_W/2}" y="{ALL_Y+26}" text-anchor="middle" '
          'font-family="sans-serif" font-size="14" font-weight="bold" '
          'fill="#92400e">__all__ = 131 项</text>')
L.append(f'<text x="{ALL_X+ALL_W/2}" y="{ALL_Y+48}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11.5" '
          'fill="#78350f">字母序公开门面清单</text>')
L.append(f'<text x="{ALL_X+ALL_W/2}" y="{ALL_Y+72}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11.5" '
          'fill="#b91c1c" font-weight="bold">含 3 项在册未导入</text>')
L.append(f'<text x="{ALL_X+ALL_W/2}" y="{ALL_Y+92}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" '
          'fill="#7c2d12">(builtin/ir/triton：访问即</text>')
L.append(f'<text x="{ALL_X+ALL_W/2}" y="{ALL_Y+108}" text-anchor="middle" '
          'font-family="sans-serif" font-size="11" '
          'fill="#7c2d12">AttributeError)</text>')

foot_y = h - 26
L.append(f'<text x="{LEFT_PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">__init__.py:L6-L111 依序 re-export 四模块（standard 18 / core 81 / math 17 / random 10）→ L124-L256 声明 __all__=131 项；'
          '声明不是保证，__all__ 是门面清单，不代表名字真的可导入。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-tl-namespace.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
