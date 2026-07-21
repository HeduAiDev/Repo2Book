#!/usr/bin/env python3
"""state-table 模板（双列清单）：scope 关键字实参 → scope.scope 属性的翻译表。
左列=调用写法，右列=生成的 mlir_attrs 结果；异常/静默丢弃行标警示色。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "关键字 → 属性翻译表：白名单外的写法不报错，只是静默不生效"
SUBTITLE = "_extract_scope_attributes 只认 ast.Constant 关键字实参；_build_mlir_attrs_from_scope_attrs 逐项翻译"
ROWS = [
    ("默认（无关键字）", "{noinline}（1 个默认属性）", "normal"),
    ('core_mode="cube"', "{noinline, tcore_type<CUBE>}", "normal"),
    ('core_mode="vector", noinline=False', "{tcore_type<VECTOR>}（noinline 被 pop 掉）", "normal"),
    ('core_mode="vector", disable_auto_sync=True', "多出 hivm.disable_auto_sync = #bool<true>", "normal"),
    ('core_mode="vector", disable_auto_sync=False', "什么都不加（属性数回落到 2）", "normal"),
    ('core_mode="aicore"（不在白名单）', "静默无 tcore_type，只剩 {noinline}", "warn"),
    ('位置参数写法 scope("vector")', "keywords 为空 ⇒ 无 tcore_type", "warn"),
    ("core_mode=mode_var（变量，非常量）", "被 _extract_scope_attributes 丢弃 ⇒ 无 tcore_type", "warn"),
    ("my_list=[7, 9]（ast.List）", "同样被丢弃（不是 ast.Constant）", "warn"),
]
COLOR = {"normal": ("#eff6ff", "#1d4ed8", "#1e3a8a"),
         "warn": ("#fff7ed", "#c2410c", "#7c2d12")}

PAD, TOP = 40, 96
LEFT_W, RIGHT_W, GAP = 430, 560, 16
ROW_H = 42
w = PAD * 2 + LEFT_W + GAP + RIGHT_W
h = TOP + ROW_H * len(ROWS) + 96

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

hy = TOP - 8
L.append(f'<text x="{PAD+16}" y="{hy}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#64748b">{esc("调用写法")}</text>')
L.append(f'<text x="{PAD+LEFT_W+GAP+16}" y="{hy}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#64748b">{esc("生成的 scope.scope 属性 / 结果")}</text>')

for i, (lhs, rhs, kind) in enumerate(ROWS):
    y = TOP + i * ROW_H
    fill, stroke, text_fill = COLOR[kind]
    L.append(f'<rect x="{PAD}" y="{y}" width="{LEFT_W}" height="{ROW_H-6}" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{PAD+16}" y="{y+(ROW_H-6)/2+5}" font-family="sans-serif" '
              f'font-size="12.5" fill="{text_fill}">{esc(lhs)}</text>')
    arrow_x1 = PAD + LEFT_W + 2
    arrow_x2 = PAD + LEFT_W + GAP - 2
    ay = y + (ROW_H - 6) / 2
    L.append(f'<line x1="{arrow_x1}" y1="{ay}" x2="{arrow_x2}" y2="{ay}" '
              f'stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<path d="M{arrow_x2-5},{ay-4} L{arrow_x2+1},{ay} L{arrow_x2-5},{ay+4} Z" '
              f'fill="{stroke}"/>')
    rx = PAD + LEFT_W + GAP
    L.append(f'<rect x="{rx}" y="{y}" width="{RIGHT_W}" height="{ROW_H-6}" rx="6" '
              f'fill="white" stroke="{stroke}" stroke-width="1.3"/>')
    mark = "⚠ " if kind == "warn" else ""
    weight_attr = 'font-weight="bold" ' if kind == "warn" else ''
    L.append(f'<text x="{rx+16}" y="{y+(ROW_H-6)/2+5}" font-family="sans-serif" '
              f'font-size="12.5" fill="{text_fill}" {weight_attr}>'
              f'{esc(mark+rhs)}</text>')

fy1 = TOP + ROW_H * len(ROWS) + 26
fy2 = fy1 + 20
L.append(f'<text x="{PAD}" y="{fy1}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("带 tcore_type 的用例：6 / 10（统计口径：带上 tcore_type 的用例数 / 总用例数）")}</text>')
L.append(f'<text x="{PAD}" y="{fy2}" font-family="sans-serif" font-size="11" '
          f'fill="#c2410c">'
          f'{esc("⚠ 标记的 4 行不报任何错误——core_mode 拼错、用位置参数、传变量或 list，都会让 tcore_type 悄悄消失。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m4-kwargs-to-attrs.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
