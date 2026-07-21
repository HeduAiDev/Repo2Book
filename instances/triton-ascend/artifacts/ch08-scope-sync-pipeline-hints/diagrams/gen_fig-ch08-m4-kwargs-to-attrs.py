#!/usr/bin/env python3
"""state-table 模板（双列清单）：scope 关键字实参 → scope.scope 属性的翻译表。
左列=调用写法，右列=生成的 mlir_attrs 结果；异常/静默丢弃行标警示色。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "关键字 → 属性翻译表：白名单外的写法不报错，只是静默不生效"
SUBTITLE = "_extract_scope_attributes 只认 ast.Constant 关键字实参；_build_mlir_attrs_from_scope_attrs 逐项翻译"
# 十行 = 十个用例全列（不抽样），读者可就地数出「带 tcore_type 的用例 6 / 10」
ROWS = [
    ('scope("vector")（位置参数写法）', "keywords 为空 ⇒ 无 tcore_type，只剩 {noinline}", "warn", False),
    ('scope(core_mode="cube")', "{noinline, tcore_type<CUBE>}", "normal", True),
    ('scope(core_mode="vector")', "{noinline, tcore_type<VECTOR>}", "normal", True),
    ('scope(core_mode="vector", noinline=False)', "{tcore_type<VECTOR>}（noinline 被 pop 掉）", "normal", True),
    ('scope(core_mode="vector", disable_auto_sync=True)', "多出 hivm.disable_auto_sync = #bool<true>（共 3 项）", "normal", True),
    ('scope(core_mode="vector", disable_auto_sync=False)', "什么都不加（属性数回落到 2）", "normal", True),
    ('scope(core_mode="aicore")（不在白名单）', "静默无 tcore_type，只剩 {noinline}", "warn", False),
    ("scope(core_mode=mode_var)（变量，非常量）", "被 _extract_scope_attributes 丢弃 ⇒ 无 tcore_type", "warn", False),
    ("scope(feature_a=True)", "{noinline, feature_a = #bool<true>}（其余原样透传）", "normal", False),
    ('scope(core_mode="cube", my_hint=3, my_tag="x", my_list=[7, 9])',
     "my_list 被丢弃（ast.List 不是 ast.Constant），其余三项照常落地", "warn", True),
]
COLOR = {"normal": ("#eff6ff", "#1d4ed8", "#1e3a8a"),
         "warn": ("#fff7ed", "#c2410c", "#7c2d12")}

PAD, TOP = 40, 96
LEFT_W, RIGHT_W, GAP, TC_W = 470, 560, 16, 96
ROW_H = 42
w = PAD * 2 + LEFT_W + GAP + RIGHT_W + GAP + TC_W
h = TOP + ROW_H * len(ROWS) + 116

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
TC_X = PAD + LEFT_W + GAP + RIGHT_W + GAP
L.append(f'<text x="{TC_X+TC_W/2}" y="{hy}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#64748b">{esc("tcore_type")}</text>')

for i, (lhs, rhs, kind, has_tc) in enumerate(ROWS):
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
    tc_fill, tc_stroke = ("#dcfce7", "#15803d") if has_tc else ("#f1f5f9", "#94a3b8")
    L.append(f'<rect x="{TC_X}" y="{y}" width="{TC_W}" height="{ROW_H-6}" rx="6" '
              f'fill="{tc_fill}" stroke="{tc_stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="{TC_X+TC_W/2}" y="{y+(ROW_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{tc_stroke}">{esc("✓ 有" if has_tc else "— 无")}</text>')

fy1 = TOP + ROW_H * len(ROWS) + 26
fy2 = fy1 + 20
L.append(f'<text x="{PAD}" y="{fy1}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("带 tcore_type 的用例：6 / 10 —— 上表十行即全部用例，右列「✓ 有」正好 6 行。")}</text>')
# ⚠ 四行的「后果」分两类：前三行丢的是 tcore_type，末行丢的是 my_list 这一项本身
for k, line in enumerate([
        "⚠ 标记的 4 行都不报任何错误：core_mode 拼错、写成位置参数、传变量这 3 行，让 tcore_type 悄悄消失；",
        "传 list 那行 tcore_type 照常落地（core_mode=\"cube\" 是 ast.Constant，照样过滤器），悄悄消失的是 my_list 这一项。"]):
    L.append(f'<text x="{PAD}" y="{fy2+k*19}" font-family="sans-serif" font-size="11" '
              f'fill="#c2410c">{esc(line)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m4-kwargs-to-attrs.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
