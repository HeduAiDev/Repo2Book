#!/usr/bin/env python3
"""fig-ch07-03-core-pipe-mode-to-hivm — layout 模板（core/pipe/mode 三行落 hivm 属性）。
三行：CORE/PIPE/MODE 各自的枚举取值 -> _make_attrs 的 builder 调用 -> IR 属性名。
下方图例列出三个枚举的全部成员（本例取值加粗标出），底部一条分支标注非 __builtin_
前缀时额外的 symbol/bitcode 属性。全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "core / pipe / mode：三个 Python 枚举落成三条 hivm 属性"
SUBTITLE = "注册期声明的语言层枚举，在 _make_attrs 里一一对应地翻成 IR 属性——这是硬件模型接缝的落点"

ROWS = [
    ("CORE", "core = CORE.VECTOR", "get_core_type_attr(op.core.value)", "hivm.tcore_type"),
    ("PIPE", "pipe = PIPE.PIPE_V", "get_pipe_attr(op.pipe.value)", "hivm.pipe"),
    ("MODE", "mode = MODE.SIMT", "get_vf_mode_attr(op.mode.value)", "hivm.vf_mode"),
]

COL_A_W, COL_B_W, COL_C_W = 230, 300, 190
GAP_AB, GAP_BC = 60, 60
ROW_H = 74
ROW_GAP = 34
PAD = 40
TOP = 108

col_a_x = PAD
col_b_x = col_a_x + COL_A_W + GAP_AB
col_c_x = col_b_x + COL_B_W + GAP_BC
w = col_c_x + COL_C_W + PAD

row_y = [TOP + i * (ROW_H + ROW_GAP) for i in range(len(ROWS))]
grid_bottom = row_y[-1] + ROW_H

LEGEND = [
    ("CORE", 4, ["VECTOR（本例）", "CUBE", "CUBE_OR_VECTOR", "CUBE_AND_VECTOR"]),
    ("PIPE", 8, ["PIPE_S", "PIPE_V（本例）", "PIPE_M", "PIPE_MTE1", "PIPE_MTE2", "PIPE_MTE3", "PIPE_ALL", "PIPE_FIX"]),
    ("MODE", 3, ["SIMD", "SIMT（本例）", "MIX"]),
]
LEG_TOP = grid_bottom + 56
LEG_ROW_H = 30

CALLOUT_TOP = LEG_TOP + LEG_ROW_H * len(LEGEND) + 44
CALLOUT_H = 90

h = CALLOUT_TOP + CALLOUT_H + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头
for x, wcol, text in [(col_a_x, COL_A_W, "注册期声明（Python 枚举）"),
                       (col_b_x, COL_B_W, "_make_attrs 的 builder 调用"),
                       (col_c_x, COL_C_W, "IR 属性名")]:
    L.append(f'<text x="{x+wcol/2}" y="{TOP-18}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#334155">{esc(text)}</text>')

COLORS = {"A": ("#e0f2fe", "#0369a1"), "B": ("#fef3c7", "#b45309"), "C": ("#dcfce7", "#15803d")}
for i, (dim, a_text, b_text, c_text) in enumerate(ROWS):
    y = row_y[i]
    cy = y + ROW_H / 2
    for (x, wcol, text, key) in [(col_a_x, COL_A_W, a_text, "A"),
                                   (col_b_x, COL_B_W, b_text, "B"),
                                   (col_c_x, COL_C_W, c_text, "C")]:
        fill, stroke = COLORS[key]
        L.append(f'<rect x="{x}" y="{y}" width="{wcol}" height="{ROW_H}" rx="10" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        L.append(f'<text x="{x+wcol/2}" y="{cy-2}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(text)}</text>')
    L.append(f'<text x="{col_a_x+14}" y="{y+18}" font-family="sans-serif" font-size="10.5" '
              f'font-weight="bold" fill="{COLORS["A"][1]}">{esc(dim)}</text>')
    # 箭头 A->B, B->C
    L.append(f'<line x1="{col_a_x+COL_A_W}" y1="{cy}" x2="{col_b_x}" y2="{cy}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<line x1="{col_b_x+COL_B_W}" y1="{cy}" x2="{col_c_x}" y2="{cy}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')

# 图例：三个枚举的全部成员
L.append(f'<text x="{PAD}" y="{LEG_TOP-20}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc("三个枚举的全部成员（粗体=本例取值）")}</text>')
for i, (dim, cnt, members) in enumerate(LEGEND):
    ly = LEG_TOP + i * LEG_ROW_H
    L.append(f'<text x="{PAD}" y="{ly}" font-family="sans-serif" font-size="12" '
              f'font-weight="bold" fill="#334155">{esc(f"{dim}（共 {cnt} 个）：")}</text>')
    label_w = 150
    body = " / ".join(members)
    L.append(f'<text x="{PAD+label_w}" y="{ly}" font-family="sans-serif" font-size="11.5" '
              f'fill="#475569">{esc(body)}</text>')

# 分支标注
L.append(f'<rect x="{PAD}" y="{CALLOUT_TOP}" width="{w-PAD*2}" height="{CALLOUT_H}" rx="10" '
          f'fill="#f1f5f9" stroke="#64748b" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{PAD+16}" y="{CALLOUT_TOP+24}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#334155">{esc("分支：非 __builtin_ 前缀时的额外属性")}</text>')
L.append(f'<text x="{PAD+16}" y="{CALLOUT_TOP+46}" font-family="sans-serif" font-size="11.5" '
          f'fill="#475569">{esc("op.name 不以 __builtin_ 开头 → _make_attrs 额外强制 symbol + bitcode 两个属性；")}</text>')
L.append(f'<text x="{PAD+16}" y="{CALLOUT_TOP+66}" font-family="sans-serif" font-size="11.5" '
          f'fill="#475569">{esc("必出现的 IR 属性数 = 3（tcore_type/pipe/vf_mode）+ 可选 extra_attr——本例（内建算子）共 4 条，没有 symbol/bitcode。")}</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11"  '
          f'fill="#64748b">{esc("_index_select 类本身与 __builtin_ 前缀哑类兜底的默认取值恰好一致：都是 VECTOR / PIPE_V / SIMT。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch07-03-core-pipe-mode-to-hivm.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
