#!/usr/bin/env python3
"""fig-m3-unsupported-census — 分组横向条形(改自 state-table 的行列思路):
pytest_ut 里 40 处生效的 skip/xfail 标记,按 reason 归成三组——五主类(33)、
零星半支持(6)、唯一 xfail 哨兵(1),条长 ∝ 数值,合计脚注核对 40。
全部坐标由循环/常量计算,文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "反面清单:40 处生效 skip/xfail,按 reason 精确归类"
SUBTITLE = "计数口径 = 标记出现次数(marker occurrences),非文件数;对应活跃无条件 skip/xfail 文件数 = 22(pytest_ut 20 + autotune_ut 2)"

GROUPS = [
    ("五主类 · 确定性暂不支持(合计 33)", "#1d4ed8", [
        ("waiting for TA to support", 13, "test_device_print_script.py:L50 等"),
        ("bishengir / compiler to support", 9, "test_pow.py:L47/L76/L105 等"),
        ("NPUIR updated in April 回退", 5, "test_dot.py:L128"),
        ("UB overflow(硬件)", 3, "test_09_persistent_matmul.py 等"),
        ("attn_cp 整批", 3, "test_attn_cp.py:L486-L496"),
    ]),
    ("零星 · flaky/半支持(合计 6)", "#b45309", [
        ("flaky: randomly failed", 4, "多文件偶发"),
        ("atomic_cas full tensor 有问题", 1, "test_atomic_cas.py:L171"),
        ("expm1 failed sometimes", 1, "test_expm1.py"),
    ]),
    ("唯一 xfail 哨兵(合计 1)", "#15803d", [
        ("xfail: allow_tf32", 1, "test_dot.py:L140-L141"),
    ]),
]

MAX_VAL = 13
PAD = 40
W = 1280
LABEL_W = 300
BAR_MAX_W = 560
BAR_H = 26
ROW_GAP = 10
GROUP_HEAD_H = 30
GROUP_GAP = 26
TOP = 100

L = []
row_defs = []  # (group_color, label, value, note)
group_heights = []
for gname, gcolor, rows in GROUPS:
    group_heights.append(GROUP_HEAD_H + len(rows) * (BAR_H + ROW_GAP) - ROW_GAP)

y = TOP
group_y = []
for gh in group_heights:
    group_y.append(y)
    y += gh + GROUP_GAP
H = y - GROUP_GAP + 70

L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="19" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

for (gname, gcolor, rows), gy in zip(GROUPS, group_y):
    L.append(f'<text x="{PAD}" y="{gy+16}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="{gcolor}">{esc(gname)}</text>')
    ry = gy + GROUP_HEAD_H
    for label, val, note in rows:
        bar_w = BAR_MAX_W * val / MAX_VAL
        label_x = PAD + LABEL_W
        L.append(f'<text x="{label_x-10}" y="{ry+BAR_H/2+4}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="12.5" fill="#1e293b">{esc(label)}</text>')
        L.append(f'<rect x="{label_x}" y="{ry}" width="{bar_w}" height="{BAR_H}" rx="4" '
                  f'fill="{gcolor}" fill-opacity="0.85"/>')
        L.append(f'<text x="{label_x+bar_w+10}" y="{ry+BAR_H/2+4}" font-family="sans-serif" '
                  f'font-size="13" font-weight="bold" fill="{gcolor}">{esc(str(val))}</text>')
        note_x = label_x + BAR_MAX_W + 46
        L.append(f'<text x="{note_x}" y="{ry+BAR_H/2+4}" font-family="sans-serif" '
                  f'font-size="10.5" fill="#94a3b8">{esc(note)}</text>')
        ry += BAR_H + ROW_GAP

foot_y = H - 40
L.append(f'<line x1="{PAD}" y1="{foot_y-18}" x2="{W-PAD}" y2="{foot_y-18}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#334155">'
          f'{esc("合计核对:13+9+5+3+3=33  +  4+1+1=6  +  1  =  40")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("硬件条件跳(skipif,如 test_no_tiling_axis_parse.py:L92 ‘only support A5’)会在 A5 上真跑,不计入这 40")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-unsupported-census.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
