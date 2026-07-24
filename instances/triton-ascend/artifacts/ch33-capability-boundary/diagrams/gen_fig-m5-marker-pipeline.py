#!/usr/bin/env python3
"""fig-m5-marker-pipeline — state-machine 模板:skip/skipif/xfail 三种标记在
pytest 收集->执行->报告流水线里各截一处;只有 xfail 让 kernel 真执行,
XPASS(X)是唯一的回归哨兵。三条水平泳道,各自从"收集"出发到终态。
全部坐标由循环/常量计算,文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "skip / skipif / xfail:谁真执行 kernel,谁能报回归信号"
SUBTITLE = "39 处 skip(止血,恒定)+ 1 处 skipif 硬件条件(A5 上真跑)+ 1 处 xfail(唯一回归哨兵)"

STAGE_LABELS = ["收集(collection)", "执行?(execution)", "终态(报告状态符)"]

# 每条泳道: (名字, 颜色, 载体file:L, 各阶段描述list[3], 终态高亮?)
LANES = [
    ("@pytest.mark.skip", "#475569",
     "test_dot.py:L128 test_dot_2",
     ["照常收集", "跳过 —— kernel 根本不执行", "s(skipped)恒定,底层修好也不主动提示"],
     False),
    ("@pytest.mark.skipif(not is_compile_on_910_95)", "#0284c7",
     "test_no_tiling_axis_parse.py:L92 test_permute_simt",
     ["照常收集", "条件真(非 A5)跳过;条件假(A5)真执行对拍", "非 A5:s;A5:pass/fail 照常——边界随硬件浮动"],
     False),
    ("@pytest.mark.xfail", "#b91c1c",
     "test_dot.py:L140 test_dot_2_allow_tf32",
     ["照常收集", "真执行 kernel,期望失败", "失败→x(计入通过);意外通过→X(XPASS 回归哨兵)"],
     True),
]

PAD = 40
W = 1360
TOP = 100
COL_W = (W - 2 * PAD - 220) / 3
LANE_H = 96
LANE_GAP = 22
NAME_COL_W = 220

H = TOP + 30 + len(LANES) * (LANE_H + LANE_GAP) - LANE_GAP + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="19" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头(收集/执行?/终态)
head_y = TOP
for i, lbl in enumerate(STAGE_LABELS):
    cx = PAD + NAME_COL_W + i * COL_W + COL_W / 2
    L.append(f'<text x="{cx}" y="{head_y}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#334155">{esc(lbl)}</text>')

y = TOP + 30
for name, color, cite, stages, hi in LANES:
    L.append(f'<rect x="{PAD}" y="{y}" width="{W-2*PAD}" height="{LANE_H}" rx="8" '
              f'fill="{color}" fill-opacity="0.06" stroke="{color}" stroke-width="1.2"/>')
    L.append(f'<text x="{PAD+14}" y="{y+22}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<text x="{PAD+14}" y="{y+40}" font-family="sans-serif" font-size="10.5" '
              f'fill="#64748b">{esc(cite)}</text>')
    for i, desc in enumerate(stages):
        cx0 = PAD + NAME_COL_W + i * COL_W
        cell_w = COL_W - 16
        cell_y = y + 50
        cell_h = LANE_H - 62
        is_final = (i == 2)
        fill = color if (is_final and hi) else "white"
        stroke = color
        text_fill = "white" if (is_final and hi) else "#1e293b"
        L.append(f'<rect x="{cx0}" y="{cell_y}" width="{cell_w}" height="{cell_h}" rx="6" '
                  f'fill="{fill}" fill-opacity="{1.0 if (is_final and hi) else 0.9}" '
                  f'stroke="{stroke}" stroke-width="{2 if (is_final and hi) else 1}"/>')
        # 文本换行(按 12 字宽粗切)
        words = desc
        maxlen = 16
        lines = []
        cur = ""
        for ch in words:
            cur += ch
            if len(cur) >= maxlen and ch in "、,;—)":
                lines.append(cur)
                cur = ""
        if cur:
            lines.append(cur)
        n = len(lines)
        ty0 = cell_y + cell_h / 2 - (n - 1) * 7 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx0+cell_w/2}" y="{ty0+k*14}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10.5" '
                      f'fill="{text_fill}">{esc(line)}</text>')
        if i < 2:
            ax1 = cx0 + cell_w
            ax2 = cx0 + COL_W
            ay = cell_y + cell_h / 2
            L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" stroke="#94a3b8" '
                      f'stroke-width="1.3" marker-end="url(#a)"/>')
    y += LANE_H + LANE_GAP

foot_y = H - 22
L.append(f'<line x1="{PAD}" y1="{foot_y-16}" x2="{W-PAD}" y2="{foot_y-16}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("仅 xfail 让 kernel 真执行,故仅它的终态能自发从 x 翻到 X 以示边界移动(红色高亮格)——skip/skipif 的跳过分支恒为 s,不会自我更新")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m5-marker-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
