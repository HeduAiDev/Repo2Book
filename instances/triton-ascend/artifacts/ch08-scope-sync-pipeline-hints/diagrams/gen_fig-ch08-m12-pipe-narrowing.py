#!/usr/bin/env python3
"""layout 模板（三级收窄台阶）：PIPE 15→8→8、TCoreType 4→4→2——
定义 / 导出 / 可用 三列，两行（PIPE / TCoreType），掉队项灰显列出名字。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "同一条收窄链，第二次出现：定义 → 导出 → 可用，逐级变窄"
SUBTITLE = "ch05 讲地址空间时的三级台阶，这里 PIPE 与 TCoreType 各走一遍"

STAGES = ["① .td 定义", "② pybind 导出", "③ 语言层可用"]
PAD, TOP = 40, 100
GUTTER = 130           # 最左行标签栏：行名与色块垂直居中、离框左边 ≥16px
STAGE_W, STAGE_GAP = 300, 90
BOX_H = 100            # 按内容（标题行 + 至多 3 条注解行 + 上下 padding）压缩
LANE_GAP = 44
LANE_H = BOX_H + LANE_GAP
w = PAD * 2 + GUTTER + STAGE_W * 3 + STAGE_GAP * 2
h = TOP + 14 + BOX_H * 2 + LANE_GAP + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="32" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12.5" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

stage_x = [PAD + GUTTER + i * (STAGE_W + STAGE_GAP) for i in range(3)]
for i, name in enumerate(STAGES):
    L.append(f'<text x="{stage_x[i]+STAGE_W/2}" y="{TOP}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#334155">{esc(name)}</text>')

def lane(y, label, counts, boxes, color):
    """boxes: list of (title, subtitle_lines, is_dropped) per stage"""
    L.append(f'<text x="{stage_x[0]-16}" y="{y+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="{color[1]}">{esc(label)}</text>')
    for i in range(3):
        bx = stage_x[i]
        title, sub, dropped = boxes[i]
        fill = "#f1f5f9" if dropped else color[0]
        stroke = "#94a3b8" if dropped else color[1]
        bh = BOX_H
        L.append(f'<rect x="{bx}" y="{y-bh/2}" width="{STAGE_W}" height="{bh}" rx="10" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        L.append(f'<text x="{bx+STAGE_W/2}" y="{y-bh/2+28}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="15" font-weight="bold" '
                  f'fill="{stroke}">{esc(title)}</text>')
        y0 = y - bh/2 + 50
        for k, line in enumerate([s for s in sub if s]):
            L.append(f'<text x="{bx+STAGE_W/2}" y="{y0+k*16}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10.5" fill="{stroke}">'
                      f'{esc(line)}</text>')
        if i < 2:
            ax1 = bx + STAGE_W + 6
            ax2 = stage_x[i+1] - 6
            L.append(f'<line x1="{ax1}" y1="{y}" x2="{ax2}" y2="{y}" '
                      f'stroke="#64748b" stroke-width="1.8" marker-end="url(#a)"/>')

y1 = TOP + 14 + BOX_H/2
pipe_boxes = [
    ("15 档", ["HIVMAttrs.td:L220-L234", "含 MTE4/MTE5/V2/两个 VIRTUAL_*", "PIPE_NUM、PIPE_UNASSIGNED(=99)"], False),
    ("8 档", ["ascend_ir.cc:L428-L435", "掉队 7 档（灰显于下方）", ""], False),
    ("8 档", ["core.py:L111-L119", "Python class PIPE", "与导出档数持平"], False),
]
lane(y1, "PIPE", None, pipe_boxes, ("#dbeafe", "#1d4ed8"))

y2 = y1 + LANE_H
tcore_boxes = [
    ("4 档", ["CUBE / VECTOR /", "CUBE_OR_VECTOR / CUBE_AND_VECTOR", ""], False),
    ("4 档", ["pybind 全部导出", "（无收窄）", ""], False),
    ("2 档", ["code_generator.py:L84-L93", "白名单 (\"cube\", \"vector\")", "另 2 档从 scope 到不了"], False),
]
lane(y2, "TCoreType", None, tcore_boxes, ("#ede9fe", "#6d28d9"))

# dropped items panel
dy = y2 + BOX_H/2 + 44
dh = 96
L.append(f'<rect x="{PAD}" y="{dy}" width="{w-2*PAD}" height="{dh}" rx="9" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.4"/>')
L.append(f'<text x="{PAD+18}" y="{dy+24}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#475569">'
          f'{esc("PIPE 在 pybind 处掉队的 7 档（第 15→8 步）：")}</text>')
L.append(f'<text x="{PAD+18}" y="{dy+44}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">'
          f'{esc("PIPE_MTE4, PIPE_MTE5, PIPE_V2, VIRTUAL_PIPE_MTE2_L1A, VIRTUAL_PIPE_MTE2_L1B, PIPE_NUM, PIPE_UNASSIGNED")}</text>')
L.append(f'<text x="{PAD+18}" y="{dy+68}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#475569">'
          f'{esc("TCoreType 从 scope(core_mode=...) 到不了的 2 档：")}</text>')
L.append(f'<text x="{PAD+18}" y="{dy+88}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">'
          f'{esc("CUBE_OR_VECTOR、CUBE_AND_VECTOR —— 掉队各档的硬件含义源码未给出依据，见 open_question")}</text>')

foot_y = dy + dh + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("PIPE：15 → 8 → 8；TCoreType：4 → 4 → 2——底层 IR 表达力与语言层可写出来的东西之间，永远隔着一层人为的窄口。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m12-pipe-narrowing.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
