#!/usr/bin/env python3
"""fig-m12-predicated-drain: state-table — 谓词化收尾:stage 0 的取数在
iv>=ub-2 时被 mask 关闭,早 stage 临尾逐个熄灭,排空折进主循环、无独立 epilogue 块。
数字来自 explainer.json m12.figure_specs[0].numbers:
  2 = matmul_sm90_ns3.ttgir.mlir:%78 = ub - 2(maxStage=2)
  0 = matmul_sm90_ns3.ttgir.mlir:%66 async_wait num=0(循环后排空)
  0 = pipeline_dump_summary.json: sm90_ns3.ttgir_scf_if=0(无独立 epilogue 分支)
  6 = ringbuffer_trace.json: trip=6 时谓词在 iter>=4(=trip-2)关闭预取
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "谓词化收尾:不 peel,靠临尾谓词逐 stage 熄灭"
SUBTITLE = "peelEpilogue=false,supportDynamicLoops=true(matmul_sm90_ns3.ttgir.mlir)"

BADGES = [
    ("ub − 2", "maxStage=2 => stage0 边界"),
    ("scf_if = 0", "无独立 epilogue 分支"),
    ("async_wait = 0", "循环后一次性排空"),
    ("trip=6 => iter≥4 关闭", "谓词临尾提前 2 拍生效"),
]

COLS = ["稳态体开头", "async_copy 处", "循环之后(无独立 epilogue)"]
ROW_LABELS = ["构造", "作用", "IR 证据"]
CELLS = {
    "构造": [
        "%78 = ub − 2\n%79 = cmpi slt, iv, %78",
        "splat %79 作 mask\n喂 async_copy_global_to_local",
        "warp_group_dot_wait pendings=0\nasync_wait num=0\nlocal_dealloc",
    ],
    "作用": [
        "stage 0 的谓词:\n临尾前 2 拍起不再发取数",
        "谓词假 => 该拍 cp.async\n被 mask 关闭(不越界取数)",
        "排空最后在飞的 dot 与 copy、\n释放缓冲(收尾靠这几条,非 emitEpilogue)",
    ],
    "IR 证据": ["%78, %79", "%94, %95, %98, %99", "%65, %66"],
}

BADGE_W, BADGE_H, BADGE_GAP = 320, 62, 22
LABEL_W, COL_W, ROW_H, HEADER_H = 108, 340, 84, 36
PAD, TOP = 40, 108

w = PAD * 2 + LABEL_W + COL_W * len(COLS)
badges_top = TOP
badges_h = BADGE_H + 30
table_top = badges_top + badges_h + 30
h = table_top + HEADER_H + ROW_H * len(ROW_LABELS) + 76

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 徽标行:4 个关键数字
bw = (w - 2 * PAD - BADGE_GAP * (len(BADGES) - 1)) / len(BADGES)
for i, (num, note) in enumerate(BADGES):
    x = PAD + i * (bw + BADGE_GAP)
    L.append(f'<rect x="{x}" y="{badges_top}" width="{bw}" height="{BADGE_H}" rx="10" '
              'fill="#fef2f2" stroke="#dc2626" stroke-width="1.8"/>')
    L.append(f'<text x="{x+bw/2}" y="{badges_top+26}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" font-weight="bold" '
              f'fill="#b91c1c">{esc(num)}</text>')
    L.append(f'<text x="{x+bw/2}" y="{badges_top+46}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#7f1d1d">{esc(note)}</text>')

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{table_top}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{table_top+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

row_y = [table_top + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]
for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 9 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*17}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="#0f172a">'
                      f'{esc(line)}</text>')

for i in range(len(ROW_LABELS) + 1):
    y = table_top + HEADER_H + i * ROW_H
    L.append(f'<line x1="{PAD+LABEL_W}" y1="{y}" x2="{w-PAD}" y2="{y}" '
              'stroke="#e2e8f0" stroke-width="1"/>')

foot_y = table_top + HEADER_H + ROW_H * len(ROW_LABELS) + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#374151">本例 trip=6:谓词在 iter&gt;=4(=trip−2)起把 stage0 的取数关闭;'
          f'stage1、stage2 无需谓词(最后 stage 本就每拍执行)。</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig-m12-predicated-drain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
