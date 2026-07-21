#!/usr/bin/env python3
"""state-table 模板：GetCore 的 2×2 落核翻转表 + 非法 sender 兜底 + Python 层默认 pipe 侧注。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "GetCore：同一个 sender，set 落发方核、wait 落收方核"
SUBTITLE = "third_party/ascend/ascend_ir.cc:L93-L113——纯函数，输入只有 (opName, sender)，两层二分支、无循环"
COLS = ["sender = \"cube\"", "sender = \"vector\""]
ROW_LABELS = ["sync_block_set", "sync_block_wait"]
CELLS = [
    ["TCoreType::CUBE", "TCoreType::VECTOR"],
    ["TCoreType::VECTOR", "TCoreType::CUBE"],
]
CELL_COLOR = [
    [("#dbeafe", "#1d4ed8"), ("#ede9fe", "#6d28d9")],
    [("#ede9fe", "#6d28d9"), ("#dbeafe", "#1d4ed8")],
]

LABEL_W, COL_W, PAD, TOP, HEADER_H, ROW_H = 190, 260, 34, 106, 42, 64
w = PAD * 2 + LABEL_W + COL_W * len(COLS) + 40
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 300
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS)):
        cx = col_x[j]
        fill, stroke = CELL_COLOR[i][j]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" fill="{stroke}" '
                  f'font-weight="bold">{esc(CELLS[i][j])}</text>')

# non-legal sender note box
note_y = row_y[-1] + ROW_H + 30
note_w = LABEL_W + COL_W * len(COLS)
L.append(f'<rect x="{PAD}" y="{note_y}" width="{note_w}" height="52" rx="8" '
          'fill="#fff7ed" stroke="#c2410c" stroke-width="1.5"/>')
aicore_note = 'sender = "aicore"（非 cube/vector）→ std::runtime_error'
L.append(f'<text x="{PAD+16}" y="{note_y+22}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#7c2d12">'
          f'{esc(aicore_note)}</text>')
L.append(f'<text x="{PAD+16}" y="{note_y+40}" font-family="sans-serif" font-size="11" '
          f'fill="#9a3412">'
          f'{esc("C++ 侧再兜一次底：只支持 cube 或 vector 作为 sender（ascend_ir.cc:L103-L105）")}</text>')

# side info: python-layer default pipe pairing (context, computed in M7/M8)
side_y = note_y + 52 + 26
L.append(f'<rect x="{PAD}" y="{side_y}" width="{note_w}" height="66" rx="8" '
          'fill="#f8fafc" stroke="#64748b" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+16}" y="{side_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#334155">'
          f'{esc("对照：Python 层默认 pipe（不显式指定时）")}</text>')
L.append(f'<text x="{PAD+16}" y="{side_y+42}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("cube 发 → PIPE_FIX / PIPE_MTE2")}</text>')
L.append(f'<text x="{PAD+16}" y="{side_y+58}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("vector 发 → PIPE_MTE3 / PIPE_MTE2")}</text>')

foot_y = side_y + 66 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("翻转表只有 2×2=4 格，却支撑起整个核间同步的落核规则：set 落 sender 核、wait 落另一核，两者恒互补。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m9-getcore-flip.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
