#!/usr/bin/env python3
"""state-table 模板：两侧 pipe 缺省配对表——只有都不给才走缺省，单边给直接 TypeError。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "缺省 pipe 配对：两侧都留空才生效，单边给 = TypeError"
SUBTITLE = '触发缺省的条件：sender_pipe is None and receiver_pipe is None（core.py:L213）；Python 层可选 pipe 档数 = 8'
COLS = ["cube 发（不给 pipe）", "vector 发（不给 pipe）", "只给 sender_pipe", "两侧都显式给"]
ROW_LABELS = ["sender_pipe", "receiver_pipe", "结果"]
CELLS = {
    "sender_pipe": ["PIPE_FIX（缺省）", "PIPE_MTE3（缺省）", "PIPE_V（显式）", "PIPE_V（显式）"],
    "receiver_pipe": ["PIPE_MTE2（缺省）", "PIPE_MTE2（缺省）", "None（留空）", "PIPE_MTE1（显式）"],
    "结果": ["OK", "OK", "TypeError", "OK"],
}
STATUS = {"结果": ["ok", "ok", "err", "ok"]}
COLOR = {"ok": ("#ecfdf5", "#047857"), "err": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, PAD, TOP, HEADER_H, ROW_H = 130, 250, 34, 106, 42, 48
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 74
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
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        text = CELLS[row][j]
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        else:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="{text_fill}" '
                  f'{weight_attr}>{esc(text)}</text>')

fy1 = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + 26
fy2 = fy1 + 20
L.append(f'<text x="{PAD}" y="{fy1}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("收方 pipe 在两种缺省配对里恒为 PIPE_MTE2（core.py:L215-L220）——最容易踩的坑不是「另一边用缺省」，而是单边给等于把另一边留成 None。")}</text>')
L.append(f'<text x="{PAD}" y="{fy2}" font-family="sans-serif" font-size="11" '
          f'fill="#c2410c">'
          f'{esc("这些 pipe 各自对应哪条硬件队列、为何收方总是 MTE2，源码未给出依据——不臆测，见 open_question。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m8-default-pipe-pairing.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
