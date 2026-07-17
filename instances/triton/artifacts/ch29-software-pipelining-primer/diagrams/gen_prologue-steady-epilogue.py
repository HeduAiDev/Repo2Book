#!/usr/bin/env python3
"""figure_id: prologue-steady-epilogue
claim: 展开器把循环重写成三段:prologue 用 numStages-1 段逐格灌满流水线
(在飞 1→numStages)、稳态 kernel loop 满并发运转、epilogue 用 numStages-1 段
逐格排空——num_stages=3 时 prologue/epilogue 各 2 段。
数据来自 PipelineExpander.cpp:287(emitPrologue for i in [0,maxStage))、
:305-307(stages[op]>i 跳过)与 explainer/traces/derive_schedule.out.json
schedule_by_num_stages[num_stages=3](max_stage=2, prologue_parts=2,
epilogue_parts=2)。state-table 模板:列=展开后的段,行=该段的关键量。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "PipelineExpander 把循环展开成三段(num_stages=3, maxStage=2)"
SUBTITLE = "prologue 逐段发射 stages[op]≤i 的 op(PipelineExpander.cpp:287,305-307);num_stages-1=2 段填流水、2 段排空"

COLS = ["prologue i=0", "prologue i=1", "稳态 kernel loop\n(每趟)", "epilogue i=0", "epilogue i=1"]
ROW_LABELS = ["发射条件", "本段新发射的 (迭代, stage)", "累计在飞迭代"]

CELLS = {
    "发射条件": [
        "stages[op] ≤ 0",
        "stages[op] ≤ 1",
        "stage 0..2 全发",
        "stages[op] ≥ 1(排空)",
        "stages[op] ≥ 2",
    ],
    "本段新发射的 (迭代, stage)": [
        "(迭代0, load)",
        "(迭代1, load)\n+(迭代0, wait)",
        "第k趟:(k+2,load)\n+(k+1,wait)+(k,dot)",
        "(末, wait)\n+(末-1, dot)",
        "(末, dot)",
    ],
    "累计在飞迭代": ["1", "2", "3(满)", "2", "1"],
}

# 语义色:prologue=蓝(填)、稳态=绿(满)、epilogue=橙(排)
STATUS = {"累计在飞迭代": ["build", "build", "full", "drain", "drain"]}
COLOR = {
    "build": ("#eff6ff", "#1d4ed8"),
    "full": ("#ecfdf5", "#047857"),
    "drain": ("#fff7ed", "#c2410c"),
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 210, 218, 84, 54, 118, 36
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 70
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

# 每列所属阶段的顶部色带(便于一眼分组:prologue/稳态/epilogue)
PHASE_OF_COL = ["prologue", "prologue", "steady", "epilogue", "epilogue"]
PHASE_BAND_COLOR = {"prologue": "#3b82f6", "steady": "#22c55e", "epilogue": "#f59e0b"}
PHASE_LABEL = {"prologue": "prologue(填流水)", "steady": "稳态", "epilogue": "epilogue(排空)"}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

# 阶段色带(表格上方,分组标注)
band_y = TOP - 30
i = 0
while i < len(COLS):
    phase = PHASE_OF_COL[i]
    j = i
    while j < len(COLS) and PHASE_OF_COL[j] == phase:
        j += 1
    x0 = col_x[i]
    x1 = col_x[j - 1] + (COL_W - 8)
    color = PHASE_BAND_COLOR[phase]
    L.append(f'<rect x="{x0}" y="{band_y}" width="{x1-x0}" height="20" rx="6" '
              f'fill="{color}" fill-opacity="0.18" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{(x0+x1)/2}" y="{band_y+14}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{color}">{esc(PHASE_LABEL[phase])}</text>')
    i = j

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    lines = name.split("\n")
    n = len(lines)
    y0 = TOP + (HEADER_H-6)/2 - (n-1)*8 + 4
    for k, line in enumerate(lines):
        L.append(f'<text x="{x+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="white" '
                  f'font-weight="bold">{esc(line)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    lbl_lines = row.split("\n")
    ln = len(lbl_lines)
    ly0 = ry + ROW_H/2 - (ln-1)*8 + 4
    for k, line in enumerate(lbl_lines):
        L.append(f'<text x="{PAD+LABEL_W-16}" y="{ly0+k*16}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="#374151">{esc(line)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="6" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        fsize = 15 if status else 12
        y0 = ry + ROW_H/2 - (n-1)*9 + 5
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*17}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{fsize}" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')
    # 行分隔线
    if i > 0:
        L.append(f'<line x1="{PAD+8}" y1="{ry}" x2="{col_x[-1]+COL_W-8}" y2="{ry}" '
                  'stroke="#e2e8f0"/>')

foot_y = row_y[-1] + ROW_H + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("蓝=在飞迭代数递增(填流水线)、绿=满并发稳态、橙=在飞迭代数递减(排空)——prologue 与 epilogue 段数都恰是 num_stages-1=2")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("num_stages=5 时 prologue/epilogue 各变为 4 段——深度越大,填/排的固定开销越大")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("prologue-steady-epilogue.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
