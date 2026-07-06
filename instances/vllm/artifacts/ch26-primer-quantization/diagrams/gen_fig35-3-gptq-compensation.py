#!/usr/bin/env python3
"""fig35-3-gptq-compensation: 二阶补偿把每列的量化误差预先摊到未量化列上，
重构误差比朴素 RTN 降 77%，且与 lazy-batch 分块大小无关。
before-after 两态：左=RTN 无补偿(整行误差)，右=GPTQ 逐列量化+级联补偿(4 步)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "GPTQ 二阶补偿：量化第 j 列的误差立刻摊到第 j+1..3 列，整行重构误差降 77%"
SUBTITLE = "W_row = [0.1, 0.95, -0.4, 0.55]，3-bit 网格，逐列从左到右量化"

# left panel: RTN (no compensation) — single summary box
LEFT_TITLE = "RTN（各列独立就近取整，无补偿）"
LEFT_BOX = ["W_row = [0.1, 0.95, -0.4, 0.55]", "逐列各自取整，互不影响", "reconstruction error = 0.0253"]

# right panel: GPTQ cascade, 4 steps
STEPS = [
    {"col": "col 0", "w": "0.1", "q": "0.1929", "err": "-0.0267",
     "rest_before": "[0.95, -0.4, 0.55]", "rest_after": "[0.9172, -0.3525, 0.4069]"},
    {"col": "col 1", "w": "0.9172", "q": "0.9643", "err": "-0.0553",
     "rest_before": "[-0.3525, 0.4069]", "rest_after": "[-0.3619, 0.3703]"},
    {"col": "col 2", "w": "-0.3619", "q": "-0.3857", "err": "0.0250",
     "rest_before": "[0.3703]", "rest_after": "[0.3945]"},
    {"col": "col 3", "w": "0.3945", "q": "0.3857", "err": "0.0094",
     "rest_before": "(块末，无剩余列)", "rest_after": ""},
]

PAD = 40
TOP = 100
LEFT_W = 280
RIGHT_X = PAD + LEFT_W + 60
STEP_W = 620
STEP_H = 92
STEP_GAP = 14
w = RIGHT_X + STEP_W + PAD
FOOT_N = 3
right_ry_end = (TOP + 24) + len(STEPS) * (STEP_H + STEP_GAP)  # after loop, right column cursor
final_y = right_ry_end + 4
foot_y = final_y + 66
h = foot_y + (FOOT_N - 1) * 18 + 26

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# left panel
L.append(f'<text x="{PAD}" y="{TOP}" font-family="sans-serif" font-size="13.5" '
          f'fill="#0f172a">{esc(LEFT_TITLE)}</text>')
by = TOP + 24
for i, line in enumerate(LEFT_BOX):
    is_last = (i == len(LEFT_BOX) - 1)
    box_fill = "#fee2e2" if is_last else "#f1f5f9"
    box_stroke = "#b91c1c" if is_last else "#94a3b8"
    L.append(f'<rect x="{PAD}" y="{by}" width="{LEFT_W}" height="52" rx="6" '
              f'fill="{box_fill}" stroke="{box_stroke}" stroke-width="{2 if is_last else 1}"/>')
    text_fill = "#b91c1c" if is_last else "#334155"
    L.append(f'<text x="{PAD+LEFT_W/2}" y="{by+30}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="{text_fill}">{esc(line)}</text>')
    if i < len(LEFT_BOX) - 1:
        L.append(f'<line x1="{PAD+LEFT_W/2}" y1="{by+52}" x2="{PAD+LEFT_W/2}" y2="{by+70}" '
                  'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    by += 70

# right panel
L.append(f'<text x="{RIGHT_X}" y="{TOP}" font-family="sans-serif" font-size="13.5" '
          f'fill="#0f172a">GPTQ（逐列量化 + 即时补偿到未量化列）</text>')
ry = TOP + 24
for i, s in enumerate(STEPS):
    is_last = (i == len(STEPS) - 1)
    box_fill = "#dbeafe" if not is_last else "#ecfdf5"
    box_stroke = "#1d4ed8" if not is_last else "#047857"
    L.append(f'<rect x="{RIGHT_X}" y="{ry}" width="{STEP_W}" height="{STEP_H}" rx="6" '
              f'fill="{box_fill}" stroke="{box_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{RIGHT_X+16}" y="{ry+20}" font-family="sans-serif" font-size="12.5" '
              f'fill="#0f172a">{esc(s["col"])}: w={esc(s["w"])} -&gt; q={esc(s["q"])}  '
              f'(err_scaled={esc(s["err"])})</text>')
    L.append(f'<text x="{RIGHT_X+16}" y="{ry+42}" font-family="sans-serif" font-size="11.5" '
              f'fill="#475569">剩余列补偿前: {esc(s["rest_before"])}</text>')
    if s["rest_after"]:
        L.append(f'<text x="{RIGHT_X+16}" y="{ry+62}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#b91c1c">补偿后 -&gt; {esc(s["rest_after"])}</text>')
    else:
        L.append(f'<text x="{RIGHT_X+16}" y="{ry+62}" font-family="sans-serif" font-size="11.5" '
                  f'fill="#475569">{esc(s["rest_before"]) if False else ""}</text>')
    if i < len(STEPS) - 1:
        L.append(f'<line x1="{RIGHT_X+STEP_W/2}" y1="{ry+STEP_H}" x2="{RIGHT_X+STEP_W/2}" '
                  f'y2="{ry+STEP_H+STEP_GAP-2}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    ry += STEP_H + STEP_GAP

final_y = ry + 4
L.append(f'<rect x="{RIGHT_X}" y="{final_y}" width="{STEP_W}" height="40" rx="6" '
          f'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
L.append(f'<text x="{RIGHT_X+STEP_W/2}" y="{final_y+26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" fill="#047857">'
          f'reconstruction error = 0.0057（比 RTN 的 0.0253 降 77.33%）</text>')

foot_y = final_y + 66
foot_lines = [
    "col 0 的误差 err_scaled = -0.0267 让 col 1-3 从 [0.95, -0.4, 0.55] 偏移到 [0.9172, -0.3525, 0.4069]；",
    "此后每一列都带着历史补偿量进入量化，直到整行量化完毕。",
    "lazy-batch blocksize = 1 / 2 / 4 结果完全相同（均为 0.0057）——分块只是效率重排，不是另一套算法。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*18}" font-family="sans-serif" font-size="12" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig35-3-gptq-compensation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
