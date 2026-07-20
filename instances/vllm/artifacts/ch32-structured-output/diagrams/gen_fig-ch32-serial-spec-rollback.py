#!/usr/bin/env python3
"""fig-ch32-serial-spec-rollback: 一个投机请求的 1+k 行掩码是「推进-填行-再推进-...-整体回滚」
的产物,回滚让本步对语法状态的净影响为零。
template: state-table"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

PAD = 50
LABEL_W = 130
COLW2 = 175
COLS = ["行 0(位置 0)", "行 1(位置 1)", "行 2(位置 2)", "循环结束"]
PANEL_GAP = 40
panel_full_w = LABEL_W + COLW2 * len(COLS)
W = PAD * 2 + panel_full_w * 2 + PANEL_GAP
H = 460  # tightened to content extent

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("投机请求的 1+k 行掩码:推进-填行...-整体回滚,净位移归零")}</text>')
L.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#64748b">{esc("语法:位置0允许{5,7};位置1允许{9};位置2允许{11,13}(k=2,行数/请求=1+k=3)")}</text>')

TOP = 90
HEADER_H = 32
ROW_H = 46

PANELS = [
    {
        "title": "rA  草稿 [5, 9](全合法)",
        "color": "#2563eb",
        "rows": [
            ("本位置 token", ["5", "9", "-1(补齐位)", "—"]),
            ("填行前 FSM 位置", ["0", "1", "2", "2"]),
            ("写入内容(允许集)", ["{5,7}", "{9}", "{11,13}", "rollback(2)"]),
            ("填行后 FSM 位置", ["1", "2", "2", "0"]),
        ],
    },
    {
        "title": "rB  草稿 [5, -1](第 2 位被语法过滤补 -1)",
        "color": "#059669",
        "rows": [
            ("本位置 token", ["5", "-1(补齐位)", "-1", "—"]),
            ("填行前 FSM 位置", ["0", "1", "1", "1"]),
            ("写入内容(允许集)", ["{5,7}", "{9}", "整行 -1(全允许)", "rollback(1)"]),
            ("填行后 FSM 位置", ["1", "1", "1", "0"]),
        ],
    },
]

for p, panel in enumerate(PANELS):
    px = PAD + p * (panel_full_w + PANEL_GAP)
    py = TOP
    L.append(f'<text x="{px}" y="{py-14}" font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="{panel["color"]}">{esc(panel["title"])}</text>')
    for j, cname in enumerate(COLS):
        cx = px + LABEL_W + j * COLW2
        is_rollback = (j == len(COLS) - 1)
        fill = "#fde68a" if is_rollback else panel["color"]
        L.append(f'<rect x="{cx}" y="{py}" width="{COLW2-6}" height="{HEADER_H}" rx="4" '
                  f'fill="{fill}"/>')
        tcol = "#78350f" if is_rollback else "white"
        L.append(f'<text x="{cx+(COLW2-6)/2}" y="{py+HEADER_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="{tcol}">{esc(cname)}</text>')
    for i, (rlabel, vals) in enumerate(panel["rows"]):
        ry = py + HEADER_H + i * ROW_H
        L.append(f'<text x="{px+LABEL_W-10}" y="{ry+ROW_H/2+4}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                  f'fill="#374151">{esc(rlabel)}</text>')
        is_fsm_row = "FSM" in rlabel
        for j, val in enumerate(vals):
            cx = px + LABEL_W + j * COLW2
            is_rollback_col = (j == len(COLS) - 1)
            highlight = is_fsm_row and is_rollback_col
            fill = "#dcfce7" if highlight else ("#f8fafc" if i % 2 == 0 else "white")
            stroke = "#16a34a" if highlight else "#e2e8f0"
            L.append(f'<rect x="{cx}" y="{ry}" width="{COLW2-6}" height="{ROW_H-4}" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if highlight else 1}"/>')
            fs = 10.5 if len(val) > 10 else 12
            tw = "bold" if highlight else "normal"
            tc = "#166534" if highlight else "#1e293b"
            L.append(f'<text x="{cx+(COLW2-6)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{fs}" font-weight="{tw}" '
                      f'fill="{tc}">{esc(val)}</text>')

FOOT_Y = TOP + HEADER_H + len(PANELS[0]["rows"]) * ROW_H + 40
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="70" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+26}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("rA 推进 2 步后 rollback(2),rB 推进 1 步后 rollback(1)——两者填行后 FSM 位置都精确回到本步开始时的 0")}</text>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+48}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("缓冲与裁剪:预分配 max_num_seqs x (1+k) = 4x3 = 12 行,本步两请求共用 6 行,返回前裁到 6")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-serial-spec-rollback.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
