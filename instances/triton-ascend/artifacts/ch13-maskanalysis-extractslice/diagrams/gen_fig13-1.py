#!/usr/bin/env python3
"""fig13-1 layout 模板:MaskState 五字段 x 三形态(标量/裸 range/矩形掩码)。
三卡片并排,每卡片列 5 个字段行(填的高亮、空的灰淡),底部标 isMask()与产出者。
数据取自 explainer m1.worked_example.table(与正文/dossier 同源)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

FIELDS = ["start", "end", "offsets", "dims", "scalar"]

CARDS = [
    {
        "title": "标量态",
        "values": {"start": "—", "end": "—", "offsets": "—", "dims": "—", "scalar": "10"},
        "is_mask": False,
        "producer": "parseConstant / parseIntScalar",
    },
    {
        "title": "裸 range 态",
        "values": {"start": "0", "end": "16", "offsets": "—", "dims": "—", "scalar": "—"},
        "is_mask": False,
        "producer": "parseMakeRange 叶子",
    },
    {
        "title": "矩形掩码态",
        "values": {"start": "—", "end": "—", "offsets": "[0]", "dims": "[10]", "scalar": "—"},
        "is_mask": True,
        "producer": "parseCmp 熔合后",
    },
]

CARD_W, CARD_H, GAP, PAD, TOP = 250, 250, 40, 50, 100
ROW_H = 28
FILLED_FILL, FILLED_STROKE = "#dbeafe", "#2563eb"
EMPTY_FILL, EMPTY_STROKE = "#f8fafc", "#cbd5e1"
MASK_TRUE_STROKE = "#16a34a"
MASK_FALSE_STROKE = "#94a3b8"

w = PAD * 2 + CARD_W * 3 + GAP * 2
h = TOP + CARD_H + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{40}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17" font-weight="bold" fill="#0f172a">'
     f'{esc("MaskState:五字段,三选一的载体")}</text>',
     f'<text x="{w/2}" y="{64}" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#475569">'
     f'{esc("字段数 = 5(start / end / offsets / dims / scalar,MaskAnalysis.h:L52-L56)")}</text>']

for i, card in enumerate(CARDS):
    x = PAD + i * (CARD_W + GAP)
    stroke = MASK_TRUE_STROKE if card["is_mask"] else MASK_FALSE_STROKE
    sw = 3 if card["is_mask"] else 1.5
    L.append(f'<rect x="{x}" y="{TOP}" width="{CARD_W}" height="{CARD_H}" rx="10" '
              f'fill="white" stroke="{stroke}" stroke-width="{sw}"/>')
    # 注:环境字体(Droid Sans Fallback)对"量"字加粗会糊成实心块,标题避开 font-weight=bold,
    # 改用更大字号 + 颜色区分承担强调。
    L.append(f'<text x="{x+CARD_W/2}" y="{TOP+27}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="16" fill="#0f172a">{esc(card["title"])}</text>')
    ry = TOP + 46
    for f in FIELDS:
        v = card["values"][f]
        filled = v != "—"
        fill = FILLED_FILL if filled else EMPTY_FILL
        bstroke = FILLED_STROKE if filled else EMPTY_STROKE
        L.append(f'<rect x="{x+16}" y="{ry}" width="{CARD_W-32}" height="{ROW_H-6}" rx="5" '
                  f'fill="{fill}" stroke="{bstroke}"/>')
        L.append(f'<text x="{x+26}" y="{ry+ROW_H/2-2}" font-family="sans-serif" font-size="12" '
                  f'fill="#334155">{esc(f)}</text>')
        L.append(f'<text x="{x+CARD_W-26}" y="{ry+ROW_H/2-2}" text-anchor="end" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{"#1e3a8a" if filled else "#94a3b8"}">{esc(v)}</text>')
        ry += ROW_H
    ry += 6
    mask_txt = "isMask() = true" if card["is_mask"] else "isMask() = false"
    L.append(f'<text x="{x+CARD_W/2}" y="{ry+10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{stroke}">{esc(mask_txt)}</text>')
    L.append(f'<text x="{x+CARD_W/2}" y="{ry+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#64748b">{esc(card["producer"])}</text>')

foot_y = TOP + CARD_H + 55
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="monospace" '
          f'font-size="12" fill="#0f172a">'
          f'{esc("isMask() ⇔ !start && !end && !scalar && dims!=0 && offsets!=0")}</text>')
L.append(f'<text x="{w/2}" y="{foot_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#334155">'
          f'{esc("只有矩形态满足 isMask()——它是唯一能发射切片的合法终态,标量与 range 都是解析中间态。")}</text>')
L.append('</svg>')

out = Path(__file__).with_name("fig13-1.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
