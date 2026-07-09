#!/usr/bin/env python3
"""paper-fig-eagle2-fig4: 论文精髓图重绘。
重绘自 arXiv:2406.16858 Fig.4（§1 Introduction）——用具体例子对比静态树与动态树：
query="10+2"（下一 token 难猜,是"="还是"+"之外的符号)时两法都长 2 个候选；
query="10+2="（下一 token 几乎确定是"1"）时 EAGLE 仍固定长 2 个候选（浪费),
EAGLE-2 则只长 1 个候选（把省下的算力挪去别处)。布局与原图一致(两法×两个 query 的
2x2 网格,每格一个 query 框 + 若干候选圆)，配色/字体套本书视觉语言，文字译中。
全部坐标由循环计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "EAGLE vs EAGLE-2：同一个「够不够确定」，树该不该跟着变形"
SUB = "重绘自 arXiv:2406.16858 Fig.4：query=“10+2=”时下一 token 几乎唯一确定为“1”，EAGLE 仍固定长出 2 个候选（浪费），EAGLE-2 只长 1 个"

# 每个方法一列,每列两个 query 场景(行)
METHODS = ["EAGLE（静态树）", "EAGLE-2（动态树）"]
ROWS = [
    ("query = “10+2”（难猜，下一 token 不确定）", {
        "EAGLE（静态树）": ["=", "+"],
        "EAGLE-2（动态树）": ["=", "+"],
    }),
    ("query = “10+2=”（好猜，下一 token 几乎唯一确定）", {
        "EAGLE（静态树）": ["1", "3"],
        "EAGLE-2（动态树）": ["1"],
    }),
]

COL_W, ROW_H = 420, 260
QBOX_W, QBOX_H = 300, 46
CAND_R = 24
PAD, TOP = 44, 150

W = PAD * 2 + COL_W * len(METHODS)
H = TOP + ROW_H * len(ROWS) + 50

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="62" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUB)}</text>']

# 列标题
for ci, method in enumerate(METHODS):
    cx = PAD + ci * COL_W + COL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-14}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="15" font-weight="bold" fill="#1e40af">{esc(method)}</text>')
if len(METHODS) == 2:
    mid_x = PAD + COL_W
    L.append(f'<line x1="{mid_x}" y1="{TOP-40}" x2="{mid_x}" y2="{TOP+ROW_H*len(ROWS)-20}" '
              'stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4,4"/>')

for ri, (row_label, cand_map) in enumerate(ROWS):
    ry0 = TOP + ri * ROW_H
    L.append(f'<text x="{PAD}" y="{ry0+8}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#334155">{esc(row_label)}</text>')
    qy = ry0 + 34
    for ci, method in enumerate(METHODS):
        cx = PAD + ci * COL_W + COL_W / 2
        # query box
        L.append(f'<rect x="{cx-QBOX_W/2:.1f}" y="{qy:.1f}" width="{QBOX_W}" height="{QBOX_H}" rx="8" '
                  f'fill="#dbeafe" stroke="#1e40af" stroke-width="1.5"/>')
        qtext = row_label.split("“")[1].split("”")[0]
        L.append(f'<text x="{cx:.1f}" y="{qy+QBOX_H/2+5:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="15" font-weight="bold" fill="#1e3a5f">{esc(qtext)}</text>')
        cands = cand_map[method]
        n = len(cands)
        cy = qy + QBOX_H + 90
        spread = 90
        xs_c = [cx] if n == 1 else [cx - spread/2, cx + spread/2]
        for cxi, tok in zip(xs_c, cands):
            L.append(f'<line x1="{cx:.1f}" y1="{qy+QBOX_H:.1f}" x2="{cxi:.1f}" y2="{cy-CAND_R:.1f}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
            L.append(f'<circle cx="{cxi:.1f}" cy="{cy:.1f}" r="{CAND_R}" fill="#fde8d7" '
                      'stroke="#c2410c" stroke-width="1.5"/>')
            L.append(f'<text x="{cxi:.1f}" y="{cy+6:.1f}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="16" font-weight="bold" fill="#c2410c">{esc(tok)}</text>')
        note = f"长 {n} 个候选" + ("（浪费）" if (ri == 1 and method == METHODS[0]) else
                                    "（刚好够）" if (ri == 1 and method == METHODS[1]) else "")
        note_color = "#b91c1c" if "浪费" in note else ("#15803d" if "刚好够" in note else "#64748b")
        L.append(f'<text x="{cx:.1f}" y="{cy+CAND_R+26:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="12" font-weight="bold" fill="{note_color}">{esc(note)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-eagle2-fig4.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
