"""04-stream-interval-throttle: stream_interval=4 send-decision table (DELTA)."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 1000, 470
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')


def text(x, y, txt, fs=13, col="#0f172a", anchor="middle", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(txt)}</text>'
    )


def cell(x, y, cw, ch, fill, stroke="#cbd5e1"):
    L.append(
        f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


text(w / 2, 32, "stream_interval = 4 的发送决策（DELTA 模式）", 18, "#0f172a", "middle", "bold")
text(w / 2, 54,
     "发送条件：完成 or 首 token(sent_tokens_offset==0) or 攒够 4 个新 token",
     13, "#94a3b8")

# Table
rows = [
    ("到达 token#", "0", "1", "2", "3", "4", "5", "6", "7(完成)"),
    ("已去 token 数", "1", "2", "3", "4", "5", "6", "7", "8"),
    ("距上次已发", "—", "1", "2", "3", "4", "1", "2", "—"),
    ("决策", "发", "抑", "抑", "抑", "发", "抑", "抑", "强发"),
    ("sent_tokens_offset", "1", "1", "1", "1", "5", "5", "5", "8"),
    ("本次 DELTA 增量", "[t0]", "—", "—", "—", "[t1..t4]", "—", "—", "[t5..t7]"),
]
ncol = len(rows[0])
x0 = 40
y0 = 80
label_w = 180
col_w = (w - x0 * 2 - label_w) / (ncol - 1)
row_h = 50

for ri, row in enumerate(rows):
    ry = y0 + ri * row_h
    for ci, val in enumerate(row):
        if ci == 0:
            cx = x0
            cw = label_w
            tx = x0 + 10
            anchor = "start"
            fill = "#f1f5f9"
            mono = False
            col = "#334155"
            weight = "bold"
        else:
            cx = x0 + label_w + (ci - 1) * col_w
            cw = col_w
            tx = cx + cw / 2
            anchor = "middle"
            mono = True
            col = "#0f172a"
            weight = "normal"
            fill = "white"
            if ri == 0:
                fill = "#e2e8f0"
                weight = "bold"
            if ri == 3:  # 决策 row
                if val == "发":
                    fill = "#dcfce7"
                    col = "#15803d"
                    weight = "bold"
                elif val == "强发":
                    fill = "#fef3c7"
                    col = "#b45309"
                    weight = "bold"
                elif val == "抑":
                    fill = "#fee2e2"
                    col = "#b91c1c"
            if ri == 5 and val != "—":
                fill = "#eff6ff"
                col = "#1d4ed8"
                weight = "bold"
        cell(cx, ry, cw, row_h, fill)
        text(tx, ry + row_h / 2 + 5, val, 13, col, anchor, weight, mono)

# legend
ly = y0 + len(rows) * row_h + 36
text(x0, ly, "发 = 满足条件，造 RequestOutput 入队", 13, "#15803d", "start", "bold")
text(x0 + 330, ly, "抑 = 返回 None，不发", 13, "#b91c1c", "start", "bold")
text(x0 + 560, ly, "强发 = finished 强制发余量", 13, "#b45309", "start", "bold")
text(x0, ly + 26,
     "sent_tokens_offset 只在「实际发送」时推进 → DELTA 增量首尾相接，无重叠无丢失。",
     13, "#475569", "start")

L.append('</svg>')
open("04-stream-interval-throttle.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote 04-stream-interval-throttle.svg")
