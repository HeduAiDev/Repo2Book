"""03-delta-merge-mailbox: DELTA merge timeline when producer outruns consumer."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 980, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')


def text(x, y, txt, fs=14, col="#0f172a", anchor="middle", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(txt)}</text>'
    )


def box(x, y, bw, bh, fill, stroke, lines, fs=13, rx=6, mono=True):
    L.append(
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    n = len(lines)
    total = n * (fs + 3)
    sy = y + bh / 2 - total / 2 + fs
    for i, ln in enumerate(lines):
        text(x + bw / 2, sy + i * (fs + 3), ln,
             fs if i == 0 else fs - 1,
             "#0f172a" if i == 0 else "#475569",
             "middle", "bold" if i == 0 else "normal", mono)


def arrow(x1, y1, x2, y2, col="#475569", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
        f'stroke-width="2"{d} marker-end="url(#a)"/>'
    )


text(w / 2, 30, "RequestOutputCollector：消费者慢时 DELTA 归并", 18, "#0f172a", "middle", "bold")
text(w / 2, 52, "aggregate = (output_kind == DELTA)；put 调 RequestOutput.add 拼接", 13, "#94a3b8")

# Three lanes
lane_x = 60
prod_y = 110
slot_y = 250
cons_y = 430
text(lane_x, prod_y - 14, "生产者 output_handler", 14, "#d97706", "start", "bold")
text(lane_x, slot_y - 22, "邮箱单槽 self.output", 14, "#a21caf", "start", "bold")
text(lane_x, cons_y - 14, "消费者 generate()（落后）", 14, "#2563eb", "start", "bold")

# producer puts
cols_x = [260, 470, 680]
put_labels = [
    ['put(d1)', 'text="你"', "token=[101]"],
    ['put(d2)', 'text="好"', "token=[102]"],
    ['put(d3)', 'text="吗"', "token=[103]"],
]
for i, px in enumerate(cols_x):
    box(px - 70, prod_y, 140, 56, "#fef3c7", "#d97706", put_labels[i])

# slot states
slot_labels = [
    ['槽 = d1', 'text="你"', "set(Event)"],
    ['add(d1,d2)', 'text="你好"', "拼接"],
    ['add(..,d3)', 'text="你好吗"', "拼接"],
]
slot_fill = ["#dcfce7", "#fae8ff", "#fae8ff"]
slot_stroke = ["#16a34a", "#a21caf", "#a21caf"]
for i, px in enumerate(cols_x):
    box(px - 75, slot_y, 150, 64, slot_fill[i], slot_stroke[i], slot_labels[i])
    arrow(px, prod_y + 56, px, slot_y)

# consumer get
gx = 820
box(gx - 75, cons_y, 170, 70, "#dbeafe", "#2563eb",
    ['get() 取走', 'text="你好吗"', "token=[101,102,103]", "clear(Event)"])
# arrow from last slot to consumer
arrow(cols_x[-1] + 75, slot_y + 32, gx - 75, cons_y + 20)
text((cols_x[-1] + 75 + gx - 75) / 2, slot_y + 64,
     "消费者一次 get 拿到合并结果，无重叠无丢失", 12, "#16a34a", "middle", "bold")

# arrows between slot states (mutation in place)
for i in range(len(cols_x) - 1):
    arrow(cols_x[i] + 75, slot_y + 32, cols_x[i + 1] - 75, slot_y + 32, "#a21caf", "5 3")

# contrast box: CUMULATIVE
text(lane_x, 510, "对比 CUMULATIVE 模式：aggregate=False → put 整体替换 self.output，只保留最后一份全量",
     13, "#64748b", "start")
text(lane_x, 532, "内存恒为「每请求一槽」O(1)，归并是背压安全阀：消费慢也不会无界堆积",
     13, "#64748b", "start")

L.append('</svg>')
open("03-delta-merge-mailbox.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote 03-delta-merge-mailbox.svg")
