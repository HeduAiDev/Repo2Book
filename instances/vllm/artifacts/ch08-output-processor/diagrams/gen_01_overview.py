"""01-stage3-overview: single-producer multi-consumer dataflow."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 1180, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')


def box(x, y, bw, bh, fill, stroke, lines, fs=15, tcol="#0f172a", bold0=True, rx=8):
    L.append(
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    n = len(lines)
    total = n * (fs + 4)
    sy = y + bh / 2 - total / 2 + fs
    for i, ln in enumerate(lines):
        weight = "bold" if (i == 0 and bold0) else "normal"
        col = tcol if i == 0 else "#475569"
        size = fs if i == 0 else fs - 2
        L.append(
            f'<text x="{x + bw / 2}" y="{sy + i * (fs + 4)}" font-family="sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{col}" '
            f'text-anchor="middle">{esc(ln)}</text>'
        )


def arrow(x1, y1, x2, y2, col="#475569", mk="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
        f'stroke-width="2"{d} marker-end="url(#{mk})"/>'
    )


def label(x, y, txt, col="#475569", fs=13, anchor="middle", weight="normal"):
    L.append(
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(txt)}</text>'
    )


# Title
label(w / 2, 30, "Stage 3：单生产者 — 多消费者扇出", "#0f172a", 18, "middle", "bold")

# Column 1: EngineCore
ec_x, ec_y, ec_w, ec_h = 30, 270, 140, 100
box(ec_x, ec_y, ec_w, ec_h, "#e0e7ff", "#6366f1",
    ["EngineCore", "(独立进程)", "一步产出整批"])

# Column 2: output_handler (single producer)
oh_x, oh_y, oh_w, oh_h = 250, 250, 190, 140
box(oh_x, oh_y, oh_w, oh_h, "#fef3c7", "#d97706",
    ["output_handler", "单个背景 asyncio 任务", "(唯一生产者)", "按 chunk_size 分块",
     "调 process_outputs"])

arrow(ec_x + ec_w, ec_y + ec_h / 2, oh_x, oh_y + oh_h / 2)
label((ec_x + ec_w + oh_x) / 2, ec_y + ec_h / 2 - 12, "get_output_async()", "#475569", 12)
label((ec_x + ec_w + oh_x) / 2, ec_y + ec_h / 2 + 16,
      "list[EngineCoreOutput]", "#94a3b8", 11)

# Column 3: per-request RequestState + Collector mailbox  (N rows)
rs_x, rs_w = 510, 150
mb_x, mb_w = 700, 130
gen_x, gen_w = 880, 150
client_x, client_w = 1070, 90

n = 3
row_h = 90
gap = 30
start_y = 130
labels = [
    ("RequestState 1", "去 token / logprobs"),
    ("RequestState 2", "去 token / logprobs"),
    ("RequestState N", "去 token / logprobs"),
]
mb_labels = ["Collector 1", "Collector 2", "Collector N"]
gen_labels = [("generate() 1", "get_nowait/get"),
              ("generate() 2", "get_nowait/get"),
              ("generate() N", "get_nowait/get")]

for i in range(n):
    y = start_y + i * (row_h + gap)
    cy = y + row_h / 2
    box(rs_x, y, rs_w, row_h, "#dcfce7", "#16a34a", list(labels[i]))
    box(mb_x, y, mb_w, row_h, "#fae8ff", "#a21caf",
        [mb_labels[i], "单槽邮箱", "asyncio.Event"])
    box(gen_x, y, gen_w, row_h, "#dbeafe", "#2563eb", list(gen_labels[i]))
    box(client_x, y, client_w, row_h, "#f1f5f9", "#64748b", ["客户端", f"流 {i + 1}"])
    # output_handler routes to each RequestState (by req_id)
    arrow(oh_x + oh_w, oh_y + oh_h / 2 + (i - 1) * 8, rs_x, cy)
    # RequestState put into its collector
    arrow(rs_x + rs_w, cy, mb_x, cy)
    # generate pulls from collector
    arrow(mb_x + mb_w, cy, gen_x, cy)
    # yield to client
    arrow(gen_x + gen_w, cy, client_x, cy)

label((rs_x + rs_w + mb_x) / 2, start_y - 8, "put()", "#a21caf", 12, "middle", "bold")
label((mb_x + mb_w + gen_x) / 2, start_y - 8, "get()", "#2563eb", 12, "middle", "bold")
label((gen_x + gen_w + client_x) / 2, start_y - 8, "yield", "#475569", 12)
label(rs_x + rs_w / 2, start_y - 8, "按 req_id 路由", "#16a34a", 12, "middle", "bold")

# reqs_to_abort reverse arrow: output_handler -> EngineCore
ab_y = ec_y + ec_h + 70
L.append(
    f'<path d="M {oh_x + 30} {oh_y + oh_h} '
    f'C {oh_x + 30} {ab_y}, {ec_x + ec_w / 2} {ab_y}, '
    f'{ec_x + ec_w / 2} {ec_y + ec_h}" fill="none" stroke="#dc2626" '
    f'stroke-width="2" stroke-dasharray="6 4" marker-end="url(#r)"/>'
)
label((oh_x + ec_x + ec_w / 2) / 2 + 10, ab_y + 18,
      "reqs_to_abort：停止串触发 → 反向 abort EngineCore", "#dc2626", 12)

# sleep(0) annotation
label(oh_x + oh_w / 2, oh_y + oh_h + 24,
      "chunk 间 await asyncio.sleep(0) 让出事件循环", "#94a3b8", 11)

L.append('</svg>')
open("01-stage3-overview.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote 01-stage3-overview.svg")
