"""02-process-outputs-loop: vertical flowchart of the single loop."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 760, 880
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


def text(x, y, txt, fs=14, col="#0f172a", anchor="middle", weight="normal"):
    L.append(
        f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
        f'fill="{col}" text-anchor="{anchor}" font-weight="{weight}">{esc(txt)}</text>'
    )


def box(x, y, bw, bh, fill, stroke, lines, fs=14, rx=8):
    L.append(
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    n = len(lines)
    total = n * (fs + 4)
    sy = y + bh / 2 - total / 2 + fs
    for i, ln in enumerate(lines):
        weight = "bold" if i == 0 else "normal"
        col = "#0f172a" if i == 0 else "#475569"
        size = fs if i == 0 else fs - 2
        text(x + bw / 2, sy + i * (fs + 4), ln, size, col, "middle", weight)


def diamond(cx, cy, hw, hh, fill, stroke, lines, fs=13):
    pts = f"{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}"
    L.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    sy = cy - n * (fs + 2) / 2 + fs
    for i, ln in enumerate(lines):
        text(cx, sy + i * (fs + 2), ln, fs, "#0f172a", "middle",
             "bold" if i == 0 else "normal")


def arrow(x1, y1, x2, y2, col="#475569", mk="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
        f'stroke-width="2"{d} marker-end="url(#{mk})"/>'
    )


text(w / 2, 30, "process_outputs：唯一遍历整批的单循环", 18, "#0f172a", "middle", "bold")
text(w / 2, 52, "for engine_core_output in engine_core_outputs:", 13, "#94a3b8")

cx = 330
bw = 380
x = cx - bw / 2
y = 75
steps = [
    ("#fee2e2", "#dc2626",
     ["取 req_state = request_states.get(req_id)", "为 None（已 abort）→ continue 跳过"], 62),
    ("#fef9c3", "#ca8a02",
     ["① stats：_update_stats_from_output", "（metrics 子系统，本步累计）"], 56),
    ("#e0e7ff", "#6366f1",
     ["② prefill 翻转：is_prefilling = False", "记 num_cached_tokens（首次）"], 56),
    ("#dcfce7", "#16a34a",
     ["③ detokenizer.update(new_token_ids, ...)", "增量去 token + check_stop_strings",
      "命中停止串 → finish_reason = STOP"], 74),
    ("#dbeafe", "#2563eb",
     ["④ logprobs_processor.update_from_output", "增量累积 sample / prompt logprobs"], 56),
    ("#fae8ff", "#a21caf",
     ["⑤ make_request_output(...)", "三道闸门：FINAL_ONLY / stream_interval / 父聚合"], 56),
]
ys = []
for fill, stroke, lines, bh in steps:
    box(x, y, bw, bh, fill, stroke, lines)
    ys.append((y, bh))
    y += bh + 26

# arrows between step boxes
for i in range(len(ys) - 1):
    y0, h0 = ys[i]
    y1, _ = ys[i + 1]
    arrow(cx, y0 + h0, cx, y1)

# decision: returned a RequestOutput?
dy = y + 12
diamond(cx, dy, 130, 50, "#f1f5f9", "#64748b",
        ["返回了", "RequestOutput？"])
arrow(cx, ys[-1][0] + ys[-1][1], cx, dy - 50)

# yes branch: dispatch
disp_y = dy + 70
box(x, disp_y, bw, 64, "#fff7ed", "#ea580c",
    ["queue 非空 → queue.put()  (AsyncLLM)",
     "否则 → request_outputs.append()  (LLMEngine)"])
arrow(cx, dy + 50, cx, disp_y)
text(cx + 14, dy + 64, "是", 13, "#16a34a", "start", "bold")

# no branch (None): skip to finish check
text(cx + 140, dy + 4, "否（被节流 / 聚合未齐）", 12, "#94a3b8", "start")

# finish check
fin_dy = disp_y + 64 + 38
diamond(cx, fin_dy, 130, 46, "#f1f5f9", "#64748b",
        ["finish_reason", "is not None？"])
arrow(cx, disp_y + 64, cx, fin_dy - 46)

# finish branch
finbox_y = fin_dy + 64
box(x, finbox_y, bw, 78, "#fee2e2", "#dc2626",
    ["_finish_request：注销三张映射表",
     "EngineCore 未完成 → reqs_to_abort.append",
     "_update_stats_from_finished"])
arrow(cx, fin_dy + 46, cx, finbox_y)
text(cx + 14, fin_dy + 60, "是", 13, "#16a34a", "start", "bold")

# loop back arrow (next iteration)
text(cx, finbox_y + 78 + 26, "↻ 下一个 engine_core_output", 14, "#64748b", "middle", "bold")
arrow(cx, finbox_y + 78, cx, finbox_y + 78 + 12)

L.append('</svg>')
open("02-process-outputs-loop.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote 02-process-outputs-loop.svg")
