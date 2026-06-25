"""06-parallel-sampling-fanout: 横向三栏数据流——扇出与归并。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1180, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
         '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# column headers
cols = [
    (30, 320, "调用方 / generate()"),
    (370, 440, "AsyncLLM.add_request（本进程）"),
    (840, 310, "EngineCore（独立进程）"),
]
for x, cw, title in cols:
    L.append(f'<text x="{x + cw/2}" y="34" text-anchor="middle" font-size="17" font-weight="bold" fill="#0f172a">{esc(title)}</text>')

def box(x, y, bw, bh, fill, stroke, lines, fs=14, tcolor="#0f172a", weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    start = y + bh/2 - (n-1)*(fs+4)/2 + fs/2 - 2
    for i, ln in enumerate(lines):
        wt = weight if i == 0 else "normal"
        L.append(f'<text x="{x + bw/2}" y="{start + i*(fs+4)}" text-anchor="middle" font-size="{fs}" font-weight="{wt}" fill="{tcolor}">{esc(ln)}</text>')

# Left: caller request
box(40, 230, 290, 90, "#eff6ff", "#2563eb",
    ["generate(prompt, n=3, seed=42)", "外部 request_id = R", "对外只看到一个请求 / 一个队列"], fs=14, weight="bold")

# Middle top: assign_request_id
box(390, 70, 400, 56, "#f1f5f9", "#64748b",
    ["assign_request_id: R → 内部 id  R-ab12cd34", "external_req_id = R 留底"], fs=13, weight="bold")

# Middle: ParentRequest container
L.append(f'<rect x="378" y="150" width="424" height="320" rx="10" fill="#fefce8" stroke="#ca8a04" stroke-width="1.8"/>')
L.append(f'<text x="590" y="174" text-anchor="middle" font-size="15" font-weight="bold" fill="#854d0e">ParentRequest（扇出 + 归并状态）</text>')
L.append(f'<text x="590" y="194" text-anchor="middle" font-size="12" fill="#854d0e">child_requests 跟踪 n 路完成</text>')

children = [
    ("0_R-ab12cd34", "n=1  seed=42", "#dcfce7", "#16a34a"),
    ("1_R-ab12cd34", "n=1  seed=43", "#dbeafe", "#2563eb"),
    ("2_R-ab12cd34", "n=1  seed=44", "#fae8ff", "#a21caf"),
]
cy0 = 210
for i, (cid, params, fill, stroke) in enumerate(children):
    cy = cy0 + i * 80
    box(398, cy, 384, 60, fill, stroke, [f"child {i}:  request_id = {cid}", params], fs=13, weight="bold")
    # arrow to EngineCore (x2=850 = EngineCore left edge)
    L.append(f'<line x1="782" y1="{cy+30}" x2="850" y2="{cy+30}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')

# arrow assign -> parent (y1=126=assign bottom, y2=150=parent top)
L.append(f'<line x1="590" y1="126" x2="590" y2="150" stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
# arrow caller -> assign (from caller right-center to assign left-center)
L.append(f'<line x1="330" y1="275" x2="390" y2="98" stroke="#2563eb" stroke-width="1.8" marker-end="url(#a)"/>')

# Right: EngineCore
L.append(f'<rect x="850" y="190" width="300" height="220" rx="10" fill="#f8fafc" stroke="#334155" stroke-width="1.8"/>')
L.append(f'<text x="1000" y="216" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">连续批处理调度</text>')
L.append(f'<text x="1000" y="236" text-anchor="middle" font-size="12" fill="#334155">3 个 child = 3 个普通独立请求</text>')
L.append(f'<text x="1000" y="254" text-anchor="middle" font-size="12" fill="#334155">引擎侧无任何 n&gt;1 批量语义</text>')
for i, (_, _, fill, stroke) in enumerate(children):
    cy = 274 + i * 42
    L.append(f'<rect x="880" y="{cy}" width="240" height="32" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    L.append(f'<text x="1000" y="{cy+21}" text-anchor="middle" font-size="12" fill="#0f172a">seq {i}：与他人请求平等竞争资源</text>')

# Merge path: EngineCore outputs -> ParentRequest.get_outputs -> single RequestOutput -> caller
L.append(f'<text x="590" y="500" text-anchor="middle" font-size="13" font-weight="bold" fill="#16a34a">归并回流：ParentRequest.get_outputs → RequestOutput(request_id = R) → 同一队列</text>')
# return arrow EngineCore bottom -> parent bottom
L.append(f'<path d="M 1000 410 L 1000 524 L 802 524" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#ag)"/>')
# parent -> caller (endpoint y=320 = caller box bottom)
L.append(f'<path d="M 378 524 L 185 524 L 185 320" fill="none" stroke="#16a34a" stroke-width="1.8" marker-end="url(#ag)"/>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch06-input-processor/diagrams/06-parallel-sampling-fanout.svg", "w").write('\n'.join(L))
print("wrote fanout svg")
