#!/usr/bin/env python3
"""ch31 图03：一个 generate 调用内单请求的生命周期（4 泳道时序）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1120, 980
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ard" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#a855f7"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W//2}" y="32" text-anchor="middle" font-size="20" font-weight="bold" fill="#0f172a">一次 generate 调用内，单个请求的生命周期</text>')

# 4 泳道
lanes = [
    ('LLM', '#dbeafe', '#3b82f6'),
    ('LLMEngine', '#dcfce7', '#16a34a'),
    ('OutputProcessor', '#fef9c3', '#ca8a04'),
    ('EngineCore（后台进程）', '#fae8ff', '#a21caf'),
]
lane_top = 52
lane_h = H - lane_top - 20
n = len(lanes)
lane_w = (W - 40) / n
cxs = []
for i, (name, fill, stroke) in enumerate(lanes):
    x = 20 + i * lane_w
    cxs.append(x + lane_w / 2)
    L.append(f'<rect x="{x}" y="{lane_top}" width="{lane_w}" height="{lane_h}" fill="{fill}" fill-opacity="0.25" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<rect x="{x}" y="{lane_top}" width="{lane_w}" height="30" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + lane_w/2}" y="{lane_top+20}" text-anchor="middle" font-size="14" font-weight="bold" fill="#0f172a">{esc(name)}</text>')

LLM, ENG, OUT, CORE = cxs
# Centers: LLM=155, ENG=425, OUT=695, CORE=965
# Lane boundaries: 20, 290, 560, 830, 1100


def node(cx, ytop, w, h, lines, fill, stroke, fs=12):
    """Draw node centered at cx, top at ytop. Returns bottom y."""
    L.append(f'<rect x="{cx-w/2}" y="{ytop}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    cnt = len(lines)
    cyy = ytop + h / 2 - (cnt - 1) * (fs + 3) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        fw = 'bold' if i == 0 else 'normal'
        L.append(f'<text x="{cx}" y="{cyy + i*(fs+3)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="#0f172a">{esc(ln)}</text>')
    return ytop + h


def harrow(x1, y, x2, label=None, label_x=None, dash=None, color="#475569", marker="ar"):
    """Draw horizontal arrow from x1 to x2 at height y.
    label_x: explicit x for label center (defaults to midpoint).
              Position in the target lane to avoid overlapping source lane.
    """
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2.5" marker-end="url(#{marker})"{d}/>')
    if label:
        mx = label_x if label_x is not None else (x1 + x2) / 2
        L.append(f'<text x="{mx}" y="{y-7}" text-anchor="middle" font-size="11" fill="{color}">{esc(label)}</text>')


# Node widths
NW = 190  # standard node width

# Gap constants: space between node bottom and next node top
# Arrow drawn in the middle of the gap.
# Gap must be >= 28px.
GAP = 34

# ── Section 1: add_request registration ─────────────────────────────────

y = 100
# Node 1: LLM._add_request
y1b = node(LLM, y, NW, 52, ['_add_request', 'output_kind=FINAL_ONLY', '自增 request_id'], '#bfdbfe', '#2563eb')

# Arrow: LLM right → LLMEngine left
# LLM node right: 155+95=250, ENG node left: 425-100=325
# Label placed in ENG lane area, above ENG node
arr1_y = y1b + GAP // 2
n2_top = y1b + GAP
# Label centered in gap zone between LLM right (250) and ENG left (325): midpoint 287.5
# Since LLM lane is [20,290] and ENG lane is [290,560], midpoint 287.5 is just at boundary.
# Place label in ENG lane center to avoid LLM node: x=390 (safely inside ENG lane)
harrow(LLM + NW//2, arr1_y, ENG - NW//2 - 5, 'add_request(id, prompt, params)',
       label_x=390)

# Node 2: LLMEngine.add_request
y2b = node(ENG, n2_top, NW + 20, 46, ['add_request', 'input_processor.process_inputs'], '#bbf7d0', '#16a34a')

# Arrow: LLMEngine → OutputProcessor
# ENG node right: 425+105=530, OUT node left: 695-95=600 → gap only 70px
# Arrow spans 530→600. Label in OUT lane: center at 695
arr2_y = y2b + GAP // 2
n3_top = y2b + GAP
harrow(ENG + (NW+20)//2, arr2_y, OUT - NW//2 - 5,
       '① output_processor.add_request',
       label_x=695)   # center in OUT lane

# Node 3: OutputProcessor.add_request
y3b = node(OUT, n3_top, NW, 42, ['add_request', '建 RequestState（待装配）'], '#fef08a', '#ca8a04')

# Arrow: LLMEngine → EngineCore (crosses OUT lane)
# ENG node right: 530, CORE node left: 965-100=865
# Label in CORE lane center: 965
arr3_y = y3b + GAP // 2
n4_top = y3b + GAP
harrow(ENG + (NW+20)//2, arr3_y, CORE - NW//2 - 5,
       '② engine_core.add_request',
       label_x=965)   # center in CORE lane

# Node 4: EngineCore.add_request
y4b = node(CORE, n4_top, NW + 10, 42, ['收下请求入调度', '（SyncMPClient 经 ZMQ 送达）'], '#f5d0fe', '#a21caf')

# ParentRequest 旁注（虚线）— below node4
pr_top = y4b + 14
L.append(f'<rect x="{ENG-155}" y="{pr_top}" width="310" height="40" rx="7" fill="#faf5ff" stroke="#a855f7" stroke-width="1.6" stroke-dasharray="6 4"/>')
L.append(f'<text x="{ENG}" y="{pr_top+16}" text-anchor="middle" font-size="11" font-weight="bold" fill="#7e22ce">n&gt;1：add_request 内 ParentRequest 扇出</text>')
L.append(f'<text x="{ENG}" y="{pr_top+32}" text-anchor="middle" font-size="11" fill="#7e22ce">①② 各重复 n 次（并行采样子请求）</text>')
pr_bot = pr_top + 40

# ── Separator: _run_engine loop ───────────────────────────────────────────
sep_y = pr_bot + 18
L.append(f'<line x1="40" y1="{sep_y}" x2="{W-40}" y2="{sep_y}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 5"/>')
loop_top = sep_y + 10
L.append(f'<rect x="{LLM-150}" y="{loop_top}" width="300" height="36" rx="7" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{LLM}" y="{loop_top+23}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1d4ed8">_run_engine: while step()  循环 ↻</text>')
loop_bot = loop_top + 36

# ── step() arrow: LLM loop box right → LLMEngine left ────────────────────
arr5_y = loop_bot + GAP // 2
n5_top = loop_bot + GAP
# LLM loop box: x=5, w=300 → right=305. ENG node left: 325.
# Label in ENG lane, x=390
harrow(LLM + 150, arr5_y, ENG - (NW+20)//2 - 5, 'step()',
       label_x=390)

# Node 5: LLMEngine.step / get_output
y5b = node(ENG, n5_top, NW + 20, 42, ['step', '① get_output'], '#bbf7d0', '#16a34a')

# Arrow: ENG → CORE get_output
arr6_y = y5b + GAP // 2
n6_top = y5b + GAP
harrow(ENG + (NW+20)//2, arr6_y, CORE - NW//2 - 5,
       'get_output()（阻塞取队列）',
       label_x=965, color="#a21caf")

# Node 6: EngineCore outputs_queue
y6b = node(CORE, n6_top, NW + 10, 38, ['后台线程喂 outputs_queue', '→ 主线程阻塞取出'], '#f5d0fe', '#a21caf', fs=11)

# Arrow: ENG → OUT process_outputs
arr7_y = y6b + GAP // 2
n7_top = y6b + GAP
harrow(ENG + (NW+20)//2, arr7_y, OUT - NW//2 - 5,
       '② process_outputs',
       label_x=695, color="#ca8a04")

# Node 7: OutputProcessor.process_outputs
y7b = node(OUT, n7_top, NW + 10, 40, ['去 token 化 / 装配 RequestOutput', '判定 finished'], '#fef08a', '#ca8a04', fs=11)

# Node 8: LLMEngine abort/stats (no arrow, just next step below)
n8_top = y7b + 20
y8b = node(ENG, n8_top, NW + 20, 38, ['③ abort 停止串触发的请求', '④（记 stats）'], '#bbf7d0', '#16a34a', fs=11)

# Return arrow: ENG left → LLM right (finished outputs)
# ENG node left: 325, LLM node right: 250. Arrow goes rightward… no, LEFT (ENG→LLM).
# LLM node right: 155+95=250, ENG node left: 325
# Arrow from x1=325 to x2=250 (going left).
# Label: centered in LLM lane, clear of LLM collect node below.
arr9_y = y8b + GAP // 2
n9_top = y8b + GAP
# Label x=155 (LLM lane center) — will be inside LLM node below?
# LLM node top = n9_top = y8b + GAP. Label at arr9_y = y8b + GAP//2 → above node top ✓
# But text bounding box extends upward from arr9_y-7. Need cx not in ENG node range.
# LLM lane center=155, ENG node x=[320,530]. cx=155 is in LLM lane — safe.
L.append(f'<line x1="{ENG - (NW+20)//2}" y1="{arr9_y}" x2="{LLM + NW//2}" y2="{arr9_y}" stroke="#475569" stroke-width="2.5" marker-end="url(#ar)"/>')
L.append(f'<text x="{LLM}" y="{arr9_y - 7}" text-anchor="middle" font-size="11" fill="#475569">finished 的 RequestOutput</text>')

# Node 9: LLM collect & sort
node(LLM, n9_top, NW + 20, 52, ['收集 finished 输出', '循环至 has_unfinished_requests()=False', 'sorted(by request_id) 返回'], '#dcfce7', '#16a34a', fs=11)

L.append('</svg>')
open('03-request-lifecycle.svg', 'w').write('\n'.join(L))
print('wrote 03-request-lifecycle.svg')
