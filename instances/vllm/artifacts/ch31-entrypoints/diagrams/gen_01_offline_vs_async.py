#!/usr/bin/env python3
"""ch31 图01：离线同步路径 vs ch04 异步路径，并排对比。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1040, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '<marker id="arb" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{W//2}" y="34" text-anchor="middle" font-size="20" font-weight="bold" fill="#0f172a">离线同步路径（本章）vs 异步流式路径（第 4 章）</text>')


def box(x, y, w, h, lines, fill, stroke, tcolor="#0f172a", fs=14, bold_first=True, rx=8):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 4) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        fw = 'bold' if (i == 0 and bold_first) else 'normal'
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+4)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcolor}">{esc(ln)}</text>')


def varrow(x, y1, y2, marker="ar"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#475569" stroke-width="2.5" marker-end="url(#{marker})"/>')


# 左栏：离线（高亮）
LX = 70
LW = 380
LCX = LX + LW / 2
L.append(f'<rect x="{LX-22}" y="56" width="{LW+44}" height="{H-110}" rx="14" fill="#eff6ff" stroke="#3b82f6" stroke-width="2.5"/>')
L.append(f'<text x="{LCX}" y="82" text-anchor="middle" font-size="16" font-weight="bold" fill="#1d4ed8">离线 · 同步阻塞</text>')

ly = 100
steps_l = [
    (['LLM.generate(prompts)'], '#dbeafe', '#3b82f6'),
    (['_render_and_add_requests', '批量提交 · output_kind=FINAL_ONLY'], '#dbeafe', '#3b82f6'),
    (['_run_engine', 'while has_unfinished_requests():', '    LLMEngine.step()'], '#bfdbfe', '#2563eb'),
    (['SyncMPClient.get_output()', '阻塞 outputs_queue.get()'], '#dbeafe', '#3b82f6'),
    (['后台收数 daemon 线程', '（EngineCoreOutputQueueThread）'], '#e0f2fe', '#0284c7'),
    (['ZMQ output_socket'], '#e0f2fe', '#0284c7'),
    (['后台进程 EngineCore', '（独立进程，连续批处理整批）'], '#cffafe', '#0891b2'),
]
heights_l = [46, 56, 70, 56, 56, 40, 56]
prev_bottom = None
for (lines, fill, stroke), bh in zip(steps_l, heights_l):
    if prev_bottom is not None:
        varrow(LCX, prev_bottom, ly)
        ly += 16
    box(LX, ly, LW, bh, lines, fill, stroke)
    prev_bottom = ly + bh
    ly += bh
# 末尾 sorted
varrow(LCX, prev_bottom, prev_bottom + 16)
box(LX, prev_bottom + 16, LW, 40, ['末尾 sorted(by request_id)'], '#dcfce7', '#16a34a', tcolor="#166534")

# 右栏：异步（灰显）
RX = 590
RW = 380
RCX = RX + RW / 2
L.append(f'<rect x="{RX-22}" y="56" width="{RW+44}" height="{H-110}" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2.5" stroke-dasharray="7 5"/>')
L.append(f'<text x="{RCX}" y="82" text-anchor="middle" font-size="16" font-weight="bold" fill="#64748b">异步 · 事件循环（对照）</text>')

ry = 100
steps_r = [
    (['AsyncLLM.generate(...)'], '#f1f5f9', '#94a3b8'),
    (['add_request', 'output_kind=DELTA · 逐步流式'], '#f1f5f9', '#94a3b8'),
    (['asyncio 事件循环', 'async for out in generator:'], '#e2e8f0', '#94a3b8'),
    (['背景 output_handler 协程', '不停 await get_output_async()'], '#f1f5f9', '#94a3b8'),
    (['AsyncMPClient.get_output_async()', 'await asyncio.Queue.get()'], '#f1f5f9', '#94a3b8'),
    (['ZMQ output_socket'], '#f1f5f9', '#94a3b8'),
    (['后台进程 EngineCore', '（同样独立进程）'], '#f1f5f9', '#94a3b8'),
]
heights_r = [46, 56, 56, 56, 56, 40, 56]
prev_bottom = None
for (lines, fill, stroke), bh in zip(steps_r, heights_r):
    if prev_bottom is not None:
        L.append(f'<line x1="{RCX}" y1="{prev_bottom}" x2="{RCX}" y2="{ry}" stroke="#94a3b8" stroke-width="2.5" marker-end="url(#arg)"/>')
        ry += 16
    box(RX, ry, RW, bh, lines, fill, stroke, tcolor="#475569")
    prev_bottom = ry + bh
    ry += bh
L.append(f'<line x1="{RCX}" y1="{prev_bottom}" x2="{RCX}" y2="{prev_bottom+16}" stroke="#94a3b8" stroke-width="2.5" marker-end="url(#arg)"/>')
box(RX, prev_bottom + 16, RW, 40, ['逐 token DELTA 流式吐给客户端'], '#f1f5f9', '#94a3b8', tcolor="#475569")

# 中间两条核心区别标注
my = 230
L.append(f'<rect x="{LX+LW+30}" y="{my}" width="{RX-LX-LW-60}" height="48" rx="10" fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
L.append(f'<text x="{(LX+LW+RX)/2}" y="{my+20}" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">同步阻塞</text>')
L.append(f'<text x="{(LX+LW+RX)/2}" y="{my+38}" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">vs 事件循环</text>')
my2 = 330
L.append(f'<rect x="{LX+LW+30}" y="{my2}" width="{RX-LX-LW-60}" height="48" rx="10" fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
L.append(f'<text x="{(LX+LW+RX)/2}" y="{my2+20}" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">FINAL_ONLY 批量</text>')
L.append(f'<text x="{(LX+LW+RX)/2}" y="{my2+38}" text-anchor="middle" font-size="13" font-weight="bold" fill="#92400e">vs DELTA 流式</text>')

# 底注
L.append(f'<rect x="70" y="{H-44}" width="{W-140}" height="34" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>')
L.append(f'<text x="{W//2}" y="{H-22}" text-anchor="middle" font-size="13" fill="#166534">两侧 EngineCore 都跑在后台进程：区别只在主进程侧的驱动方式（阻塞线程队列 vs 事件循环协程），不依赖 in-process。</text>')

L.append('</svg>')
open('01-offline-vs-async-spine.svg', 'w').write('\n'.join(L))
print('wrote 01-offline-vs-async-spine.svg')
