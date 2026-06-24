#!/usr/bin/env python3
"""ch31 图03：一个 generate 调用内单请求的生命周期（4 泳道时序）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1120, 760
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


def node(cx, y, w, h, lines, fill, stroke, fs=12):
    L.append(f'<rect x="{cx-w/2}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    cnt = len(lines)
    cyy = y + h / 2 - (cnt - 1) * (fs + 3) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        fw = 'bold' if i == 0 else 'normal'
        L.append(f'<text x="{cx}" y="{cyy + i*(fs+3)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="#0f172a">{esc(ln)}</text>')


def harrow(x1, y, x2, label=None, dash=None, color="#475569", marker="ar"):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2.5" marker-end="url(#{marker})"{d}/>')
    if label:
        mx = (x1 + x2) / 2
        L.append(f'<text x="{mx}" y="{y-7}" text-anchor="middle" font-size="11" fill="{color}">{esc(label)}</text>')


y = 100
# 1. LLM._add_request
node(LLM, y, 180, 50, ['_add_request', 'output_kind=FINAL_ONLY', '自增 request_id'], '#bfdbfe', '#2563eb')
y += 70
# 2. -> LLMEngine.add_request
harrow(LLM + 90, y, ENG - 95, 'add_request(id, prompt, params)')
node(ENG, y - 18, 180, 46, ['add_request', 'input_processor.process_inputs'], '#bbf7d0', '#16a34a')
y += 60
# 3. 双注册：-> OutputProcessor
harrow(ENG + 90, y, OUT - 95, '① output_processor.add_request')
node(OUT, y - 18, 180, 42, ['add_request', '建 RequestState（待装配）'], '#fef08a', '#ca8a04')
y += 56
# 4. 双注册：-> EngineCore
harrow(ENG + 90, y, CORE - 100, '② engine_core.add_request')
node(CORE, y - 18, 190, 42, ['收下请求入调度', '（SyncMPClient 经 ZMQ 送达）'], '#f5d0fe', '#a21caf')
y += 60
# ParentRequest 旁注（虚线）
L.append(f'<rect x="{ENG-150}" y="{y-12}" width="300" height="40" rx="7" fill="#faf5ff" stroke="#a855f7" stroke-width="1.6" stroke-dasharray="6 4"/>')
L.append(f'<text x="{ENG}" y="{y+4}" text-anchor="middle" font-size="11" font-weight="bold" fill="#7e22ce">n&gt;1：add_request 内 ParentRequest 扇出</text>')
L.append(f'<text x="{ENG}" y="{y+20}" text-anchor="middle" font-size="11" fill="#7e22ce">①② 各重复 n 次（并行采样子请求）</text>')
y += 56

# 分隔：进入 _run_engine 循环
L.append(f'<line x1="40" y1="{y}" x2="{W-40}" y2="{y}" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 5"/>')
L.append(f'<rect x="{LLM-150}" y="{y+8}" width="300" height="36" rx="7" fill="#bfdbfe" stroke="#2563eb" stroke-width="2"/>')
L.append(f'<text x="{LLM}" y="{y+31}" text-anchor="middle" font-size="13" font-weight="bold" fill="#1d4ed8">_run_engine: while step()  循环 ↻</text>')
y += 64

# step 一拍
harrow(LLM + 90, y, ENG - 95, 'step()')
node(ENG, y - 18, 190, 42, ['step', '① get_output'], '#bbf7d0', '#16a34a')
y += 54
harrow(ENG + 95, y, CORE - 100, 'get_output()（阻塞取队列）', color="#a21caf")
node(CORE, y - 16, 190, 38, ['后台线程喂 outputs_queue', '→ 主线程阻塞取出'], '#f5d0fe', '#a21caf', fs=11)
y += 52
harrow(ENG + 95, y, OUT - 95, '② process_outputs', color="#ca8a04")
node(OUT, y - 16, 190, 40, ['去 token 化 / 装配 RequestOutput', '判定 finished'], '#fef08a', '#ca8a04', fs=11)
y += 52
node(ENG, y - 8, 200, 38, ['③ abort 停止串触发的请求', '④（记 stats）'], '#bbf7d0', '#16a34a', fs=11)
y += 50
# 收集 + 排序
L.append(f'<line x1="{ENG-95}" y1="{y}" x2="{LLM+95}" y2="{y}" stroke="#475569" stroke-width="2.5" marker-end="url(#ar)"/>')
L.append(f'<text x="{(ENG+LLM)/2}" y="{y-7}" text-anchor="middle" font-size="11" fill="#475569">finished 的 RequestOutput</text>')
node(LLM, y - 18, 200, 50, ['收集 finished 输出', '循环至 has_unfinished_requests()=False', 'sorted(by request_id) 返回'], '#dcfce7', '#16a34a', fs=11)

L.append('</svg>')
open('03-request-lifecycle.svg', 'w').write('\n'.join(L))
print('wrote 03-request-lifecycle.svg')
