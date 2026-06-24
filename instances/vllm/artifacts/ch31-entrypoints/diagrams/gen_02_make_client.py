#!/usr/bin/env python3
"""ch31 图02：make_client 三分支决策树——离线默认是 SyncMPClient，不是 InprocClient。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1080, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W//2}" y="34" text-anchor="middle" font-size="20" font-weight="bold" fill="#0f172a">make_client 三分支：离线默认走 SyncMPClient（不是 InprocClient）</text>')


def box(x, y, w, h, lines, fill, stroke, tcolor="#0f172a", fs=14, rx=8, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 4) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        fw = 'bold' if i == 0 else 'normal'
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+4)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcolor}">{esc(ln)}</text>')


def line(x1, y1, x2, y2, marker="ar", color="#475569", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5" marker-end="url(#{marker})"{d}/>')


def edgelabel(x, y, txt, color="#1d4ed8"):
    tw = len(txt) * 8 + 16
    L.append(f'<rect x="{x-tw/2}" y="{y-13}" width="{tw}" height="22" rx="5" fill="white" stroke="{color}" stroke-width="1.2"/>')
    L.append(f'<text x="{x}" y="{y+2}" text-anchor="middle" font-size="12" font-weight="bold" fill="{color}">{esc(txt)}</text>')


# 根
ROOTX, ROOTW = W/2 - 200, 400
box(ROOTX, 56, ROOTW, 66, ['from_engine_args', '检查 envs.VLLM_ENABLE_V1_MULTIPROCESSING', '（默认值 True）'], '#e0e7ff', '#6366f1', fs=14)
root_bottom = 122
root_cx = W/2

# 三列出口
col_default_cx = 250
col_inproc_cx = 620
col_async_cx = 920

# 分支线
y_split = 180
line(root_cx, root_bottom, root_cx, y_split-30, marker=None) if False else None
L.append(f'<line x1="{root_cx}" y1="{root_bottom}" x2="{root_cx}" y2="{y_split-40}" stroke="#475569" stroke-width="2.5"/>')
# 水平分流
L.append(f'<line x1="{col_default_cx}" y1="{y_split-40}" x2="{col_async_cx}" y2="{y_split-40}" stroke="#475569" stroke-width="2.5"/>')
line(col_default_cx, y_split-40, col_default_cx, y_split, color="#1d4ed8")
line(col_inproc_cx, y_split-40, col_inproc_cx, y_split, color="#475569")
line(col_async_cx, y_split-40, col_async_cx, y_split, color="#94a3b8", marker="arg")
edgelabel(col_default_cx, y_split-58, "env=True（默认）", "#1d4ed8")
edgelabel(col_inproc_cx, y_split-58, "env=0", "#475569")
edgelabel(col_async_cx, y_split-58, "asyncio_mode=True", "#94a3b8")

# 分支1：默认 SyncMPClient（高亮）
bw = 320
b1x = col_default_cx - bw/2
L.append(f'<rect x="{b1x-14}" y="{y_split-12}" width="{bw+28}" height="392" rx="14" fill="#eff6ff" stroke="#3b82f6" stroke-width="2.5"/>')
L.append(f'<text x="{col_default_cx}" y="{y_split+12}" text-anchor="middle" font-size="14" font-weight="bold" fill="#1d4ed8">⭐ 默认路径</text>')
box(b1x, y_split+24, bw, 56, ['enable_multiprocessing 被强翻 True', 'multiprocess_mode=True, asyncio_mode=False'], '#dbeafe', '#3b82f6', fs=13)
line(col_default_cx, y_split+80, col_default_cx, y_split+108, color="#1d4ed8")
box(b1x, y_split+108, bw, 44, ['make_client', '命中 (mp=True, async=False)'], '#dbeafe', '#3b82f6', fs=13)
line(col_default_cx, y_split+152, col_default_cx, y_split+180, color="#1d4ed8")
box(b1x, y_split+180, bw, 188, [
    'SyncMPClient',
    '后台进程 EngineCore',
    '+ ZMQ output_socket',
    '+ 后台收数 daemon 线程',
    '+ outputs_queue（阻塞队列）',
    'get_output() = outputs_queue.get()',
], '#bfdbfe', '#2563eb', fs=13, tcolor="#0f172a")

# 分支2：env=0 → InprocClient
b2x = col_inproc_cx - bw/2
L.append(f'<text x="{col_inproc_cx}" y="{y_split+12}" text-anchor="middle" font-size="13" font-weight="bold" fill="#b45309">测试 / 调试 / V0 风格回退</text>')
box(b2x, y_split+24, bw, 56, ['enable_multiprocessing 维持 False', 'multiprocess_mode=False'], '#fef3c7', '#d97706', fs=13)
line(col_inproc_cx, y_split+80, col_inproc_cx, y_split+108)
box(b2x, y_split+108, bw, 44, ['make_client', '落到最后一行（else）'], '#fef3c7', '#d97706', fs=13)
line(col_inproc_cx, y_split+152, col_inproc_cx, y_split+180)
box(b2x, y_split+180, bw, 110, [
    'InprocClient',
    '真·进程内，无 ZMQ',
    '无后台进程 / 无后台线程',
    'get_output() 直接 engine_core.step_fn()',
], '#fde68a', '#d97706', fs=13)

# 分支3：async → AsyncMP（灰显）
bw3 = 240
b3x = col_async_cx - bw3/2
box(b3x, y_split+24, bw3, 56, ['mp=True, async=True', 'make_async_mp_client'], '#f1f5f9', '#94a3b8', fs=12, tcolor="#475569", dash="6 4")
L.append(f'<line x1="{col_async_cx}" y1="{y_split+80}" x2="{col_async_cx}" y2="{y_split+108}" stroke="#94a3b8" stroke-width="2.5" marker-end="url(#arg)" stroke-dasharray="6 4"/>')
box(b3x, y_split+108, bw3, 70, ['AsyncMPClient', '（第 4 章 AsyncLLM）', '本章不命中'], '#f1f5f9', '#94a3b8', fs=12, tcolor="#475569", dash="6 4")

L.append('</svg>')
open('02-make-client-decision.svg', 'w').write('\n'.join(L))
print('wrote 02-make-client-decision.svg')
