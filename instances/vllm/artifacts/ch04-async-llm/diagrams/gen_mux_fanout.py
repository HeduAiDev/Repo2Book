#!/usr/bin/env python3
"""fig-mux-fanout: 一个 EngineCore 批 -> 按 req_id 分发回 N 个 per-request 队列。"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


C_TXT = "#1e293b"
C_MUT = "#64748b"
C_A = "#2563eb"
C_B = "#059669"
C_C = "#d97706"
REQS = [("req_A", C_A), ("req_B", C_B), ("req_C", C_C)]


def build():
    w, h = 1180, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    for mk, col in [("am", C_MUT), ("aA", C_A), ("aB", C_B), ("aC", C_C)]:
        L.append(f'<marker id="{mk}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{col}"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="30" y="34" font-family="sans-serif" font-size="20" font-weight="bold" fill="{C_TXT}">{esc("多路复用解扇出：一个 EngineCore 批 → 按 req_id 分回 N 个队列")}</text>')
    L.append(f'<text x="30" y="58" font-family="sans-serif" font-size="13" fill="{C_MUT}">{esc("IPC 单出口扇入  →  process_outputs 按 req_id 解扇出  →  per-request 队列互不排队")}</text>')

    mk_of = {C_A: "aA", C_B: "aB", C_C: "aC"}

    # 左：一个 EngineCoreOutputs 批
    bx, by, bw = 60, 130, 230
    L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="290" rx="8" fill="#f8fafc" stroke="#94a3b8" stroke-width="2"/>')
    L.append(f'<text x="{bx+bw//2}" y="{by-10}" text-anchor="middle" font-family="monospace" font-size="13" font-weight="bold" fill="{C_TXT}">{esc("EngineCoreOutputs 批")}</text>')
    L.append(f'<text x="{bx+bw//2}" y="{by+22}" text-anchor="middle" font-family="monospace" font-size="10" fill="{C_MUT}">{esc("get_output_async() 一次拉回")}</text>')
    item_y = {}
    for i, (rid, col) in enumerate(REQS):
        iy = by + 50 + i * 75
        L.append(f'<rect x="{bx+20}" y="{iy}" width="{bw-40}" height="56" rx="5" fill="white" stroke="{col}" stroke-width="2"/>')
        L.append(f'<text x="{bx+bw//2}" y="{iy+22}" text-anchor="middle" font-family="monospace" font-size="12" font-weight="bold" fill="{col}">{esc("EngineCoreOutput")}</text>')
        L.append(f'<text x="{bx+bw//2}" y="{iy+42}" text-anchor="middle" font-family="monospace" font-size="11" fill="{col}">{esc("request_id = " + rid)}</text>')
        item_y[rid] = iy + 28

    # 中：process_outputs for 循环 + 查找表
    mx, my, mw, mh = 420, 150, 280, 250
    L.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="2"/>')
    L.append(f'<text x="{mx+mw//2}" y="{my-10}" text-anchor="middle" font-family="monospace" font-size="13" font-weight="bold" fill="#4338ca">{esc("output_handler.process_outputs")}</text>')
    L.append(f'<text x="{mx+20}" y="{my+30}" font-family="monospace" font-size="11" fill="{C_TXT}">{esc("for o in engine_core_outputs:")}</text>')
    L.append(f'<text x="{mx+34}" y="{my+50}" font-family="monospace" font-size="11" fill="{C_TXT}">{esc("req_id = o.request_id")}</text>')
    L.append(f'<text x="{mx+34}" y="{my+70}" font-family="monospace" font-size="11" fill="{C_TXT}">{esc("st = request_states[req_id]")}</text>')
    L.append(f'<text x="{mx+34}" y="{my+90}" font-family="monospace" font-size="11" fill="{C_TXT}">{esc("st.queue.put(out)")}</text>')
    # 查找表
    ty = my + 110
    L.append(f'<rect x="{mx+18}" y="{ty}" width="{mw-36}" height="120" rx="5" fill="white" stroke="#94a3b8" stroke-width="1.4"/>')
    L.append(f'<text x="{mx+mw//2}" y="{ty+18}" text-anchor="middle" font-family="monospace" font-size="11" font-weight="bold" fill="{C_MUT}">{esc("request_states: req_id → RequestState")}</text>')
    row_y = {}
    for i, (rid, col) in enumerate(REQS):
        ry = ty + 36 + i * 26
        L.append(f'<text x="{mx+34}" y="{ry}" font-family="monospace" font-size="11" fill="{col}">{esc(rid + "  →  queue " + rid[-1])}</text>')
        row_y[rid] = ry - 4

    # 左 -> 中（扇入）
    for rid, col in REQS:
        L.append(f'<path d="M{bx+bw},{item_y[rid]} C{(bx+bw+mx)//2},{item_y[rid]} {(bx+bw+mx)//2},{my+50} {mx},{my+55}" fill="none" stroke="{col}" stroke-width="2" marker-end="url(#{mk_of[col]})"/>')

    # 右：三个独立队列 + generate
    rx = 850
    for i, (rid, col) in enumerate(REQS):
        qy = 140 + i * 120
        L.append(f'<rect x="{rx}" y="{qy}" width="200" height="50" rx="6" fill="#fffbeb" stroke="{col}" stroke-width="2.2"/>')
        L.append(f'<text x="{rx+100}" y="{qy+20}" text-anchor="middle" font-family="monospace" font-size="11" font-weight="bold" fill="{col}">{esc("RequestOutputCollector")}</text>')
        L.append(f'<text x="{rx+100}" y="{qy+38}" text-anchor="middle" font-family="monospace" font-size="11" fill="{col}">{esc("queue " + rid[-1])}</text>')
        # generate 协程
        L.append(f'<rect x="{rx+40}" y="{qy+62}" width="120" height="34" rx="5" fill="#eef2ff" stroke="{col}" stroke-width="1.6"/>')
        L.append(f'<text x="{rx+100}" y="{qy+84}" text-anchor="middle" font-family="monospace" font-size="11" fill="{C_TXT}">{esc("generate() " + rid[-1])}</text>')
        # 中 -> 右（解扇出），从查找表对应行出发
        L.append(f'<path d="M{mx+mw-20},{row_y[rid]} C{(mx+mw+rx)//2},{row_y[rid]} {(mx+mw+rx)//2},{qy+25} {rx},{qy+25}" fill="none" stroke="{col}" stroke-width="2.4" marker-end="url(#{mk_of[col]})"/>')
        # queue -> generate
        L.append(f'<path d="M{rx+100},{qy+50} L{rx+100},{qy+62}" fill="none" stroke="{col}" stroke-width="2" marker-end="url(#{mk_of[col]})"/>')

    # 说明
    L.append(f'<text x="{(bx+bw+mx)//2}" y="120" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" fill="{C_MUT}">{esc("扇入：单 IPC 出口")}</text>')
    L.append(f'<text x="{(mx+mw+rx)//2}" y="120" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" fill="{C_MUT}">{esc("解扇出：per-request 多队列")}</text>')
    L.append(f'<text x="{rx+100}" y="510" text-anchor="middle" font-family="sans-serif" font-size="11" fill="{C_TXT}">{esc("每路 generate 只等自己的队列 → 尾延迟与并发数解耦")}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1]
    svg = build()
    Path(base + '.svg').write_text(svg)
    print(f"SVG {len(svg)}B")
    assert subprocess.run(['xmllint', '--noout', base + '.svg']).returncode == 0
    vs = Path('/mnt/e/Laboratory/Repo2Book/.claude/skills/svg-diagram/scripts/validate_svg.py')
    r = subprocess.run([sys.executable, str(vs), base + '.svg'], capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout
    subprocess.run(['convert', '-density', '150', base + '.svg', base + '.png'], check=True)
    print(f"PNG {os.path.getsize(base+'.png')//1024}KB")
