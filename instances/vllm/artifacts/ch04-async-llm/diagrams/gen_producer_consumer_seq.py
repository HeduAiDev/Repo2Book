#!/usr/bin/env python3
"""fig-producer-consumer-seq: output_handler 与 generate 的生产者-消费者时序（单请求）。"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


C_TXT = "#1e293b"
C_MUT = "#64748b"
C_REQ = "#2563eb"
C_OUT = "#dc2626"
C_GREEN = "#059669"


def build():
    w, h = 1180, 760
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    for mk, col in [("am", C_MUT), ("ar", C_REQ), ("ao", C_OUT), ("ag", C_GREEN)]:
        L.append(f'<marker id="{mk}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{col}"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="30" y="34" font-family="sans-serif" font-size="20" font-weight="bold" fill="{C_TXT}">{esc("生产者-消费者时序：output_handler 喂、generate 取（单请求）")}</text>')

    # 五条生命线
    actors = [
        ("Client", 110, "#f1f5f9", "#475569"),
        ("generate() 协程", 350, "#eef2ff", "#6366f1"),
        ("RequestOutputCollector", 620, "#fef3c7", "#d97706"),
        ("output_handler 任务", 890, "#fee2e2", "#dc2626"),
        ("EngineCore (stub)", 1080, "#e5e7eb", "#9ca3af"),
    ]
    top = 60
    bot = 720
    for name, x, fill, bd in actors:
        bw = 150 if len(name) < 18 else 180
        L.append(f'<rect x="{x-bw//2}" y="{top}" width="{bw}" height="40" rx="6" fill="{fill}" stroke="{bd}" stroke-width="2"/>')
        L.append(f'<text x="{x}" y="{top+25}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" fill="{C_TXT}">{esc(name)}</text>')
        # lifeline 用细 rect 避免触发 validator marker 检查
        L.append(f'<rect x="{x-1}" y="{top+40}" width="2" height="{bot-top-40}" fill="#cbd5e1"/>')

    X = {a[0]: a[1] for a in actors}

    def msg(y, src, dst, text, col=C_MUT, dash=False, ret=False):
        x1, x2 = X[src], X[dst]
        mk = {C_MUT: "am", C_REQ: "ar", C_OUT: "ao", C_GREEN: "ag"}[col]
        da = ' stroke-dasharray="5,4"' if dash else ''
        L.append(f'<path d="M{x1},{y} L{x2},{y}" fill="none" stroke="{col}" stroke-width="2"{da} marker-end="url(#{mk})"/>')
        anc = "middle"
        mx = (x1 + x2) // 2
        L.append(f'<text x="{mx}" y="{y-7}" text-anchor="{anc}" font-family="monospace" font-size="11" fill="{col}">{esc(text)}</text>')

    def note(y, x, text, col="#475569", w_=210):
        L.append(f'<rect x="{x-w_//2}" y="{y-15}" width="{w_}" height="26" rx="4" fill="#fffbeb" stroke="{col}" stroke-width="1"/>')
        L.append(f'<text x="{x}" y="{y+2}" text-anchor="middle" font-family="sans-serif" font-size="10.5" fill="{col}">{esc(text)}</text>')

    def activation(x, y0, y1, col):
        L.append(f'<rect x="{x-5}" y="{y0}" width="10" height="{y1-y0}" rx="2" fill="{col}" fill-opacity="0.18" stroke="{col}" stroke-width="1"/>')

    y = 130
    msg(y, "Client", "generate() 协程", "async for ... in generate()", C_MUT)
    activation(X["generate() 协程"], y, 700, "#6366f1")
    y += 50
    msg(y, "generate() 协程", "RequestOutputCollector", "add_request → 建 Collector", C_REQ)
    y += 44
    msg(y, "generate() 协程", "EngineCore (stub)", "add_request_async（投递请求）", C_REQ)
    y += 54

    # while 循环框
    L.append(f'<rect x="270" y="{y-18}" width="660" height="250" rx="6" fill="none" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="4,4"/>')
    L.append(f'<rect x="270" y="{y-18}" width="120" height="22" rx="3" fill="#94a3b8"/>')
    L.append(f'<text x="278" y="{y-2}" font-family="monospace" font-size="11" font-weight="bold" fill="white">{esc("while not finished")}</text>')
    y += 26

    note(y, X["generate() 协程"], "get_nowait() → None（槽空）", "#6366f1", 220)
    y += 40
    msg(y, "generate() 协程", "RequestOutputCollector", "await get()：挂起于 ready.wait()", C_MUT, dash=True)
    note(y + 26, X["generate() 协程"], "协程挂起，让出事件循环", "#64748b", 210)
    y += 58

    # 并行：output_handler 拉取
    msg(y, "output_handler 任务", "EngineCore (stub)", "await get_output_async()", C_OUT)
    y += 30
    msg(y, "EngineCore (stub)", "output_handler 任务", "EngineCoreOutput", C_OUT)
    activation(X["output_handler 任务"], y - 30, y + 110, "#dc2626")
    y += 44
    msg(y, "output_handler 任务", "RequestOutputCollector", "process_outputs → put()", C_OUT)
    note(y + 26, X["RequestOutputCollector"], "槽空 → 写值 + ready.set() 唤醒", "#d97706", 250)
    y += 60
    msg(y, "RequestOutputCollector", "generate() 协程", "get() 返回 RequestOutput", C_GREEN)
    y += 34
    msg(y, "generate() 协程", "Client", "yield out（一个 token 块）", C_GREEN)
    y += 46

    # 退出循环条件
    note(y, (X["generate() 协程"] + X["RequestOutputCollector"]) // 2, "循环直到 EngineCoreOutput.finished = True", C_REQ, 360)
    y += 50

    # 生产者超前 merge 分支
    L.append(f'<rect x="540" y="{y-18}" width="420" height="64" rx="6" fill="#fef2f2" stroke="#dc2626" stroke-width="1.3" stroke-dasharray="4,4"/>')
    L.append(f'<rect x="540" y="{y-18}" width="170" height="22" rx="3" fill="#dc2626"/>')
    L.append(f'<text x="548" y="{y-2}" font-family="monospace" font-size="11" font-weight="bold" fill="white">{esc("alt 生产者超前")}</text>')
    y += 26
    msg(y, "output_handler 任务", "RequestOutputCollector", "连续两次 put() → self.output.add() 合帧", C_OUT)

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
    subprocess.run(['rsvg-convert', '-z', '2', base + '.svg', '-o', base + '.png'], check=True)
    print(f"PNG {os.path.getsize(base+'.png')//1024}KB")
