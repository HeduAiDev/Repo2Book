#!/usr/bin/env python3
"""ch17-pipe-topology: ready/death pipe ownership diagram.
Shows which ends of each pipe the parent (Executor) vs child (WorkerProc) hold,
and which ends are closed after proc.start().
"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 980, 480
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="arrg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
         '<marker id="arro" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#ea580c"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def txt(x, y, t, fs=13, anchor="middle", fill="#1e293b", fw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{fw}" fill="{fill}">{esc(t)}</text>')

def box(x, y, w, h, fill, stroke, sw=1.5, rx=6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def arrow(x1, y1, x2, y2, color="#475569", mk="arr", dash=False, sw=2):
    d = ' stroke-dasharray="6,4"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mk})"{d}/>')

# Title
txt(W//2, 34, "ready / death 两条管道：端点归属与关闭顺序", 17, "middle", "#0f172a", "bold")
txt(W//2, 56, "父进程关掉已交给子进程的那端；自己留下读端 + death 写端", 13, "middle", "#475569")

# Process boxes
px_cx = 200   # parent center x
cx_cx = 680   # child center x
bw, bh = 220, 56
by = 80

box(px_cx - bw//2, by, bw, bh, "#eef2ff", "#6366f1", 2)
txt(px_cx, by + bh//2 + 6, "Executor（父进程）", 15, "middle", "#3730a3", "bold")

box(cx_cx - bw//2, by, bw, bh, "#f0fdf4", "#16a34a", 2)
txt(cx_cx, by + bh//2 + 6, "WorkerProc（子进程）", 15, "middle", "#15803d", "bold")

# ---- Ready pipe ----
ry = 190  # y center of ready pipe row
txt(W//2, ry - 48, "Ready Pipe（子 → 父，报告就绪）", 13, "middle", "#15803d", "bold")

# ready_reader (parent keeps)
box(px_cx - 80, ry - 20, 160, 40, "#dcfce7", "#16a34a", 1.5)
txt(px_cx, ry + 5, "ready_reader  ✓ 保留", 12, "middle", "#15803d", "bold")

# ready_writer (child sends READY; parent closes after spawn)
box(cx_cx - 80, ry - 20, 160, 40, "#fef9c3", "#ca8a04", 1.5)
txt(cx_cx, ry - 3, "ready_writer  ✓ 子持有", 12, "middle", "#713f12", "bold")
txt(cx_cx, ry + 13, "（发完 READY 后 worker_main 置 None）", 10, "middle", "#92400e")

# Arrow: child writes → parent reads
arrow(cx_cx - 85, ry, px_cx + 85, ry, "#16a34a", "arrg", sw=2)
txt(W//2, ry - 9, "send(READY, handle)", 11, "middle", "#166534")

# note: parent closes ready_writer immediately after spawn
txt(px_cx + 100, ry + 36, "proc.start() 后 → 父立刻 ready_writer.close()", 10.5, "middle", "#7c3aed")

# ---- Death pipe ----
dy = 320  # y center of death pipe row
txt(W//2, dy - 48, "Death Pipe（父 → 子，感知父进程退出）", 13, "middle", "#dc2626", "bold")

# death_writer (parent keeps)
box(px_cx - 80, dy - 20, 160, 40, "#fef2f2", "#dc2626", 1.5)
txt(px_cx, dy - 3, "death_writer  ✓ 保留", 12, "middle", "#991b1b", "bold")
txt(px_cx, dy + 13, "（关闭即触发子进程 EOFError）", 10, "middle", "#b91c1c")

# death_reader (child holds; parent closes after spawn)
box(cx_cx - 80, dy - 20, 160, 40, "#fff7ed", "#ea580c", 1.5)
txt(cx_cx, dy - 3, "death_reader  ✓ 子持有", 12, "middle", "#9a3412", "bold")
txt(cx_cx, dy + 13, "（阻塞 recv()，收 EOFError 后退出）", 10, "middle", "#c2410c")

# Arrow: parent closes death_writer → child reads EOF
arrow(px_cx + 85, dy, cx_cx - 85, dy, "#dc2626", "arro", dash=True, sw=2)
txt(W//2, dy - 9, "父退出 / shutdown → EOF → child 自清理", 11, "middle", "#991b1b")

# note: parent closes death_reader immediately after spawn
txt(cx_cx - 60, dy + 36, "proc.start() 后 → 父立刻 death_reader.close()", 10.5, "middle", "#7c3aed")

# Legend bottom
lx, ly = 40, 410
txt(lx, ly, "图例：", 11.5, "start", "#0f172a", "bold")
items = [
    ("#dcfce7", "#16a34a", "父进程持有（保留）"),
    ("#fef9c3", "#ca8a04", "子进程持有（spawn 时传入）"),
    ("#fef2f2", "#dc2626", "父进程持有 death 写端 → 关闭即触发 EOF"),
]
for i, (f, s, label) in enumerate(items):
    bx = lx + 60 + i * 270
    box(bx, ly + 12, 18, 16, f, s, 1.2, 3)
    txt(bx + 24, ly + 24, label, 11, "start", "#334155")

svg = '\n'.join(L) + '\n</svg>\n'
out = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch17-worker-and-executor/diagrams/ch17-pipe-topology.svg"
open(out, "w", encoding="utf-8").write(svg)
print(f"wrote {out}")
