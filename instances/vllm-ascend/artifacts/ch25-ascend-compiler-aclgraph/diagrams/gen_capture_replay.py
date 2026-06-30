#!/usr/bin/env python3
"""fig25-2: ACLGraphWrapper.__call__ 的分桶 capture / replay 状态机 + 207008 兜底。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1180, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="#1e293b">ACLGraphWrapper.__call__：按 BatchDescriptor 分桶，每形状捕一张 NPUGraph</text>')

def box(x, y, bw, bh, fill, stroke, lines, fs=16, tcol="#1e293b", rx=12, bold0=True):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    n = len(lines)
    total = n * (fs + 5)
    sy = y + (bh - total)/2 + fs
    for i, ln in enumerate(lines):
        fw = 'bold' if (i == 0 and bold0) else 'normal'
        L.append(f'<text x="{x+bw/2}" y="{sy + i*(fs+5)}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(ln)}</text>')

def diamond(cx, cy, rw, rh, lines, fs=15):
    pts = f"{cx},{cy-rh} {cx+rw},{cy} {cx},{cy+rh} {cx-rw},{cy}"
    L.append(f'<polygon points="{pts}" fill="#fef9c3" stroke="#ca8a04" stroke-width="2"/>')
    n = len(lines)
    sy = cy - (n-1)*(fs+3)/2 + fs/2 - 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{cx}" y="{sy + i*(fs+3)}" text-anchor="middle" font-size="{fs}" fill="#713f12">{esc(ln)}</text>')

def arrow(x1, y1, x2, y2, col="#475569", mk="a", wd=2.2):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{wd}" marker-end="url(#{mk})"/>')

def label(x, y, t, col="#475569", fs=14, anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" fill="{col}">{esc(t)}</text>')

cx = w/2
# entry box
box(cx-180, 60, 360, 64, "#e0f2fe", "#0284c7",
    ["取 forward_context", "batch_descriptor + cudagraph_runtime_mode"], fs=15)
arrow(cx, 124, cx, 150)

# diamond1: mode match?
diamond(cx, 192, 150, 44, ["runtime_mode 匹配?"])
# NO -> straight run
box(cx+250, 162, 230, 60, "#f1f5f9", "#94a3b8",
    ["否 → 直跑 runnable()", "（profile / warmup / 无图）"], fs=14)
arrow(cx+150, 192, cx+250, 192)
arrow(cx, 236, cx, 262)
label(cx-20, 256, "是", "#16a34a", 15, "end")

# bucket lookup
box(cx-200, 262, 400, 56, "#ede9fe", "#7c3aed",
    ["查 concrete_aclgraph_entries[batch_descriptor]"], fs=15, tcol="#6d28d9")
arrow(cx, 318, cx, 344)

# diamond2: entry.aclgraph is None?
diamond(cx, 388, 160, 46, ["entry.aclgraph", "is None ?"])

# LEFT branch: capture (first seen)
capx = 70
# L-route: horizontal from diamond to capture-column center, then down into box
L.append(f'<line x1="{cx-160}" y1="388" x2="{capx+200}" y2="388" stroke="#475569" stroke-width="2.2"/>')
arrow(capx+200, 388, capx+200, 450)
label(cx-250, 372, "是 / 首见此形状", "#b45309", 14, "middle")
box(capx, 450, 400, 150, "#fff7ed", "#ea580c",
    ["首见：捕获 (capture)",
     "validate_cudagraph_capturing_enabled()",
     "记 input_addresses = [x.data_ptr()…]",
     "aclgraph = torch.npu.NPUGraph()",
     "with torch.npu.graph(aclgraph, pool):",
     "    output = runnable(*args)"], fs=14, tcol="#9a3412")
# 207008 callout
box(capx, 624, 400, 76, "#fef2f2", "#dc2626",
    ["except RuntimeError → 命中 207008 ?",
     "_is_stream_resource_capture_error →",
     "改写成带调参指引的 RuntimeError"], fs=13, tcol="#b91c1c")
arrow(capx+200, 600, capx+200, 624, col="#dc2626", mk="r")
# capture done -> weak ref store
label(capx+200, 720, "捕获成功 → weak_ref 存 entry，本次返回真 output", "#9a3412", 13.5, "middle")

# RIGHT branch: replay (already captured)
repx = w - 70 - 400
L.append(f'<line x1="{cx+160}" y1="388" x2="{repx+200}" y2="388" stroke="#475569" stroke-width="2.2"/>')
arrow(repx+200, 388, repx+200, 450)
label(cx+250, 372, "否 / 已捕获", "#16a34a", 14, "middle")
box(repx, 450, 400, 150, "#f0fdf4", "#16a34a",
    ["已捕获：重放 (replay)",
     "[DEBUG] 断言 input 地址与捕获时一致",
     "FULL 模式 & 非 draft-eagle:",
     "    current_stream().synchronize()",
     "entry.aclgraph.replay()",
     "return entry.output（复用捕获池张量）"], fs=14, tcol="#15803d")

L.append('</svg>')
open("fig25-2-capture-replay.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote fig25-2-capture-replay.svg")
