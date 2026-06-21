#!/usr/bin/env python3
"""03-rpc-correlation-flow: call_utility RPC 时序（correlation-id 在单向流上配对）。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

w, h = 940, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def txt(x, y, s, size=13, anchor="middle", fill="#1e293b", weight="normal"):
    L.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

# 三条生命线
lanes = [(160, "client 调用方", "#1d4ed8"), (480, "ZMQ ROUTER↔DEALER", "#94a3b8"), (790, "engine busy loop", "#15803d")]
for lx, name, col in lanes:
    L.append(f'<rect x="{lx-110}" y="30" width="220" height="40" rx="8" fill="white" stroke="{col}" stroke-width="2"/>')
    txt(lx, 56, name, 14, weight="bold", fill=col)
    L.append(f'<line x1="{lx}" y1="70" x2="{lx}" y2="520" stroke="{col}" stroke-width="1.5" stroke-dasharray="4 4"/>')

def step(y, x1, x2, label, sub=None):
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>')
    mid = (x1 + x2) / 2
    txt(mid, y - 8, label, 12, weight="bold")
    if sub:
        txt(mid, y + 16, sub, 11, fill="#64748b")

def note(x, y, lines, fill="#fef9c3", stroke="#ca8a04"):
    bw = 240
    bh = 22 * len(lines) + 16
    L.append(f'<rect x="{x-bw/2}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    for i, ln in enumerate(lines):
        txt(x, y + 22 + i * 22, ln, 11.5, fill="#1e293b")

note(160, 90, ["call_id = uuid1().int >> 64",
               "future = Future()",
               "utility_results[call_id] = future"])
step(200, 160, 790, "① send UTILITY 帧", "(client_index, call_id, method, args)")
note(790, 230, ["_handle_client_request 分派 UTILITY",
                "method = getattr(self, method_name)",
                "result = method(*conv_args)",
                "output = UtilityOutput(call_id, result)"], "#dcfce7", "#15803d")
step(360, 790, 160, "② 回发 UtilityOutput", "(call_id, result) 经 PUSH→PULL")
note(160, 390, ["_process_utility_output(output)",
                "future = utility_results.pop(call_id)",
                "future.set_result(...)  → 唤醒 await"], "#dbeafe", "#2563eb")
txt(470, 500, "单向 ZMQ 流没有内建 request-response：靠唯一 call_id 把发出的 Future 与回来的输出配对（correlation-id 模式）",
    12.5, fill="#7e22ce", weight="bold")

L.append('</svg>')
open("03-rpc-correlation-flow.svg", "w").write('\n'.join(L))
print("wrote 03-rpc-correlation-flow.svg")
