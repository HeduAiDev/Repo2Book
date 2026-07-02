#!/usr/bin/env python3
"""swimlane 模板:跨组件时序协议。示例:调度器与 Worker 的一步 RPC 往返。
用法:python3 example-swimlane.py  → 同目录 example-swimlane.svg
改造点:LANES(泳道)与 EVENTS(时刻,from,to,标签)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Scheduler", "Worker0", "Worker1"]
EVENTS = [  # (from_lane, to_lane, label) 按时间序
    ("Scheduler", "Worker0", "execute_model(batch=8)"),
    ("Scheduler", "Worker1", "execute_model(batch=8)"),
    ("Worker0", "Scheduler", "sampled_ids[8]"),
    ("Worker1", "Scheduler", "sampled_ids[8]"),
]
LANE_W, TOP, STEP, PAD = 220, 70, 60, 40
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 120
h = TOP + STEP * (len(EVENTS) + 1) + PAD
X = {name: PAD + 60 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
for name, x in X.items():  # 泳道头 + 生命线
    L.append(f'<rect x="{x-55}" y="{TOP-40}" width="110" height="28" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-10}" x2="{x}" y2="{h-PAD}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')
for i, (src, dst, label) in enumerate(EVENTS):  # 消息箭头:端点取自生命线 x
    y = TOP + STEP * (i + 1)
    x1, x2 = X[src], X[dst]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
             'stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(x1+x2)/2}" y="{y-7}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{PAD-14}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')
L.append('</svg>')
out = Path(__file__).with_name("example-swimlane.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
