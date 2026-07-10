#!/usr/bin/env python3
"""fig36-7-vllm-chain: swimlane 模板。vLLM v1 EAGLE propose() 的链式草稿——
泳道=输入构造/Draft Head 前向/argmax 输出,事件按时间序在泳道间跳转,
呈现"回喂上一步 token+特征"的链式结构(非树)。
数字来自 explainer.json fig36-7 numbers(traces/chain.json)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["输入构造", "Draft Head 前向", "argmax 输出"]
EVENTS = [
    ("输入构造", "Draft Head 前向", "首遍输入 [3,2,4]（左移+塞 next_token=4）"),
    ("Draft Head 前向", "argmax 输出", "step1 pred feature → argmax"),
    ("argmax 输出", "Draft Head 前向", "draft_token=3（c=0.354）回喂"),
    ("Draft Head 前向", "argmax 输出", "step2 pred feature → argmax"),
    ("argmax 输出", "Draft Head 前向", "draft_token=1（c=0.304）回喂"),
    ("Draft Head 前向", "argmax 输出", "step3 pred feature → argmax"),
    ("argmax 输出", "Draft Head 前向", "draft_token=3（c=0.592，链中最自信）回喂"),
    ("Draft Head 前向", "argmax 输出", "step4 pred feature → argmax → draft_token=0（c=0.259）"),
]
LANE_W, TOP, STEP, PAD = 420, 90, 52, 40
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 160
h = TOP + STEP * (len(EVENTS) + 1) + PAD + 50
X = {name: PAD + 80 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="{PAD-8}" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc("vLLM v1 EAGLE 的链式草稿（propose()）")}</text>']

for name, x in X.items():
    L.append(f'<rect x="{x-95}" y="{TOP-46}" width="190" height="30" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-26}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-16}" x2="{x}" y2="{h-PAD-40}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1)
    x1, x2 = X[src], X[dst]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
             'stroke-width="1.5" marker-end="url(#a)"/>')
    anchor = "middle"
    tx = (x1 + x2) / 2
    L.append(f'<text x="{tx}" y="{y-7}" text-anchor="{anchor}" '
             f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{PAD-8}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')

foot_y = h - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("每步只取 1 个 argmax token，回喂上一步 (token, feature) 继续；无兄弟分支、无树注意力掩码。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" fill="#94a3b8">'
          f'{esc("源码注释 dummy_run 仍留 FIXME: tree-based specdec——EAGLE-2 的树式扩展尚未接入这条路径。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig36-7-vllm-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
