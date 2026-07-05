#!/usr/bin/env python3
"""fig31-6-decode-vs-prefill: 同一份 KV cache,decode 走吸收路径、prefill 走物化
路径;swimlane 骨架,decode/prefill 两泳道各自与"KV Cache"泳道交互,末尾汇合验证
输出恒等。落地锚点 vllm_ascend/attention/mla_v1.py forward() L1718。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["Decode(吸收路径)", "KV Cache(576 元素/token)", "Prefill(物化路径)"]
EVENTS = [
    ("Decode(吸收路径)", "KV Cache(576 元素/token)", "ql_nope = q_nope @ W_UK_T(落潜空间)"),
    ("KV Cache(576 元素/token)", "Decode(吸收路径)", "对缓存 c_kv 直接打分(不物化 k)"),
    ("Decode(吸收路径)", "Decode(吸收路径)", "_v_up_proj 乘 W_UV 还原输出"),
    ("Prefill(物化路径)", "KV Cache(576 元素/token)", "kv_b_proj 物化 full k_nope/value"),
    ("KV Cache(576 元素/token)", "Prefill(物化路径)", "标准注意力(全量物化 k/v)"),
]
LANE_W, TOP, STEP, PAD = 320, 90, 62, 40
w = PAD*2 + LANE_W*(len(LANES)-1) + 260
h = TOP + STEP*(len(EVENTS)+1) + 150
X = {name: PAD + 130 + i*LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("同一套权重、同一份缓存:decode 走吸收,prefill 走物化")}</text>']

colors = {"Decode(吸收路径)": "#1d4ed8", "KV Cache(576 元素/token)": "#0f172a",
          "Prefill(物化路径)": "#b45309"}
for name, x in X.items():
    c = colors[name]
    L.append(f'<rect x="{x-110}" y="{TOP-42}" width="220" height="30" rx="6" '
             f'fill="{c}"/>')
    L.append(f'<text x="{x}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="white">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-8}" x2="{x}" y2="{h-150}" '
             f'stroke="#94a3b8" stroke-dasharray="4,4"/>')

self_loop_dx = 70
for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP*(i+1)
    x1, x2 = X[src], X[dst]
    if src == dst:
        # self-loop for "Decode does its own final up-projection"
        lx = x1 + self_loop_dx
        L.append(f'<path d="M {x1} {y} C {lx} {y-14}, {lx} {y+14}, {x1} {y+2}" '
                 'fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{x1+22}" y="{y-16}" text-anchor="start" font-family="sans-serif" '
                 f'font-size="11.5" fill="#334155">{esc(label)}</text>')
    else:
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
                 'stroke-width="1.5" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{PAD-6}" y="{y+4}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="11" fill="#64748b">t{i+1}</text>')

foot_top = TOP + STEP*(len(EVENTS)+1) + 20
foot_w = w - 2*PAD
L.append(f'<rect x="{PAD}" y="{foot_top}" width="{foot_w}" height="96" rx="10" '
         'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{PAD+foot_w/2}" y="{foot_top+32}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#92400e">'
         f'{esc("两路径共享同一份 576 元素/token 缓存,输出逐位相等——最大绝对差 0.0")}</text>')
L.append(f'<text x="{PAD+foot_w/2}" y="{foot_top+62}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="12.5" fill="#92400e">'
         f'{esc("落地锚点:vllm_ascend/attention/mla_v1.py forward()(L1718)按 decode/prefill 分派——回指第 20 章")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig31-6-decode-vs-prefill.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
