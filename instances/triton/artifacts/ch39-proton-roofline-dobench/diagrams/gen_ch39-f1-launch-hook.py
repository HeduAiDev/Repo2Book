#!/usr/bin/env python3
"""ch39-f1-launch-hook: proton 零侵入钩子挂在发射路径上(swimlane 改)。
claim: 发射器在 cuLaunchKernel 前后各回调一次类级钩子,把 scope 挂到发射路径上,不改一行核代码。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["调用方\nkernel.run()", "launch_enter_hook\n(类级钩子槽位)", "cuLaunchKernel\n(发射,见第37章)", "launch_exit_hook\n(类级钩子槽位)"]
EVENTS = [
    ("调用方\nkernel.run()", "launch_enter_hook\n(类级钩子槽位)", "① enter_hook(metadata)"),
    ("launch_enter_hook\n(类级钩子槽位)", "cuLaunchKernel\n(发射,见第37章)", "放行:发射真正开始"),
    ("cuLaunchKernel\n(发射,见第37章)", "launch_exit_hook\n(类级钩子槽位)", "② exit_hook(metadata)"),
]
LANE_W, TOP, STEP, PAD = 250, 150, 68, 40
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 140
h = TOP + STEP * (len(EVENTS) + 1) + PAD + 30
X = {name: PAD + 70 + i * LANE_W for i, name in enumerate(LANES)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# top banner: one-time injection fact
L.append(f'<rect x="{PAD}" y="14" width="{w-2*PAD}" height="44" rx="8" fill="#fef9c3" stroke="#ca8a04"/>')
L.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#854d0e">register_triton_hook() 一次性动作:类级属性赋值,对全体已编译 kernel 立即生效</text>')
L.append(f'<text x="{w/2}" y="49" text-anchor="middle" font-family="sans-serif" font-size="11" '
         f'fill="#a16207">钩子槽位默认值 = None(python/triton/compiler/compiler.py:L343-L348) —— 未注册时零调用</text>')

for name, x in X.items():  # 泳道头 + 生命线
    lines = name.split("\n")
    L.append(f'<rect x="{x-90}" y="{TOP-46}" width="180" height="40" rx="6" '
             'fill="#e2e8f0" stroke="#64748b"/>')
    for k, line in enumerate(lines):
        L.append(f'<text x="{x}" y="{TOP-30+k*16}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="12" font-weight="bold" fill="#0f172a">{esc(line)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-6}" x2="{x}" y2="{h-PAD-20}" '
             'stroke="#94a3b8" stroke-dasharray="4,4"/>')

for i, (src, dst, label) in enumerate(EVENTS):
    y = TOP + STEP * (i + 1) - 20
    x1, x2 = X[src], X[dst]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#334155" '
             'stroke-width="2" marker-end="url(#a)"/>')
    mid = (x1 + x2) / 2
    L.append(f'<rect x="{mid-115}" y="{y-24}" width="230" height="20" rx="4" fill="white"/>')
    L.append(f'<text x="{mid}" y="{y-9}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#1d4ed8">{esc(label)}</text>')

foot_y = h - PAD + 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">每次发射恰好 2 次钩子回调(enter + exit)——发射器把 enter_hook/exit_hook '
         f'透传给 self.run(python/triton/compiler/compiler.py:L421-L423)。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f1-launch-hook.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
