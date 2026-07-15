#!/usr/bin/env python3
"""fig-m10-seams: 双语栈接缝图 —— 把 Python<->C++/MLIR 的四道接缝按一生的时间轴
排成泳道图。lanes=Python/C++,MLIR(libtriton.so)/LLVM/ptxas(独立子进程)/
CUDA driver(launcher)。数字取自 explainer figure_specs['fig-m10-seams'].numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


LANES = [
    ("Python", "JITFunction / compile() 编排"),
    ("C++ / MLIR", "libtriton.so"),
    ("LLVM", ""),
    ("ptxas", "独立子进程"),
    ("CUDA driver", "launcher"),
]

# (from_lane_idx, to_lane_idx, 主标签, [附加证据行...])
EVENTS = [
    (0, 1, "pybind 进 libtriton.so · compiler.py:L273",
     ["追踪（make_ir）与全部 pass 都在这跑", "首个 pass=add_inliner · compiler.py:L191"]),
    (1, 2, "to_module 跨 MLIR → LLVM 世界 · compiler.py:L291",
     ["llir 实测 150 行（跨 LLVM 后的证据）"]),
    (2, 3, "起独立子进程（进程边界）· compiler.py:L341", []),
    (3, 4, "CudaLauncher 现场编译 C launcher · driver.py:L439", []),
]

LANE_W, TOP, PAD = 280, 96, 44
STEP = 130
w = PAD * 2 + LANE_W * (len(LANES) - 1) + 200
h = TOP + STEP * len(EVENTS) + 190

X = {i: PAD + 100 + i * LANE_W for i in range(len(LANES))}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="28" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">双语栈的四道接缝：一生时间轴上控制权去哪了</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("Python 只是编排者；ch01 已讲透其中三道的内部机制，本图只按时间轴排成一列、标各归后续哪一章")}</text>']

# 泳道头 + 生命线
for i, (name, sub) in enumerate(LANES):
    x = X[i]
    L.append(f'<rect x="{x-90}" y="{TOP-46}" width="180" height="{40 if sub else 28}" rx="6" '
              'fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-26 if sub else TOP-27}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
              f'fill="#0f172a">{esc(name)}</text>')
    if sub:
        L.append(f'<text x="{x}" y="{TOP-10}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" fill="#475569">{esc(sub)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-4}" x2="{x}" y2="{h-70}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

# 事件：每条按时间往下排，箭头从 from 泳道 -> to 泳道
for i, (fi, ti, label, evid) in enumerate(EVENTS):
    y = TOP + 30 + i * STEP
    x1, x2 = X[fi], X[ti]
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#1d4ed8" '
              'stroke-width="2" marker-end="url(#a)"/>')
    L.append(f'<circle cx="{x1}" cy="{y}" r="4" fill="#1d4ed8"/>')
    L.append(f'<text x="{(x1+x2)/2}" y="{y-10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#1d4ed8">{esc(f"{chr(0x2460+i)} " + label)}</text>')
    for k, ev in enumerate(evid):
        L.append(f'<text x="{(x1+x2)/2}" y="{y+16+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#475569">{esc(ev)}</text>')

foot_y = h - 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("① ② ③ ④ 按时间序：Python 只在两端露面，中段全在 C++/MLIR/LLVM/独立进程里跑")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("① ④ 的内部机制 ch01 已讲透（三条双语接缝），不重讲；② ③ 是本图新增的两处")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m10-seams.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
