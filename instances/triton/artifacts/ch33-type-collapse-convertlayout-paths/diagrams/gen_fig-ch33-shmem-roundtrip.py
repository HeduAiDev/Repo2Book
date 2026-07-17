#!/usr/bin/env python3
"""fig-ch33-shmem-roundtrip: swimlane——共享内存往返 2 次迭代:store->barrier->load,
每迭代 store/load 各 2 op(4 元素/inVec=2),迭代间界起再插一次 barrier,累计 3 次 barrier。
transferWithinBlockImpl 主循环:ConvertLayoutOpToLLVM.cpp:L606-L644。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["各线程(寄存器)", "Shared Memory scratch"]
LANE_W = 560
TOP = 100
STEP = 62
PAD = 40
LANE_HDR_HALF = 95  # 泳道头矩形半宽,需覆盖最长标签 "Shared Memory scratch"
X = {LANES[0]: PAD + 110, LANES[1]: PAD + 110 + LANE_W}
w = X[LANES[1]] + LANE_HDR_HALF + PAD

# 事件序列: (kind, label, iter_label) —— iter_label 非 None 时在该行左侧标注迭代号
EVENTS = [
    ("store", "store x2 op(4 元素,inVec=2)", "迭代 0"),
    ("barrier", "barrier(bar.sync) — 累计 1", None),
    ("load", "load x2 op(4 元素,outVec=2)", None),
    ("barrier", "barrier(迭代 1 开头,i!=0) — 累计 2", "迭代 1"),
    ("store", "store x2 op(4 元素,inVec=2)", None),
    ("barrier", "barrier(bar.sync) — 累计 3", None),
    ("load", "load x2 op(4 元素,outVec=2)", None),
]
h = TOP + STEP * (len(EVENTS) + 2) + PAD + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="17" font-weight="bold" '
     'fill="#0f172a">共享内存往返:store-&gt;barrier-&gt;load(最贵路径)</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     '构造示例:inVals=8,iterations=2,inVec=outVec=2(ConvertLayoutOpToLLVM.cpp:L606-L644)</text>']

for name, x in X.items():
    L.append(f'<rect x="{x-LANE_HDR_HALF}" y="{TOP-40}" width="{2*LANE_HDR_HALF}" height="28" '
              'rx="6" fill="#e2e8f0" stroke="#64748b"/>')
    L.append(f'<text x="{x}" y="{TOP-21}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{TOP-10}" x2="{x}" y2="{h-PAD-40}" '
              'stroke="#94a3b8" stroke-dasharray="4,4"/>')

y = TOP + 30
for i, (kind, label, iter_label) in enumerate(EVENTS):
    if iter_label:
        L.append(f'<text x="{10}" y="{y+4}" font-family="sans-serif" font-size="11" '
                  f'font-weight="bold" fill="#64748b">{esc(iter_label)}</text>')
    if kind == "store":
        x1, x2 = X[LANES[0]], X[LANES[1]]
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#b45309" '
                  'stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#b45309">{esc(label)}</text>')
    elif kind == "load":
        x1, x2 = X[LANES[1]], X[LANES[0]]
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#0369a1" '
                  'stroke-width="2" marker-end="url(#a)"/>')
        L.append(f'<text x="{(x1+x2)/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#0369a1">{esc(label)}</text>')
    else:  # barrier: 全宽虚线红色
        L.append(f'<line x1="{X[LANES[0]]-40}" y1="{y}" x2="{X[LANES[1]]+40}" y2="{y}" '
                  'stroke="#b91c1c" stroke-width="2" stroke-dasharray="6,4"/>')
        L.append(f'<text x="{(X[LANES[0]]+X[LANES[1]])/2}" y="{y-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#b91c1c">{esc(label)}</text>')
    y += STEP

# padding 标注在 shared memory 泳道底部(含 bank 数,呼应 spec.numbers)
py = y + 10
L.append(f'<rect x="{X[LANES[1]]-125}" y="{py}" width="250" height="64" rx="8" '
          'fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.2"/>')
L.append(f'<text x="{X[LANES[1]]}" y="{py+19}" text-anchor="middle" font-family="sans-serif" '
          'font-size="11" font-weight="bold" fill="#5b21b6">padding=2(防 bank 冲突)</text>')
L.append(f'<text x="{X[LANES[1]]}" y="{py+36}" text-anchor="middle" font-family="sans-serif" '
          'font-size="10" fill="#5b21b6">outOrd[0] 加 max(inVec,outVec)</text>')
L.append(f'<text x="{X[LANES[1]]}" y="{py+53}" text-anchor="middle" font-family="sans-serif" '
          'font-size="10" font-weight="bold" fill="#5b21b6">Shared Memory:32 个 bank,每 bank 4 字节</text>')

foot_y = py + 64 + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          'font-weight="bold" fill="#0f172a">合计:8 元素 store + 8 元素 load + 3 次 barrier '
          '(2xiterations-1)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          'fill="#64748b">store 端地址=srcLayout.invertAndCompose(sharedLayout)(L514)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+38}" font-family="sans-serif" font-size="11" '
          'fill="#64748b">load 端地址=dstLayout.invertAndCompose(sharedLayout)(L523)</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch33-shmem-roundtrip.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
