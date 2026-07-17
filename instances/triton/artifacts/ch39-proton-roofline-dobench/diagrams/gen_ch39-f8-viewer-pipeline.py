#!/usr/bin/env python3
"""ch39-f8-viewer-pipeline: viewer 主管线(flow)。
claim: hatchet json→调用树→inclusive 上卷→派生 flop/s·byte/s·util→过滤 metadata 桶→打印树。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def esc_lines(lines): return [esc(x) if not x.startswith("__RAW__") else x[7:] for x in lines]

STAGES = [
    (["hatchet json 输入", "third_party/proton/test/example_cuda.json"], "#f1f5f9", "#64748b", "#1e293b"),
    (["database.pop(1) 拆分", "[0]=调用树(喂 hatchet) / [1]=device_info"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["update_inclusive_columns", "子节点计数上卷到父 → ROOT time = 叶子之和"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["derive_metrics", "原始计数 → flop/s · byte/s · util"], "#e0f2fe", "#0369a1", "#0c4a6e"),
    (["filter_frames 过滤", "剔除 __proton_launch_metadata 桶"], "#fef3c7", "#b45309", "#78350f"),
    (["打印 hatchet 调用树"], "#dcfce7", "#15803d", "#14532d"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 300, 66, 46, 40, 130
ANNOT_COL_W = 380
w = PAD * 2 + BOX_W + ANNOT_COL_W
h = TOP + len(STAGES) * (BOX_H + GAP) - GAP + PAD + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#1e40af">roofline viewer 主管线</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'third_party/proton/proton/viewer.py —— hatchet 是第三方库,点到即止</text>']

x = PAD
y = TOP
box_ys = []
for i, (lines, fill, stroke, tc) in enumerate(STAGES):
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    y0 = y + BOX_H/2 - (n-1)*10 + 5
    for k, line in enumerate(lines):
        fs = 13 if k == 0 else 11.5
        wt = 'font-weight="bold" ' if k == 0 else ''
        L.append(f'<text x="{x+BOX_W/2}" y="{y0+k*19}" text-anchor="middle" font-family="sans-serif" '
                 f'font-size="{fs}" {wt}fill="{tc}">{esc(line)}</text>')
    box_ys.append(y)
    y += BOX_H + GAP

# arrows between consecutive stages, with annotation on right side for key ones
ANNOT = {
    0: "viewer.py:L31-L35 (database.pop(1))",
    1: "viewer.py:L198 (update_inclusive_columns)",
    3: "hook.py:L4 (桶名 __proton_launch_metadata)",
}
for i in range(len(STAGES) - 1):
    y1 = box_ys[i] + BOX_H
    y2 = box_ys[i+1]
    L.append(f'<line x1="{x+BOX_W/2}" y1="{y1}" x2="{x+BOX_W/2}" y2="{y2}" '
             f'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')

for i, note in ANNOT.items():
    ny = box_ys[i] + BOX_H + GAP/2
    L.append(f'<text x="{x+BOX_W+20}" y="{ny+4}" font-family="sans-serif" font-size="10.5" '
             f'fill="#94a3b8">{esc(note)}</text>')

foot_y = h - 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" fill="#64748b">'
         f'json 两半划分:[0] 调用树用于构建 hatchet GraphFrame,[1] device_info 提供峰值算力/带宽——'
         f'roofline 的分母就来自这里。</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch39-f8-viewer-pipeline.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
