#!/usr/bin/env python3
"""fig-m5-three-passes-each-touch: flow 模板(3 泳道)。
Coalesce / AccelerateMatmul / Pipeliner 各在这个核的 TTGIR 上留下一处可指认痕迹。
数字全部来自 _attn_fwd.ttgir + third_party/nvidia/backend/compiler.py(spec.numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "三个 TTGIR 优化 pass 在 _attn_fwd 上各留一处痕迹"

LANES = [
    ("Coalesce", "#0284c7", "#e0f2fe",
     "third_party/nvidia/backend/compiler.py:L220",
     "tt.load 访存模式(无合并提示)",
     "选出访存合并的 #blocked",
     "sizePerThread=[8,1]", "每线程连续搬 8 元素 → 带宽",
     "_attn_fwd.ttgir:L1"),
    ("AccelerateMatmul", "#7c3aed", "#ede9fe",
     "third_party/nvidia/backend/compiler.py:L227",
     "tt.dot(无布局)",
     "matmul → Tensor Core",
     "结果 tensor<128x64xf32,#mma>", "versionMajor=2, instrShape=[16,8] → 算力",
     "_attn_fwd.ttgir:L174"),
    ("Pipeliner", "#b45309", "#fef3c7",
     "third_party/nvidia/backend/compiler.py:L239",
     "K/V tt.load(逐块同步等待)",
     "软件流水:双缓冲 + 异步预取",
     "memdesc<2x64x64xf16> · async_copy_global_to_local",
     "num_stages=3,首维=2 → 延迟隐藏",
     "_attn_fwd.ttgir:L122,123,132,152"),
]

LANE_W, PAD, TOP = 420, 40, 130
GAP = 30
w = PAD * 2 + 3 * LANE_W + 2 * GAP
BOX_W = LANE_W
h = TOP + 430

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>']

lane_x = [PAD + i * (LANE_W + GAP) for i in range(3)]
for i, (name, stroke, fill, passloc, before, action, after1, after2, ttgirloc) in enumerate(LANES):
    x = lane_x[i]
    cx = x + LANE_W / 2
    y = TOP
    # pass 名称胶囊
    L.append(f'<rect x="{x}" y="{y}" width="{LANE_W}" height="34" rx="17" '
              f'fill="{stroke}"/>')
    L.append(f'<text x="{cx}" y="{y+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="white">{esc(name)}</text>')
    y2 = y + 34 + 12
    L.append(f'<text x="{cx}" y="{y2}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">{esc(passloc)}</text>')

    # 输入框
    by = y2 + 20
    bh = 56
    L.append(f'<rect x="{x+10}" y="{by}" width="{LANE_W-20}" height="{bh}" rx="8" '
              'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>')
    L.append(f'<text x="{cx}" y="{by+22}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#64748b">接收(pass 前)</text>')
    L.append(f'<text x="{cx}" y="{by+42}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" fill="#334155">{esc(before)}</text>')

    # 标签 + 箭头(标签在上,箭头单独一段在下,不重叠)
    ay1 = by + bh
    ay2 = ay1 + 46
    L.append(f'<text x="{cx}" y="{ay1+15}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" font-weight="bold" fill="{stroke}">{esc(action)}</text>')
    L.append(f'<line x1="{cx}" y1="{ay1+22}" x2="{cx}" y2="{ay2-4}" '
              f'stroke="{stroke}" stroke-width="2" marker-end="url(#a)"/>')

    # 输出框(高亮)
    oy = ay2
    oh = 92
    L.append(f'<rect x="{x+10}" y="{oy}" width="{LANE_W-20}" height="{oh}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{cx}" y="{oy+20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="bold" fill="{stroke}">产出(pass 后)</text>')
    L.append(f'<text x="{cx}" y="{oy+42}" text-anchor="middle" font-family="monospace" '
              f'font-size="11" fill="#0f172a">{esc(after1)}</text>')
    for k, ln in enumerate(after2.split(" → ")):
        pass
    L.append(f'<text x="{cx}" y="{oy+62}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#0f172a">{esc(after2)}</text>')
    L.append(f'<text x="{cx}" y="{oy+oh-12}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="9.5" fill="#64748b">{esc(ttgirloc)}</text>')

foot_lines = [
    "结论:同一段 TTGIR,三个 pass 各管一件事——Coalesce 让相邻线程读相邻地址(带宽),",
    "AccelerateMatmul 把矩阵乘换成 Tensor Core 指令(算力),Pipeliner 用双缓冲+异步拷贝",
    "把『搬下一块』和『算这一块』重叠(延迟隐藏)。这三处就是 part-6 三章在一个真核上的落点。",
]
foot_y0 = h - 20 - (len(foot_lines) - 1) * 18
for i, fl in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(fl)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m5-three-passes-each-touch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
