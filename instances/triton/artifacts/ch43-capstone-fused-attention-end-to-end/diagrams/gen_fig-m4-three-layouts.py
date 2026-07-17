#!/usr/bin/env python3
"""fig-m4-three-layouts: layout 模板。
TTGIR 里第一次出现的三种布局(#blocked/#mma/#shared)+ dot_op 转换。
数字全部来自 _attn_fwd.ttgir(spec.numbers)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "convert_to_ttgpuir 派生的三种布局(_attn_fwd)"
SUBTITLE = "模块属性:ttg.num-warps=4, threads-per-warp=32, num-ctas=1, target=cuda:120(_attn_fwd.ttgir:L46)"

CARDS = [
    ("#blocked", "访存布局", "#93c5fd", "#1d4ed8",
     ["sizePerThread=[8,1]", "threadsPerWarp=[8,4]", "warpsPerCTA=[1,4]", "order=[0,1]"],
     "L1"),
    ("#mma", "Tensor Core 布局", "#c4b5fd", "#6d28d9",
     ["nvidia_mma versionMajor=2", "warpsPerCTA=[4,1]", "instrShape=[16,8]"],
     "L11"),
    ("#shared", "共享内存布局", "#86efac", "#15803d",
     ["swizzled_shared vec=8", "perPhase=1, maxPhase=8"],
     "L12"),
]

CARD_W, CARD_H, GAP, PAD, TOP = 300, 158, 40, 40, 132
w = PAD * 2 + 3 * CARD_W + 2 * GAP
h = TOP + CARD_H + 260

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#6366f1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

card_x = [PAD + i * (CARD_W + GAP) for i in range(3)]
for i, (name, role, fill, stroke, params, loc) in enumerate(CARDS):
    x = card_x[i]
    L.append(f'<rect x="{x}" y="{TOP}" width="{CARD_W}" height="{CARD_H}" rx="10" '
              f'fill="{fill}" fill-opacity="0.22" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x+CARD_W/2}" y="{TOP+30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="18" font-weight="bold" fill="{stroke}">{esc(name)}</text>')
    L.append(f'<text x="{x+CARD_W/2}" y="{TOP+50}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(role)} · {esc(loc)}</text>')
    for k, p in enumerate(params):
        L.append(f'<text x="{x+CARD_W/2}" y="{TOP+74+k*20}" text-anchor="middle" '
                  f'font-family="monospace" font-size="12" fill="#0f172a">{esc(p)}</text>')

# dot_op 转换区
dot_y = TOP + CARD_H + 56
L.append(f'<text x="{PAD}" y="{dot_y}" font-family="sans-serif" font-size="13.5" '
          f'font-weight="bold" fill="#0f172a">{esc("tt.dot 操作数戴上 dot_op<parent=#mma>")}</text>')

box_w, box_h = 300, 50
b1x, b2x = PAD, PAD + box_w + 220
by = dot_y + 24
L.append(f'<rect x="{b1x}" y="{by}" width="{box_w}" height="{box_h}" rx="8" '
          'fill="#ede9fe" stroke="#6d28d9" stroke-width="1.5"/>')
L.append(f'<text x="{b1x+box_w/2}" y="{by+21}" text-anchor="middle" font-family="monospace" '
          f'font-size="11.5" fill="#4c1d95">dot_op&lt;opIdx=0,parent=#mma,kWidth=2&gt;</text>')
L.append(f'<text x="{b1x+box_w/2}" y="{by+38}" text-anchor="middle" font-family="monospace" '
          f'font-size="11.5" fill="#4c1d95">dot_op&lt;opIdx=1,parent=#mma,kWidth=2&gt;</text>')
L.append(f'<text x="{b1x+box_w/2}" y="{by+box_h+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">L172,L173,L174</text>')

L.append(f'<rect x="{b2x}" y="{by}" width="{box_w}" height="{box_h}" rx="8" '
          'fill="#fce7f3" stroke="#be185d" stroke-width="1.5"/>')
L.append(f'<text x="{b2x+box_w/2}" y="{by+21}" text-anchor="middle" font-family="monospace" '
          f'font-size="11" fill="#831843">P:tensor&lt;128x64xf16,#mma&gt;</text>')
L.append(f'<text x="{b2x+box_w/2}" y="{by+38}" text-anchor="middle" font-family="monospace" '
          f'font-size="11" fill="#831843">→ dot_op&lt;opIdx=0,parent=#mma&gt;</text>')
L.append(f'<text x="{b2x+box_w/2}" y="{by+box_h+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">ttg.convert_layout · L202</text>')

amx = b1x + box_w + 8
amx2 = b2x - 8
amy = by + box_h / 2
L.append(f'<line x1="{amx}" y1="{amy}" x2="{amx2}" y2="{amy}" '
          'stroke="#6366f1" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(amx+amx2)/2}" y="{amy-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#6366f1">P 喂第二个 dot 前</text>')

foot_lines = [
    "结论:TTGIR = TTIR + 布局注解。同一个 tt.dot,结果类型从裸 tensor 变成 tensor<...xf32,#mma>;",
    "#blocked 管全局访存怎么分给 4 个 warp,#shared 管数据在 SRAM 怎么 swizzle 避 bank 冲突;",
    "ttg.convert_layout 是布局之间的显式搬运(有真实开销)。",
]
foot_y0 = h - 20 - (len(foot_lines) - 1) * 18
for i, fl in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y0+i*18}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(fl)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-three-layouts.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
