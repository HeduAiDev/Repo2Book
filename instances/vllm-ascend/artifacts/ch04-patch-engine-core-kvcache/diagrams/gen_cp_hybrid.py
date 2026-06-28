#!/usr/bin/env python3
"""CP+hybrid 前缀缓存的解锁路径：去断言 + 有效块大小缩放。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append('<defs>'
         '<marker id="ag" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#0891b2"/></marker>'
         '</defs>')

L.append(f'<text x="{W/2}" y="40" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">解锁 CP + hybrid 前缀缓存：去断言 + 有效块大小缩放</text>')
L.append(f'<text x="{W/2}" y="68" font-family="sans-serif" font-size="14" fill="#64748b" text-anchor="middle">原版禁止 hybrid 开 context parallel；昇腾把 block_size 乘上 CP 因子，重新对齐命中长度</text>')

# 上半：被划掉的原版门
gx, gy, gw, gh = 60, 100, 460, 110
L.append(f'<rect x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" fill="#fef2f2" stroke="#dc2626" stroke-width="2"/>')
L.append(f'<text x="{gx+gw/2}" y="{gy+30}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#b91c1c" text-anchor="middle">vLLM 基座：直接拒绝</text>')
L.append(f'<text x="{gx+24}" y="{gy+58}" font-family="monospace" font-size="13" fill="#7f1d1d">assert dcp_world_size == 1</text>')
L.append(f'<text x="{gx+24}" y="{gy+80}" font-family="monospace" font-size="13" fill="#7f1d1d">assert pcp_world_size == 1</text>')
L.append(f'<text x="{gx+24}" y="{gy+100}" font-family="monospace" font-size="12.5" fill="#7f1d1d">resolve(): 多组+CP &#62;1 → raise ValueError</text>')
# 划掉
L.append(f'<line x1="{gx+10}" y1="{gy+10}" x2="{gx+gw-10}" y2="{gy+gh-10}" stroke="#dc2626" stroke-width="2.5" stroke-opacity="0.55"/>')
L.append(f'<line x1="{gx+gw-10}" y1="{gy+10}" x2="{gx+10}" y2="{gy+gh-10}" stroke="#dc2626" stroke-width="2.5" stroke-opacity="0.55"/>')

# 右侧昇腾说明
nx = gx + gw + 40
L.append(f'<rect x="{nx}" y="{gy}" width="560" height="{gh}" rx="10" fill="#ecfeff" stroke="#0891b2" stroke-width="2"/>')
L.append(f'<text x="{nx+24}" y="{gy+30}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0e7490">昇腾：MLA / SWA-MLA 各自实现了 CP</text>')
L.append(f'<text x="{nx+24}" y="{gy+58}" font-family="sans-serif" font-size="13" fill="#155e75">去掉两条 assert，保存 dcp / pcp；</text>')
L.append(f'<text x="{nx+24}" y="{gy+80}" font-family="sans-serif" font-size="13" fill="#155e75">把 raise 换成 lcm / gcd 真实计算；</text>')
L.append(f'<text x="{nx+24}" y="{gy+100}" font-family="sans-serif" font-size="13" fill="#155e75">命中长度改按「有效块大小」对齐。</text>')

# 流水线：spec.block_size → ×CP → ×compress → effective → lcm → 命中对齐
fy = 290
boxes = [
    ("spec.block_size", "每类 attention\n各自的逻辑块", "#475569", "#f1f5f9"),
    ("× dcp × pcp", "CP 因子\n(dcp_world×pcp_world)", "#0891b2", "#ecfeff"),
    ("× compress_ratio", "DeepseekV4 C128\n压缩放大", "#7c3aed", "#f5f3ff"),
    ("effective_block_size", "_get_effective\n_block_size()", "#0891b2", "#cffafe"),
    ("lcm(各类有效块)", "lcm_block_size\n命中长度的整除单位", "#0e7490", "#ecfeff"),
]
n = len(boxes)
bw = 196
gap = (W - 80 - n * bw) / (n - 1)
bx = 40
bh = 96
cx_list = []
for i, (head, sub, col, bg) in enumerate(boxes):
    L.append(f'<rect x="{bx}" y="{fy}" width="{bw}" height="{bh}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2"/>')
    L.append(f'<text x="{bx+bw/2}" y="{fy+34}" font-family="monospace" font-size="13.5" font-weight="bold" fill="{col}" text-anchor="middle">{esc(head)}</text>')
    for j, sl in enumerate(sub.split("\n")):
        L.append(f'<text x="{bx+bw/2}" y="{fy+58+j*19}" font-family="sans-serif" font-size="12" fill="#475569" text-anchor="middle">{esc(sl)}</text>')
    cx_list.append((bx, bx + bw))
    bx += bw + gap

for i in range(n - 1):
    x1 = cx_list[i][1]
    x2 = cx_list[i + 1][0]
    L.append(f'<line x1="{x1}" y1="{fy+bh/2}" x2="{x2-2}" y2="{fy+bh/2}" stroke="#0891b2" stroke-width="2.5" marker-end="url(#ag)"/>')

L.append(f'<text x="60" y="{fy-22}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">有效块大小怎么算（_get_effective_block_size）</text>')

# 底部：scheduler vs hash
sy = 440
L.append(f'<rect x="40" y="{sy}" width="540" height="92" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="60" y="{sy+28}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">scheduler_block_size</text>')
L.append(f'<text x="60" y="{sy+52}" font-family="monospace" font-size="13" fill="#334155">= lcm(group_block_sizes) × dcp × pcp</text>')
L.append(f'<text x="60" y="{sy+74}" font-family="sans-serif" font-size="12.5" fill="#64748b">调度器对 num_computed_tokens 取整的不变量（CP 因子只乘进这里）</text>')

L.append(f'<rect x="600" y="{sy}" width="540" height="92" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="620" y="{sy+28}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a">hash_block_size</text>')
L.append(f'<text x="620" y="{sy+52}" font-family="monospace" font-size="13" fill="#334155">= gcd(group_block_sizes)</text>')
L.append(f'<text x="620" y="{sy+74}" font-family="sans-serif" font-size="12.5" fill="#64748b">算 block hash 的最细粒度，每组 block_size 必被它整除</text>')

L.append('</svg>')
open("cp_hybrid.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote cp_hybrid.svg")
