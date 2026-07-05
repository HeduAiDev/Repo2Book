#!/usr/bin/env python3
"""ch22 压缩 KV 的 block 换算：逻辑 token 按 compress_ratio 压成物理 KV slot。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 620
TXT = "#1e293b"
SUB = "#64748b"
VLLM = "#0f766e"
ASC = "#7c3aed"
LOG = "#2563eb"   # 逻辑 token 蓝
PHY = "#b45309"   # 物理 block 橙

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 12 10" refX="11" refY="5" markerWidth="11" markerHeight="9" orient="auto"><path d="M0,0 L12,5 L0,10 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">压缩 MLA 的 block 换算：每 compress_ratio 个逻辑 token 压成 1 个物理 KV slot</text>')
L.append(f'<text x="{W/2}" y="66" text-anchor="middle" font-size="14" fill="{SUB}">示例 compress_ratio=4、block_size=2（每图块仅示意，真实值更大）</text>')

# 逻辑 token 行
ly = 120
L.append(f'<text x="60" y="{ly-16}" font-size="15" font-weight="bold" fill="{LOG}">逻辑 token 序列（共 16 个）</text>')
tw = 60
for i in range(16):
    x = 70 + i * tw
    L.append(f'<rect x="{x}" y="{ly}" width="{tw-6}" height="40" rx="5" fill="#eff6ff" stroke="{LOG}" stroke-width="1.3"/>')
    L.append(f'<text x="{x+(tw-6)/2}" y="{ly+26}" text-anchor="middle" font-size="13" fill="{LOG}">t{i}</text>')
# compress_ratio 分组括号
for g in range(4):
    x0 = 70 + g * 4 * tw
    x1 = x0 + 4 * tw - 6
    L.append(f'<path d="M{x0},{ly+50} L{x0},{ly+60} L{x1},{ly+60} L{x1},{ly+50}" fill="none" stroke="{LOG}" stroke-width="1.4"/>')
    L.append(f'<text x="{(x0+x1)/2}" y="{ly+78}" text-anchor="middle" font-size="12.5" fill="{SUB}">4 个一组</text>')

# 压缩箭头
L.append(f'<text x="{W/2}" y="248" text-anchor="middle" font-size="14" font-weight="bold" fill="{PHY}">// compress_ratio</text>')
sw = 160
for g in range(4):
    x0 = 70 + g * 4 * tw + (4 * tw - 6) / 2
    x1 = 140 + g * 220 + sw / 2
    L.append(f'<line x1="{x0}" y1="210" x2="{x1}" y2="270" stroke="#94a3b8" stroke-width="1.4" marker-end="url(#ar)"/>')

# 物理 KV slot 行（4 个）
py = 282
L.append(f'<text x="60" y="{py-8}" font-size="15" font-weight="bold" fill="{PHY}">物理 KV slot（16 // 4 = 4 个）</text>')
for i in range(4):
    x = 140 + i * 220
    L.append(f'<rect x="{x}" y="{py}" width="{sw}" height="44" rx="6" fill="#fff7ed" stroke="{PHY}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+sw/2}" y="{py+28}" text-anchor="middle" font-size="14" fill="{PHY}">slot {i}</text>')

# 物理 block 分组（block_size=2 → 每 2 slot 一个物理 block）
by = 362
for b in range(2):
    x0 = 140 + b * 2 * 220
    x1 = x0 + sw + 220
    L.append(f'<rect x="{x0-8}" y="{by}" width="{x1-x0+16}" height="46" rx="8" fill="none" stroke="{PHY}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    L.append(f'<text x="{(x0+x1)/2}" y="{by+29}" text-anchor="middle" font-size="13" fill="{PHY}">物理 block {b}（2 slot = 8 逻辑 token）</text>')
L.append(f'<text x="{W/2}" y="{by+72}" text-anchor="middle" font-size="13.5" fill="{SUB}">logical_block_size = block_size × compress_ratio = 2 × 4 = 8 逻辑 token —— 前缀命中按这个粒度对齐</text>')

# 公式条
fy = 470
L.append(f'<rect x="80" y="{fy}" width="1120" height="118" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="105" y="{fy+34}" font-size="14" fill="{TXT}"><tspan font-weight="bold" fill="{ASC}">CompressAttentionManager</tspan>：物理 block 数 = ceil( (num_tokens // compress_ratio) / block_size )</text>')
L.append(f'<text x="105" y="{fy+64}" font-size="14" fill="{TXT}"><tspan font-weight="bold" fill="{VLLM}">FullAttentionManager</tspan>（父类，不缩放）：物理 block 数 = ceil( num_tokens / block_size )</text>')
L.append(f'<text x="105" y="{fy+96}" font-size="13.5" fill="{SUB}">唯一差别：入口处 num_tokens //= compress_ratio。其余 block 管理算法逐字复用父类——这就是「只改换算比例」。</text>')

L.append('</svg>')
open("compress_scaling.svg", "w").write('\n'.join(L))
print("wrote compress_scaling.svg")
