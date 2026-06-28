#!/usr/bin/env python3
"""Sparse-C8 把 MLA 的 3 元组 KV 扩成 4 元组：int8 省显存 + fp16 scale 补偿。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 540
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="42" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">Sparse-C8：MLA 的 KV 从 3 元组扩成 4 元组</text>')
L.append(f'<text x="{W/2}" y="70" font-family="sans-serif" font-size="14" fill="#64748b" text-anchor="middle">page_size_bytes 重算的根因：indexer key 降精度省显存，再补一条量化 scale</text>')


def tuple_row(y, label, items, total_note):
    L.append(f'<text x="40" y="{y-14}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    bx = 40
    bw = 250
    gap = 14
    bh = 92
    for it in items:
        name, dtype, col, bg, store = it
        L.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{bh}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2"/>')
        L.append(f'<text x="{bx+bw/2}" y="{y+34}" font-family="monospace" font-size="15" font-weight="bold" fill="{col}" text-anchor="middle">{esc(name)}</text>')
        L.append(f'<text x="{bx+bw/2}" y="{y+62}" font-family="sans-serif" font-size="13" fill="#334155" text-anchor="middle">存：{esc(it[4])}</text>')
        # dtype chip
        L.append(f'<rect x="{bx+bw/2-46}" y="{y+72}" width="92" height="22" rx="11" fill="{col}"/>')
        L.append(f'<text x="{bx+bw/2}" y="{y+87}" font-family="monospace" font-size="12" font-weight="bold" fill="white" text-anchor="middle">{esc(dtype)}</text>')
        bx += bw + gap
    L.append(f'<text x="{bx+16}" y="{y+bh/2+5}" font-family="sans-serif" font-size="13" fill="#475569">{esc(total_note)}</text>')
    return bx


BF = "#2563eb"
BFG = "#eff6ff"
INT8 = "#dc2626"
INT8G = "#fef2f2"
SCALE = "#059669"
SCALEG = "#ecfdf5"

# 原 3 元组
y1 = 130
tuple_row(
    y1, "原 MLA（3 元组，全 bf16）",
    [
        ("kv_cache[0]", "bfloat16", BF, BFG, "kv_lora", ),
        ("kv_cache[1]", "bfloat16", BF, BFG, "k_rope", ),
        ("kv_cache[2]", "bfloat16", BF, BFG, "indexer key", ),
    ],
    "",
)
# 修正 store 文本（上面占位）
# 重新写 store 标签需要传第 5 项——已在 items[4]

# 箭头
midx = W/2
L.append(f'<defs><marker id="ar" markerWidth="11" markerHeight="11" refX="8" refY="4" orient="auto"><path d="M0,0 L9,4 L0,8 z" fill="#7c3aed"/></marker></defs>')
ay = 252
L.append(f'<line x1="{midx}" y1="{ay}" x2="{midx}" y2="{ay+40}" stroke="#7c3aed" stroke-width="2.5" marker-end="url(#ar)"/>')
L.append(f'<rect x="{midx-230}" y="{ay+6}" width="460" height="28" rx="14" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.2"/>')
L.append(f'<text x="{midx}" y="{ay+25}" font-family="sans-serif" font-size="13.5" fill="#6d28d9" text-anchor="middle">cache_sparse_c8 = True：indexer key 转 int8 + 补 fp16 scale</text>')

# Sparse-C8 4 元组
y2 = 360
tuple_row(
    y2, "Sparse-C8（4 元组）",
    [
        ("kv_cache[0]", "bfloat16", BF, BFG, "kv_lora"),
        ("kv_cache[1]", "bfloat16", BF, BFG, "k_rope"),
        ("kv_cache[2]", "int8", INT8, INT8G, "indexer key"),
        ("kv_cache[3]", "float16", SCALE, SCALEG, "key scale"),
    ],
    "",
)

# bottom note
ny = 470
L.append(f'<rect x="40" y="{ny}" width="{W-80}" height="56" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="60" y="{ny+24}" font-family="sans-serif" font-size="13" fill="#334155">kv_cache[2] 从 bf16 降到 int8：省一半字节；新增 kv_cache[3] 存量化 scale 作补偿。</text>')
L.append(f'<text x="60" y="{ny+44}" font-family="sans-serif" font-size="13" fill="#334155">A5 设备把 kv_lora+k_rope 合并成单 CKV（fp8），A3 分开存（bf16）——同一公式按 qk_rope_head_dim==0 分流。</text>')

L.append('</svg>')
open("sparse_c8.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote sparse_c8.svg")
