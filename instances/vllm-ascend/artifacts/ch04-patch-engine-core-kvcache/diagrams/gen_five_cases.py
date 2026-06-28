#!/usr/bin/env python3
"""本章 5 个 KV-cache patch 案例总览：原算法 → 昇腾约束 → 重绑定手法。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W = 1480
margin = 24
# 列宽
cols = [
    ("案例", 250),
    ("原算法（vLLM 基座）", 400),
    ("昇腾约束", 360),
    ("重绑定手法（复用第 3 章技法）", 422),
]
xstarts = [margin]
for _, w in cols:
    xstarts.append(xstarts[-1] + w)

rows = [
    {
        "case": ["① block_size", "16 → 128"],
        "orig": ["kernel block 取 backend 支持", "的最小值（CUDA mamba2 = 16）；", "block_size 对齐推迟到执行期"],
        "constraint": ["NPU mamba kernel 不支持 16；", "硬件要求所有 cache tensor 连续"],
        "tech": ["技法③ 方法替换", "整体顶替 verify_and_update_config", "把对齐数硬钉 128 并前移到构图期"],
        "col": "#2563eb",
    },
    {
        "case": ["② MLAAttentionSpec", "子类化"],
        "orig": ["page_size_bytes 走父类；", "MLA 的 KV 是 3 元组、全 bf16"],
        "constraint": ["DSA / Sparse-C8 把 indexer key", "从 bf16 降 int8 + 存 fp16 scale", "→ KV 变 4 元组（A5/A3 dtype 不同）"],
        "tech": ["技法① 整类替换", "AscendMLAAttentionSpec 重写", "page_size_bytes；技法⑤ 三处别名重绑"],
        "col": "#7c3aed",
    },
    {
        "case": ["③ CP + hybrid", "前缀缓存"],
        "orig": ["assert dcp==1 / pcp==1；", "多组 block size + CP>1 直接", "raise ValueError（PR #40860）"],
        "constraint": ["昇腾对 MLA / SWA-MLA 层", "各自独立实现 context parallel"],
        "tech": ["技法① 去断言 + _effective_block_size", "技法②/③ 工厂 + resolve 改 lcm/gcd", "技法⑤ 补绑 from-import 旧引用"],
        "col": "#0891b2",
    },
    {
        "case": ["④ bind_kv_cache", "跳 NPU raise"],
        "orig": ["同 layer_index 多 layer_name 时,", "非 CUDA/CPU/XPU 平台走", "else: raise NotImplementedError"],
        "constraint": ["NPU 初始化 KV cache 会撞上", "这条 raise（临时补丁，带 TODO）"],
        "tech": ["技法③ 方法替换", "重绑 bind_kv_cache，每 index", "只取 layer_names[0]，绕过 raise"],
        "col": "#d97706",
    },
    {
        "case": ["⑤ int32", "slot_mapping"],
        "orig": ["slot_mapping 默认 int64", "（CUDA reshape_and_cache 接受）"],
        "constraint": ["vllm-ascend 的 reshape_and_cache", "算子要求 slot_mapping 为 int32"],
        "tech": ["子类覆盖（非 patch）", "del 父类张量后以 int32 重建", "立即 gc 不浪费显存"],
        "col": "#dc2626",
    },
]

header_h = 50
pad = 14
line_h = 21
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 0">']  # height patched later

body = []
body.append(f'<rect x="0" y="0" width="{W}" height="100%" fill="white"/>')
title_y = 40
body.append(f'<text x="{W/2}" y="{title_y}" font-family="sans-serif" font-size="25" font-weight="bold" fill="#0f172a" text-anchor="middle">本章 5 个 KV-cache patch 案例：原算法 → 昇腾约束 → 重绑定手法</text>')
sub_y = title_y + 26
body.append(f'<text x="{W/2}" y="{sub_y}" font-family="sans-serif" font-size="14" fill="#64748b" text-anchor="middle">每个案例都对照 vLLM 基座，点名复用第 3 章的 5 类重绑技法（本章不再重讲技法本身）</text>')

table_top = sub_y + 28
# header row
hy = table_top
for i, (name, w) in enumerate(cols):
    x = xstarts[i]
    body.append(f'<rect x="{x}" y="{hy}" width="{w}" height="{header_h}" fill="#1e293b"/>')
    body.append(f'<text x="{x+w/2}" y="{hy+header_h/2+5}" font-family="sans-serif" font-size="15" font-weight="bold" fill="white" text-anchor="middle">{esc(name)}</text>')

y = hy + header_h
for r in rows:
    cells = [r["case"], r["orig"], r["constraint"], r["tech"]]
    nlines = max(len(c) for c in cells)
    rh = pad * 2 + nlines * line_h
    # row background tint
    body.append(f'<rect x="{margin}" y="{y}" width="{W-2*margin}" height="{rh}" fill="{r["col"]}" fill-opacity="0.05"/>')
    # left accent bar
    body.append(f'<rect x="{margin}" y="{y}" width="6" height="{rh}" fill="{r["col"]}"/>')
    for ci, (name, w) in enumerate(cols):
        x = xstarts[ci]
        # cell border
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{rh}" fill="none" stroke="#cbd5e1" stroke-width="1"/>')
        lines = cells[ci]
        # vertical centering
        block_h = len(lines) * line_h
        ty0 = y + (rh - block_h) / 2 + 16
        for li, ln in enumerate(lines):
            if ci == 0:
                fw, fs, fill, anchor, tx = ("bold", 15, r["col"], "middle", x + w/2)
            elif ci == 3:
                fw, fs, fill, anchor, tx = ("normal", 12.5, "#334155", "start", x + 14)
            else:
                fw, fs, fill, anchor, tx = ("normal", 13, "#334155", "start", x + 14)
            body.append(f'<text x="{tx}" y="{ty0+li*line_h}" font-family="sans-serif" font-size="{fs}" font-weight="{fw}" fill="{fill}" text-anchor="{anchor}">{esc(ln)}</text>')
    y += rh

H = y + margin
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">'
L.extend(body)
L.append('</svg>')
open("five_cases.svg", "w", encoding="utf-8").write('\n'.join(L))
print(f"wrote five_cases.svg  ({W}x{H})")
