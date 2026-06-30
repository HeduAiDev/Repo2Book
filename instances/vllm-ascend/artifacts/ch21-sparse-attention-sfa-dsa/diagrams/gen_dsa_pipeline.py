#!/usr/bin/env python3
"""ch21 DSA 流水线：元数据期预建 qli/sas，前向期低秩 prolog → 选 top-512 → 稀疏注意力 → o_proj。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 860
TXT = "#1e293b"
SUB = "#64748b"
BUILD = "#7c3aed"  # 元数据期 紫
FWD = "#1d4ed8"    # 前向期 蓝
IDX = "#b45309"    # 索引器 橙
SP = "#15803d"     # 稀疏注意力 绿

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#475569"/></marker>')
L.append('<marker id="arp" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#7c3aed"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">DSA 一层 forward：元数据期预建索引元数据，前向期选 top-512 再算</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-size="14" fill="{SUB}">decode 与 prefill 两路对称，下图以 prefill 路为例（_forward_prefill）</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 4) + s * 0.34
        col = stroke if i == 0 else TXT
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i==0 else "normal"}" fill="{col}">{esc(t)}</text>')


def varrow(x, y1, y2, color="#475569"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#ar)"/>')


# ===== 左：元数据期（build）=====
BX = 285
L.append(f'<rect x="70" y="110" width="430" height="320" rx="12" fill="#faf5ff" stroke="{BUILD}" stroke-width="1.5" stroke-dasharray="6 4"/>')
L.append(f'<text x="{BX}" y="138" text-anchor="middle" font-size="16" font-weight="bold" fill="{BUILD}">元数据期 · AscendDSAMetadataBuilder.build</text>')

box(BX, 155, 360, 46, "#f5f3ff", BUILD, [("split_decodes_and_prefills", 14), ("按 query_len 拆 decode / prefill", 11.5)])
varrow(BX, 201, 224, BUILD)
box(BX, 224, 360, 46, "#f5f3ff", BUILD, [("build_prefill_metadata / build_decode_metadata", 13), ("各装配一份元数据", 11.5)])
varrow(BX, 270, 295, BUILD)
box(BX, 295, 390, 58, "#ede9fe", BUILD,
    [("npu_quant_lightning_indexer_metadata", 13.5),
     ("sparse_count = index_topk = 512", 12.5),
     ("sparse_mode = 3   →  qli_metadata", 12)])
varrow(BX, 353, 376, BUILD)
box(BX, 376, 390, 44, "#ede9fe", BUILD,
    [("npu_sparse_attn_sharedkv_metadata", 13), ("→  sas_metadata（稀疏注意力元数据）", 11.5)])

# ===== 右：前向期（forward）=====
FX = 920
L.append(f'<text x="{FX}" y="100" text-anchor="middle" font-size="16" font-weight="bold" fill="{FWD}">前向期 · AscendDSAImpl.forward</text>')

steps = [
    (FWD, "#eff6ff", [("hidden_states 拆 [decode | prefill]", 14), ("o_proj_input 两段各写各的", 11.5)]),
    (FWD, "#eff6ff", [("MLA 式低秩 prolog（内联）", 14), ("q = wq_b(q_norm(wq_a(h)))，kv = kv_norm(wkv(h))", 11.5), ("各自 inplace_partial_rotary_mul 加 RoPE", 11.5)]),
    (IDX, "#fff7ed", [("indexer_select_qli", 14), ("投影 q + compressor 压缩 KV", 11.5), ("量化写 indexer KV cache", 11.5)]),
    (IDX, "#ffedd5", [("npu_quant_lightning_indexer", 13.5), ("sparse_count = index_topk = 512", 12), ("→ compress_topk_idxs（top-512 索引）", 11.5)]),
    (SP, "#f0fdf4", [("npu_sparse_attn_sharedkv", 14), ("cmp_sparse_indices = compress_topk_idxs", 11.5), ("只对 top-512 个 KV 算注意力", 12)]),
    (SP, "#f0fdf4", [("o_proj：wo_a → wo_b", 14), ("（先 inplace 逆 RoPE）", 11.5)]),
]
y = 118
hs = [56, 70, 70, 64, 70, 52]
ys = []
for i, (stroke, fill, lines) in enumerate(steps):
    h = hs[i]
    box(FX, y, 430, h, fill, stroke, lines)
    ys.append((y, h))
    if i < len(steps) - 1:
        varrow(FX, y + h, y + h + 22)
    y += h + 22

# ===== 元数据 → 前向 的喂入（虚线）=====
# qli_metadata → npu_quant_lightning_indexer (step index 3)
q_y = ys[3][0] + ys[3][1] / 2
L.append(f'<path d="M 480 324 C 620 324, 640 {q_y}, 700 {q_y}" fill="none" stroke="{BUILD}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#arp)"/>')
L.append(f'<text x="600" y="{ (324+q_y)/2 - 8 }" text-anchor="middle" font-size="12" fill="{BUILD}">qli_metadata</text>')
# sas_metadata → npu_sparse_attn_sharedkv (step index 4)
s_y = ys[4][0] + ys[4][1] / 2
L.append(f'<path d="M 480 398 C 600 398, 630 {s_y}, 700 {s_y}" fill="none" stroke="{BUILD}" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#arp)"/>')
L.append(f'<text x="600" y="{ (398+s_y)/2 + 16 }" text-anchor="middle" font-size="12" fill="{BUILD}">sas_metadata</text>')

# 底注
L.append(f'<text x="{W/2}" y="820" text-anchor="middle" font-size="13.5" fill="{SUB}">「选哪些 KV」在元数据期就把索引元数据备好；前向期只做轻量索引器打分选 top-512，再把这 512 个喂给稀疏内核。</text>')

L.append('</svg>')
open("dsa_pipeline.svg", "w").write('\n'.join(L))
print("wrote dsa_pipeline.svg")
