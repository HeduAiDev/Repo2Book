#!/usr/bin/env python3
"""ch19 五态机分流表：行=五态，列=[batch 形态 | block_table 来源 | KV 来源 | forward_impl 落点 | sparse_mode]。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


# 列定义：标题 + 宽度
cols = [
    ("AscendAttentionState", 215),
    ("batch 形态（一句话）", 250),
    ("block_table 来源", 175),
    ("KV 来源", 165),
    ("forward_impl 落点", 215),
    ("sparse_mode ✱", 150),
]
rows = [
    ("PrefillNoCache", "纯新 prefill，无历史 KV", "None", "本步算的 K/V", "fused-infer", "3 因果"),
    ("PrefillCacheHit", "带前缀缓存的 prefill", "block_tables[:bs]", "paged cache（view）", "fused-infer", "3 因果"),
    ("DecodeOnly", "纯单 token 解码", "block_tables", "paged cache（view）", "paged ✦ / fused 回落", "—（PA）/ 3"),
    ("ChunkedPrefill", "prefill 与 decode 混批", "block_tables", "paged cache（view）", "fused-infer", "3 因果"),
    ("SpecDecoding", "投机解码，每 req 多 token", "block_tables", "paged cache（view）", "fused-infer", "3 因果"),
]

# 颜色
HEAD_BG = "#1e3a8a"
HEAD_FG = "#ffffff"
ROW_BG = ["#f1f5f9", "#ffffff"]
DECODE_BG = "#fef3c7"  # DecodeOnly 行高亮（唯一可能走 paged）
BORDER = "#cbd5e1"
TXT = "#1e293b"
ACCENT = "#b45309"

pad_x = 14
row_h = 46
head_h = 52
title_h = 64
x0 = 20
y0 = title_h + 20

col_x = [x0]
for _, wd in cols:
    col_x.append(col_x[-1] + wd)
W = col_x[-1] + x0
H = y0 + head_h + row_h * len(rows) + 92

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">五态机：当前 batch 落在哪个状态，就走哪条 torch_npu 算子路径</text>')
L.append(f'<text x="{W/2}" y="58" text-anchor="middle" font-size="14" fill="#64748b">行 = AscendAttentionState 五态　·　✦ 唯有 DecodeOnly 在满足 using_paged_attention 多门槛 + 无滑窗时走 paged，其余皆 fused-infer</text>')

# 表头
hy = y0
for i, (name, wd) in enumerate(cols):
    cx = col_x[i]
    L.append(f'<rect x="{cx}" y="{hy}" width="{wd}" height="{head_h}" fill="{HEAD_BG}" stroke="{BORDER}"/>')
    L.append(f'<text x="{cx + wd/2}" y="{hy + head_h/2 + 5}" text-anchor="middle" font-size="14.5" font-weight="bold" fill="{HEAD_FG}">{esc(name)}</text>')

# 数据行
for r, row in enumerate(rows):
    ry = y0 + head_h + r * row_h
    is_decode = row[0] == "DecodeOnly"
    bg = DECODE_BG if is_decode else ROW_BG[r % 2]
    for c, val in enumerate(row):
        cx = col_x[c]
        wd = cols[c][1]
        L.append(f'<rect x="{cx}" y="{ry}" width="{wd}" height="{row_h}" fill="{bg}" stroke="{BORDER}"/>')
        if c == 0:
            fw, fs, fill = "bold", 15, (ACCENT if is_decode else TXT)
        elif c == 4 and is_decode:
            fw, fs, fill = "bold", 13.5, ACCENT
        else:
            fw, fs, fill = "normal", 13.5, TXT
        L.append(f'<text x="{cx + wd/2}" y="{ry + row_h/2 + 5}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{fill}">{esc(val)}</text>')

# 脚注
fy = y0 + head_h + row_h * len(rows) + 30
L.append(f'<text x="{x0}" y="{fy}" font-size="13.5" fill="#475569">KV「paged cache（view）」= 把 (2,num_blocks,block_size,...) 的 cache 张量 view 成 (num_block, block_size, -1) 喂算子；本步新算的 K/V 已先经 reshape_and_cache 写回。</text>')
L.append(f'<text x="{x0}" y="{fy + 22}" font-size="13.5" fill="#475569">✱ sparse_mode 实由 attn_metadata.causal 与 sliding_window 决定（0 非因果 / 3 因果 / 4 滑窗），与五态正交（见 §19.6）；</text>')
L.append(f'<text x="{x0}" y="{fy + 42}" font-size="13.5" fill="#475569">上表列的是标准因果 MHA 情形——非因果 pooling 模型即便落在 ChunkedPrefill 也走 sparse_mode=0。</text>')

L.append('</svg>')
open("ch19-state-machine.svg", "w").write('\n'.join(L))
print("wrote ch19-state-machine.svg", W, H)
