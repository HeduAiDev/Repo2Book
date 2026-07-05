#!/usr/bin/env python3
"""ch20 chunked-context 在线 softmax 合并：长 context 分块各算注意力，连同新 token 段一起喂 npu_attention_update。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 660
TXT = "#1e293b"
SUB = "#64748b"
NEW = "#15803d"
CHK = "#b45309"
MRG = "#7c3aed"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">chunked-context：分块算历史注意力，再用 LSE 在线 softmax 合并</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-size="14" fill="{SUB}">长 context 切 num_chunks 块（每块 workspace 有界）→ 各出 (out, lse) → npu_attention_update 一把合并（等价基座 merge_attn_states）</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 4) + s * 0.34
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i==0 else "normal"}" fill="{stroke if i==0 else TXT}">{esc(t)}</text>')


# 新 token 段（prefix）
NX = 230
box(NX, 110, 300, 64, "#f0fdf4", NEW,
    [("新 token 段（当前 query）", 14), ("_forward_prefill 已算出", 12), ("prefix_out, prefix_lse", 12.5)])

# chunk 行
chunks = ["chunk 0", "chunk 1", "chunk …", "chunk n-1"]
cy0 = 210
for i, c in enumerate(chunks):
    cy = cy0 + i * 80
    # read
    box(NX - 70, cy, 200, 56, "#fff7ed", CHK,
        [(f"读 {c} 的 kv_c", 13.5), ("从分页 cache", 11.5)])
    L.append(f'<line x1="{NX+30}" y1="{cy+28}" x2="{NX+128}" y2="{cy+28}" stroke="#64748b" stroke-width="2" marker-end="url(#ar)"/>')
    box(NX + 240, cy, 220, 56, "#fff7ed", CHK,
        [("kv_b_proj 解压 → k_nope,v", 12.5), ("FIA → (out_i, lse_i)", 12.5)])

# 列表汇聚框
LX = 720
box(LX, 110, 260, 64, "#f5f3ff", MRG,
    [("out_list / lse_list", 14.5), ("[ prefix, out_0, …, out_{n-1} ]", 12)])
# 箭头：prefix → list（进 out_list 左边）
L.append(f'<line x1="{NX+150}" y1="120" x2="{LX-132}" y2="128" stroke="{NEW}" stroke-width="2" marker-end="url(#ar)"/>')
# 箭头：各 chunk 的 (out,lse) 汇入 out_list 底边（扇入，避开下方 merge 框）
for i in range(len(chunks)):
    cy = cy0 + i * 80 + 28
    ex = LX - 60 + i * 18
    L.append(f'<line x1="{NX+352}" y1="{cy}" x2="{ex}" y2="176" stroke="{CHK}" stroke-width="1.4" marker-end="url(#ar)" opacity="0.65"/>')

# merge
varrow_x = LX
L.append(f'<line x1="{LX}" y1="174" x2="{LX}" y2="222" stroke="{MRG}" stroke-width="2.4" marker-end="url(#ar)"/>')
box(LX, 222, 300, 70, "#ede9fe", MRG,
    [("npu_attention_update", 15.5),
     ("在线 softmax：按 lse_max 数值稳定加权", 12)])
L.append(f'<line x1="{LX}" y1="292" x2="{LX}" y2="332" stroke="{MRG}" stroke-width="2.4" marker-end="url(#ar)"/>')
box(LX, 332, 240, 48, "#f5f3ff", MRG, [("output_final (T,H,D)", 14.5)])

# 公式
fy = 470
L.append(f'<rect x="700" y="{fy-30}" width="540" height="120" rx="10" fill="#faf5ff" stroke="#ddd6fe" stroke-width="1.4"/>')
L.append(f'<text x="970" y="{fy}" text-anchor="middle" font-size="14" font-weight="bold" fill="{MRG}">在线 softmax 合并（数值稳定）</text>')
L.append(f'<text x="970" y="{fy+30}" text-anchor="middle" font-size="15" fill="{TXT}">out = Σᵢ exp(lseᵢ − lse_max)·outᵢ ⁄ Σᵢ exp(lseᵢ − lse_max)</text>')
L.append(f'<text x="970" y="{fy+58}" text-anchor="middle" font-size="13" fill="{SUB}">lse_max = maxᵢ lseᵢ；先减最大值再取指数，避免溢出</text>')

# 左下注解
L.append(f'<text x="{NX}" y="560" text-anchor="middle" font-size="13" fill="{SUB}">分块保证每块 workspace 有界：</text>')
L.append(f'<text x="{NX}" y="584" text-anchor="middle" font-size="12.5" fill="{SUB}">max_context_chunk = workspace ⁄ 带历史的 prefill 数</text>')
L.append(f'<text x="{NX}" y="606" text-anchor="middle" font-size="12.5" fill="{SUB}">→ round_down 到 block_size；num_chunks = cdiv(max_ctx, chunk)</text>')

L.append('</svg>')
open("chunked.svg", "w").write('\n'.join(L))
print("wrote chunked.svg")
