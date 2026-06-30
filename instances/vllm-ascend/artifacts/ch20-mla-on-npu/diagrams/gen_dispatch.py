#!/usr/bin/env python3
"""ch20 forward 分流泳道：fused_qkv_a_proj 拆 q_c/kv_no_split，按 decode/prefill 走两条算子路径。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1280, 720
TXT = "#1e293b"
SUB = "#64748b"
DEC = "#15803d"   # decode 绿
PRE = "#b45309"   # prefill 橙
COM = "#1d4ed8"   # 公共蓝

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">forward 的真实分流：decode 走吸收 MQA，prefill 走解压 MHA</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-size="14" fill="{SUB}">同一个 forward 内按 has_decode / has_prefill 双路 —— forward_mqa / forward_mha 在本版只是基类概念名，不被调用</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 4) + s * 0.34
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i==0 else "normal"}" fill="{stroke if i==0 else TXT}">{esc(t)}</text>')


def varrow(x, y1, y2, label=None, color="#64748b"):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" stroke-width="2" marker-end="url(#ar)"/>')
    if label:
        L.append(f'<text x="{x+10}" y="{(y1+y2)/2+4}" font-size="12" fill="#475569">{esc(label)}</text>')


CX = W / 2
# 顶部公共
box(CX, 92, 230, 42, "#eff6ff", COM, [("hidden_states", 15)])
varrow(CX, 134, 162)
box(CX, 162, 360, 50, "#eff6ff", COM, [("fused_qkv_a_proj → q_c, kv_no_split", 14.5), ("（q_a_layernorm 归一化 q_c）", 12)])
varrow(CX, 212, 244)
box(CX, 244, 360, 48, "#eff6ff", COM, [("split_decodes_and_prefills", 14.5), ("query_len ≤ decode_threshold → decode", 12)])

# 分叉
DX, PX = 330, 950
L.append(f'<line x1="{CX}" y1="292" x2="{CX}" y2="312" stroke="#64748b" stroke-width="2"/>')
L.append(f'<line x1="{DX}" y1="312" x2="{PX}" y2="312" stroke="#64748b" stroke-width="2"/>')
varrow(DX, 312, 348, color=DEC)
varrow(PX, 312, 348, color=PRE)

# decode 道
L.append(f'<text x="{DX}" y="338" text-anchor="middle" font-size="15" font-weight="bold" fill="{DEC}">decode 道 · MQA 吸收</text>')
dsteps = [
    ("_q_proj_and_k_up_proj", "bmm(q_nope, W_UK_T) → ql_nope"),
    ("exec_kv_decode", "npu_kv_rmsnorm_rope_cache 写 cache"),
    ("_forward_decode", "FIA_score_v2(ql, kv_c, kv_c)"),
    ("_v_up_proj", "latent 输出 → V"),
]
y = 348
for i, (a, b) in enumerate(dsteps):
    box(DX, y, 340, 50, "#f0fdf4", DEC, [(a, 14.5), (b, 12)])
    if i < len(dsteps) - 1:
        varrow(DX, y + 50, y + 78, color=DEC)
    y += 78

# prefill 道
L.append(f'<text x="{PX}" y="338" text-anchor="middle" font-size="15" font-weight="bold" fill="{PRE}">prefill 道 · MHA 解压</text>')
psteps = [
    ("q_proj（满维 q_nope/q_pe）", "不吸收"),
    ("exec_kv_prefill", "同算子 is_output_kv=True"),
    ("kv_b_proj 显式解压", "→ k_nope, value"),
    ("_forward_prefill + _compute_prefill_context", "FIA(TND) + 历史 chunk 合并"),
]
y = 348
for i, (a, b) in enumerate(psteps):
    box(PX, y, 360, 50, "#fff7ed", PRE, [(a, 13.5), (b, 12)])
    if i < len(psteps) - 1:
        varrow(PX, y + 50, y + 78, color=PRE)
    y += 78

# 汇合
my = 660
box(CX, my, 380, 48, "#eff6ff", COM, [("o_proj_input  →  o_proj", 15), ("decode 段写 [:nd]，prefill 段写 [nd:]", 12)])
# 汇流箭头
L.append(f'<line x1="{DX}" y1="660" x2="{DX}" y2="640" stroke="{DEC}" stroke-width="2"/>')
L.append(f'<line x1="{PX}" y1="660" x2="{PX}" y2="640" stroke="{PRE}" stroke-width="2"/>')
L.append(f'<line x1="{DX}" y1="640" x2="{PX}" y2="640" stroke="#94a3b8" stroke-width="2"/>')
varrow(CX, 640, 660)

L.append('</svg>')
open("dispatch.svg", "w").write('\n'.join(L))
print("wrote dispatch.svg")
