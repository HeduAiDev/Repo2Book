#!/usr/bin/env python3
"""ch19 内核替换对照：左 vLLM CUDA 线，右昇腾 NPU 线；契约不变只换内核。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))

# 三组对应：(职责, CUDA 算子, NPU 算子)
maps = [
    ("变长前向（prefill / 混批）", "flash_attn_varlen_func", "npu_fused_infer_attention_score(_v2)"),
    ("纯解码前向", "（同上，cu_seqlens 变长）", "_npu_paged_attention"),
    ("写 KV 回分页 cache", "reshape_and_cache_flash", "_npu_reshape_and_cache"),
    ("workspace（算子暂存显存）", "内核内部自管，无显式步", "*_get_workspace 预取并缓存"),
]

W, H = 1180, 620
LB = "#0e7490"   # CUDA 蓝绿
RB = "#b45309"   # NPU 橙
TXT = "#1e293b"
BORDER = "#cbd5e1"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">把 FlashAttention 的 CUDA 内核，逐一换成 torch_npu 算子</text>')
L.append(f'<text x="{W/2}" y="64" text-anchor="middle" font-size="14.5" fill="#64748b">后端契约 AttentionBackend / Builder / Impl 不变 —— 只换内核。这正是 OOT 插件「顶替内核、不改引擎」的范式。</text>')

# 列头
col_l_x, col_r_x, col_w = 315, 730, 375
mid_label_x = 158
L.append(f'<text x="{col_l_x + col_w/2}" y="108" text-anchor="middle" font-size="17" font-weight="bold" fill="{LB}">vLLM 基座 · CUDA 线</text>')
L.append(f'<text x="{col_l_x + col_w/2}" y="128" text-anchor="middle" font-size="13" fill="#64748b">vllm/v1/attention/backends/flash_attn.py</text>')
L.append(f'<text x="{col_r_x + col_w/2}" y="108" text-anchor="middle" font-size="17" font-weight="bold" fill="{RB}">vLLM-Ascend · NPU 线</text>')
L.append(f'<text x="{col_r_x + col_w/2}" y="128" text-anchor="middle" font-size="13" fill="#64748b">vllm_ascend/attention/attention_v1.py</text>')

y = 160
box_h = 88
gap = 24
for i, (role, cuda, npu) in enumerate(maps):
    cy = y + i * (box_h + gap)
    mid_y = cy + box_h / 2
    # 职责标签（左侧）
    L.append(f'<text x="{mid_label_x}" y="{mid_y - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="{TXT}">{esc(role.split(chr(40))[0].strip())}</text>')
    extra = role[role.find("（"):] if "（" in role else ""
    if extra:
        L.append(f'<text x="{mid_label_x}" y="{mid_y + 12}" text-anchor="middle" font-size="12.5" fill="#64748b">{esc(extra)}</text>')
    # 左框 CUDA
    L.append(f'<rect x="{col_l_x}" y="{cy}" width="{col_w}" height="{box_h}" rx="9" fill="#ecfeff" stroke="{LB}" stroke-width="1.6"/>')
    L.append(f'<text x="{col_l_x + col_w/2}" y="{mid_y + 6}" text-anchor="middle" font-size="15.5" font-weight="bold" fill="{LB}">{esc(cuda)}</text>')
    # 右框 NPU
    L.append(f'<rect x="{col_r_x}" y="{cy}" width="{col_w}" height="{box_h}" rx="9" fill="#fff7ed" stroke="{RB}" stroke-width="1.6"/>')
    L.append(f'<text x="{col_r_x + col_w/2}" y="{mid_y + 6}" text-anchor="middle" font-size="15.5" font-weight="bold" fill="{RB}">{esc(npu)}</text>')
    # 箭头
    L.append(f'<line x1="{col_l_x + col_w + 6}" y1="{mid_y}" x2="{col_r_x - 8}" y2="{mid_y}" stroke="#64748b" stroke-width="2" marker-end="url(#ar)"/>')
    L.append(f'<text x="{(col_l_x + col_w + col_r_x)/2}" y="{mid_y - 8}" text-anchor="middle" font-size="12" fill="#94a3b8">换内核</text>')

# 底注
by = y + len(maps) * (box_h + gap) + 8
L.append(f'<text x="{W/2}" y="{by}" text-anchor="middle" font-size="13.5" fill="#475569">workspace 预取是 NPU 算子相对 CUDA flash 内核的特有节拍：图捕获路径先量出算子所需暂存显存并缓存，再把真正算子录进 ACL 图。</text>')

L.append('</svg>')
open("ch19-kernel-swap.svg", "w").write('\n'.join(L))
print("wrote ch19-kernel-swap.svg", W, H)
