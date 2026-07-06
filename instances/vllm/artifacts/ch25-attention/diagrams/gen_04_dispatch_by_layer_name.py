#!/usr/bin/env python3
"""04-forward-context-dispatch-by-layer-name: 按 layer_name 取数 + 写/算两算子。"""
import xml.sax.saxutils as xs

OUT = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch24-attention/diagrams/04-forward-context-dispatch-by-layer-name.svg"


def esc(s):
    return xs.escape(str(s))


W, H = 1040, 600
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ad" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#8b5cf6"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, lines, fs=12.5, tcol="#0f172a", weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs * 0.35
    for i, t in enumerate(lines):
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+3)}" font-family="sans-serif" '
                 f'font-size="{fs}" font-weight="{weight}" text-anchor="middle" fill="{tcol}">{esc(t)}</text>')


def label(x, y, t, fs=12, col="#475569", weight="normal", anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
             f'font-weight="{weight}" text-anchor="{anchor}" fill="{col}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, marker="a", col="#475569", dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
             f'stroke-width="2"{d} marker-end="url(#{marker})"/>')


label(W/2, 30, "按 layer_name 分发：先写 cache，再算+读", 18, "#0f172a", "bold")

# forward_context 仓库（右侧）
FX = 760
L.append(f'<rect x="{FX}" y="80" width="260" height="420" rx="10" fill="#faf5ff" stroke="#e9d5ff"/>')
label(FX + 130, 102, "forward_context", 13, "#7c3aed", "bold")
label(FX + 130, 120, "model_runner 那头按 layer_name 装", 10.5, "#64748b")
ctx_items = [
    ("attn_metadata", '{"layer.0.attn": md, ...}'),
    ("slot_mapping", '{"layer.0.attn": tensor, ...}'),
    ("no_compile_layers", '{"layer.0.attn": Attention}'),
    ("→ attn_layer.kv_cache", '真实 paged 显存'),
]
cy = 140
for k, v in ctx_items:
    box(FX + 16, cy, 228, 56, "#f3e8ff", "#a855f7", [k, v], fs=11)
    cy += 70

# 左侧主时序链
def step(y, lines, fill, stroke, fs=12.5, weight="normal", w=300, x=60):
    box(x, y, w, 54, fill, stroke, lines, fs=fs, weight=weight)
    return y

x0 = 60
box(x0, 80, 300, 50, "#e0e7ff", "#6366f1",
    ["Attention.forward(q, k, v)", "层 prefix = layer.0.attn"], fs=12.5, weight="bold")
box(x0, 160, 300, 50, "#dbeafe", "#3b82f6",
    ["reshape q / k / v", "use_direct_call → 直调"], fs=12)

# 写算子
box(x0, 250, 300, 56, "#fee2e2", "#ef4444",
    ["unified_kv_cache_update(k, v,", "  layer_name)  —— 写"], fs=12, weight="bold")
box(x0, 326, 300, 40, "#fef2f2", "#dc2626",
    ["impl.do_kv_cache_update（写 KV）"], fs=11.5)

# 算+读算子
box(x0, 410, 300, 56, "#dcfce7", "#22c55e",
    ["unified_attention_with_output(", "  query, ..., layer_name)  —— 算+读"], fs=11.5, weight="bold")
box(x0, 486, 300, 40, "#f0fdf4", "#16a34a",
    ["impl.forward（读 paged KV 算注意力）"], fs=11.5)

arrow(210, 130, 210, 160)
arrow(210, 210, 210, 250)
arrow(210, 306, 210, 326)
arrow(210, 366, 210, 410)
arrow(210, 466, 210, 486)

# dummy_dep 串写→算
L.append(f'<path d="M 360 290 Q 430 350 360 430" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#a)"/>')
label(445, 360, "kv_cache_dummy_dep", 11, "#b45309", "bold")
label(445, 376, "串一条写→算假依赖", 10.5, "#92400e")
label(445, 392, "防 torch.compile 乱序", 10.5, "#92400e")

# get_attention_context 取数（从两个算子虚线连到 forward_context）
label(575, 250, "get_attention_context(layer_name)", 11.5, "#7c3aed", "bold")
arrow(360, 278, FX, 200, marker="ad", col="#8b5cf6", dash=True)
arrow(360, 438, FX, 230, marker="ad", col="#8b5cf6", dash=True)
label(575, 470, "按 layer_name 取本层 kv_cache /", 11, "#7c3aed")
label(575, 486, "attn_metadata / slot_mapping", 11, "#7c3aed")

label(W/2, H - 16, "一次前向里几十个注意力层各有自己的 KV cache 与 metadata；layer_name 是把它们串起来的键", 12, "#475569")

L.append('</svg>')
open(OUT, "w").write('\n'.join(L))
print("ok", OUT)
