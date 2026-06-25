#!/usr/bin/env python3
"""四级模块嵌套与 prefix 串联（树形图）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1300, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

GREEN = "#dcfce7"
GREEN_E = "#16a34a"
BLUE = "#dbeafe"
BLUE_E = "#2563eb"
AMBER = "#fef3c7"
AMBER_E = "#d97706"
GRAY = "#f1f5f9"
GRAY_E = "#94a3b8"


def box(x, y, w, h, title, prefix, fill, edge):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1.6"/>'
    )
    L.append(
        f'<text x="{x + w / 2}" y="{y + 20}" font-size="15" font-weight="bold" '
        f'text-anchor="middle" fill="#111827" font-family="sans-serif">{esc(title)}</text>'
    )
    if prefix:
        L.append(
            f'<text x="{x + w / 2}" y="{y + 38}" font-size="11.5" '
            f'text-anchor="middle" fill="#475569" font-family="monospace">{esc(prefix)}</text>'
        )


def arrow(x1, y1, x2, y2):
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>'
    )


# Level 1
box(440, 30, 300, 48, "LlamaForCausalLM", "prefix=''", GREEN, GREEN_E)
# Level 2: model + lm_head
box(250, 140, 300, 48, "LlamaModel", "model", BLUE, BLUE_E)
box(640, 140, 230, 44, "ParallelLMHead", "lm_head", GRAY, GRAY_E)
arrow(560, 78, 420, 140)
arrow(640, 78, 730, 140)
# Level 2 children of model: embed, layers, norm
box(60, 248, 220, 44, "VocabParallelEmbedding", "model.embed_tokens", GRAY, GRAY_E)
box(300, 248, 240, 48, "LlamaDecoderLayer ×N", "model.layers.{i}", BLUE, BLUE_E)
box(570, 248, 150, 44, "RMSNorm", "model.norm", GRAY, GRAY_E)
arrow(330, 188, 170, 248)
arrow(400, 188, 410, 248)
arrow(470, 188, 600, 248)
# Level 3 children of decoder layer
box(40, 360, 165, 48, "input_layernorm", "...layers.0.input_layernorm", GRAY, GRAY_E)
box(225, 360, 175, 48, "LlamaAttention", "...layers.0.self_attn", AMBER, AMBER_E)
box(420, 360, 200, 48, "post_attention_layernorm", "...post_attention_layernorm", GRAY, GRAY_E)
box(640, 360, 165, 48, "LlamaMLP", "...layers.0.mlp", AMBER, AMBER_E)
for cx in (122, 312, 520, 722):
    arrow(420, 296, cx, 360)
# Level 4 children of attention
ay = 480
box(40, ay, 175, 56, "QKVParallelLinear", "...self_attn.qkv_proj", GREEN, GREEN_E)
box(232, ay, 130, 48, "rotary_emb", "...self_attn (rope)", GRAY, GRAY_E)
box(378, ay, 165, 56, "Attention", "...self_attn.attn", GREEN, GREEN_E)
box(560, ay, 175, 56, "RowParallelLinear", "...self_attn.o_proj", GREEN, GREEN_E)
for cx in (127, 297, 460, 647):
    arrow(312, 408, cx, ay)
# Level 4 children of MLP
my = 480
box(770, my, 215, 56, "MergedColumnParallelLinear", "...mlp.gate_up_proj", GREEN, GREEN_E)
box(1000, my, 120, 48, "SiluAndMul", "...mlp.act_fn", GRAY, GRAY_E)
box(1135, my, 130, 56, "RowParallel", "down_proj", GREEN, GREEN_E)
for cx in (877, 1060, 1200):
    arrow(722, 408, cx, my)

# prefix annotation banner
L.append(
    f'<rect x="40" y="600" width="1220" height="120" rx="8" '
    f'fill="#fffbeb" stroke="#d97706" stroke-width="1.4"/>'
)
L.append(
    '<text x="60" y="628" font-size="14.5" font-weight="bold" fill="#92400e" '
    'font-family="sans-serif">一根 prefix 串起两件事</text>'
)
notes = [
    "构造时逐级拼接：model.layers.0.self_attn.qkv_proj —— 这正是 checkpoint 里该权重的全名（权重匹配靠它）。",
    "走到 Attention 那一层，prefix 又被当作 layer_name 注册进 static_forward_context —— 运行期按它取回本层的 kv_cache 与 attn_metadata。",
    "所以：同一根 prefix，既定位「静态权重」，又定位「动态 KV 缓存」。",
]
for i, n in enumerate(notes):
    L.append(
        f'<text x="60" y="{652 + i * 22}" font-size="12.5" fill="#78350f" '
        f'font-family="sans-serif">{esc(n)}</text>'
    )

L.append("</svg>")
open(
    "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch22-model-definitions/diagrams/01-module-nesting.svg",
    "w",
    encoding="utf-8",
).write("\n".join(L))
print("wrote 01-module-nesting.svg")
